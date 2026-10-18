"""Search on training-only time folds; freeze a candidate before one test evaluation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.common import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, choose_review_threshold,
    classification_metrics, load_tickets, make_features, predictions_frame, review_metrics,
)
from scripts.report import render_saved_report
from scripts.train import build_models, write_json


THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.05), 2)


def build_candidates():
    """Fixed modest search, specified before opening the external test dataset."""
    candidates = build_models()
    estimators = {}
    for depth, leaf in [(None, 1), (8, 3), (6, 8)]:
        estimators[f"rf_depth{depth}_leaf{leaf}"] = RandomForestClassifier(
            n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
            max_features=1.0, random_state=42, n_jobs=1,
        )
    for leaf in [3, 10]:
        estimators[f"extra_trees_leaf{leaf}"] = ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=leaf, random_state=42, n_jobs=1,
        )
    for leaves, regularization in [(3, 1.0), (7, 1.0), (15, 5.0)]:
        estimators[f"hgb_leaves{leaves}_l2{regularization}"] = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=leaves,
            min_samples_leaf=15, l2_regularization=regularization,
            early_stopping=False, random_state=42,
        )
    for depth in [1, 2]:
        estimators[f"gradient_boosting_depth{depth}"] = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.04, max_depth=depth,
            min_samples_leaf=12, random_state=42,
        )
    for depth in [1, 2, 3]:
        estimators[f"decision_tree_depth{depth}"] = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=12, random_state=42,
        )
    for regularization in [0.5, 2.0, 8.0]:
        estimators[f"rbf_svm_c{regularization}"] = SVC(
            C=regularization, kernel="rbf", probability=True, random_state=42,
        )
    for name, estimator in estimators.items():
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ])
        candidates[name] = Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])
    for knots in [3, 5]:
        preprocessing = ColumnTransformer([
            ("numeric", Pipeline([("spline", SplineTransformer(n_knots=knots, degree=2)),
                                  ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ])
        candidates[f"spline_logistic_knots{knots}"] = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(C=0.2, max_iter=2000, random_state=42)),
        ])
    return candidates


def error_counts(y, probability, threshold=0.5):
    y, probability = np.asarray(y), np.asarray(probability)
    prediction = probability >= threshold
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    return {"false_positives": fp, "false_negatives": fn, "errors": fp + fn,
            "accuracy": float(np.mean(y == prediction))}


def select_candidate(rows, reference):
    """Require both observed error counts to be no worse than the old model."""
    eligible = [row for row in rows if row["false_positives"] <= reference["false_positives"]
                and row["false_negatives"] <= reference["false_negatives"]]
    if not eligible:
        raise ValueError("No candidate satisfies both training-CV error constraints")
    return min(eligible, key=lambda row: (
        row["errors"], row["brier_score"], abs(row["classification_threshold"] - 0.5), row["model"],
    ))


def validate_training(frame):
    if frame.fault_confirmed.isna().any() or set(frame.fault_confirmed.unique()) != {0, 1}:
        raise ValueError("Training requires known labels from both classes")
    if len(frame) < 40:
        raise ValueError("At least 40 training rows are required for time folds")


def validate_test(training, testing):
    if testing.fault_confirmed.isna().any():
        raise ValueError("Test rows must have known labels")
    if set(training.ticket_id) & set(testing.ticket_id):
        raise ValueError("Training and test ticket IDs must not overlap")
    train_end = pd.to_datetime(training.ticket_time, utc=True).max()
    test_start = pd.to_datetime(testing.ticket_time, utc=True).min()
    if train_end >= test_start:
        raise ValueError("Every test timestamp must be after training")


def experiment_summary(selection, comparison):
    old, new = comparison["old_model"], comparison["optimized_model"]
    reference, selected = selection["reference_training_oof"], selection["selected_training_oof"]
    verdict = "测试样本观察到零错误，仍须独立验收" if new["errors"] == 0 else "未达到两类错误都为零的目标"
    return f"""# 模型优化试验结果

**{verdict}。本轮候选没有替换原模型。**

| 数据范围 | 模型 | 空跑误报 FP | 真故障漏判 FN | 总错误 |
| --- | --- | ---: | ---: | ---: |
| 训练集内时间验证 {selection['training_oof_rows']} 条 | 旧随机森林 | {reference['false_positives']} | {reference['false_negatives']} | {reference['errors']} |
| 训练集内时间验证 {selection['training_oof_rows']} 条 | 训练内选中候选 | {selected['false_positives']} | {selected['false_negatives']} | {selected['errors']} |
| 测试集 {comparison['test_rows']} 条 | 旧随机森林 | {old['false_positives']} | {old['false_negatives']} | {old['errors']} |
| 测试集 {comparison['test_rows']} 条 | 候选 {selection['model']} | {new['false_positives']} | {new['false_negatives']} | {new['errors']} |

测试准确率：旧模型 {old['accuracy']:.2%}，候选 {new['accuracy']:.2%}；测试两类错误均不变差：{comparison['both_error_counts_not_worse_on_test']}。

## 试验边界

- 仅使用固定的训练集，比较 {selection['candidate_models']} 种模型配置与 {selection['thresholds_per_model']} 个阈值，共 {selection['candidate_models'] * selection['thresholds_per_model']} 种组合。训练时间验证零错误组合数为 {selection['zero_error_training_oof_candidates']}。
- 选择规则先要求 FP、FN 都不高于旧模型的训练内结果，再最小化总错误；平局比较 Brier 分数、距 0.50 的距离和名称。所有选择在打开测试集之前保存到 selection.json。
- 选中分类阈值为 {selection['classification_threshold']:.2f}。测试集只评估这一个已冻结候选；观察测试回退后没有继续调参或挑选另一个模型。
- 模型与阈值搜索会使训练内交叉验证结果偏乐观；本次测试表现说明这种提升没有稳定迁移。没有把训练集上的高分当作零误判证明。
- 验收验证集未读取、未重新推理、未用于训练；原数据、原模型与既有本地验收结果均保留。
- 没有伪造样本、改标签、随机重分数据或用人工复核代替错误计数。仅靠这一周的有限特征，无法保证未来每条工单都判对。

## 复现与验收

运行 `uv run --locked python -m scripts.optimize` 可复现实验。`training_search.csv` 保存全部组合；`training_oof_predictions.csv` 保存训练内逐单预测；`test_comparison.json` 保存旧新模型的测试对比。候选保存在 `artifacts/optimized/`，默认预测仍使用原模型。

如果用户需要验收该候选，单独运行：

```bash
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/optimized/fault_model.joblib --output-dir outputs/user_acceptance/optimized
```

候选模型 SHA-256：{comparison['model_sha256']}。
"""


def optimize(train_path, test_path, output_dir, artifact_dir):
    # Default workflow reads only train/test and has no separate validation argument.
    training, train_quality = load_tickets(train_path)
    validate_training(training)
    training = training.assign(_utc=pd.to_datetime(training.ticket_time, utc=True)).sort_values(
        ["_utc", "ticket_id"]).drop(columns="_utc").reset_index(drop=True)
    x, y = make_features(training), training.fault_confirmed.astype(int)
    candidates = build_candidates()
    folds = list(TimeSeriesSplit(n_splits=3).split(x))
    for fit, check in folds:
        if y.iloc[fit].nunique() != 2:
            raise ValueError("Every time-fold training portion must contain both classes")
        if pd.Timestamp(training.ticket_time.iloc[fit[-1]]) >= pd.Timestamp(training.ticket_time.iloc[check[0]]):
            raise ValueError("Time-fold boundaries must not split simultaneous tickets")
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows, fold_rows, oof = [], [], {}
    for name, prototype in candidates.items():
        probability = np.full(len(y), np.nan)
        for fold, (fit, check) in enumerate(folds, 1):
            candidate = clone(prototype).fit(x.iloc[fit], y.iloc[fit])
            probability[check] = candidate.predict_proba(x.iloc[check])[:, 1]
            fold_rows.append({"model": name, "fold": fold, "fit_rows": len(fit),
                              "check_rows": len(check), "classification_threshold": 0.5,
                              **error_counts(y.iloc[check], probability[check])})
        covered = np.isfinite(probability)
        oof[name] = probability
        score = classification_metrics(y[covered], probability[covered])
        for threshold in THRESHOLDS:
            candidate_rows.append({"model": name, "classification_threshold": float(threshold),
                                   "brier_score": score["brier_score"],
                                   **error_counts(y[covered], probability[covered], threshold)})
        print(f"Training CV complete: {name}", flush=True)
    reference = next(row for row in candidate_rows
                     if row["model"] == "random_forest" and row["classification_threshold"] == 0.5)
    chosen = select_candidate(candidate_rows, reference)
    name, threshold = chosen["model"], chosen["classification_threshold"]
    covered = np.isfinite(oof[name])
    review_threshold, review_rows = choose_review_threshold(y[covered], oof[name][covered], recall_floor=0.95)
    fitted = clone(candidates[name]).fit(x, y)
    selection = {
        "model": name, "classification_threshold": threshold, "review_threshold": review_threshold,
        "objective": "Minimize FP + FN with both training-OOF counts no worse than original random_forest at 0.50",
        "tie_breakers": ["Brier score", "distance to 0.50", "model name"],
        "training_oof_rows": int(covered.sum()), "candidate_models": len(candidates),
        "thresholds_per_model": len(THRESHOLDS), "random_seed": 42,
        "reference_training_oof": reference, "selected_training_oof": chosen,
        "zero_error_training_oof_candidates": sum(row["errors"] == 0 for row in candidate_rows),
        "test_used_for_selection": False, "final_validation_evaluated": False,
        "train_quality": train_quality,
        "warning": "Search-selected CV results are optimistic; zero observed errors would not guarantee future zero errors.",
    }
    metadata = {"model_name": name, "classification_threshold": threshold, "review_threshold": review_threshold,
                "features": FEATURES, "training_input_sha256": train_quality["input_sha256"],
                "sklearn_version": sklearn.__version__, "selection_scope": "training_only_timeseries_cv",
                "mode": "advisory_only", "variant": "optimized_v2", "review_reference_floor": 0.95}
    # Persist frozen choices before reading any test features or labels.
    model_path = artifact_dir / "fault_model.joblib"
    joblib.dump({**metadata, "pipeline": fitted}, model_path, compress=3)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write_json(artifact_dir / "model_metadata.json", metadata)
    write_json(output_dir / "selection.json", selection)
    pd.DataFrame(candidate_rows).sort_values(["errors", "brier_score", "model"]).to_csv(
        output_dir / "training_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output_dir / "training_folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review_rows).to_csv(output_dir / "training_review_thresholds.csv", index=False, encoding="utf-8-sig")
    oof_frame = training.loc[covered, ["ticket_id", "ticket_time", "fault_confirmed"]].copy()
    for model_name, probabilities in oof.items():
        oof_frame[model_name] = probabilities[covered]
    oof_frame.to_csv(output_dir / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")

    testing, test_quality = load_tickets(test_path)
    validate_test(training, testing)
    test_probability = fitted.predict_proba(make_features(testing))[:, 1]
    # Original model is rebuilt from the same fixed training rows for a fair paired comparison.
    original = clone(candidates["random_forest"]).fit(x, y)
    old_probability = original.predict_proba(make_features(testing))[:, 1]
    metrics = {"data_quality": test_quality, "classification_threshold": threshold,
               "classification": classification_metrics(testing.fault_confirmed, test_probability, threshold),
               "review": review_metrics(testing.fault_confirmed, test_probability < review_threshold),
               "review_threshold": review_threshold, "model": name, "model_sha256": metadata["model_sha256"],
               "evaluation_role": "test", "review_reference_floor": 0.95,
               "optimization_goal": "zero_errors_both_classes",
               "previous_model_error_counts": error_counts(testing.fault_confirmed, old_probability)}
    write_json(output_dir / "metrics.json", metrics)
    predictions = predictions_frame(testing, test_probability, review_threshold, threshold)
    predictions["fault_confirmed"] = testing.fault_confirmed
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    comparison = {
        "test_rows": len(testing), "model_sha256": metadata["model_sha256"],
        "old_model": error_counts(testing.fault_confirmed, old_probability),
        "optimized_model": error_counts(testing.fault_confirmed, test_probability, threshold),
        "always_fault_baseline": error_counts(testing.fault_confirmed, np.ones(len(testing))),
        "classification_threshold": threshold, "final_validation_evaluated": False,
    }
    comparison["zero_error_goal_reached_on_test"] = comparison["optimized_model"]["errors"] == 0
    comparison["both_error_counts_not_worse_on_test"] = all(
        comparison["optimized_model"][key] <= comparison["old_model"][key]
        for key in ["false_positives", "false_negatives"])
    write_json(output_dir / "test_comparison.json", comparison)
    (output_dir / "comparison.md").write_text(experiment_summary(selection, comparison), encoding="utf-8")
    render_saved_report(output_dir)
    print(json.dumps({"selection": chosen, "test_comparison": comparison}, ensure_ascii=False, indent=2))
    return selection, comparison


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=Path("datasets/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("datasets/test.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/optimization"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts/optimized"), type=Path)
    args = parser.parse_args()
    try:
        optimize(args.train, args.test, args.output_dir, args.artifact_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Optimization aborted: {exc}\n")


if __name__ == "__main__":
    main()
