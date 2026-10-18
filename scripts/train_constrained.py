"""Maximize true-negative detection subject to zero observed training false negatives."""

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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from scripts.common import FEATURES, classification_metrics, load_tickets, make_features, predictions_frame, review_metrics
from scripts.fault_features import FaultInteractions, INTERACTION_CATEGORICAL, INTERACTION_NUMERIC
from scripts.optimize import error_counts, validate_test, validate_training
from scripts.report import console_summary, render_saved_report
from scripts.train import build_models, write_json
from scripts.train_recall import split_training_calibration


def build_constrained_candidates():
    """Fixed search declared before external test access; no resampling or fake data."""
    candidates = build_models()
    estimators = {}
    for weight in [1, 4, 12]:
        for c in [0.1, 1.0, 10.0]:
            estimators[f"interactions_logistic_c{c}_faultweight{weight}"] = LogisticRegression(
                C=c, class_weight={0: 1, 1: weight}, max_iter=3000, random_state=42)
    for weight in [1, 4]:
        for c in [0.5, 2.0]:
            estimators[f"interactions_rbf_c{c}_faultweight{weight}"] = SVC(
                C=c, class_weight={0: 1, 1: weight}, probability=True, random_state=42)
        for leaf in [3, 12]:
            estimators[f"interactions_rf_leaf{leaf}_faultweight{weight}"] = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=leaf, max_features=0.8,
                class_weight={0: 1, 1: weight}, random_state=42, n_jobs=1)
        for leaves in [3, 7]:
            estimators[f"interactions_hgb_leaves{leaves}_faultweight{weight}"] = HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.05, max_leaf_nodes=leaves,
                min_samples_leaf=15, l2_regularization=5.0,
                class_weight={0: 1, 1: weight}, early_stopping=False, random_state=42)
    members = [
        ("logistic", estimators["interactions_logistic_c1.0_faultweight4"]),
        ("forest", estimators["interactions_rf_leaf12_faultweight4"]),
        ("boosting", estimators["interactions_hgb_leaves3_faultweight4"]),
    ]
    estimators["interactions_soft_voting"] = VotingClassifier(estimators=members, voting="soft")
    for name, estimator in estimators.items():
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), INTERACTION_NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), INTERACTION_CATEGORICAL),
        ])
        candidates[name] = Pipeline([
            ("interactions", FaultInteractions()), ("preprocessing", preprocessing), ("classifier", estimator),
        ])
    return candidates


def daily_folds(frame):
    """Each check day is strictly later than every fitting row; never split a day."""
    days = sorted(frame.business_day.unique())
    if len(days) < 2:
        raise ValueError("At least two fitting business days are required")
    folds = []
    for day in days[1:]:
        fit = np.flatnonzero(frame.business_day.lt(day))
        check = np.flatnonzero(frame.business_day.eq(day))
        if frame.iloc[fit].fault_confirmed.nunique() != 2:
            raise ValueError("Every fitting period needs both classes")
        validate_test(frame.iloc[fit], frame.iloc[check])
        folds.append((fit, check))
    return folds


def strict_threshold(labels, probability):
    """Largest threshold allowed by all observed faults, without decimal rounding.

    Classification uses p >= threshold for fault. Equality is deliberately a fault.
    This constraint applies to these observed labels, not to unknown future labels.
    """
    labels, probability = np.asarray(labels), np.asarray(probability, dtype=float)
    if labels.ndim != 1 or probability.shape != labels.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Aligned known binary labels are required")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid fault scores")
    if not (labels == 1).any():
        raise ValueError("At least one real fault is needed for threshold calibration")
    minimum = float(probability[labels == 1].min())
    # One ULP of tolerance keeps saved floating-point round trips on the fault side.
    return max(0.0, float(np.nextafter(minimum, -np.inf)))


def constrained_counts(labels, probability, threshold):
    counts = error_counts(labels, probability, threshold)
    counts["true_negatives"] = int((np.asarray(labels) == 0).sum()) - counts["false_positives"]
    counts["true_positives"] = int((np.asarray(labels) == 1).sum()) - counts["false_negatives"]
    return counts


def select_constrained(rows):
    """A single missed fault always loses to a candidate with zero observed misses."""
    eligible = [row for row in rows if row["false_negatives"] == 0]
    if not eligible:
        raise ValueError("No candidate satisfies the zero-observed-FN constraint")
    return min(eligible, key=lambda row: (
        -row["true_negatives"], -row["days_with_true_negatives"], row["brier_score"], row["model"],
    ))


def adoption_gate(comparison):
    """Passing a finite test only qualifies a candidate for user acceptance."""
    reasons = []
    if comparison["faults"] == 0:
        reasons.append("no_real_faults_in_test")
    if comparison["candidate"]["false_negatives"]:
        reasons.append("real_faults_missed")
    if comparison["candidate"]["true_negatives"] == 0:
        reasons.append("no_false_alarms_identified")
    return {"status": "rejected" if reasons else "awaiting_user_acceptance", "reasons": reasons,
            "requirements": {"false_negatives": 0, "true_negatives_minimum": 1},
            "model_sha256": comparison["model_sha256"], "default_model_replaced": False,
            "user_accepted": None, "future_zero_fn_guaranteed": False}


def train_and_freeze(train_path, output_dir, artifact_dir, *, candidates=None,
                     candidate_details=None, selector=None, variant="constrained_specificity_v1"):
    training, quality = load_tickets(train_path)
    validate_training(training)
    fitting, calibration = split_training_calibration(training)
    x, y = make_features(fitting), fitting.fault_confirmed.astype(int)
    folds = daily_folds(fitting)
    covered = np.zeros(len(fitting), dtype=bool)
    for _, check in folds:
        covered[check] = True
    candidates = build_constrained_candidates() if candidates is None else candidates
    candidate_details = {} if candidate_details is None else candidate_details
    selector = select_constrained if selector is None else selector
    rows, saved_oof, fold_rows = [], {}, []
    for name, prototype in candidates.items():
        oof = np.full(len(fitting), np.nan)
        for fit, check in folds:
            estimator = clone(prototype).fit(x.iloc[fit], y.iloc[fit])
            oof[check] = estimator.predict_proba(x.iloc[check])[:, 1]
        threshold = strict_threshold(y[covered], oof[covered])
        day_hits = 0
        for fold, (fit, check) in enumerate(folds, start=1):
            counts = constrained_counts(y.iloc[check], oof[check], threshold)
            day_hits += counts["true_negatives"] > 0
            fold_rows.append({"model": name, "fold": fold, "fit_rows": len(fit),
                              "check_rows": len(check), "day": fitting.business_day.iloc[check[0]],
                              "classification_threshold": threshold, **counts})
        score = classification_metrics(y[covered], oof[covered], threshold)
        rows.append({**candidate_details.get(name, {}), "model": name, "classification_threshold": threshold,
                     "days_with_true_negatives": int(day_hits), "brier_score": score["brier_score"],
                     **constrained_counts(y[covered], oof[covered], threshold)})
        saved_oof[name] = oof
        print(f"Constrained training CV complete: {name}", flush=True)
    selected = selector(rows)
    estimator = clone(candidates[selected["model"]]).fit(x, y)
    probability = estimator.predict_proba(make_features(calibration))[:, 1]
    calibration_threshold = strict_threshold(calibration.fault_confirmed, probability)
    threshold = min(selected["classification_threshold"], calibration_threshold)
    calibrated_counts = constrained_counts(calibration.fault_confirmed, probability, threshold)
    assert calibrated_counts["false_negatives"] == 0
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "model": selected["model"], "classification_threshold": threshold, "review_threshold": threshold,
        "objective": "Maximize true negatives subject to zero observed training OOF false negatives",
        "tie_breakers": ["more check days with true negatives", "lower Brier score", "model name"],
        "threshold_rule": "one float step below minimum observed real-fault score; equality is a fault",
        "training_rows": len(training), "estimator_fit_rows": len(fitting), "calibration_rows": len(calibration),
        "training_oof_rows": int(covered.sum()), "training_oof_faults": int(y[covered].eq(1).sum()),
        "training_oof_false_alarms": int(y[covered].eq(0).sum()),
        "candidate_models": len(candidates), "daily_folds": len(folds), "selected_training_oof": selected,
        "calibration_day": calibration.business_day.iloc[0], "calibration_threshold": calibration_threshold,
        "calibration_results": calibrated_counts, "test_used_for_selection": False,
        "final_validation_evaluated": False, "random_seed": 42, "train_quality": quality,
    }
    for key in ["feature_set", "selected_features", "excluded_features", "feature_count", "algorithm"]:
        if key in selected:
            selection[key] = selected[key]
    metadata = {
        "model_name": selected["model"], "classification_threshold": threshold, "review_threshold": threshold,
        "review_reference_floor": 1.0, "features": selection.get("selected_features", FEATURES),
        "input_features": FEATURES, "training_input_sha256": quality["input_sha256"],
        "sklearn_version": sklearn.__version__, "selection_scope": "training_only_daily_folds_then_training_day_calibration",
        "mode": "advisory_only", "variant": variant, "optimization_goal": "zero_false_negatives",
        "future_zero_fn_guaranteed": False, "automatic_dispatch_cancellation_enabled": False,
        "estimator_fit_rows": len(fitting), "calibration_rows": len(calibration),
    }
    model_path = artifact_dir / "fault_model.joblib"
    joblib.dump({**metadata, "pipeline": estimator}, model_path, compress=3)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    selection["model_sha256"] = metadata["model_sha256"]
    write_json(artifact_dir / "model_metadata.json", metadata)
    write_json(output_dir / "selection.json", selection)
    pd.DataFrame(rows).sort_values(["false_negatives", "true_negatives", "brier_score"], ascending=[True, False, True]).to_csv(
        output_dir / "training_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output_dir / "training_folds.csv", index=False, encoding="utf-8-sig")
    oof = fitting.loc[covered, ["ticket_id", "business_day", "fault_confirmed"]].copy()
    for name, probability in saved_oof.items():
        oof[name] = probability[covered]
    oof.to_csv(output_dir / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")
    probability = estimator.predict_proba(make_features(calibration))[:, 1]
    output = predictions_frame(calibration, probability, threshold, threshold)
    output["fault_confirmed"] = calibration.fault_confirmed
    output["dispatch_action"] = "preserve_dispatch"
    output.to_csv(output_dir / "calibration_predictions.csv", index=False, encoding="utf-8-sig")
    return training, selection, model_path


def evaluate_frozen(training, selection, model_path, test_path, output_dir, baseline_path,
                    *, experiment_renderer=None):
    """Read external test only after decisions have been saved, and do not tune on it."""
    output_dir = Path(output_dir)
    if json.loads((output_dir / "selection.json").read_text(encoding="utf-8")) != selection:
        raise ValueError("Selection must be frozen on disk before test access")
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != selection["model_sha256"]:
        raise ValueError("Frozen candidate fingerprint does not match")
    bundle = joblib.load(model_path)
    threshold = bundle["classification_threshold"]
    if threshold != selection["classification_threshold"]:
        raise ValueError("Frozen classification threshold does not match selection")
    control = None
    if control_info := selection.get("matched_full_features_control"):
        control_path = Path(control_info["path"])
        if hashlib.sha256(control_path.read_bytes()).hexdigest() != control_info["model_sha256"]:
            raise ValueError("Frozen full-feature control fingerprint does not match")
        control = joblib.load(control_path)
        if control["classification_threshold"] != control_info["classification_threshold"]:
            raise ValueError("Frozen full-feature control threshold does not match")
    testing, quality = load_tickets(test_path)
    validate_test(training, testing)
    probability = bundle["pipeline"].predict_proba(make_features(testing))[:, 1]
    counts = constrained_counts(testing.fault_confirmed, probability, threshold)
    baseline = joblib.load(baseline_path)
    old_probability = baseline["pipeline"].predict_proba(make_features(testing))[:, 1]
    old = constrained_counts(testing.fault_confirmed, old_probability, baseline.get("classification_threshold", 0.5))
    faults = int(testing.fault_confirmed.eq(1).sum())
    comparison = {"test_rows": len(testing), "faults": faults, "false_alarms": int(testing.fault_confirmed.eq(0).sum()),
                  "model_sha256": selection["model_sha256"], "baseline": old, "candidate": counts,
                  "always_fault_baseline": constrained_counts(testing.fault_confirmed, np.ones(len(testing)), 0.5),
                  "zero_misses_observed": faults > 0 and counts["false_negatives"] == 0,
                  "nontrivial_goal_observed": faults > 0 and counts["false_negatives"] == 0 and counts["true_negatives"] > 0,
                  "future_zero_fn_guaranteed": False, "final_validation_evaluated": False}
    if control is not None:
        control_probability = control["pipeline"].predict_proba(make_features(testing))[:, 1]
        control_threshold = control["classification_threshold"]
        comparison["matched_full_features_control"] = {
            "model_sha256": control_info["model_sha256"], "classification_threshold": control_threshold,
            "counts": constrained_counts(testing.fault_confirmed, control_probability, control_threshold),
            "classification": classification_metrics(testing.fault_confirmed, control_probability, control_threshold),
        }
        control_output = predictions_frame(testing, control_probability, control_threshold, control_threshold)
        control_output["fault_confirmed"] = testing.fault_confirmed
        control_output["dispatch_action"] = "preserve_dispatch"
        control_output.to_csv(output_dir / "full_features_control_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {"data_quality": quality, "model": bundle["model_name"], "model_sha256": selection["model_sha256"],
               "classification_threshold": threshold, "review_threshold": threshold, "review_reference_floor": 1.0,
               "classification": classification_metrics(testing.fault_confirmed, probability, threshold),
               "review": review_metrics(testing.fault_confirmed, probability < threshold),
               "evaluation_role": "test", "optimization_goal": "zero_false_negatives", "previous_model_error_counts": old}
    output = predictions_frame(testing, probability, threshold, threshold)
    output["fault_confirmed"] = testing.fault_confirmed
    output["dispatch_action"] = "preserve_dispatch"
    output.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "test_comparison.json", comparison)
    write_json(output_dir / "adoption_gate.json", adoption_gate(comparison))
    renderer = experiment_markdown if experiment_renderer is None else experiment_renderer
    (output_dir / "experiment.md").write_text(renderer(selection, comparison), encoding="utf-8")
    summary = render_saved_report(output_dir)
    print(console_summary(summary, output_dir))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return comparison


def experiment_markdown(selection, comparison):
    old, new, always = comparison["baseline"], comparison["candidate"], comparison["always_fault_baseline"]
    if comparison["nontrivial_goal_observed"]:
        verdict = "本次测试观察到真实故障零漏判，同时识别了一部分空跑；尚待用户验收，不保证未来零漏判"
    elif comparison["zero_misses_observed"]:
        verdict = "本次测试没有漏判，但未识别任何空跑，尚无节省空跑的分类价值"
    else:
        verdict = "测试仍存在真实故障漏判，未达到目标，不能据此取消或延后派单"
    return f"""# 先零漏判，再尽量识别空跑

**{verdict}。默认模型未替换。**

| 测试集 {comparison['test_rows']} 条 | 原随机森林 | 本轮候选 | 全判故障参照 |
| --- | ---: | ---: | ---: |
| 真实故障正确识别 TP | {old['true_positives']} | {new['true_positives']} | {always['true_positives']} |
| 真实故障漏判 FN | {old['false_negatives']} | {new['false_negatives']} | {always['false_negatives']} |
| 空跑正确识别 TN | {old['true_negatives']} | {new['true_negatives']} | {always['true_negatives']} |
| 空跑误报 FP | {old['false_positives']} | {new['false_positives']} | {always['false_positives']} |

## 训练约束

- 固定训练文件 {selection['training_rows']} 条，前段 {selection['estimator_fit_rows']} 条用于 {selection['daily_folds']} 个扩展日期折，最后一个训练日 {selection['calibration_rows']} 条仅校准阈值；不再回填拟合。
- 对比 {selection['candidate_models']} 种固定配置，含原始特征基线、非线性交互、故障类别权重 1/4/12，以及软投票集成；没有伪造样本或改标签。
- 第一目标：时间折预测 FN=0。第二目标：TN 最大；平局依次比较能识别空跑的日期数、Brier 分数、模型名。仅用训练标签。
- 阈值取真实故障最低分的前一个浮点数，边界归为故障；相较固定 0.05 余量和两位小数取整，在同一零训练漏判约束下保留更多识别空跑的空间。最后训练日只能降低阈值。
- 选中 {selection['model']}，分类与复核阈值均为 {selection['classification_threshold']:.17g}。训练内时间验证 {selection['training_oof_rows']} 条、真实故障 {selection['training_oof_faults']} 条、空跑 {selection['training_oof_false_alarms']} 条，选中候选识别空跑 {selection['selected_training_oof']['true_negatives']} 条；这些是参与选择的数据，不能作为独立有效性证明。
- 所有候选、阈值和模型指纹在读取测试文件前冻结；只评估这个候选，不根据本次测试调参。该测试集已在历史实验中反复查看，因此仍需新数据检验。验收集本轮未读取、未推理。

## 业务边界

输出为真实的 0/1 二分类；没有把漏判改成“待复核”后从错误计数中剔除。预测 0 只触发远程确认建议，所有工单保留派单。**当前数据不能证明任何未来工单都不漏判；避免无人处理仍需要派单、确认和超时升级的业务闭环。**

## 复现与用户验收

训练并测试：`uv run --locked python -m scripts.train_constrained`

用户自行验收（本轮未执行）：

```powershell
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/constrained/fault_model.joblib --output-dir outputs/user_acceptance/constrained
```

离线研究预测（先检查 adoption_gate.json；被拒绝的候选不得用于自动处理工单）：

```powershell
uv run --locked python -m scripts.predict --input outputs/predict_input_example.csv --model artifacts/constrained/fault_model.joblib --output outputs/constrained/new_predictions.csv
```

模型指纹：{selection['model_sha256']}。根目录 README.md 和 AI_LOG.md 保持不变。
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=Path("datasets/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("datasets/test.csv"), type=Path)
    parser.add_argument("--baseline-model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/constrained"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts/constrained"), type=Path)
    args = parser.parse_args()
    try:
        training, selection, model_path = train_and_freeze(args.train, args.output_dir, args.artifact_dir)
        evaluate_frozen(training, selection, model_path, args.test, args.output_dir, args.baseline_model)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Constrained training aborted; preserve dispatch: {exc}\n")


if __name__ == "__main__":
    main()
