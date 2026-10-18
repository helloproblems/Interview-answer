"""Controlled feature ablation for zero observed misses and maximum false-alarm detection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.common import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, make_features
from scripts.train import build_models, write_json
from scripts.train_constrained import constrained_counts, evaluate_frozen, strict_threshold, train_and_freeze
from scripts.train_recall import split_training_calibration


AGE, HEALTH = "telemetry_age_minutes", "reported_health_score"
FEATURE_SETS = {
    "all_features": FEATURES,
    **{f"without_{column}": [item for item in FEATURES if item != column] for column in FEATURES},
    "age_only": [AGE],
    "age_health": [AGE, HEALTH],
    "age_health_device": [AGE, HEALTH, "device_type"],
    "age_health_region": [AGE, HEALTH, "region"],
    "numeric_only": NUMERIC_FEATURES,
}
ALGORITHMS = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
TIE_BREAKERS = ["more check days with true negatives", "fewer feature columns", "lower Brier score", "model name"]


def build_feature_pipeline(columns, estimator):
    """Physically drop excluded inputs before fitting; labels and IDs are forbidden."""
    if not columns or len(set(columns)) != len(columns) or not set(columns).issubset(FEATURES):
        raise ValueError("Feature subset must contain unique, permitted predictor columns")
    numeric = [item for item in NUMERIC_FEATURES if item in columns]
    categorical = [item for item in CATEGORICAL_FEATURES if item in columns]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return Pipeline([("preprocessing", ColumnTransformer(transformers, remainder="drop")),
                     ("classifier", clone(estimator))])


def build_ablation_candidates():
    prototypes, candidates, details = build_models(), {}, {}
    for feature_set, columns in FEATURE_SETS.items():
        for algorithm in ALGORITHMS:
            name = f"{feature_set}__{algorithm}"
            candidates[name] = build_feature_pipeline(columns, prototypes[algorithm].named_steps["classifier"])
            details[name] = {"feature_set": feature_set, "algorithm": algorithm,
                             "selected_features": columns, "feature_count": len(columns),
                             "excluded_features": [item for item in FEATURES if item not in columns]}
    return candidates, details


def select_feature_candidate(rows):
    eligible = [row for row in rows if row["false_negatives"] == 0]
    if not eligible:
        raise ValueError("No feature subset satisfies zero observed training misses")
    return min(eligible, key=lambda row: (
        -row["true_negatives"], -row["days_with_true_negatives"], row["feature_count"], row["brier_score"], row["model"],
    ))


def train_feature_selection(train_path, output_dir, artifact_dir):
    """Freeze the chosen subset and a matched all-column control without test access."""
    candidates, details = build_ablation_candidates()
    training, selection, model_path = train_and_freeze(
        train_path, output_dir, artifact_dir, candidates=candidates, candidate_details=details,
        selector=select_feature_candidate, variant="feature_ablation_v1")
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    selection["tie_breakers"] = TIE_BREAKERS
    selection["feature_sets_compared"] = len(FEATURE_SETS)
    selection["algorithms_compared"] = ALGORITHMS
    selection["feature_sets"] = FEATURE_SETS
    search = pd.read_csv(output_dir / "training_search.csv", float_precision="round_trip")
    ablation = search.copy()
    references = search[search.feature_set.eq("all_features")].set_index("algorithm")
    ablation["true_negative_change_vs_full_same_algorithm"] = [
        int(row.true_negatives - references.loc[row.algorithm, "true_negatives"]) for row in ablation.itertuples()]
    ablation["brier_change_vs_full_same_algorithm"] = [
        float(row.brier_score - references.loc[row.algorithm, "brier_score"]) for row in ablation.itertuples()]
    ablation.to_csv(output_dir / "feature_ablation.csv", index=False, encoding="utf-8-sig")

    # A paired control isolates column removal from algorithm, fit rows and calibration changes.
    control_name = f"all_features__{selection['algorithm']}"
    reference = search[search.model.eq(control_name)].iloc[0]
    fitting, calibration = split_training_calibration(training)
    control_pipeline = clone(candidates[control_name]).fit(make_features(fitting), fitting.fault_confirmed.astype(int))
    probability = control_pipeline.predict_proba(make_features(calibration))[:, 1]
    threshold = min(float(reference.classification_threshold), strict_threshold(calibration.fault_confirmed, probability))
    control_path = artifact_dir / "full_features_control.joblib"
    bundle = {"pipeline": control_pipeline, "model_name": control_name, "features": FEATURES,
              "classification_threshold": threshold, "review_threshold": threshold,
              "variant": "feature_ablation_matched_control", "mode": "advisory_only"}
    joblib.dump(bundle, control_path, compress=3)
    selection["matched_full_features_control"] = {
        "path": control_path.as_posix(), "model_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "model": control_name, "classification_threshold": threshold,
        "estimator_fit_rows": len(fitting), "calibration_rows": len(calibration),
        "calibration_results": constrained_counts(calibration.fault_confirmed, probability, threshold),
    }
    # Selection and both artifacts are now final; external test access has not occurred.
    write_json(output_dir / "selection.json", selection)
    return training, selection, model_path


def experiment_markdown(selection, comparison):
    new, control, original = comparison["candidate"], comparison["matched_full_features_control"]["counts"], comparison["baseline"]
    if new["false_negatives"]:
        verdict = "仍有真实故障漏判，未达到目标，不替换默认模型"
    elif not new["true_negatives"]:
        verdict = "没有漏判，但未识别任何空跑，不能视为有效改进，不替换默认模型"
    else:
        verdict = "本次测试零漏判且识别了部分空跑，等待用户验收，不保证未来零漏判"
    selected = selection["selected_training_oof"]
    kept = ", ".join(selection["selected_features"])
    removed = ", ".join(selection["excluded_features"]) or "无"
    return f"""# 按零漏判目标筛选训练列

**{verdict}。**

## 本轮选择

- 算法：{selection['algorithm']}；保留特征：{kept}；不参与拟合的特征：{removed}。
- 工单 ID、真实标签和绝对日期始终不是预测特征。时间戳仅用于计算遥测延迟与按时间分组；原始 CSV 列没有删除。
- 本轮不加入交互项，固定 {selection['feature_sets_compared']} 个特征组合 × {len(selection['algorithms_compared'])} 个算法，共 {selection['candidate_models']} 个候选。相同算法在各组合中使用相同超参数、训练行、验证日期和阈值规则。
- 先要求训练内时间验证 FN=0，再使 TN 最大；平局依次比较识别空跑的日期数、更少的特征列、Brier 分数和名称。只用训练集选择；校准日只可降低阈值。
- 选中候选在训练内时间验证中：故障漏判 {selected['false_negatives']}/{selection['training_oof_faults']}，识别空跑 {selected['true_negatives']}/{selection['training_oof_false_alarms']}。这些数据参与了选择，不是独立有效性证据。

## 冻结后的测试结果

| 测试集 {comparison['test_rows']} 条 | 原默认随机森林 | 同算法全部 5 列对照 | 本轮选中列 |
| --- | ---: | ---: | ---: |
| 故障漏判 FN / {comparison['faults']} | {original['false_negatives']} | {control['false_negatives']} | {new['false_negatives']} |
| 识别空跑 TN / {comparison['false_alarms']} | {original['true_negatives']} | {control['true_negatives']} | {new['true_negatives']} |
| 空跑误报 FP | {original['false_positives']} | {control['false_positives']} | {new['false_positives']} |

同算法对照使用相同的 {selection['estimator_fit_rows']} 条拟合数据、{selection['calibration_rows']} 条训练内校准数据。两者的阈值按相同规则独立确定，均在读取测试数据前冻结；没有按测试结果再换列、调参或选择赢家。默认模型的训练范围不同，仅作历史参照。

删列的价值须结合上述配对结果判断；不能仅因训练内得分提高，就断言某列永远是干扰项。`feature_ablation.csv` 给出同算法的全部删列对比，`training_folds.csv` 给出逐日计数，`full_features_control_predictions.csv` 保留对照逐单结果。

测试集已经在历史实验中查看过，并非新的盲测。本轮验收集未读取、未推理。候选 `predicted_fault_confirmed` 仍为 0/1，漏判照实计数；所有预测仅作建议，不能直接取消派单。

## 复现与验收

训练与测试：`uv run --locked python -m scripts.select_features`

仅由用户自行执行验收：

```powershell
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/feature_selection/fault_model.joblib --output-dir outputs/user_acceptance/feature_selection
```

CSV 输入接口仍保留原有字段；模型内部的列选择器会丢弃未选中的列，无需改写原始数据文件。模型指纹：{selection['model_sha256']}。

README.md 和 AI_LOG.md 本轮不修改。
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=Path("datasets/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("datasets/test.csv"), type=Path)
    parser.add_argument("--baseline-model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/feature_selection"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts/feature_selection"), type=Path)
    args = parser.parse_args()
    try:
        training, selection, model_path = train_feature_selection(args.train, args.output_dir, args.artifact_dir)
        evaluate_frozen(training, selection, model_path, args.test, args.output_dir, args.baseline_model,
                        experiment_renderer=experiment_markdown)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Feature selection aborted; preserve dispatch: {exc}\n")


if __name__ == "__main__":
    main()
