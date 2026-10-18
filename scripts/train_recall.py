"""Prioritize zero observed false negatives, with training-only time calibration."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from scripts.common import FEATURES, classification_metrics, load_tickets, make_features, predictions_frame, review_metrics
from scripts.optimize import build_candidates, error_counts, validate_test, validate_training
from scripts.report import console_summary, render_saved_report
from scripts.train import write_json


def zero_fn_threshold(labels, probabilities, margin=0.05):
    """Subtract a fixed margin from the least-confident known fault; round down."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Calibration requires aligned known binary labels")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid calibration probabilities")
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("Margin must be within [0, 1]")
    if not (labels == 1).any():
        raise ValueError("Zero-FN calibration requires observed real faults")
    minimum = float(probabilities[labels == 1].min())
    threshold = math.floor(max(0.0, minimum - margin) * 100) / 100
    return threshold, minimum


def split_training_calibration(training):
    """Keep the last training business day out of estimator fitting/selection."""
    training = training.assign(_utc=pd.to_datetime(training.ticket_time, utc=True)).sort_values(
        ["_utc", "ticket_id"]).drop(columns="_utc").reset_index(drop=True)
    last_day = training.business_day.max()
    fitting = training[training.business_day.lt(last_day)].reset_index(drop=True)
    calibration = training[training.business_day.eq(last_day)].reset_index(drop=True)
    validate_training(fitting)
    validate_test(fitting, calibration)
    if not calibration.fault_confirmed.eq(1).any():
        raise ValueError("The final training day must contain real faults for calibration")
    return fitting, calibration


def select_recall_candidate(rows):
    eligible = [r for r in rows if r["false_negatives"] == 0]
    if not eligible:
        raise ValueError("No training-OOF candidate meets zero observed false negatives")
    return min(eligible, key=lambda r: (r["false_positives"], r["brier_score"], r["model"]))


def comparison_markdown(selection, comparison):
    old, new = comparison["old_model"], comparison["high_recall_model"]
    verdict = "测试集观察到零漏判" if comparison["zero_false_negatives_observed"] else "测试集仍存在漏判，不能作为达标候选"
    upper = comparison["zero_fn_rate_one_sided_95pct_upper"]
    evidence = (
        f"观察到零漏判，但不保证未来零漏判；在独立同分布伯努利抽样假设下，漏判率单侧 95% 上界约为 {upper:.2%}。"
        if upper is not None else f"仍有 {new['false_negatives']} 条真实故障漏判，零漏判目标未达到，不计算零事件置信上界。"
    )
    return f"""# 零漏判优先模型试验

**{verdict}；最终验收集尚未评估。**

| 测试集 {comparison['test_rows']} 条 | 原模型 | 零漏判优先候选 |
| --- | ---: | ---: |
| 真实故障漏判 FN | {old['false_negatives']} | {new['false_negatives']} |
| 空跑误报 FP | {old['false_positives']} | {new['false_positives']} |
| 正确识别的空跑 | {comparison['false_alarms'] - old['false_positives']} | {comparison['false_alarms'] - new['false_positives']} |
| 准确率 | {old['accuracy']:.2%} | {new['accuracy']:.2%} |

目标改为先保证观察样本中的 FN=0，再尽量减少 FP；不再要求两类错误同时改善。测试预测为无故障的数量为 {comparison['predicted_no_fault_count']}，是否等同于全判故障：{comparison['all_predicted_fault']}。

## 训练和校准

- 使用原有训练文件 {selection['training_rows']} 条；前几天 {selection['estimator_fit_rows']} 条用于训练和 3 折扩展时间验证，最后一天 {selection['calibration_rows']} 条仅校准阈值。
- 在前段训练数据内比较 {selection['candidate_models']} 种配置。每个阈值取该模型时间验证中真实故障最低分减去固定 {selection['safety_margin']:.2f}，向下取整到两位小数；在零漏判约束下选择误报最少者，平局比较 Brier 分数和名称。
- 选中 {selection['model']}。校准集中的真实故障最低分再次限制阈值，只允许降低，不允许升高。最终分类阈值 {selection['classification_threshold']:.2f}，复核阈值相同，训练后不再把校准行加入拟合。
- 模型、阈值与 selection.json 在读取测试文件前保存。测试集已在前几轮被查看，因此不是全新的盲测；本轮不按测试结果继续调参。用户验收集不读取、不推理。

## 可以和不可以得出的结论

测试集真实故障 {comparison['faults']} 条。{evidence}时间相关、分布变化与重复开发会限制这个解释。

这一版是概率模型的保守分类策略，不是无故障证明。若要求对任意未来样本绝不输出错误的“无故障”，仅凭当前特征无法同时提供这种保证和自动排除能力；需要保留人工确认。原默认模型不自动替换，候选通过用户验收后再决定使用。

## 运行

训练：`uv run --locked python -m scripts.train_recall`

由用户验收：

```bash
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/high_recall/fault_model.joblib --output-dir outputs/user_acceptance/high_recall
```

新数据预测：

```bash
uv run --locked python -m scripts.predict --input outputs/predict_input_example.csv --model artifacts/high_recall/fault_model.joblib --output outputs/high_recall/new_predictions.csv
```

模型 SHA-256：{comparison['model_sha256']}。本轮未修改根目录 README.md 或 AI_LOG.md。
"""


def train_recall(train_path, test_path, baseline_path, output_dir, artifact_dir, margin=0.05):
    training, train_quality = load_tickets(train_path)
    validate_training(training)
    fitting, calibration = split_training_calibration(training)
    x, y = make_features(fitting), fitting.fault_confirmed.astype(int)
    candidates = build_candidates()
    folds = list(TimeSeriesSplit(n_splits=3).split(x))
    for fit, check in folds:
        if y.iloc[fit].nunique() != 2:
            raise ValueError("Every time-fold fitting portion must contain both classes")
        if pd.Timestamp(fitting.ticket_time.iloc[fit[-1]]) >= pd.Timestamp(fitting.ticket_time.iloc[check[0]]):
            raise ValueError("Time folds must not split simultaneous tickets")
    rows, probabilities = [], {}
    for name, prototype in candidates.items():
        oof = np.full(len(y), np.nan)
        for fit, check in folds:
            pipeline = clone(prototype).fit(x.iloc[fit], y.iloc[fit])
            oof[check] = pipeline.predict_proba(x.iloc[check])[:, 1]
        covered = np.isfinite(oof)
        threshold, minimum = zero_fn_threshold(y[covered], oof[covered], margin)
        score = classification_metrics(y[covered], oof[covered], threshold)
        rows.append({"model": name, "classification_threshold": threshold,
                     "min_fault_probability": minimum, "brier_score": score["brier_score"],
                     **error_counts(y[covered], oof[covered], threshold)})
        probabilities[name] = oof
        print(f"Recall CV complete: {name}", flush=True)
    selected = select_recall_candidate(rows)
    pipeline = clone(candidates[selected["model"]]).fit(x, y)
    calibration_probability = pipeline.predict_proba(make_features(calibration))[:, 1]
    calibrated_threshold, calibration_minimum = zero_fn_threshold(
        calibration.fault_confirmed, calibration_probability, margin)
    threshold = min(selected["classification_threshold"], calibrated_threshold)
    calibration_score = error_counts(calibration.fault_confirmed, calibration_probability, threshold)
    if calibration_score["false_negatives"] != 0:
        raise ValueError("Calibration failed zero observed false negatives")
    selection = {"model": selected["model"], "classification_threshold": threshold,
                 "review_threshold": threshold, "safety_margin": margin,
                 "objective": "Zero observed training-OOF false negatives, then minimum false positives",
                 "training_rows": len(training), "estimator_fit_rows": len(fitting),
                 "calibration_rows": len(calibration), "calibration_day": calibration.business_day.iloc[0],
                 "calibration_faults": int(calibration.fault_confirmed.eq(1).sum()),
                 "candidate_models": len(candidates), "training_oof_rows": int(covered.sum()),
                 "selected_training_oof": selected, "calibration_min_fault_probability": calibration_minimum,
                 "calibration_results": calibration_score, "test_used_for_selection": False,
                 "final_validation_evaluated": False, "random_seed": 42, "train_quality": train_quality}
    metadata = {"model_name": selected["model"], "classification_threshold": threshold,
                "review_threshold": threshold, "review_reference_floor": 1.0, "features": FEATURES,
                "training_input_sha256": train_quality["input_sha256"], "sklearn_version": sklearn.__version__,
                "selection_scope": "training_only_cv_then_held_out_training_day_calibration",
                "mode": "advisory_only", "variant": "high_recall_v3",
                "optimization_goal": "zero_false_negatives", "future_zero_fn_guaranteed": False,
                "estimator_fit_rows": len(fitting), "calibration_rows": len(calibration)}
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "fault_model.joblib"
    joblib.dump({**metadata, "pipeline": pipeline}, model_path, compress=3)
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write_json(artifact_dir / "model_metadata.json", metadata)
    write_json(output_dir / "selection.json", selection)
    pd.DataFrame(rows).sort_values(["false_negatives", "false_positives", "brier_score"]).to_csv(
        output_dir / "training_search.csv", index=False, encoding="utf-8-sig")
    oof_frame = fitting.loc[covered, ["ticket_id", "ticket_time", "fault_confirmed"]].copy()
    for name, probability in probabilities.items():
        oof_frame[name] = probability[covered]
    oof_frame.to_csv(output_dir / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")
    calibration_output = predictions_frame(calibration, calibration_probability, threshold, threshold)
    calibration_output["fault_confirmed"] = calibration.fault_confirmed
    calibration_output.to_csv(output_dir / "calibration_predictions.csv", index=False, encoding="utf-8-sig")

    # Test access begins only after the full candidate and decision threshold are frozen.
    testing, test_quality = load_tickets(test_path)
    validate_test(training, testing)
    test_probability = pipeline.predict_proba(make_features(testing))[:, 1]
    baseline = joblib.load(baseline_path)
    old_probability = baseline["pipeline"].predict_proba(make_features(testing))[:, 1]
    old_counts = error_counts(testing.fault_confirmed, old_probability, baseline.get("classification_threshold", 0.5))
    new_counts = error_counts(testing.fault_confirmed, test_probability, threshold)
    faults = int(testing.fault_confirmed.eq(1).sum())
    false_alarms = int(testing.fault_confirmed.eq(0).sum())
    comparison = {"test_rows": len(testing), "faults": faults, "false_alarms": false_alarms,
                  "old_model": old_counts, "high_recall_model": new_counts,
                  "model_sha256": metadata["model_sha256"], "baseline_model_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
                  "classification_threshold": threshold, "final_validation_evaluated": False,
                  "zero_false_negatives_observed": faults > 0 and new_counts["false_negatives"] == 0,
                  "all_predicted_fault": bool((test_probability >= threshold).all()),
                  "predicted_no_fault_count": int((test_probability < threshold).sum()),
                  "future_zero_fn_guaranteed": False}
    comparison["zero_fn_rate_one_sided_95pct_upper"] = 1 - 0.05 ** (1 / faults) if comparison["zero_false_negatives_observed"] else None
    write_json(output_dir / "test_comparison.json", comparison)
    metrics = {"data_quality": test_quality, "model": selected["model"],
               "model_sha256": metadata["model_sha256"], "classification_threshold": threshold,
               "review_threshold": threshold, "review_reference_floor": 1.0,
               "classification": classification_metrics(testing.fault_confirmed, test_probability, threshold),
               "review": review_metrics(testing.fault_confirmed, test_probability < threshold),
               "evaluation_role": "test", "optimization_goal": "zero_false_negatives",
               "previous_model_error_counts": old_counts}
    write_json(output_dir / "metrics.json", metrics)
    predictions = predictions_frame(testing, test_probability, threshold, threshold)
    predictions["fault_confirmed"] = testing.fault_confirmed
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "comparison.md").write_text(comparison_markdown(selection, comparison), encoding="utf-8")
    summary = render_saved_report(output_dir)
    print(console_summary(summary, output_dir))
    print(json.dumps(comparison, indent=2))
    return selection, comparison


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=Path("datasets/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("datasets/test.csv"), type=Path)
    parser.add_argument("--baseline-model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/high_recall"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts/high_recall"), type=Path)
    parser.add_argument("--margin", default=0.05, type=float)
    args = parser.parse_args()
    try:
        train_recall(args.train, args.test, args.baseline_model, args.output_dir, args.artifact_dir, args.margin)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Recall training aborted: {exc}\n")


if __name__ == "__main__":
    main()
