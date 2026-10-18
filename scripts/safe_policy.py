"""A conservative output contract: no automatic no-fault verdict, with explicit abstentions."""

import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.common import classification_metrics, load_tickets, make_features, predictions_frame, review_metrics
from scripts.train import write_json


POLICY = "review_before_clear"


def safe_predictions_frame(frame, probability, review_threshold, classification_threshold=0.5):
    """Never serialize a no-fault prediction; low scores are explicitly unresolved."""
    output = predictions_frame(frame, probability, review_threshold, classification_threshold)
    pending = output.recommendation.eq("remote_review") | output.predicted_fault_confirmed.eq(0)
    output["predicted_fault_confirmed"] = pd.Series(1, index=output.index, dtype="Int64").mask(pending)
    output["prediction_status"] = np.where(pending, "requires_confirmation", "fault_suspected")
    output["risk_label_zh"] = np.where(pending, "待复核，不能确认无故障", "疑似故障")
    output["recommendation"] = np.where(pending, "remote_review", "normal_dispatch")
    output["recommendation_reason"] = np.where(
        pending, "Unresolved: verify fresh telemetry; preserve dispatch until confirmed",
        "Fault suspected: preserve current dispatch",
    )
    output["dispatch_action"] = "preserve_dispatch"
    return output


def decision_metrics(labels, output, classification_threshold):
    labels = np.asarray(labels)
    if len(labels) != len(output) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Decision metrics require aligned known labels")
    prediction = pd.to_numeric(output.predicted_fault_confirmed, errors="raise")
    if not prediction.dropna().eq(1).all():
        raise ValueError("Safe policy must never output an automatic no-fault verdict")
    pending = prediction.isna().to_numpy()
    if not output.prediction_status.eq(np.where(pending, "requires_confirmation", "fault_suspected")).all():
        raise ValueError("Prediction status disagrees with unresolved decisions")
    faults, false_alarms = int((labels == 1).sum()), int((labels == 0).sum())
    return {
        "samples": len(labels), "faults": faults, "false_alarms": false_alarms,
        "auto_fault_predictions": int((~pending).sum()), "pending_reviews": int(pending.sum()),
        "auto_no_fault_predictions": 0, "auto_no_fault_errors": 0,
        "automatic_classification_coverage": float((~pending).mean()),
        "real_faults_auto_flagged": int(((labels == 1) & ~pending).sum()),
        "real_faults_pending_review": int(((labels == 1) & pending).sum()),
        "false_alarms_auto_flagged": int(((labels == 0) & ~pending).sum()),
        "false_alarms_pending_review": int(((labels == 0) & pending).sum()),
        "full_workflow_accuracy": None, "user_accepted": None,
        "underlying_classifier": classification_metrics(labels, output.fault_probability, classification_threshold),
        "review": review_metrics(labels, pending),
    }


def console_summary(metrics, output_dir):
    d = metrics["decision_metrics"]
    raw = d["underlying_classifier"]
    scope = "测试集" if metrics["evaluation_role"] == "test" else "用户验收集"
    return f"""保守输出策略 · {scope}
自动判为无故障：0 条（规则禁止，不代表模型零漏判）
疑似故障：{d['auto_fault_predictions']} 条；待复核：{d['pending_reviews']} 条；自动分类覆盖率：{d['automatic_classification_coverage']:.2%}
真实故障 {d['faults']} 条：{d['real_faults_auto_flagged']} 条标记疑似故障，{d['real_faults_pending_review']} 条待复核
空跑 {d['false_alarms']} 条：{d['false_alarms_auto_flagged']} 条仍标记疑似故障，{d['false_alarms_pending_review']} 条待复核
基模型二分类诊断：漏判 {raw['confusion_matrix_0_1'][1][0]} 条、误报 {raw['confusion_matrix_0_1'][0][1]} 条
待复核不计为预测正确，不能声称模型召回率 100%；派单全部保留。
报告：{Path(output_dir) / 'report.html'}"""


def render_safe_report(output_dir, metrics=None, output=None):
    output_dir = Path(output_dir)
    if metrics is None:
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if output is None:
        output = pd.read_csv(output_dir / "predictions.csv", dtype={"ticket_id": "string"}, encoding="utf-8-sig")
    if output.empty or output.ticket_id.isna().any() or output.ticket_id.duplicated().any():
        raise ValueError("Safe report requires nonempty unique ticket IDs")
    probability = pd.to_numeric(output.fault_probability, errors="raise")
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Invalid saved probabilities")
    threshold, review_threshold = metrics["classification_threshold"], metrics["review_threshold"]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in [threshold, review_threshold]):
        raise ValueError("Invalid saved thresholds")
    pending = probability.lt(max(threshold, review_threshold))
    if not output.predicted_fault_confirmed.isna().eq(pending).all():
        raise ValueError("Saved unresolved decisions disagree with thresholds")
    if not output.recommendation.eq(np.where(pending, "remote_review", "normal_dispatch")).all():
        raise ValueError("Saved recommendations disagree with unresolved decisions")
    if not output.dispatch_action.eq("preserve_dispatch").all():
        raise ValueError("Safe policy must preserve dispatch")
    observed = decision_metrics(output.fault_confirmed, output, threshold)
    if observed != metrics["decision_metrics"]:
        # CSV floating-point round trips may change ranking scores in the last bit.
        for key, value in observed.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    expected = metrics["decision_metrics"][key][inner_key]
                    if isinstance(inner_value, (int, float)):
                        if not math.isclose(inner_value, expected, rel_tol=1e-9, abs_tol=1e-9):
                            raise ValueError("Saved safe-policy metrics disagree with predictions")
                    elif inner_value != expected:
                        raise ValueError("Saved safe-policy metrics disagree with predictions")
            elif value != metrics["decision_metrics"][key]:
                raise ValueError("Saved safe-policy metrics disagree with predictions")
    d = observed
    tn, fp = d["underlying_classifier"]["confusion_matrix_0_1"][0]
    fn, tp = d["underlying_classifier"]["confusion_matrix_0_1"][1]
    scope = "测试集 · 非最终验收" if metrics["evaluation_role"] == "test" else "用户验收参考"
    summary = f"""# 保守输出策略：禁止自动判为无故障

{scope}，共 {d['samples']} 条。输出只含“疑似故障”和“待复核”；待复核的 predicted_fault_confirmed 留空，绝不填 0。

| 真实标签 | 疑似故障 | 待复核 | 自动判为无故障 |
| --- | ---: | ---: | ---: |
| 真实故障 | {d['real_faults_auto_flagged']} | {d['real_faults_pending_review']} | 0 |
| 空跑 | {d['false_alarms_auto_flagged']} | {d['false_alarms_pending_review']} | 0 |

自动分类覆盖率 {d['automatic_classification_coverage']:.2%}；待复核 {d['pending_reviews']} 条，不能计为正确或丢弃。完整流程准确率与用户验收结论均未确定。所有工单保留派单，复核前不自动撤单或延后处理。

**这是输出安全约束，不是模型 100% 准确率或零漏判能力。** 基模型按 {threshold:.2f} 二分类仍有 {fn} 条故障漏判、{fp} 条空跑误报。保留原始概率便于审计；低于复核阈值 {review_threshold:.2f} 或分类阈值的结果均改为待确认。

若要自动给出“无故障”，当前有限特征无法保证不漏掉未来的真实故障，需要额外的可靠确认流程。本策略没有实际确认或消除任何空跑。

模型指纹：{metrics['model_sha256']}；数据指纹：{metrics['data_quality']['input_sha256']}。
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_json(output_dir / "summary.json", metrics)
    escape = lambda value: html.escape(str(value), quote=True)
    rows = []
    for row in output.sort_values("prediction_status", ascending=False).to_dict("records"):
        cells = [row["ticket_id"], row["region"], row["device_type"],
                 "真实故障" if row["fault_confirmed"] == 1 else "空跑",
                 row["risk_label_zh"], f"{row['fault_probability']:.2%}", "保留派单"]
        rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>保守策略 · {scope}</title>
<style>body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f3f6fa;color:#18304a;margin:0}}main{{max-width:1150px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}p{{line-height:1.8}}.note{{padding:20px;background:#fff4e8;border-left:5px solid #ce5c28;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,section{{background:white;padding:22px;border-radius:10px;border:1px solid #dce5ee}}.value{{font-size:32px;font-weight:bold;margin:12px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dde6ee}}th{{background:#eef4f9}}section{{margin-top:20px}}.scroll{{overflow:auto;max-height:500px}}.muted{{color:#5b6f84}}a{{color:#135f9b}}footer{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}main{{padding:16px}}}}</style></head><body><main>
<div class="muted">{scope}</div><h1>低分工单待复核，不自动判定无故障</h1>
<div class="note"><strong>自动输出“无故障”：禁止</strong><p>这是输出约束，不代表模型已实现零漏判。待复核不计为预测正确，所有工单继续保留派单。</p></div>
<div class="cards"><div class="card">疑似故障<div class="value">{d['auto_fault_predictions']}</div>自动分类覆盖率 {d['automatic_classification_coverage']:.2%}</div><div class="card">待复核<div class="value">{d['pending_reviews']}</div>含 {d['real_faults_pending_review']} 条真实故障、{d['false_alarms_pending_review']} 条空跑</div><div class="card">自动判为无故障<div class="value">0</div>没有自动确认或消除空跑</div></div>
<section><h2>每类工单最终去了哪里</h2><table><tr><th>真实标签</th><th>疑似故障</th><th>待复核</th><th>自动无故障</th></tr><tr><th>真实故障</th><td>{d['real_faults_auto_flagged']}</td><td>{d['real_faults_pending_review']}</td><td>0</td></tr><tr><th>空跑</th><td>{d['false_alarms_auto_flagged']}</td><td>{d['false_alarms_pending_review']}</td><td>0</td></tr></table><p>基模型按 {threshold:.2f} 二分类仍有 <b>{fn} 条故障漏判、{fp} 条空跑误报</b>。本策略保留概率，将原本会判为无故障或需要复核的工单标为待确认。复核阈值 {review_threshold:.2f}。</p><p class="muted">本页共 {d['samples']} 条，待复核 {d['pending_reviews']} 条均保留在分母中；完整流程准确率尚未确定。用户验收集由用户自行执行，不能把本测试页面视为验收通过。</p></section>
<section><h2>逐单结果</h2><p><a href="predictions.csv">下载逐单 CSV</a> · <a href="summary.md">文字说明</a> · <a href="metrics.json">完整指标</a></p><div class="scroll"><table><thead><tr><th>工单</th><th>片区</th><th>设备</th><th>真实标签</th><th>保守输出</th><th>基模型概率</th><th>派单</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<footer class="muted"><p>模型：{escape(metrics['model'])}；模型指纹：{escape(metrics['model_sha256'])}</p><p>“不得自动判为无故障”由输出规则保证；它不是对未来物理状态的保证，也不代替现场或可靠远程确认。</p></footer></main></body></html>"""
    (output_dir / "report.html").write_text(page, encoding="utf-8")
    return metrics


def evaluate_safe(frame, quality, bundle, model_path, output_dir, evaluation_role="user_acceptance"):
    if frame.fault_confirmed.isna().any():
        raise ValueError("Safe evaluation requires known labels")
    probability = bundle["pipeline"].predict_proba(make_features(frame))[:, 1]
    threshold = bundle.get("classification_threshold", 0.5)
    output = safe_predictions_frame(frame, probability, bundle["review_threshold"], threshold)
    output["fault_confirmed"] = frame.fault_confirmed
    metrics = {"data_quality": quality, "decision_policy": POLICY, "model": bundle["model_name"],
               "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
               "evaluation_role": evaluation_role, "classification_threshold": threshold,
               "review_threshold": bundle["review_threshold"],
               "decision_metrics": decision_metrics(frame.fault_confirmed, output, threshold)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "metrics.json", metrics)
    render_safe_report(output_dir, metrics, output)
    return metrics


def build_safe_bundle(base_model, artifact_dir):
    baseline = joblib.load(base_model)
    bundle = {**baseline, "decision_policy": POLICY, "variant": "safe_dispatch_v1",
              "classification_threshold": baseline.get("classification_threshold", 0.5),
              "underlying_model_sha256": hashlib.sha256(Path(base_model).read_bytes()).hexdigest(),
              "automatic_no_fault_verdicts_enabled": False}
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "fault_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    metadata = {k: v for k, v in bundle.items() if k != "pipeline"}
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    write_json(artifact_dir / "model_metadata.json", metadata)
    return model_path, bundle


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts/safe_dispatch"), type=Path)
    parser.add_argument("--test", default=Path("datasets/test.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/safe_dispatch"), type=Path)
    args = parser.parse_args()
    try:
        path, bundle = build_safe_bundle(args.base_model, args.artifact_dir)
        frame, quality = load_tickets(args.test)
        metrics = evaluate_safe(frame, quality, bundle, path, args.output_dir, "test")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Safe policy aborted; preserve dispatch: {exc}\n")
    print(console_summary(metrics, args.output_dir))


if __name__ == "__main__":
    main()
