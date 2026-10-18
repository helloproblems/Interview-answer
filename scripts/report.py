"""Render existing acceptance results without training, tuning, or predicting again."""

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import classification_metrics, review_metrics


def percentage(value):
    return "无可计算样本" if value is None else f"{value:.2%}"


def prepare_report(metrics, predictions):
    """Verify saved results and derive a transparent always-fault baseline."""
    required = {"ticket_id", "fault_confirmed", "fault_probability", "predicted_fault_confirmed", "recommendation"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if predictions.empty or predictions.ticket_id.duplicated().any():
        raise ValueError("Report requires nonempty predictions with unique ticket IDs")
    labels = pd.to_numeric(predictions.fault_confirmed, errors="raise")
    probability = pd.to_numeric(predictions.fault_probability, errors="raise")
    if not labels.isin([0, 1]).all() or not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Invalid labels or probabilities in saved predictions")
    classification_threshold = float(metrics.get("classification_threshold", 0.5))
    if not math.isfinite(classification_threshold) or not 0 <= classification_threshold <= 1:
        raise ValueError("Invalid classification threshold")
    expected_prediction = probability.ge(classification_threshold).astype(int)
    if not predictions.predicted_fault_confirmed.eq(expected_prediction).all():
        raise ValueError(f"Saved classification does not match the fixed {classification_threshold} threshold")
    review_threshold = float(metrics["review_threshold"])
    if not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise ValueError("Invalid review threshold")
    review_mask = probability.lt(review_threshold)
    if not predictions.recommendation.eq(np.where(review_mask, "remote_review", "normal_dispatch")).all():
        raise ValueError("Saved recommendations do not match the recorded review threshold")
    classification = classification_metrics(labels, probability, classification_threshold)
    review = review_metrics(labels, review_mask)
    # Prevent a report from silently combining results of different evaluation runs.
    for section, actual in [("classification", classification), ("review", review)]:
        for key, value in actual.items():
            expected = metrics[section].get(key)
            if isinstance(value, (int, float)):
                if expected is None or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"Saved metrics disagree with predictions: {section}.{key}")
            elif value != expected:
                raise ValueError(f"Saved metrics disagree with predictions: {section}.{key}")
    tn, fp = classification["confusion_matrix_0_1"][0]
    fn, tp = classification["confusion_matrix_0_1"][1]
    n = len(predictions)
    baseline_accuracy = int(labels.eq(1).sum()) / n
    details = predictions.copy()
    details["classification_result"] = np.select(
        [labels.eq(1) & expected_prediction.eq(0), labels.eq(0) & expected_prediction.eq(1)],
        ["真故障漏判", "空跑未识别"], default="预测正确",
    )
    details["actual_label_zh"] = labels.map({0: "空跑 / 无故障", 1: "真实故障"})
    details["predicted_label_zh"] = expected_prediction.map({0: "预测无故障", 1: "预测有故障"})
    details["review_recommendation_zh"] = np.where(review_mask, "建议远程复核", "保留原派单")
    details["fault_probability_pct"] = probability.map(lambda value: f"{value:.2%}")
    details["_order"] = details.classification_result.map({"真故障漏判": 0, "空跑未识别": 1, "预测正确": 2})
    details = details.sort_values(["_order", "fault_probability"]).drop(columns="_order")
    summary = {
        "samples": n, "model": metrics["model"], "classification": classification, "review": review,
        "review_threshold": review_threshold, "classification_threshold": classification_threshold,
        "evaluation_role": metrics.get("evaluation_role", "user_acceptance"),
        "model_sha256": metrics.get("model_sha256"),
        "baseline_accuracy": baseline_accuracy,
        "accuracy_difference_percentage_points": (classification["accuracy"] - baseline_accuracy) * 100,
        "missed_real_faults": fn, "identified_false_alarms": tn,
        "false_alarms_not_identified": fp, "correct_real_faults": tp,
        "predicted_no_fault_precision": tn / (tn + fn) if tn + fn else None,
        "random_ranking_false_alarm_ap": int(labels.eq(0).sum()) / n,
        "error_count": fn + fp,
        "interpretation": "当前分类阈值未优于全部判故障基线" if classification["accuracy"] <= baseline_accuracy else "当前分类准确率高于全部判故障基线",
        "review_reference_floor": metrics.get("review_reference_floor", 0.95),
        "meets_review_reference_floor": review["faults_outside_review_rate"] is not None and review["faults_outside_review_rate"] >= metrics.get("review_reference_floor", 0.95),
        "user_accepted": None,
        "previous_model_error_counts": metrics.get("previous_model_error_counts"),
        "observed_zero_errors": fn + fp == 0,
    }
    previous = summary["previous_model_error_counts"]
    if metrics.get("optimization_goal") == "zero_errors_both_classes":
        summary["interpretation"] = (
            "本次样本观察到零错误，仍不能保证未来零误判" if fn + fp == 0
            else "零误判目标未达到；" + summary["interpretation"]
        )
    if previous is not None and (fp > previous["false_positives"] or fn > previous["false_negatives"]):
        summary["interpretation"] = "候选模型未通过旧模型对比，保留原模型；零误判目标未达到"
    return summary, details


def summary_markdown(summary, metrics):
    c, r = summary["classification"], summary["review"]
    interval = c["fault_recall_95pct_wilson"]
    interval_text = "无可计算样本" if interval is None else f"{interval[0]:.2%}～{interval[1]:.2%}"
    auc_text = "不可计算" if c["roc_auc"] is None else f"{c['roc_auc']:.4f}"
    reference = "达到" if summary["meets_review_reference_floor"] else "未达到"
    title = "测试集评估摘要（非最终验收）" if summary["evaluation_role"] == "test" else "验证集验收摘要"
    return f"""# {title}

**判读：{summary['interpretation']}。仍有 {summary['missed_real_faults']} 张真实故障被预测为无故障，不能据此自动撤单。**

本报告用于辅助验收，不代替用户确认；没有改动模型、重新训练或调整阈值。

| 观察项 | 模型结果 | 参照与解释 |
| --- | --- | --- |
| 有效样本 | {summary['samples']} | 真故障 {c['faults']}，空跑 {c['false_alarms']} |
| 准确率 | {c['accuracy']:.2%} | 全判故障 {summary['baseline_accuracy']:.2%}，差值 {summary['accuracy_difference_percentage_points']:+.2f} 个百分点 |
| 真故障召回率 | {c['fault_recall']:.2%} | {summary['correct_real_faults']}/{c['faults']}；漏判 {summary['missed_real_faults']} 张 |
| 真故障召回率 95% Wilson 区间 | {interval_text} | 小样本不确定性；不是未来安全承诺 |
| 空跑识别率 | {c['false_alarm_recall']:.2%} | {summary['identified_false_alarms']}/{c['false_alarms']} |
| 预测“无故障”的正确比例 | {percentage(summary['predicted_no_fault_precision'])} | 不能把低分直接当作设备正常 |
| ROC-AUC | {auc_text} | 随机排序参考为 0.5 |
| 空跑 Average Precision | {c['false_alarm_average_precision']:.4f} | 随机排序参考约 {summary['random_ranking_false_alarm_ap']:.4f} |

## 复核建议与成本

分类阈值固定 {summary['classification_threshold']:.2f}，复核阈值固定 {summary['review_threshold']:.2f}，不得根据本评估集重新选择。

- 建议复核 {r['reviews']} 张（{r['review_share']:.2%}），其中空跑 {r['false_alarms_reviewed']} 张、真实故障 {r['real_faults_reviewed']} 张。
- 覆盖空跑 {r['false_alarm_capture']:.2%}。真故障留在原派单通道的比例 {percentage(r['faults_outside_review_rate'])}，{reference}训练阶段提出的 {summary['review_reference_floor']:.0%} 参考目标；该目标不是已确认的业务 SLA。
- 若复核成功率为 100%，潜在净工时 {r['potential_net_hours_at_100pct_success']}；若为 50%，为 {r['false_alarms_reviewed'] * 120 * 0.5 - r['review_hours']:.0f}；若为 0%，为 {-r['review_hours']}。
- 盈亏平衡成功率：{percentage(r['break_even_review_success'])}。以上仅为情景估算，未计入无法观测的故障延误及误清除损失。

模型可能有排序价值，但排序能力不等于当前阈值已经有效，更不等于已能安全自动取消工单。最终是否接受应结合用户的业务风险标准。

## 核查文件

- report.html：可筛选的中文验收页面。
- validation_details.csv：逐单真实标签、预测结果、错误类型与复核建议。
- metrics.json / predictions.csv：原始验收输出，保持不变。
- summary.json：用于复核的汇总数据，user_accepted 为 null。

模型：{summary['model']}；模型 SHA-256：{summary['model_sha256'] or '旧版未记录'}；评估数据 SHA-256：{metrics['data_quality']['input_sha256']}。
末两天曾在初版中被查看，不能视为完全未见的盲测；样本独立、缺失回传和未来分布变化等限制仍适用。
"""


def console_summary(summary, output_dir):
    """Show the decision-relevant counts before pointing to the complete report."""
    c, r = summary["classification"], summary["review"]
    auc_text = "不可计算" if c["roc_auc"] is None else f"{c['roc_auc']:.4f}"
    title = "测试集结果（非最终验收）" if summary["evaluation_role"] == "test" else "验证集结果（尚未代表用户验收通过）"
    return f"""{title}
样本：{summary['samples']} 条；真实故障 {c['faults']} 条；空跑 {c['false_alarms']} 条
判读：{summary['interpretation']}

准确率：{c['accuracy']:.2%}；全判故障基线：{summary['baseline_accuracy']:.2%}；差值：{summary['accuracy_difference_percentage_points']:+.2f} 个百分点
真实故障识别：{summary['correct_real_faults']}/{c['faults']}（{c['fault_recall']:.2%}）；漏判：{summary['missed_real_faults']} 条
空跑识别：{summary['identified_false_alarms']}/{c['false_alarms']}（{c['false_alarm_recall']:.2%}）；未识别：{summary['false_alarms_not_identified']} 条
预测无故障的正确比例：{percentage(summary['predicted_no_fault_precision'])}；ROC-AUC：{auc_text}

复核建议：{r['reviews']} 条，包含空跑 {r['false_alarms_reviewed']} 条、真实故障 {r['real_faults_reviewed']} 条
仅作复核建议，不可据此自动撤单；排序能力不等于当前阈值已达到业务要求。
中文报告：{Path(output_dir) / 'report.html'}
文字摘要：{Path(output_dir) / 'summary.md'}
逐单明细：{Path(output_dir) / 'validation_details.csv'}"""


def render_html(summary, metrics, details):
    c, r = summary["classification"], summary["review"]
    escape = lambda value: html.escape(str(value), quote=True)
    tn, fp = c["confusion_matrix_0_1"][0]
    fn, tp = c["confusion_matrix_0_1"][1]
    interval = c["fault_recall_95pct_wilson"]
    interval_text = "无可计算样本" if interval is None else f"{interval[0]:.2%}～{interval[1]:.2%}"
    rows = []
    for row in details.to_dict("records"):
        kind = {"真故障漏判": "missed", "空跑未识别": "false", "预测正确": "correct"}[row["classification_result"]]
        values = [row["ticket_id"], row.get("region", "—"), row.get("device_type", "—"),
                  row["actual_label_zh"], row["predicted_label_zh"], row["fault_probability_pct"],
                  row["classification_result"], row["review_recommendation_zh"]]
        cells = "".join(f"<td>{escape(value)}</td>" for value in values)
        rows.append(f'<tr data-kind="{kind}" data-review="{str(row["recommendation"] == "remote_review").lower()}">{cells}</tr>')
    metric_rows = [
        ("准确率", f"{c['accuracy']:.2%}", f"全判故障 {summary['baseline_accuracy']:.2%}", f"{summary['accuracy_difference_percentage_points']:+.2f} 个百分点"),
        ("真实故障召回率", f"{c['fault_recall']:.2%}", f"{tp}/{c['faults']} 识别正确", f"漏判 {fn} 张"),
        ("空跑识别率", f"{c['false_alarm_recall']:.2%}", f"{tn}/{c['false_alarms']} 识别正确", f"仍有 {fp} 张未识别"),
        ("预测无故障的正确比例", percentage(summary["predicted_no_fault_precision"]), f"{tn}/{tn + fn}", "低分不代表已确认设备正常"),
        ("ROC-AUC", f"{c['roc_auc']:.4f}" if c["roc_auc"] is not None else "不可计算", "随机排序参考 0.5000", "衡量排序，不证明可撤单"),
        ("空跑 Average Precision", f"{c['false_alarm_average_precision']:.4f}", f"随机排序参考 {summary['random_ranking_false_alarm_ap']:.4f}", "适合观察少数类空跑排序"),
    ]
    previous = summary["previous_model_error_counts"]
    if previous is not None:
        metric_rows.insert(0, ("与旧模型对比", f"误报 {fp} / 漏判 {fn}",
                              f"旧模型误报 {previous['false_positives']} / 漏判 {previous['false_negatives']}",
                              "两类错误目标均为 0"))
    metric_html = "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>" for row in metric_rows)
    reference = "达到" if summary["meets_review_reference_floor"] else "未达到"
    dataset_label = "测试集评估 · 非最终验收" if summary["evaluation_role"] == "test" else "验证集 · 用户验收参考"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{dataset_label}</title>
<style>
:root{{font-family:Inter,'Segoe UI','Microsoft YaHei',sans-serif;color:#17243b;background:#f3f6fa;font-size:15px}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1200px;margin:auto;padding:36px 24px 60px}}h1{{font-size:30px;margin:8px 0 12px}}h2{{font-size:20px;margin:0 0 16px}}p{{line-height:1.7}}.eyebrow{{color:#47627d;font-size:13px;font-weight:700;letter-spacing:2px}}.muted{{color:#5f7185}}.banner{{border-left:5px solid #c45125;background:#fff6ed;padding:20px 24px;border-radius:8px;margin:24px 0}}.banner strong{{font-size:19px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.card,section{{background:white;border:1px solid #dce4ef;border-radius:12px;padding:22px}}.value{{font-size:32px;font-weight:750;margin:10px 0 6px}}.danger{{color:#b42318}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{margin-top:20px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:13px 12px;border-bottom:1px solid #e5ebf2;vertical-align:top}}th{{background:#f5f8fc;color:#45607a}}.scroll{{overflow-x:auto}}.matrix td{{text-align:center;font-size:23px;font-weight:700}}.matrix .bad{{background:#fff0ed;color:#b42318}}.matrix .good{{background:#edf6ff;color:#194d8f}}.pill{{display:inline-block;background:#edf3fb;color:#315777;border-radius:20px;padding:6px 12px;margin-right:8px;font-size:13px}}.filters{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}select,input[type=search]{{font:inherit;border:1px solid #bac9d9;border-radius:6px;padding:9px 12px;background:white}}input[type=range]{{width:100%;margin:20px 0}}tr[data-kind=missed]{{background:#fff4f1}}tr[data-kind=false]{{background:#fffbee}}tr[hidden]{{display:none}}a{{color:#175fc0}}footer{{font-size:13px;overflow-wrap:anywhere;margin-top:24px}}.amount{{font-size:28px;font-weight:700}}button{{background:#174f87;color:white;border:0;border-radius:6px;padding:10px 16px;cursor:pointer}}@media(max-width:800px){{.cards,.grid{{grid-template-columns:1fr 1fr}}main{{padding:20px 12px}}}}@media(max-width:520px){{.cards,.grid{{grid-template-columns:1fr}}}}@media print{{.filters,button,input[type=range]{{display:none}}body{{background:white}}main{{max-width:none;padding:0}}section,.card{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">{dataset_label}</div><h1>这个模型，是否值得使用？</h1>
<p class="muted">依据已生成的评估结果展示，最终验收由你决定。</p>
<span class="pill">{escape(metrics['data_quality']['first_business_day'])} — {escape(metrics['data_quality']['last_business_day'])}</span><span class="pill">{escape(summary['model'])}</span><span class="pill">分类 {summary['classification_threshold']:.2f} / 复核 {summary['review_threshold']:.2f}</span>
<div class="banner"><strong>{escape(summary['interpretation'])}</strong><p>有 {fn} 张真实故障被预测为无故障。排序指标需与漏判、空跑识别和复核代价一起判断；当前结果不能作为自动取消工单的依据。</p></div>
<div class="cards"><div class="card">本次评估样本<div class="value">{summary['samples']}</div><div class="muted">真故障 {c['faults']} · 空跑 {c['false_alarms']}</div></div><div class="card">真故障召回率<div class="value">{c['fault_recall']:.2%}</div><div class="muted">95% 区间 {interval_text}</div></div><div class="card">真实故障漏判<div class="value danger">{fn} 张</div><div class="muted">应优先逐条核查</div></div><div class="card">空跑识别率<div class="value">{c['false_alarm_recall']:.2%}</div><div class="muted">{tn}/{c['false_alarms']} 被判为无故障</div></div></div>
<section><h2>01 · 与简单基线相比</h2><div class="scroll"><table><thead><tr><th>指标</th><th>模型</th><th>参照 / 分母</th><th>怎样解读</th></tr></thead><tbody>{metric_html}</tbody></table></div><p class="muted">“全部判为故障”能保住真实故障，但无法减少空跑。ROC-AUC 和 Average Precision 反映排序；不代表当前分类阈值已达到业务要求。区间依赖样本独立等假设。</p></section>
<div class="grid"><section><h2>02 · 具体判错了什么</h2><table class="matrix"><thead><tr><th>真实 / 预测</th><th>预测无故障</th><th>预测有故障</th></tr></thead><tbody><tr><th>实际空跑</th><td class="good">{tn}</td><td class="bad">{fp}</td></tr><tr><th>实际真故障</th><td class="bad">{fn}</td><td class="good">{tp}</td></tr></tbody></table><p class="muted">分类阈值为 {summary['classification_threshold']:.2f}。预测无故障仍需核验，不等于可以撤单。</p></section>
<section><h2>03 · 复核建议带来的代价</h2><p>建议复核 <b>{r['reviews']} 张</b>（{r['review_share']:.2%}），其中 <b>{r['false_alarms_reviewed']} 张空跑、<span class="danger">{r['real_faults_reviewed']} 张真故障</span></b>。</p><p>覆盖空跑 {r['false_alarm_capture']:.2%}；真故障留在原派单通道比例 <b>{percentage(r['faults_outside_review_rate'])}</b>，{reference}训练阶段提出的 {summary['review_reference_floor']:.0%} 参考目标，该目标不是业务 SLA。</p><label for="success">假设复核成功率：<b id="success-label">50%</b>（尚无实测）</label><input id="success" type="range" min="0" max="100" value="50"><div>潜在净工时：<span class="amount" id="net-hours">{r['false_alarms_reviewed'] * 60 - r['review_hours']}</span></div><p class="muted">120 × {r['false_alarms_reviewed']} × 成功率 − 12 × {r['reviews']}。盈亏平衡 {percentage(r['break_even_review_success'])}。这是情景估算，未计入故障延误、误清除损失。</p></section></div>
<section><h2>04 · 逐条核查评估工单</h2><div class="filters"><label>查看 <select id="filter"><option value="all">全部工单</option><option value="missed">真故障漏判</option><option value="false">空跑未识别</option><option value="errors">所有分类错误</option><option value="correct">预测正确</option><option value="review">建议远程复核</option></select></label><input id="search" type="search" aria-label="搜索工单、片区或设备" placeholder="搜索工单、片区或设备"><span id="count" aria-live="polite">{summary['samples']} / {summary['samples']} 条</span><a href="validation_details.csv">下载明细 CSV</a></div><div class="scroll"><table id="details"><thead><tr><th>工单编号</th><th>片区</th><th>设备</th><th>真实情况</th><th>模型判断</th><th>故障概率</th><th>分类结果</th><th>复核建议</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<footer class="muted"><p>明细按真实故障漏判、空跑未识别、预测正确排序。当前报告为有限样本的验收参考，没有自动给出业务“通过”结论。末两天曾在初版被查看，不能称为完全盲测。</p><p>模型指纹：{escape(summary['model_sha256'] or '旧版未记录')}</p><p>数据指纹：{escape(metrics['data_quality']['input_sha256'])}</p><a href="summary.md">文字摘要</a> · <a href="metrics.json">原始指标</a> · <a href="predictions.csv">原始预测</a> <button onclick="window.print()">打印报告</button></footer>
</main><script>
const rows = [...document.querySelectorAll('#details tbody tr')];
function filterRows() {{ const kind = document.getElementById('filter').value; const query = document.getElementById('search').value.toLowerCase(); let shown = 0; rows.forEach(row => {{ const match = kind === 'all' || row.dataset.kind === kind || (kind === 'errors' && row.dataset.kind !== 'correct') || (kind === 'review' && row.dataset.review === 'true'); row.hidden = !(match && row.textContent.toLowerCase().includes(query)); if (!row.hidden) shown++; }}); document.getElementById('count').textContent = shown + ' / ' + rows.length + ' 条'; }}
document.getElementById('filter').addEventListener('change', filterRows); document.getElementById('search').addEventListener('input', filterRows);
document.getElementById('success').addEventListener('input', event => {{ const q = Number(event.target.value); document.getElementById('success-label').textContent = q + '%'; document.getElementById('net-hours').textContent = (120 * {r['false_alarms_reviewed']} * q / 100 - {r['review_hours']}).toFixed(0); }});
</script></body></html>"""


def render_saved_report(output_dir):
    output_dir = Path(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "predictions.csv", dtype={"ticket_id": "string"}, encoding="utf-8-sig")
    summary, details = prepare_report(metrics, predictions)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_markdown(summary, metrics), encoding="utf-8")
    details.to_csv(output_dir / "validation_details.csv", index=False, encoding="utf-8-sig")
    (output_dir / "report.html").write_text(render_html(summary, metrics, details), encoding="utf-8")
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=Path("outputs/user_acceptance"), type=Path)
    args = parser.parse_args()
    try:
        summary = render_saved_report(args.output_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Report aborted: {exc}\n")
    print(console_summary(summary, args.output_dir))


if __name__ == "__main__":
    main()
