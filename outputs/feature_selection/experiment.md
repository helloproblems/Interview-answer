# 按零漏判目标筛选训练列

**仍有真实故障漏判，未达到目标，不替换默认模型。**

## 本轮选择

- 算法：logistic_regression；保留特征：telemetry_age_minutes, reported_health_score, rated_power_kw, device_type；不参与拟合的特征：region。
- 工单 ID、真实标签和绝对日期始终不是预测特征。时间戳仅用于计算遥测延迟与按时间分组；原始 CSV 列没有删除。
- 本轮不加入交互项，固定 11 个特征组合 × 3 个算法，共 33 个候选。相同算法在各组合中使用相同超参数、训练行、验证日期和阈值规则。
- 先要求训练内时间验证 FN=0，再使 TN 最大；平局依次比较识别空跑的日期数、更少的特征列、Brier 分数和名称。只用训练集选择；校准日只可降低阈值。
- 选中候选在训练内时间验证中：故障漏判 0/253，识别空跑 3/37。这些数据参与了选择，不是独立有效性证据。

## 冻结后的测试结果

| 测试集 129 条 | 原默认随机森林 | 同算法全部 5 列对照 | 本轮选中列 |
| --- | ---: | ---: | ---: |
| 故障漏判 FN / 108 | 2 | 1 | 2 |
| 识别空跑 TN / 21 | 5 | 1 | 3 |
| 空跑误报 FP | 16 | 20 | 18 |

同算法对照使用相同的 393 条拟合数据、105 条训练内校准数据。两者的阈值按相同规则独立确定，均在读取测试数据前冻结；没有按测试结果再换列、调参或选择赢家。默认模型的训练范围不同，仅作历史参照。

删列的价值须结合上述配对结果判断；不能仅因训练内得分提高，就断言某列永远是干扰项。`feature_ablation.csv` 给出同算法的全部删列对比，`training_folds.csv` 给出逐日计数，`full_features_control_predictions.csv` 保留对照逐单结果。

测试集已经在历史实验中查看过，并非新的盲测。本轮验收集未读取、未推理。候选 `predicted_fault_confirmed` 仍为 0/1，漏判照实计数；所有预测仅作建议，不能直接取消派单。

## 复现与验收

训练与测试：`uv run --locked python -m scripts.select_features`

仅由用户自行执行验收：

```powershell
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/feature_selection/fault_model.joblib --output-dir outputs/user_acceptance/feature_selection
```

CSV 输入接口仍保留原有字段；模型内部的列选择器会丢弃未选中的列，无需改写原始数据文件。模型指纹：14fd7340b2a557395da7575991b68d94a204e6120ff501d2db6ee0f7b06a19b2。

README.md 和 AI_LOG.md 本轮不修改。
