# 先零漏判，再尽量识别空跑

**测试仍存在真实故障漏判，未达到目标，不能据此取消或延后派单。默认模型未替换。**

| 测试集 129 条 | 原随机森林 | 本轮候选 | 全判故障参照 |
| --- | ---: | ---: | ---: |
| 真实故障正确识别 TP | 106 | 105 | 108 |
| 真实故障漏判 FN | 2 | 3 | 0 |
| 空跑正确识别 TN | 5 | 0 | 0 |
| 空跑误报 FP | 16 | 21 | 21 |

## 训练约束

- 固定训练文件 498 条，前段 393 条用于 2 个扩展日期折，最后一个训练日 105 条仅校准阈值；不再回填拟合。
- 对比 26 种固定配置，含原始特征基线、非线性交互、故障类别权重 1/4/12，以及软投票集成；没有伪造样本或改标签。
- 第一目标：时间折预测 FN=0。第二目标：TN 最大；平局依次比较能识别空跑的日期数、Brier 分数、模型名。仅用训练标签。
- 阈值取真实故障最低分的前一个浮点数，边界归为故障；相较固定 0.05 余量和两位小数取整，在同一零训练漏判约束下保留更多识别空跑的空间。最后训练日只能降低阈值。
- 选中 interactions_rbf_c2.0_faultweight4，分类与复核阈值均为 0.54400414925851026。训练内时间验证 290 条、真实故障 253 条、空跑 37 条，选中候选识别空跑 4 条；这些是参与选择的数据，不能作为独立有效性证明。
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

模型指纹：0311e20a7fa062b1f93b174286cb345d6523efdc222d2ee377d7b9f224819817。根目录 README.md 和 AI_LOG.md 保持不变。
