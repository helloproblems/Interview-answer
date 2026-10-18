# 保守输出策略使用说明

本次训练候选在测试集仍漏判 1 条真实故障，没有达到零漏判要求。默认模型文件保持原样；需要显式指定 `artifacts/safe_dispatch/fault_model.joblib` 才启用以下输出规则。

## 输出含义

- `predicted_fault_confirmed=1`：疑似故障，保留派单。
- `predicted_fault_confirmed` 留空且 `prediction_status=requires_confirmation`：待复核，不能解释为无故障，也不能填充为 0。
- 所有结果的 `dispatch_action=preserve_dispatch`，复核前不撤单或延后处理。

这是禁止自动输出“无故障”的规则，不是模型达到 100% 召回率。基模型原始概率仍保留。待复核必须计入总样本数，不能算作预测正确。

测试集共 129 条：113 条疑似故障、16 条待复核。108 条真实故障中，101 条疑似故障、7 条待复核；21 条空跑中，12 条疑似故障、9 条待复核。此流程尚未实际确认或消除任何空跑。

## 从仓库根目录运行

重建保守策略并生成测试集报告：

```powershell
uv run --locked python -m scripts.safe_policy
```

对新工单预测（将输入路径替换为你的 CSV）：

```powershell
uv run --locked python -m scripts.predict --input outputs/predict_input_example.csv --model artifacts/safe_dispatch/fault_model.joblib --output outputs/safe_dispatch/new_predictions.csv
```

以下验收命令由用户自行执行；本轮没有对验收集运行推理或评估：

```powershell
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/safe_dispatch/fault_model.joblib --output-dir outputs/user_acceptance/safe_dispatch
```

查看生成目录中的 `report.html`、`predictions.csv` 和 `metrics.json`。报告分别展示自动输出、待复核数量以及基模型的漏判与误报；完整流程准确率和用户验收结论保持未确定。

测试结果位于 `outputs/safe_dispatch/`；未达标训练候选的实验记录位于 `outputs/high_recall/`。
