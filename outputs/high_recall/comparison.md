# 零漏判优先模型试验

**测试集仍存在漏判，不能作为达标候选；最终验收集尚未评估。**

| 测试集 129 条 | 原模型 | 零漏判优先候选 |
| --- | ---: | ---: |
| 真实故障漏判 FN | 2 | 1 |
| 空跑误报 FP | 16 | 21 |
| 正确识别的空跑 | 5 | 0 |
| 准确率 | 86.05% | 82.95% |

目标改为先保证观察样本中的 FN=0，再尽量减少 FP；不再要求两类错误同时改善。测试预测为无故障的数量为 1，是否等同于全判故障：False。

## 训练和校准

- 使用原有训练文件 498 条；前几天 393 条用于训练和 3 折扩展时间验证，最后一天 105 条仅校准阈值。
- 在前段训练数据内比较 22 种配置。每个阈值取该模型时间验证中真实故障最低分减去固定 0.05，向下取整到两位小数；在零漏判约束下选择误报最少者，平局比较 Brier 分数和名称。
- 选中 logistic_regression。校准集中的真实故障最低分再次限制阈值，只允许降低，不允许升高。最终分类阈值 0.21，复核阈值相同，训练后不再把校准行加入拟合。
- 模型、阈值与 selection.json 在读取测试文件前保存。测试集已在前几轮被查看，因此不是全新的盲测；本轮不按测试结果继续调参。用户验收集不读取、不推理。

## 可以和不可以得出的结论

测试集真实故障 108 条。仍有 1 条真实故障漏判，零漏判目标未达到，不计算零事件置信上界。时间相关、分布变化与重复开发会限制这个解释。

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

模型 SHA-256：f3aad80f292929d16e362fe3350988e34a0d10f5b695d388d0c9d1ea86d897a5。本轮未修改根目录 README.md 或 AI_LOG.md。
