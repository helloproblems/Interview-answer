# 模型优化试验结果

**未达到两类错误都为零的目标。本轮候选没有替换原模型。**

| 数据范围 | 模型 | 空跑误报 FP | 真故障漏判 FN | 总错误 |
| --- | --- | ---: | ---: | ---: |
| 训练集内时间验证 372 条 | 旧随机森林 | 43 | 5 | 48 |
| 训练集内时间验证 372 条 | 训练内选中候选 | 36 | 4 | 40 |
| 测试集 129 条 | 旧随机森林 | 16 | 2 | 18 |
| 测试集 129 条 | 候选 logistic_regression | 17 | 3 | 20 |

测试准确率：旧模型 86.05%，候选 84.50%；测试两类错误均不变差：False。

## 试验边界

- 仅使用固定的训练集，比较 22 种模型配置与 17 个阈值，共 374 种组合。训练时间验证零错误组合数为 0。
- 选择规则先要求 FP、FN 都不高于旧模型的训练内结果，再最小化总错误；平局比较 Brier 分数、距 0.50 的距离和名称。所有选择在打开测试集之前保存到 selection.json。
- 选中分类阈值为 0.40。测试集只评估这一个已冻结候选；观察测试回退后没有继续调参或挑选另一个模型。
- 模型与阈值搜索会使训练内交叉验证结果偏乐观；本次测试表现说明这种提升没有稳定迁移。没有把训练集上的高分当作零误判证明。
- 验收验证集未读取、未重新推理、未用于训练；原数据、原模型与既有本地验收结果均保留。
- 没有伪造样本、改标签、随机重分数据或用人工复核代替错误计数。仅靠这一周的有限特征，无法保证未来每条工单都判对。

## 复现与验收

运行 `uv run --locked python -m scripts.optimize` 可复现实验。`training_search.csv` 保存全部组合；`training_oof_predictions.csv` 保存训练内逐单预测；`test_comparison.json` 保存旧新模型的测试对比。候选保存在 `artifacts/optimized/`，默认预测仍使用原模型。

如果用户需要验收该候选，单独运行：

```bash
uv run --locked python -m scripts.evaluate --input datasets/validation.csv --model artifacts/optimized/fault_model.joblib --output-dir outputs/user_acceptance/optimized
```

候选模型 SHA-256：ab7f85f62e109f1c20c75bd49c0be3f00680f31f58a783a922f549c17497e8e3。
