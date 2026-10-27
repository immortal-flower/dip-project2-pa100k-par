# E4 指标组合优化：E1a + E2_center 概率软投票

## 结论

E4 不重新训练，也不在测试集上调参。它在验证集上对六个候选模型的概率输出进行两两组合和 0.05 步长权重搜索，最终冻结为：

```text
E4 probability = 0.45 × E1a_pos_weight + 0.55 × E2_center_SE
binary / multilabel threshold = 0.5
softmax groups = argmax after probability averaging
```

这组组合在独立测试集上同时超过统一 E0 的四项选模指标。相对 E0，九组平均 AttrAcc 提高 **0.41 个百分点**，macro mA 提高 **1.30 个百分点**，macro F1 提高 **0.66 个百分点**，instance F1 提高 **0.87 个百分点**。

## 方法与防止测试集泄漏

1. 使用 `evaluate.py --predictions` 分别保存 E0、E1a、E1b、E2_center、E3、E3b 在同一验证集上的概率。
2. 检查六份文件的样本顺序、属性顺序和标签完全一致。
3. 枚举任意两个模型及 `[0.00, 0.05, ..., 1.00]` 的软投票权重。
4. 以 E0 为基准，对“九组平均 AttrAcc、macro mA、macro F1、instance F1”四项指标的相对比值取等权平均，作为验证集组合分数。
5. 选出 E1a/E2_center 的 0.45/0.55 权重后冻结配置，才生成并读取这三个模型的测试集概率；测试集不参与模型或权重选择。

没有做逐属性阈值搜索，避免在 10,000 张验证样本、特别是极少数属性上引入额外过拟合。

## 验证集结果

| 模型 | 九组平均 AttrAcc | macro mA | macro F1 | instance F1 |
|---|---:|---:|---:|---:|
| E0 | 92.36% | 76.91% | 63.20% | 83.70% |
| E1a pos_weight | 90.78% | 78.45% | 61.05% | 80.82% |
| E2_center SE | 92.37% | 77.02% | 63.64% | 83.34% |
| **E4：0.45 E1a + 0.55 E2_center** | **92.60%** | **78.45%** | **65.53%** | **84.21%** |

E1a 提供了对正类不平衡更敏感的概率，E2_center 保留了更强的总体准确率和实例级表现；软投票把两者互补性转化为四项指标的共同提升。

## 冻结后测试集结果

| 模型 | 九组平均 AttrAcc | macro AttrAcc | macro mA | macro F1 | instance F1 |
|---|---:|---:|---:|---:|---:|
| E0 | 92.99% | 93.37% | 76.09% | 63.08% | 84.49% |
| E1a pos_weight | 91.75% | 91.79% | 78.39% | 60.11% | 82.11% |
| E2_center SE | 93.05% | 93.44% | 75.41% | 62.07% | 84.50% |
| **E4：0.45 E1a + 0.55 E2_center** | **93.40%** | **93.70%** | **77.39%** | **63.74%** | **85.36%** |
| E4 − E0（百分点） | **+0.41** | **+0.33** | **+1.30** | **+0.66** | **+0.87** |

## 稀有属性边界

总体指标共同提升不代表每个稀有属性都改善。测试集上，`upper_splice`、`long_coat`、`boots` 的 F1 比 E0 提高；`age_lt18`、`lower_stripe`、`holding` 下降。尤其 `boots`、`holding` 的测试正例仅 55 和 101 个，绝对 F1 仍很低。因此 E4 应表述为“更好的整体折中”，不能宣称彻底解决了极端少数类问题。

| 属性 | 测试正例数 | E0 F1 | E4 F1 | 变化（百分点） |
|---|---:|---:|---:|---:|
| age_lt18 | 92 | 32.79% | 15.79% | -17.00 |
| upper_splice | 447 | 74.26% | 77.83% | +3.57 |
| lower_stripe | 25 | 43.24% | 42.31% | -0.93 |
| long_coat | 448 | 52.13% | 54.39% | +2.26 |
| boots | 55 | 0.00% | 2.90% | +2.90 |
| holding | 101 | 5.50% | 3.60% | -1.90 |

完整 26 属性数据见 `test_attribute_comparison.csv`。

## 复现

先用六份验证集预测搜索权重：

```powershell
python starter_code/optimize_ensemble.py optimize `
  --attributes attributes.yaml --split val --baseline E0 --step 0.05 `
  --out-dir experiments/E4_combination `
  --prediction E0=runs/metric-combination/predictions/E0_val.npz `
  --prediction E1a=runs/metric-combination/predictions/E1a_val.npz `
  --prediction E1b=runs/metric-combination/predictions/E1b_val.npz `
  --prediction E2_center=runs/metric-combination/predictions/E2_center_val.npz `
  --prediction E3=runs/metric-combination/predictions/E3_val.npz `
  --prediction E3b=runs/metric-combination/predictions/E3b_val.npz
```

再用冻结配置进行测试集复评：

```powershell
python starter_code/optimize_ensemble.py apply `
  --attributes attributes.yaml `
  --config experiments/E4_combination/ensemble_config.json `
  --split test --out-dir experiments/E4_combination `
  --prediction E1a=runs/metric-combination/predictions/E1a_test.npz `
  --prediction E2_center=runs/metric-combination/predictions/E2_center_test.npz `
  --reference E0=runs/metric-combination/predictions/E0_test.npz
```

## 文件

- `ensemble_config.json`：冻结模型、权重、评分口径及验证预测 SHA-256。
- `val_evaluation.json`：所有单模型验证结果与最优组合。
- `weight_search.json`：全部两模型权重搜索记录。
- `test_evaluation.json`：冻结后的测试结果及输入预测 SHA-256。
- `test_attribute_comparison.csv`：E0、E1a、E2_center、E4 的 26 属性 mA/F1 对照。
- `starter_code/optimize_ensemble.py`：可复现优化与应用脚本。
- `starter_code/infer_ensemble.py`：直接读取两份 checkpoint 的冻结组合推理脚本。

模型本体仍为 Release 中的两份 `best.pt`，E4 以 `ensemble_config.json` 作为组合配置，不复制或修改权重。
