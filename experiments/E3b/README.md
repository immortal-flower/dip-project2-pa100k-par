# E3b：温和加权采样消融实验

## 目的

E3 使用最大属性权重 10，能够改善部分极稀有属性，但一个 epoch 只覆盖 47,804 张不同图片，整体 AttrAcc 与实例级 F1 略降。E3b 只把 `sampler_max_weight` 从 10 降到 5，验证降低过采样强度能否恢复样本多样性和整体性能。

模型、损失函数、数据增强、随机种子、优化器、batch size 和 epoch 均与 E0/E3 相同。

```powershell
python starter_code/analyze_sampling.py --data-root data/pa100k --out experiments/E3b --sampler-max-weight 5
python starter_code/train.py --data-root data/pa100k --attention none --sampler weighted --sampler-max-weight 5 --epochs 15 --batch-size 64 --workers 4 --seed 42 --out runs/E3b_weighted_sampler_cap5
```

## 采样强度

| 项目 | E3（上限 10） | E3b（上限 5） |
| --- | ---: | ---: |
| 一个 epoch 覆盖的不同图片 | 47,804 | 49,081 |
| 没有被抽到的图片 | 32,196 | 30,919 |
| 单张图片最大重复次数 | 10 | 9 |
| 归一化后的最大图片权重 | 3.156 | 1.693 |

| 属性 | 原始正例率 | E3 采样率 | E3b 采样率 |
| --- | ---: | ---: | ---: |
| age_lt18 | 5.19% | 7.18% | 7.33% |
| age_gt60 | 1.41% | 3.70% | 2.42% |
| lower_stripe | 0.50% | 1.66% | 0.87% |
| boots | 0.62% | 2.00% | 1.05% |
| holding | 0.96% | 3.03% | 1.62% |

![E3b 采样分布](sampling_distribution.png)

## E0、E3、E3b 结果

E3b 最佳 checkpoint 出现在第 7 轮，总训练时间 37.73 分钟。

| 项目 | E0 | E3 | E3b |
| --- | ---: | ---: | ---: |
| 九组平均 AttrAcc | **92.36%** | 92.24% | 92.32% |
| 实例级 Precision | 85.96% | 85.46% | **86.02%** |
| 实例级 Recall | **82.38%** | 82.02% | 81.76% |
| 实例级 F1 | **83.71%** | 83.28% | 83.41% |

逐属性复评使用 SHA-256 为 `991891f5…c806` 的官方 E0 Release 权重，并与 E3/E3b 在同一验证集、评估脚本和推理环境中运行：

| 属性 | E0 Recall / F1 | E3 Recall / F1 | E3b Recall / F1 |
| --- | ---: | ---: | ---: |
| age_lt18 | 63.48% / 73.93% | 56.48% / 69.75% | **69.99% / 80.00%** |
| upper_splice | 62.30% / 65.83% | 63.49% / 66.12% | **69.05% / 69.60%** |
| lower_stripe | **19.05% / 25.81%** | 14.29% / 17.65% | 9.52% / 16.00% |
| long_coat | 43.57% / 54.75% | 49.62% / 60.29% | **63.69% / 64.37%** |
| boots | 6.67% / 10.71% | **6.67% / 11.76%** | 2.22% / 4.00% |
| holding | 3.49% / 6.52% | **5.81% / 10.31%** | 2.33% / 4.35% |

E3b 对 `age_lt18`、`upper_splice` 和 `long_coat` 的提升明显，并追回大部分总体 AttrAcc；但 `lower_stripe`、`boots`、`holding` 均低于官方 E0。E3 与 E3b 揭示了明确的强度权衡，但也说明仅凭属性频率无法保证采样收益。

![E3b 的成功与失败案例](e0_e3b_error_cases.png)

## 结论

E3b 是比 E3 更均衡的设置，但仍没有在总体指标上超过 E0。若以总体性能为主，应保留 E0；E3 可用于展示 `boots`、`holding` 的小幅改善，E3b 可用于展示 `age_lt18`、`upper_splice`、`long_coat` 的改善以及采样强度权衡。五模型统一复评进一步表明，`pos_weight` 的宏 mA/Recall 最高但误报代价明显，Focal Loss 对 `boots` 最有效，没有一种不平衡处理方法在所有属性和总体指标上占优。

完整配置、逐轮指标、逐属性比较和图片均保存在本目录。模型权重保留在本地 `runs/E3b_weighted_sampler_cap5/`，不进入普通 Git 历史。
