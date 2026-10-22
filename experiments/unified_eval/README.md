# E0–E3 五模型统一复评

## 复评口径

本目录使用 A 分支 `feature/focal-loss`（Release 目标提交 `66f2720b`）的统一 `evaluate.py`，在同一台机器、同一份 PA-100K 验证集上复评 E0、E1a、E1b、E3 和 E3b。二值属性使用 `logits > 0`，年龄与方向组使用 `argmax` 后转 one-hot；宏指标对 26 个属性等权平均，实例指标先逐图计算再取平均。

- 验证样本数：10,000
- `train.csv` SHA-256：`d231656a0ca3d138272d2456d97b672f848a39665c0c0a983cc182d56db74adb`
- `val.csv` SHA-256：`c99e7cc03fab148b932d4c5e66292c5e3008b648a714a451c98ecf4d4595a88c`
- 评估环境：Python 3.9.21、PyTorch 2.5.1、torchvision 0.20.1、Pillow 11.1.0、CUDA、RTX 4060 Laptop GPU
- batch size / workers：64 / 4
- 所有权重均先通过 SHA-256 校验；测试集未参与模型选择或本次比较

## 最终五模型总表

| 模型 | 唯一方法变化 | 最佳 epoch | 九组 AttrAcc | 26 属性 AttrAcc | 宏 mA | 宏 Precision | 宏 Recall | 宏 F1 | 实例 F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 原始 BCE | 10 | **92.3602%** | **93.0588%** | 76.9107% | **71.3046%** | 59.6700% | **63.2039%** | **83.7057%** |
| E1a | BCE + `pos_weight` | 14 | 90.7811% | 91.2538% | **78.4481%** | 61.0647% | **65.5430%** | 61.0549% | 80.8171% |
| E1b | Focal Loss，γ=2 | 6 | 92.1570% | 92.9481% | 75.8296% | 69.9876% | 56.8723% | 61.3345% | 83.1092% |
| E3 | 加权采样，上限 10 | 13 | 92.2417% | 92.8704% | 76.5489% | 71.2164% | 59.3327% | 62.7465% | 83.2842% |
| E3b | 加权采样，上限 5 | 7 | 92.3241% | 92.9677% | 76.9361% | 70.3317% | 59.4467% | 62.4317% | 83.4087% |

E0 在九组 AttrAcc、26 属性 AttrAcc、宏 Precision、宏 F1 和实例 F1 上最好，是最终总体基线。E1a 获得最高宏 mA 和宏 Recall，但 Precision、总体准确率与实例 F1 明显下降，体现正类加权以更多假阳性换取召回。E1b、E3、E3b 均未在总体指标上超过 E0；E3b 是三种改进中总体最接近 E0 的方案。

## 代表性少数属性 F1

| 属性 | E0 | E1a | E1b | E3 | E3b | 最佳方法 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| age_lt18 | 73.93% | 62.30% | **82.35%** | 69.75% | 80.00% | E1b |
| upper_splice | 65.83% | 67.86% | 63.47% | 66.12% | **69.60%** | E3b |
| lower_stripe | **25.81%** | 13.51% | 0.00% | 17.65% | 16.00% | E0 |
| long_coat | 54.75% | 64.16% | 60.30% | 60.29% | **64.37%** | E3b |
| boots | 10.71% | 11.86% | **21.78%** | 11.76% | 4.00% | E1b |
| holding | 6.52% | **23.42%** | 4.26% | 10.31% | 4.35% | E1a |

不同方法改善的属性并不相同，且验证集中 `lower_stripe`、`boots`、`holding` 的正例分别只有 21、45、86 个，因此应同时报告 Recall、Precision、F1 和混淆计数，不能根据单一属性的小幅变化宣称全面有效。

## 权重与结果文件

| 模型 | 权重 SHA-256 | 来源 |
| --- | --- | --- |
| E0 | `991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806` | [E0 Release](https://github.com/immortal-flower/dip-project2-pa100k-par/releases/tag/e0-baseline) |
| E1a | `19affea68f877b29a5ebccf810e145012548a6dfccca98020ab0dea91e67565b` | [A 权重 Release](https://github.com/immortal-flower/dip-project2-pa100k-par/releases/tag/a-loss-checkpoints) |
| E1b | `d2fa32ee72fd83164422bdeece6a0096de12b4dfd951405cab87f152fc7ec7e5` | [A 权重 Release](https://github.com/immortal-flower/dip-project2-pa100k-par/releases/tag/a-loss-checkpoints) |
| E3 | `6b55bb798d471e74613d9417d1ce6dbf42ec8eab9ad4fbc7d82361336a6ebc5e` | [C 权重 Release](https://github.com/immortal-flower/dip-project2-pa100k-par/releases/tag/c-weighted-sampler) |
| E3b | `056460041a52e73973130daddb142e750d310adf7fd419a7d84463b78e59139f` | [C 权重 Release](https://github.com/immortal-flower/dip-project2-pa100k-par/releases/tag/c-weighted-sampler) |

完整逐属性、逐组、混淆计数、权重参数、评估环境和标签文件哈希见：

- [`E0_evaluation.json`](E0_evaluation.json)
- [`E1a_evaluation.json`](E1a_evaluation.json)
- [`E1b_evaluation.json`](E1b_evaluation.json)
- [`E3_evaluation.json`](E3_evaluation.json)
- [`E3b_evaluation.json`](E3b_evaluation.json)
- [`five_model_summary.json`](five_model_summary.json)

## 环境差异核查

A 服务器使用 PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、Pillow 10.4.0 和 RTX 4090。本机使用相同 E0 权重、验证集和评估脚本时，九组 AttrAcc 为 92.3602%，服务器记录为 92.3576%；差异来自少量阈值附近预测。因此最终总表全部采用上面的本机同环境结果，不混用服务器数值。
