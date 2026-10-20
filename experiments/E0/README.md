# E0 baseline 实验记录

E0 使用 ResNet18 + 9 个分组分类头，关闭 SE（`attention=none`），未启用 `pos_weight` 或加权采样。输入是原始 PA-100K 的 `data/*.jpg` 与 `train.csv`、`val.csv`；训练/验证样本数分别为 80,000/10,000。训练环境为 PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、RTX 3090。完整参数见 [`config.json`](config.json)。

| 项目 | 结果 |
| --- | ---: |
| 随机种子 | 42 |
| Epoch / batch size | 15 / 64 |
| 最佳 epoch | 10 |
| 最佳验证集九组平均 AttrAcc | 92.36% |
| 模型参数量 | 12,365,146 |
| 总训练耗时 | 19.93 分钟 |

逐轮记录在 [`metrics.jsonl`](metrics.jsonl)，最佳 epoch 的各组和逐属性 AttrAcc 在 [`summary.json`](summary.json)，训练日志在 [`log.txt`](log.txt)。[`e0_diagnostic.txt`](e0_diagnostic.txt) 补充了训练/验证正例率、验证集逐属性 mA/Precision/Recall/F1、混淆计数和错误图片候选。

注意：`summary.json` 的最佳模型选择指标是九组平均 **AttrAcc，不是 mA**。其中 age/orientation 的“逐属性 AttrAcc”是组准确率的重复值；这两个互斥组的逐类别指标应以诊断文件中的 one-vs-rest 计算为准。boots/holding 等稀有属性的高 AttrAcc 也不代表高召回率。

模型文件 `best.pt`（47.3 MiB）保存在原始 E0 归档中，计划作为 GitHub Release 附件发布，而不纳入普通 Git 历史。其 SHA-256 为 `991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806`。`last.pt`、逐图预测文件、错误图片候选压缩包与 PA-100K 原始数据不提交到仓库。
