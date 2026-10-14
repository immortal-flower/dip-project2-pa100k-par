# DIP Project 2: PA100K Pedestrian Attribute Recognition

数字图像处理课程项目二：多标签非均衡样本的行人属性分类。

本项目以助教提供的 `ResNet18 + 分组分类头` baseline 为起点，在 PA100K 数据集上完成 9 个属性组、26 个属性的联合预测。项目核心不是从零搭建模型，而是完成“提出假设 → 控制变量实验 → 指标比较 → 失败分析”的研究闭环。

## 当前状态

- [x] 建立 GitHub 协作仓库
- [x] 导入助教 baseline、属性配置和依赖文件
- [x] 收录简版与详细版项目说明
- [ ] 补齐并核验 PA100K 数据
- [ ] 跑通原始 baseline，保存统一对照结果 E0
- [ ] 完成三条单因素改进实验 E1–E3
- [ ] 汇总结果、错误案例、Demo 和项目报告

> 数据完整性提醒：目前课程资源包的 `train_list.txt` 有 80,000 条记录，但压缩包中只发现 45,491 张训练图片，缺少 34,509 张。验证集和测试集各 10,000 张且数量完整。正式训练前必须向助教确认并补齐训练集；不要在不说明的情况下直接用残缺数据训练。

## 仓库结构

```text
.
├── attributes.yaml              # 9 个属性组及损失权重配置
├── starter_code/                # 助教 baseline
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── data/
│   └── README.md                 # PA100K 本地放置说明
├── docs/                         # 两份项目说明 PDF
├── requirements.txt
└── README.md
```

PA100K 图片、标签列表、训练输出和模型权重不进入 Git 仓库。每位成员在自己的电脑上按 `data/README.md` 放置数据。

## 项目说明

- [项目二说明（简版）](docs/项目二_多标签非均衡样本的行人属性分类_简版.pdf)
- [项目二说明（详细版）](docs/项目二_多标签非均衡样本的行人属性分类_详细说明.pdf)

## 环境与 baseline

建议使用 Python 3.10+。在仓库根目录执行：

```bash
pip install -r requirements.txt
python starter_code/dataset.py
python starter_code/train.py --epochs 15 --batch-size 64 --out runs/baseline
python starter_code/infer.py --ckpt runs/baseline/best.pt --img data/pa100k/test/090001.jpg
```

显存不足时可以减小 batch size，但所有对照实验必须使用同一组训练配置。首次训练前，三人共同确认数据数量、随机种子、学习率、epoch、batch size、数据增强和模型选择标准。

## 三人分工

采用“每人负责一条完整研究线”的方式。每个人都需要完成假设、实现、训练、结果分析和对应报告章节，避免把工作简单拆成代码、训练和写报告而造成贡献失衡。

### A：损失函数与类别不平衡（E1）

研究问题：Focal Loss 是否能够改善极少数正例属性，同时避免明显损害常见属性？

- 实现 binary/multilabel 属性组的 Focal Loss。
- 运行已有 `pos_weight` 对照实验。
- 补充 mA、逐属性 Precision/Recall/F1 和实例级指标。
- 重点分析 `boots`、`holding`、`lower_stripe` 等稀有属性。
- 撰写类别不平衡、损失函数和 E0/E1 对比章节。

建议分支：`feature/focal-loss`

### B：网络结构与注意力（E2）

研究问题：在 backbone 与分类头之间加入 SE 通道注意力，能否改善服装、包和配饰等细粒度属性？

- 在 ResNet18 特征与分类头之间实现 SEBlock。
- 保持损失、采样和其他训练配置不变。
- 统计参数量、单轮训练时间和最佳验证指标。
- 重点分析 `upper`、`lower`、`accessory`、`bag` 属性组。
- 撰写网络结构、SE 原理和 E0/E2 对比章节。

建议分支：`feature/se-attention`

### C：采样策略与错误分析（E3）

研究问题：稀有属性加权采样能否提高模型对少数正例的学习能力？

- 实现 WeightedRandomSampler，并说明样本权重设计。
- 统计普通采样与加权采样后的属性出现频率。
- 完成随机 9 张训练图片及标签可视化。
- 自动导出并分析至少 5 张验证集错误案例。
- 整理至少 5 张测试图 Demo，撰写数据、采样和失败分析章节。

建议分支：`feature/weighted-sampler`

## 实验矩阵

| 实验 | 唯一改动 | 负责人 | 目的 |
|---|---|---|---|
| E0 | 无，原始 baseline | 三人共同复核 | 统一对照组 |
| E1a | 启用 `pos_weight` | A | 检查正例加权效果 |
| E1b | Focal Loss | A | 聚焦困难和稀有样本 |
| E2 | SE 注意力 | B | 验证通道注意力效果 |
| E3 | WeightedRandomSampler | C | 验证采样平衡效果 |
| E4 | 组合表现最好的方法 | 三人共同完成，可选 | 检查方法能否互补 |

E1、E2、E3 必须分别与 E0 比较。完成单因素实验之前，不把多个改动直接堆叠在一起。

统一记录以下内容：训练配置、Git commit、最佳 epoch、各属性组指标、逐属性 AttrAcc/mA/Precision/Recall/F1、训练时间、失败案例和结论。

## 协作流程

1. `main` 只保存已经复核、可以复现的版本。
2. 每个人从最新 `main` 创建自己的功能分支。
3. 一次提交只处理一个明确问题，提交信息写清实验编号。
4. 合并前至少由另一名成员复核代码、配置和结果。
5. 实验输出目录使用 `runs/<实验编号>_<简短名称>/`，不要提交图片数据和本地训练输出。
6. 任何结果都要保留；指标下降也是有效结论，但必须分析原因。

建议提交信息示例：

```text
feat(E1): add focal loss for binary attribute groups
feat(E2): add SE attention after ResNet18 backbone
feat(E3): add rare-attribute weighted sampler
docs: add E1 experiment results and analysis
```

## 必交内容

- 完整 Python 源代码、`requirements.txt` 和运行说明。
- 最佳模型文件 `best.pt` 与对应配置文件。
- 完整推理代码和至少 5 张测试图输出截图。
- baseline 与改进方法的公平对比结果。
- 至少 5 张验证集错误案例及原因分析。
- 项目报告。

课程说明要求不提交 PA100K 数据集。本仓库的 `.gitignore` 已默认排除数据图片、训练输出和模型权重；最终模型如需通过 GitHub 共享，建议使用 GitHub Release 或 Git LFS。
