# DIP Project 2: PA100K Pedestrian Attribute Recognition

数字图像处理课程项目二：多标签非均衡样本的行人属性分类。

本项目以助教提供的 `ResNet18 + 分组分类头` baseline 为起点，在 PA100K 数据集上完成 9 个属性组、26 个属性的联合预测。项目核心不是从零搭建模型，而是完成“提出假设 → 控制变量实验 → 指标比较 → 失败分析”的研究闭环。

## 当前状态

- [x] 建立 GitHub 协作仓库
- [x] 导入助教 baseline、属性配置和依赖文件
- [x] 收录简版与详细版项目说明
- [x] 补齐并核验 PA100K 数据（数据不进入仓库）
- [x] 跑通 baseline，保存统一对照结果 E0
- [ ] 完成三条单因素改进实验 E1–E3
- [ ] 汇总结果、错误案例、Demo 和项目报告

> 数据完整性提醒：课程资源包中的训练、验证图片为零字节，不能用于训练。已另行核验一份原始 PA-100K：100,000 张图片均能解码，CSV 分别包含 80,000/10,000/10,000 条记录。使用时通过 `--data-root` 指向完整数据集目录；图片不进入 Git 仓库。

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
├── experiments/E0/               # E0 配置、逐轮指标、汇总与诊断
├── docs/                         # 两份项目说明 PDF
├── requirements.txt
└── README.md
```

PA100K 图片、原始标签、完整训练目录和模型权重不进入普通 Git 历史；仅保留选定的小型实验记录。代码支持直接读取原始数据集的 `data/*.jpg` 与 `train.csv`、`val.csv`、`test.csv`，无须复制或重排图片；详见 `data/README.md`。

E0 已在服务器上完成：最佳第 10 轮，验证集九组平均 AttrAcc 为 92.36%，总耗时 19.93 分钟。配置、训练曲线和逐属性诊断见 [E0 实验记录](experiments/E0/README.md)。B 负责的共享 SE 和五种后续结构对照已完成，代码、配置、结果、结构图及案例分析见 [B 部分 SE 实验记录](experiments/E2/README.md)。这些结果在 B 功能分支上，合入 `main` 前仍需其他成员复核。

## 项目说明

- [项目二说明（简版）](docs/项目二_多标签非均衡样本的行人属性分类_简版.pdf)
- [项目二说明（详细版）](docs/项目二_多标签非均衡样本的行人属性分类_详细说明.pdf)

## 环境与 baseline

建议使用 Python 3.10+。在仓库根目录执行（将数据路径替换为本机完整 PA-100K 目录）：

```powershell
pip install -r requirements.txt
python starter_code/dataset.py --data-root "D:\ShuZiTuXiangChuLi_Lab\挑战项目2\PA-100K"
python starter_code/train.py --data-root "D:\ShuZiTuXiangChuLi_Lab\挑战项目2\PA-100K" --attention none --epochs 15 --batch-size 64 --out runs/E0_baseline
python starter_code/train.py --data-root "D:\ShuZiTuXiangChuLi_Lab\挑战项目2\PA-100K" --attention se --epochs 15 --batch-size 64 --out runs/E2_se
python starter_code/infer.py --ckpt runs/E2_se/best.pt --img "D:\ShuZiTuXiangChuLi_Lab\挑战项目2\PA-100K\data\090001.jpg"
python starter_code/compare_se.py runs/E0_baseline runs/E2_se --out runs/E0_E2_comparison.md
```

显存不足时可以减小 batch size，但 E0/E2 必须使用同一组训练配置。两次训练分别写入独立目录；每个目录包含 `config.json`、`metrics.jsonl`、`summary.json`、日志和模型权重。模型按验证集的组均 AttrAcc 选出最佳 epoch，其他指标可用统一评价程序复评。B 的结构对比图见 [SE 结构图](docs/se_architecture.svg)。

## 三人分工

### 分工原则

不按“一个人写代码、一个人训练、一个人写报告”横向切割，而是每人负责一条完整研究线。A、B、C 都必须独立完成：

```text
提出假设 → 修改代码 → 运行公平对照 → 整理指标 → 分析成败 → 撰写自己的报告章节
```

三条主线均为项目说明中的二星难度方向。A 多完成一组损失实验和评价指标；B 负责结构改造、计算成本和结构图；C 负责采样改造、数据可视化和错误案例，因此总工作量大致相当。

### 一页分工总表

| 成员 | 核心研究线 | 主要代码位置 | 必跑实验 | 必须产出 | 分支 |
|---|---|---|---|---|---|
| A | 损失函数与类别不平衡 | `losses.py`、`evaluate.py` | E1a、E1b | 完整指标程序、损失对比表、稀有属性分析、报告章节 | `feature/focal-loss` |
| B | 网络结构与注意力 | `model.py`、训练统计代码 | E2 | SE 模型、结构图、参数量/速度对比、属性组分析、报告章节 | `feature/se-attention` |
| C | 采样策略与错误分析 | `dataset.py`、`train.py`、分析脚本 | E3 | 加权采样器、分布图、错误案例、Demo、报告章节 | `feature/weighted-sampler` |

### 共同起点：E0 baseline

E0 只训练一次，作为所有人的统一对照组。开始三条改进线之前，三人共同确认以下内容并记录到实验表：

- 数据集确实完整，训练/验证/测试数量分别为 80,000 / 10,000 / 10,000。
- 固定随机种子、epoch、batch size、学习率、图像尺寸和数据增强。
- 保存 `config.json`、`log.txt`、`best.pt` 和最佳 epoch。
- 保存各属性组指标及 26 个逐属性指标。
- 记录运行 E0 的 Git commit；后续实验都从该版本建立分支。

E0 的准备工作也做明确分配：A 核对标签与评价结果，B 运行并保存 baseline，C 检查推理输出和结果文件是否齐全。三人共同签字确认后，E0 才视为冻结。

### A：损失函数与类别不平衡（E1）

**研究问题：** Focal Loss 是否能够改善极少数正例属性，同时避免明显损害常见属性？

**代码任务**

- 在 `starter_code/losses.py` 中实现 binary/multilabel 属性组的 Focal Loss。
- 保持 age、orientation 的 CrossEntropyLoss 不变。
- 在训练参数中加入清晰的损失模式选择，不能靠手动改代码切换实验。
- 在 `starter_code/evaluate.py` 中补齐：逐属性 mA、Precision、Recall、F1，以及实例级 Precision/Recall/F1。
- 保证原始 BCE、`pos_weight`、Focal Loss 都能从命令行复现。

**实验任务**

- E1a：只启用已有的 `pos_weight`，其余配置与 E0 完全一致。
- E1b：只把 BCE 换为 Focal Loss，其余配置与 E0 完全一致。
- 对比 E0/E1a/E1b，重点检查 `boots`、`holding`、`lower_stripe`、`age_lt18`。
- 分析是否出现“Recall 上升但 Precision 下降”等代价。

**A 的交付清单**

- [ ] 可配置的 Focal Loss 实现及必要注释。
- [ ] 完整评价指标代码和验证结果。
- [ ] E0/E1a/E1b 总表及稀有属性专项表。
- [ ] 至少 2 张能说明改进或失败的案例图。
- [ ] 报告中的“类别不平衡、损失函数、指标与实验结果”章节。

完成标准：其他成员按照 README 命令能够复现三种损失模式，表格中的数字能够从保存的结果文件追溯。

### B：网络结构与注意力（E2）

**研究问题：** 在 backbone 与分类头之间加入 SE 通道注意力，能否改善服装、包和配饰等细粒度属性？

**代码任务**

- 在 `starter_code/model.py` 中实现 SEBlock。
- 将 SE 接在 ResNet18 输出的 512 维特征与 9 个分类头之间。
- 加入模型参数，使 baseline 与 SE 模型可以从命令行或配置切换。
- 保证关闭 SE 时模型行为与 E0 一致，推理脚本能根据 checkpoint 正确重建模型。
- 增加参数量和训练耗时统计，不改变 A、C 所负责的损失及采样逻辑。

**实验任务**

- E2：只增加 SE，损失函数、采样器、增强和训练超参数全部保持 E0 设置。
- 对比参数量、每轮耗时、最佳 epoch、总体指标和各属性组指标。
- 重点分析 `upper`、`lower`、`accessory`、`bag`，同时检查是否牺牲其他属性。
- 绘制一张 baseline 与 SE 模型结构对比图。

**B 的交付清单**

- [x] 可开关的 SE 模型及 checkpoint 加载支持。
- [x] baseline/SE 结构对比图。
- [x] E0/E2 指标、参数量和训练时间对比表。
- [x] 至少 2 张细粒度属性成功或失败案例图（留在本地交付包；公开仓库只收录案例索引及分析，不上传数据集图片）。
- [x] 报告中的“网络结构、SE 原理、计算成本与实验结果”章节。

完成标准：关闭 SE 能复现 E0；开启 SE 的差异只有注意力模块，结构图、参数统计和实验配置能够互相对应。

### C：采样策略与错误分析（E3）

**研究问题：** 稀有属性加权采样能否让模型在训练阶段看到更多有效正例，并提高少数属性识别能力？

**代码任务**

- 在数据加载流程中实现 WeightedRandomSampler，并说明单张图片同时具有多个稀有属性时的权重计算方法。
- 加入命令行或配置开关，在普通 shuffle 和加权采样之间切换。
- 统计并可视化采样前后的属性出现频率，检查是否存在少数样本被过度重复。
- 编写数据检查/可视化脚本，随机展示 9 张训练图及其标签。
- 编写错误案例导出功能，保存图片、真实标签、预测结果和置信度。

**实验任务**

- E3：只启用 WeightedRandomSampler，模型、损失、增强和训练超参数保持 E0 设置。
- 比较 E0/E3 的稀有属性 Recall、Precision、mA 和整体指标。
- 对至少 5 张验证集错误图片进行遮挡、模糊、混淆、标注噪声等原因分析。
- 从最终候选模型中制作至少 5 张测试图 Demo。

**C 的交付清单**

- [ ] 可开关的 WeightedRandomSampler。
- [ ] 9 张训练样本标签检查图。
- [ ] 采样前后分布图及 E0/E3 结果表。
- [ ] 至少 5 张错误案例及逐张解释。
- [ ] 至少 5 张最终测试 Demo。
- [ ] 报告中的“数据分布、采样策略、错误分析与 Demo”章节。

完成标准：采样权重公式和实现一致，可以从统计图证明采样分布发生了什么变化，并能说明指标变化的原因。

### 最终整合责任

- A 汇总所有实验的统一指标总表，检查指标口径一致。
- B 负责把表现最好的方法组合成可选 E4，并检查训练/推理兼容性。
- C 汇总三人的报告章节、图片和 Demo，但 A、B 必须分别校对自己提供的文字与数字。
- 最终提交前，三人共同从干净环境执行一次 README 中的训练或推理命令。

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
