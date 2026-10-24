# B：SE 通道注意力实验

本目录是 B 成员的结构改造交付，记录 E0 与六种 SE 门控的公平对照。核心结论、九组与 26 属性结果、计算成本和逐图案例分析见 [实验报告](report.md)。原始 E2 指共享 SE；后五种为同一研究线内的后续独立结构实验。目录中 `E3_residual`、`E4_adaptive`、`E5_attribute` 是 B 线历史运行代号，**不是**项目总表中 C 负责的 E3 加权采样或可选的 E4 方法组合。

## 交付文件

- `../../starter_code/model.py`：可选关闭 SE、共享 SE、九组 SE 和逐属性 SE；关闭时保持 E0 结构。
- `../../starter_code/train.py`、`../../starter_code/infer.py`：训练开关与按 checkpoint 配置重建模型。
- `tables/overall.csv`：参数量、最佳 epoch、训练时间、组均 AttrAcc、宏平均 mA/F1。
- `tables/group_metrics.csv`、`tables/attribute_metrics.csv`：九组与 26 属性指标；`tables/config_audit.json`：控制变量核对。
- `configs/`、`summaries/`：七个模型各自的原始配置和最佳轮摘要。
- `diagrams/`：六版 SE 各自与 E0 对比的 PNG/SVG 结构图。
- `tables/cases_original.csv`、`tables/cases_additional.csv`：六版 SE 各一例修正与一例新增错误，共 12 个同图记录。
- `checkpoints/<模型代号>/best.pt`：六版 SE 的最佳权重，使用 Git LFS，不进入普通 Git 对象。
- `checkpoint_verification.txt`：服务器严格加载七个 checkpoint 的检查结果。

仓库不提交 PA-100K 图片或含图案例卡；报告保留案例文件名、真值、E0/SE 分数和目视描述。拥有原始数据集的组员可以按 `val.csv` 中的文件名定位原图。本地交付包中的 12 张案例卡未推送到此公开仓库。

## 复现

先安装 `requirements.txt`，并按根目录 `data/README.md` 准备完整 PA-100K。要下载权重，执行 `git lfs pull`。在仓库根目录运行，例如：

```bash
python starter_code/train.py --data-root /path/to/PA-100K --attention se --epochs 15 --batch-size 64 --lr 0.0003 --weight-decay 0.0005 --warmup-epochs 1 --workers 4 --hidden 256 --drop 0.3 --se-reduction 16 --seed 42 --out runs/E2_se
python starter_code/infer.py --ckpt experiments/E2/checkpoints/E2_shared/best.pt --img /path/to/PA-100K/data/090001.jpg
```

其他五版只需把 `--attention` 分别设为 `se_group`、`se_group_center`、`se_group_residual025`、`se_group_adaptive025`、`se_attr_residual025`，并使用不同的 `--out` 目录。具体参数以 `configs/` 为准，切勿更改损失、采样或增强后仍称为本次 E2 公平对照。比较脚本可用 `python starter_code/compare_se.py runs/E0_baseline runs/E2_se --out runs/E0_E2_comparison.md`。

全部结果来自一次固定随机种子的验证集运行；不同日期的共享服务器负载会影响墙钟时间，指标小幅差异不构成显著性结论。按三人协作约定，本功能分支应经另一成员复核后再合入 `main`。
