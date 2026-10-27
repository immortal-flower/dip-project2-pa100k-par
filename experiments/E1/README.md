# A：损失函数与类别不平衡

2026-10-21：正式实验和结果分析已完成。先读 [A报告章节](report_A.md)、[统一比较表](server_results/E1_comparison.md)、[26属性补充表](all_attributes.md) 和 [独立核验记录](results_audit.json)。E1a的宏mA提高1.51个百分点，但宏F1下降2.16个百分点；E1b整体未改善。正式证据在server_results，顶层e0_evaluation.json/e0_review.md仅为早期CPU临时核验。

服务器的一键运行与源码包交接见 [SERVER_HANDOFF.md](SERVER_HANDOFF.md)。run_e1.py 默认只预览命令，加--execute后才开始正式流程。

## 研究假设与预先约定

E0 诊断文件显示，验证集 boots 有 45 个正例，仅检出 3 个；holding 有 86 个正例，仅检出 3 个；lower_stripe 有 21 个正例，仅检出 4 个。高 AttrAcc 掩盖了漏检。因此，本研究比较正例加权与难例聚焦能否改善少数属性识别，并检查误报增加及常见属性性能下降的代价。

| 实验 | 相对 E0 的唯一方法变化 | 预期与可证伪条件 |
| --- | --- | --- |
| E0 | 原始 BCE | 共享已发布 best.pt；统一复评，不重训 |
| E1a | BCE 的 pos_weight = 训练负例数 / 训练正例数 | 稀有属性 Recall 或 mA 可能提高，但 Precision 可能下降；若 mA/F1 不改善，则不能仅凭 Recall 宣称有效 |
| E1b | binary/multilabel BCE 换为 Focal，gamma=2，alpha=None | 降低易分样本的贡献可能改善难例；Focal 不保证稀有正例改善，下降也应记录 |

主实验参数在训练前确定，不根据验证集反复挑选 gamma、alpha 或预测阈值。alpha=None 用来避免引入额外的正负类加权；如另试 alpha=0.25，应单列探索性实验，不替换预先约定的 E1b。Focal 会降低 BCE 组损失绝对量，从而影响其与交叉熵组的相对梯度贡献；这一点是解释结果时需要保留的边界。

age/orientation 继续使用原始交叉熵。age_lt18 作为间接影响观察项：其变化只能与共享 backbone 的联合优化相关，不能描述为对年龄组直接启用 Focal 或 pos_weight。

## 固定条件与选模

- ResNet18 ImageNet 预训练，attention=none，hidden=256，drop=0.3。
- 原始 CSV 分割 80,000/10,000/10,000；Female 转 male；不混用课程零字节图片。
- 图像 256x128；训练 Resize + 水平翻转 + ImageNet 归一化；验证不随机增强。
- seed=42，15 epochs，batch=64，AdamW lr=0.0003，weight_decay=0.0005，warmup=1 epoch，随后余弦衰减，workers=4。
- CUDA 自动混合精度方式保持 E0 的 bfloat16 设置；记录实际软件和 GPU 环境。正式比较优先使用 E0 的 PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、RTX 3090。
- 每个实验仍按原九组平均 AttrAcc 选择 best.pt。mA、P/R/F1 只用于统一复评，不另选各指标最优轮次。
- 二值阈值 logits>0；互斥组 argmax。测试集不参与参数选择和失败案例筛选。

## 执行命令

以下在仓库根目录执行。Linux 服务器把数据路径替换为实际路径。训练前保存当前已核验代码的 commit；config.json 会记录 commit、dirty 状态及环境。

```powershell
python -m unittest discover -s tests -v
python starter_code/audit_e1_data.py --data-root 'D:\2026DIP\pro2\PA-100K' --out experiments/E1/data_audit.json

# 真实图片端到端 smoke，仅4个样本、随机初始化；不属于正式实验。
$env:PA100K_SMOKE_ROOT='D:\2026DIP\pro2\PA-100K'
python -m unittest discover -s tests -p test_training_smoke.py -v

# E0：只复评共享模型，不再训练。
python starter_code/evaluate.py --ckpt checkpoints/E0/best.pt --data-root 'D:\2026DIP\pro2\PA-100K' --out runs/E0_review/evaluation.json --predictions runs/E0_review/predictions.npz
python starter_code/verify_e0.py --evaluation runs/E0_review/evaluation.json --predictions runs/E0_review/predictions.npz --out runs/E0_review/review.md

# E1a：只增加正例加权。
python starter_code/train.py --data-root 'D:\2026DIP\pro2\PA-100K' --attention none --loss bce --pos-weight --epochs 15 --batch-size 64 --lr 0.0003 --weight-decay 0.0005 --warmup-epochs 1 --workers 4 --hidden 256 --drop 0.3 --seed 42 --out runs/E1a_pos_weight

# E1b：不同时开启 pos_weight。
python starter_code/train.py --data-root 'D:\2026DIP\pro2\PA-100K' --attention none --loss focal --focal-gamma 2 --epochs 15 --batch-size 64 --lr 0.0003 --weight-decay 0.0005 --warmup-epochs 1 --workers 4 --hidden 256 --drop 0.3 --seed 42 --out runs/E1b_focal

python starter_code/evaluate.py --ckpt runs/E1a_pos_weight/best.pt --data-root 'D:\2026DIP\pro2\PA-100K' --out runs/E1a_pos_weight/evaluation.json --predictions runs/E1a_pos_weight/predictions.npz
python starter_code/evaluate.py --ckpt runs/E1b_focal/best.pt --data-root 'D:\2026DIP\pro2\PA-100K' --out runs/E1b_focal/evaluation.json --predictions runs/E1b_focal/predictions.npz
python starter_code/compare_e1.py --e0 runs/E0_review/evaluation.json --e1a runs/E1a_pos_weight/evaluation.json --e1b runs/E1b_focal/evaluation.json --out runs/E1_comparison.md
python starter_code/e1_cases.py --e0 runs/E0_review/predictions.npz --e1 runs/E1a_pos_weight/predictions.npz --name E1a --data-root 'D:\2026DIP\pro2\PA-100K' --out runs/E1a_cases
python starter_code/e1_cases.py --e0 runs/E0_review/predictions.npz --e1 runs/E1b_focal/predictions.npz --name E1b --data-root 'D:\2026DIP\pro2\PA-100K' --out runs/E1b_cases
```

E0 Release 模型 SHA-256：`991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806`。下载链接见主 README 的 E0 Release。复评默认支持 CPU（--device cpu --workers 0），不需要重新下载 ImageNet 权重。

## 指标口径

| 名称 | 定义与用途 |
| --- | --- |
| legacy_mean_group_AttrAcc | 保留 E0 九组等权平均；age/orientation 按组分类准确率，仍用于选模 |
| attributes.*.AttrAcc | 所有26个属性都按 one-vs-rest 的 (TP+TN)/N；互斥组先 argmax 再 one-hot |
| attributes.*.mA | 0.5*(TP/(TP+FN)+TN/(TN+FP)) |
| macro | 对26个属性等权平均；不等于九组平均 |
| groups | 组内属性的 one-vs-rest 指标均值；不替代 legacy 组分类准确率 |
| instance | 每图计算26维 P、R、F1，再分别跨图片平均；F1不是平均P/R的调和均值 |
| 零分母 | P/R/F1记0；某属性缺少任一真实类别时 mA=null，并从宏 mA 的有效属性集合排除 |

compare_e1.py 会拒绝训练超参数、数据标签哈希或损失模式不一致的结果。预测 NPZ 保存路径、属性顺序、真实标签、预测和概率，e1_cases.py 在核验样本对齐后按文件名顺序选取改善/退化案例。图像原因必须由人工核看，不能从数字自动推断遮挡或标注错误。

## 当前交付状态

- [x] 损失开关、Focal 稳定实现，CE保持不变。
- [x] 完整评价及原 E0 返回接口兼容。
- [x] 12项测试通过，含三种模式真实图像训练/验证、模型及指标保存 smoke、案例改善/退化与样本对齐检查，以及服务器流程固定参数检查。
- [x] 可追溯结果表和改善/退化案例导出程序。
- [x] E0 的10,000张验证图CPU复评、独立混淆计数核算及原诊断差异记录（见 e0_review.md）。
- [x] 在正式服务器上同环境复评E0/E1a/E1b；服务器E0与原始诊断26/26行混淆计数一致。
- [x] E1a/E1b 两次正式15轮训练及统一复评；最佳轮次14、6。
- [x] 最终指标表、6张人工核看的案例及结果章节。
- [x] NumPy独立重算三组预测的计数和全部指标，核对15轮日志与选模结果。
- [x] 2026-10-22已下载E1a/E1b的best.pt并独立计算SHA-256，与服务器记录一致，见weights_verification.json；权重不进入普通Git历史。
- [ ] 等B/C结果齐全后完成E0–E3统一总表；尚未提交Git或推送远端。

CPU smoke与正式实验是两种证据；最终报告现已填入服务器实际结果，不使用CPU smoke推断方法效果。

本地CPU复评记录保留但不参与正式表格。服务器统一复评的E0为legacy Acc=92.3576%、宏mA=76.8736%、实例F1=83.6997%，与原始服务器诊断完全吻合；未证明早期CPU偏差的具体原因，不作单因素归因。

复核返回预测（完整NPZ在本地runs忽略目录，不进Git）：

```powershell
python starter_code/audit_e1_results.py --suite runs/server_import_20261021/E1_suite --out experiments/E1/results_audit.json
```

原始结果ZIP位于 `D:/2026DIP/pro2/E1_results.zip`。小型证据已复制至server_results，无需将完整数据集或NPZ加入源码仓库。
