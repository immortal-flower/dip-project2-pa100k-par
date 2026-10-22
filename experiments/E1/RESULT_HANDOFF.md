# A线结果交接（2026-10-21）

2026-10-22补记：E1a/E1b权重已下载，分别位于 `D:/2026DIP/pro2/runs/best.pt` 与 `D:/2026DIP/pro2/runs/focal/best.pt`。独立计算两个文件的SHA-256，均与服务器evaluation记录一致，见 [weights_verification.json](weights_verification.json)。以下下载步骤为历史交接说明，现已完成，无需重复下载。此补记不表示重新执行模型前向。

主文：[report_A.md](report_A.md)。总表：[server_results/E1_comparison.md](server_results/E1_comparison.md)。完整26属性：[all_attributes.md](all_attributes.md)。逐图三模型分数：[selected_cases.json](selected_cases.json)。独立核验：[results_audit.json](results_audit.json)。

## 结论摘要

- E1a：宏mA 76.87%→78.38%，宏F1 63.17%→61.00%，实例F1 83.70%→80.78%。属于召回/误报取舍，不是全面提升。
- E1b：宏mA 75.85%、宏F1 61.47%、实例F1 83.13%，整体未改善；boots局部改善，lower_stripe全负预测。
- 三模型各10,000张验证图独立重算通过；E0服务器复评与原诊断26/26混淆行一致。A已完成假设、代码、对照、指标、分析及章节，仍需参与小组最终结果整合。

## 原始证据与保存边界

输入ZIP：`D:/2026DIP/pro2/E1_results.zip`，SHA-256：

```text
b81eb599efd44d78486a64b8bf0816e6e684f3083eae2c49c9cd2c19b519cba6
```

完整返回包53个条目通过ZIP CRC检查，原始NPZ/控制台记录保留于本地runs/server_import_20261021。Git交接只保留小型配置、训练日志、指标、源码清单和6张选定案例，不包含完整数据集、NPZ或模型。6张案例仅是说明性子集，不能充当小组最终“至少5张错误案例”的完整交付：其中3张是恢复正例，3张是退化错误；小组错误案例需与C线合并检查。

服务器训练快照基于2ba3659aa108dde04b8a6c8c1f9b5c51a947497e及当时的本地修改，见server_results/SOURCE_MANIFEST.json；不是一个已推送的Git提交。本次只增加结果核验/记录，不改原训练或评价逻辑。未提交、未推送，未替B/C补写不存在的实验结果。

## 下一步：另行下载E1模型

结果ZIP没有best.pt。请保留服务器原文件，另将下面两个文件下载到本机检查点目录或个人备份位置，不要加入普通Git历史：

```text
/project/IDIP/ZZS/pro2/runs/E1_suite/E1a_pos_weight/best.pt
/project/IDIP/ZZS/pro2/runs/E1_suite/E1b_focal/best.pt
```

可在服务器终端执行只读校验后，把输出与下面记录比较：

```bash
cd /project/IDIP/ZZS/pro2
sha256sum runs/E1_suite/E1a_pos_weight/best.pt runs/E1_suite/E1b_focal/best.pt
```

预期E1a：`19affea68f877b29a5ebccf810e145012548a6dfccca98020ab0dea91e67565b`；E1b：`d2fa32ee72fd83164422bdeece6a0096de12b4dfd951405cab87f152fc7ec7e5`。这些是服务器evaluation元数据记录，尚未本地独立重算模型文件哈希。不要删除训练目录或重复运行15轮流程。

权重归档后，待B/C结果返回，按相同评价口径汇总E0–E3，并检查小组推理截图、错误案例、完整报告和最终模型提交要求。
