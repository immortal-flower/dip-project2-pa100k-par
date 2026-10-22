# A线服务器运行交接

状态补记（2026-10-21）：下述流程已完成，结果见report_A.md及server_results。无需再次训练。实际使用RTX4090物理4号卡；原文的3090为对齐E0环境的初始建议，不是本次E1实际设备。E1_results.zip不含best.pt，模型需另行归档。

## 准备材料

代码使用feature/focal-loss分支的本地修改版，或本次提供的源码ZIP。源码ZIP包含SOURCE_MANIFEST.json，记录原始Git基线、dirty状态及每个文件的SHA-256；它不是已推送的新Git提交。数据集和模型不在源码包内。

服务器需要：完整PA-100K（data/下100,000张图及train/val/test.csv）、共享E0 best.pt、Python 3.10+与可用CUDA。优先沿用E0的PyTorch 2.1.2+cu118 / torchvision 0.16.2+cu118和RTX 3090；实际配置到位后再核对版本、显存、磁盘和路径。训练使用bfloat16，GPU需支持该精度。

E0权重下载链接：

https://github.com/immortal-flower/dip-project2-pa100k-par/releases/download/e0-baseline/best.pt

SHA-256：991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806。

## 先预览命令，再正式执行

在源码根目录运行，替换数据和模型路径。无--execute时只打印计划，不训练、不创建输出目录。

```bash
python starter_code/run_e1.py --data-root /path/to/PA-100K --e0-ckpt /path/to/E0/best.pt --out-root runs/E1_suite
python starter_code/run_e1.py --data-root /path/to/PA-100K --e0-ckpt /path/to/E0/best.pt --out-root runs/E1_suite --execute
```

流程固定为：标签审计→E1a正例加权训练→E1b Focal训练→同环境复评E0/E1a/E1b→核对E0历史记录→生成对比表→导出改善及退化案例。训练参数从仓库的experiments/E0/config.json读取，E1b固定gamma=2且alpha=None。

执行前检查CUDA、bfloat16能力、Python版本、E0权重哈希及输出目录。输出目录必须全新，遇到错误立即停止，不覆盖已有实验。如果中断，不要删除已产生的结果；使用E1 README中的单条命令检查和继续剩余步骤，不能把不同配置的局部结果拼接。

## 结果收回与人工分析

保留E1_suite下的config/metrics/summary/evaluation/log及best.pt和预测NPZ；源码ZIP内的manifest也会保留到套件输出目录。普通Git只提交选定的小型JSON、表格、报告和经检查的案例图，完整数据、NPZ与模型不进入普通Git历史。

E1_comparison.md包含统一总表和稀有属性表；E1a_cases、E1b_cases保存匹配样本的改善及退化示例。至少挑选2张实际查看，解释漏检改善或误报增加，不把程序的候选标记直接当作图像原因。用实际表格完成report_A.md中的结果和结论部分。

## 已知待核实事项

本地CPU E0复评与历史CUDA记录存在小幅预测差异，已记录在e0_review.md。因此正式表格必须使用本套件在服务器上新产生的三份evaluation.json，不能直接用源码包里的临时e0_evaluation.json。比较程序会拒绝临时CPU记录及评价环境不一致的结果。

本机Git作者身份尚未配置，因此修改保留在本地功能分支中，尚未提交或推送。若使用Git协作，正式训练前应以本人Git身份提交已复核代码；若通过源码ZIP运行，模型和配置会保存SOURCE_MANIFEST的哈希作为追溯依据。
