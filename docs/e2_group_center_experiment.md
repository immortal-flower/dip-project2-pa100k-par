# E2 分组 SE 的中心门控实验

研究问题：此前分组 SE 对 `backpack`、`shoulder_bag`、`handbag`、`glasses` 等目标属性未带来稳定改善。标准 SE 的通道系数为 `sigmoid(z)`，范围 `(0, 1)`，只能衰减特征。本实验检验允许相对原特征增强通道是否有帮助，**不预设一定提高性能**。

唯一结构改动：保留九组独立 SE、`512 → 32 → 512` 两层网络和原有初始化，仅将每组输出由 `x * sigmoid(z)` 改为 `x * (2 * sigmoid(z))`。新系数范围 `(0, 2)`，约以 `1` 为中心。这里的 sigmoid 是注意力模块的激活，不是损失函数。

控制变量：仍使用 ResNet18、九个分类头、原 BCEWithLogitsLoss／CrossEntropyLoss、无 pos_weight、无加权采样、相同图像尺寸与增强、15 轮、batch size 64、学习率 0.0003、AdamW、随机种子 42；训练配置检查允许的差异只有 `attention` 和输出目录。参数量应与原分组 SE 相同。

独立结果目录：`runs/E2_se_group_center`。最佳 checkpoint 仍按验证集九组平均 AttrAcc 选取。训练后统一评估四个 checkpoint（E0、共享 SE、分组 SE、中心门控分组 SE）的 26 属性 mA、Precision、Recall、F1、TP/FP/FN/TN；重点检查包、眼镜、衣袖、朝向，同时观察整体与其他属性是否退步。单次运行的差异受随机性及服务器负载影响，不能把耗时变化当成 SE 本身的加速效果。

如果新模型没有稳定超过 E0 或原分组 SE，应如实保留负面结果，不通过调阈值或改损失来掩盖本次结构对照。
