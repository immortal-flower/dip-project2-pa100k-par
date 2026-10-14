"""分组损失函数（baseline 完整实现）。

不同属性组使用不同损失：
  binary     : BCEWithLogitsLoss
  softmax    : CrossEntropyLoss
  multilabel : BCEWithLogitsLoss（每个属性独立）

鼓励修改的方向（见 README 第七节）：
  - softmax 组换成 margin softmax（AM-Softmax / ArcFace 风格）
  - binary/multilabel 组换成 Focal Loss
  - 结合 pos_weight 处理类别不平衡
"""
import torch
import torch.nn.functional as F


def group_losses(outputs, target, spec, pos_weight=None):
    """计算所有属性组的损失。

    参数:
      outputs   : {组名: (B, N) logits}
      target    : (B, num_attrs) 全局标签向量，元素为 0/1
      spec      : load_spec() 返回的属性词表
      pos_weight: (num_attrs,) 可选，正例权重（负例数/正例数）

    返回:
      {组名: loss 标量}
    """
    losses = {}
    for g in spec["groups"]:
        name, gtype, idx = g["name"], g["type"], g["indices"]
        logits = outputs[name]
        tgt = target[:, idx]
        pw = pos_weight[idx] if pos_weight is not None else None

        if gtype in ("binary", "multilabel"):
            # binary 是 N=1 的特例，二者实现相同
            losses[name] = F.binary_cross_entropy_with_logits(
                logits, tgt, pos_weight=pw)
        else:  # softmax：tgt 是 one-hot，转成类别索引
            cls = tgt.argmax(dim=1)
            losses[name] = F.cross_entropy(logits, cls)

        # 组级 loss 权重（attributes.yaml 中可配置）
        losses[name] = losses[name] * spec["loss_weights"].get(name, 1.0)
    return losses


def total_loss(losses):
    """对所有组的损失求和。"""
    return sum(losses.values())
