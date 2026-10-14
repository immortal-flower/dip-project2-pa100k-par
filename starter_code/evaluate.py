"""分组评测（baseline 完整实现）。

必做指标：每属性准确率（AttrAcc），以及组内均值。
选做指标：mA（正例/负例准确率平均）、实例级 Precision/Recall/F1，
         可在 evaluate() 基础上自行扩展。
"""
import torch


@torch.no_grad()
def evaluate(model, loader, spec, device):
    """在验证集/测试集上评测，返回:
      metrics     : {组名: 组内属性准确率均值}
      attr_detail : {属性名: 属性准确率}（供失败分析使用）
    """
    model.eval()
    stat = {g["name"]: {"correct": torch.zeros(len(g["indices"])),
                        "total": torch.zeros(len(g["indices"]))}
            for g in spec["groups"]}

    for imgs, target in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        for g in spec["groups"]:
            idx, name, gtype = g["indices"], g["name"], g["type"]
            tgt = target[:, idx]
            logits = outputs[name].cpu()

            if gtype == "softmax":
                # 组内互斥：每个样本只算一次对错，组内所有属性共享该统计
                pred = logits.argmax(dim=1)
                cls = tgt.argmax(dim=1)
                c = (pred == cls).float().sum()
                for i in range(len(idx)):
                    stat[name]["correct"][i] += c
                    stat[name]["total"][i] += imgs.size(0)
            else:
                # binary/multilabel：logits > 0 等价于 sigmoid > 0.5，逐属性统计
                pred = (logits > 0).float()
                stat[name]["correct"] += (pred == tgt).float().sum(0)
                stat[name]["total"] += torch.ones_like(tgt).sum(0)

    metrics, attr_detail = {}, {}
    for g in spec["groups"]:
        name = g["name"]
        c, t = stat[name]["correct"], stat[name]["total"]
        acc = c / t.clamp(min=1)
        metrics[name] = acc.mean().item()
        for a, i in zip(g["attrs"], range(len(g["attrs"]))):
            attr_detail[a] = acc[i].item()
    return metrics, attr_detail


def format_metrics(metrics):
    parts = [f"{k}={v * 100:.2f}" for k, v in metrics.items()]
    mean = sum(metrics.values()) / max(len(metrics), 1) * 100
    return f"mean={mean:.2f} | " + " ".join(parts)
