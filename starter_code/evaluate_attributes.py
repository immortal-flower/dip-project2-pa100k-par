"""用同一套验证规则比较 E0、共享 SE 和分组 SE 的逐属性指标。

示例：
  python starter_code/evaluate_attributes.py \
    --data-root ./PA-100K \
    --checkpoints runs/E0_baseline/best.pt runs/E2_se/best.pt \
                  runs/E2_se_group/best.pt \
    --out runs/E0_E2_group_attribute_metrics.json

年龄、朝向先 argmax，再按属性 one-vs-rest 计 TP/FP/FN/TN；其余属性
使用 logits > 0（等价于 sigmoid > 0.5），与 E0 验证规则一致。
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from dataset import PA100KDataset, build_transform
from model import PAResNet


def predictions(outputs, spec, target_shape, device):
    """将各组 logits 转成与全局标签向量同形状的布尔预测。"""
    pred = torch.zeros(target_shape, dtype=torch.bool, device=device)
    for group in spec["groups"]:
        idx, name = group["indices"], group["name"]
        logits = outputs[name]
        if group["type"] == "softmax":
            pred[:, idx] = one_hot(logits.argmax(dim=1), len(idx)).bool()
        else:
            pred[:, idx] = logits > 0
    return pred


def attribute_metrics(tp, fp, fn, tn):
    """从整数混淆计数计算逐属性指标。"""
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "positive_rate": ratio(tp + fn, tp + fp + fn + tn),
        "attr_acc": ratio(tp + tn, tp + fp + fn + tn),
        "mA": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "F1": ratio(2 * precision * recall, precision + recall),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, data_root, batch_size, workers, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    spec = checkpoint["spec"]
    image_size = args.get("image_size", [256, 128])
    model = PAResNet(
        spec, hidden=args.get("hidden", 256), drop=args.get("drop", 0.3),
        attention=args.get("attention", "none"),
        se_reduction=args.get("se_reduction", 16), pretrained=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    data_file = "val.csv" if (data_root / "val.csv").is_file() else "val_list.txt"
    dataset = PA100KDataset(str(data_root), data_file, spec,
                            build_transform(False, image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    counts = torch.zeros((spec["num_attrs"], 4), dtype=torch.long)
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        truth = target.to(device, non_blocking=True).bool()
        pred = predictions(model(images), spec, truth.shape, device)
        for column, mask in enumerate((pred & truth, pred & ~truth,
                                       ~pred & truth, ~pred & ~truth)):
            counts[:, column] += mask.sum(dim=0).cpu()

    details = {}
    for group in spec["groups"]:
        for attr, idx in zip(group["attrs"], group["indices"]):
            details[attr] = attribute_metrics(*counts[idx].tolist())
    return {
        "checkpoint": str(checkpoint_path),
        "attention": args.get("attention", "none"),
        "image_size": image_size,
        "samples": len(dataset),
        "macro_mA": sum(x["mA"] for x in details.values()) / len(details),
        "macro_F1": sum(x["F1"] for x in details.values()) / len(details),
        "attributes": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [evaluate_checkpoint(path, args.data_root, args.batch_size,
                                args.workers, device)
            for path in args.checkpoints]
    if len({tuple(run["image_size"]) for run in runs}) != 1:
        raise ValueError("Checkpoints use different image sizes")
    if len({run["samples"] for run in runs}) != 1:
        raise ValueError("Checkpoints evaluated different sample counts")
    for attr in runs[0]["attributes"]:
        positives = {run["attributes"][attr]["TP"] +
                     run["attributes"][attr]["FN"] for run in runs}
        if len(positives) != 1:
            raise ValueError(f"Checkpoints evaluated different labels for {attr}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(runs, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for run in runs:
        print(f"{run['attention']:9s} macro_mA={run['macro_mA'] * 100:.2f}% "
              f"macro_F1={run['macro_F1'] * 100:.2f}% "
              f"samples={run['samples']}")
    focus = ("male", "side", "glasses", "short_sleeve", "long_sleeve",
             "upper_plaid", "backpack", "shoulder_bag", "handbag")
    for attr in focus:
        parts = []
        for run in runs:
            metric = run["attributes"][attr]
            parts.append(f"{run['attention']}: R={metric['recall'] * 100:.2f}% "
                         f"F1={metric['F1'] * 100:.2f}% "
                         f"FP={metric['FP']} FN={metric['FN']}")
        print(f"{attr:15s} " + " | ".join(parts))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
