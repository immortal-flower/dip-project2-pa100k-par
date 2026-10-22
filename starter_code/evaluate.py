"""保留 baseline 选模口径，同时提供逐属性与实例级完整评价。

evaluate() 默认仍返回原来的二元组；return_details=True 增加第三个
字典，其中 softmax 的逐属性统计采用 one-vs-rest，而不是重复组准确率。
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


class AttributeMetrics:
    """One-vs-rest counts for every attribute, including softmax classes.

    Undefined precision/recall/F1 are 0. mA is null if either target class
    is absent; macro mA averages only attributes with both classes present.
    Instance F1 is averaged per image, not derived from averaged P and R.
    """

    def __init__(self, spec):
        self.spec = spec
        self.counts = torch.zeros(4, spec["num_attrs"], dtype=torch.int64)
        self.instance_sums = torch.zeros(3, dtype=torch.float64)
        self.samples = 0

    def update(self, pred, target):
        pred, target = pred.bool().cpu(), target.bool().cpu()
        if pred.shape != target.shape or pred.ndim != 2:
            raise ValueError("pred and target must have matching [N, attributes] shapes")
        self.counts += torch.stack([
            (pred & target).sum(0), (pred & ~target).sum(0),
            (~pred & target).sum(0), (~pred & ~target).sum(0)])
        tp = (pred & target).sum(1).double()
        p_count, t_count = pred.sum(1), target.sum(1)
        self.instance_sums += torch.stack([
            tp / p_count.clamp(min=1), tp / t_count.clamp(min=1),
            2 * tp / (p_count + t_count).clamp(min=1)]).sum(1)
        self.samples += pred.shape[0]

    def result(self):
        if not self.samples:
            raise ValueError("Cannot evaluate an empty dataset")
        detail = {}
        for attr, index in self.spec["attr2idx"].items():
            tp, fp, fn, tn = self.counts[:, index].tolist()
            detail[attr] = {
                "AttrAcc": (tp + tn) / self.samples,
                "mA": (0.5 * (tp / (tp + fn) + tn / (tn + fp))
                       if tp + fn and tn + fp else None),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "positive_support": tp + fn,
            }

        def macro(attrs):
            result = {}
            for key in ("AttrAcc", "mA", "precision", "recall", "f1"):
                values = [detail[a][key] for a in attrs if detail[a][key] is not None]
                result[key] = sum(values) / len(values) if values else None
            return result

        return {
            "schema_version": 1, "samples": self.samples,
            "macro": macro(list(detail)), "attributes": detail,
            "groups": {g["name"]: macro(g["attrs"]) for g in self.spec["groups"]},
            "instance": dict(zip(("precision", "recall", "f1"),
                                 (self.instance_sums / self.samples).tolist())),
            "definitions": {
                "threshold": "binary/multilabel logits > 0; softmax argmax then one-hot",
                "macro": "equal weight per attribute, not per group",
                "mA": "mean of positive recall and negative recall",
                "undefined": "P/R/F1=0; mA=null when either class is absent",
                "instance": "mean of per-image P, R and F1 over all 26 attributes",
            },
        }


def decode_outputs(outputs, spec):
    """Return global-order hard predictions and confidence probabilities."""
    first = next(iter(outputs.values()))
    probs = torch.zeros(first.shape[0], spec["num_attrs"])
    pred = torch.zeros_like(probs, dtype=torch.bool)
    for group in spec["groups"]:
        logits = outputs[group["name"]].detach().float().cpu()
        idx = group["indices"]
        if group["type"] == "softmax":
            probs[:, idx] = logits.softmax(1)
            pred[:, idx] = F.one_hot(logits.argmax(1), len(idx)).bool()
        else:
            probs[:, idx] = logits.sigmoid()
            pred[:, idx] = logits > 0
    return pred, probs


@torch.no_grad()
def evaluate(model, loader, spec, device, return_details=False, prediction_sink=None,
             progress_every=0):
    """在验证集/测试集上评测，返回:
      metrics     : {组名: 组内属性准确率均值}
      attr_detail : {属性名: 属性准确率}（供失败分析使用）
    """
    model.eval()
    complete = AttributeMetrics(spec) if return_details else None
    stat = {g["name"]: {"correct": torch.zeros(len(g["indices"])),
                        "total": torch.zeros(len(g["indices"]))}
            for g in spec["groups"]}

    for batch_index, (imgs, target) in enumerate(loader, 1):
        imgs = imgs.to(device)
        outputs = model(imgs)
        if complete is not None or prediction_sink is not None:
            predictions, probabilities = decode_outputs(outputs, spec)
            if complete is not None:
                complete.update(predictions, target)
            if prediction_sink is not None:
                prediction_sink(predictions, probabilities, target)
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
        if progress_every and batch_index % progress_every == 0:
            print(f"Evaluated {batch_index}/{len(loader)} batches", flush=True)

    metrics, attr_detail = {}, {}
    for g in spec["groups"]:
        name = g["name"]
        c, t = stat[name]["correct"], stat[name]["total"]
        acc = c / t.clamp(min=1)
        metrics[name] = acc.mean().item()
        for a, i in zip(g["attrs"], range(len(g["attrs"]))):
            attr_detail[a] = acc[i].item()
    if return_details:
        details = complete.result()
        details["legacy_group_AttrAcc"] = metrics
        details["legacy_mean_group_AttrAcc"] = sum(metrics.values()) / len(metrics)
        return metrics, attr_detail, details
    return metrics, attr_detail


def format_metrics(metrics):
    parts = [f"{k}={v * 100:.2f}" for k, v in metrics.items()]
    mean = sum(metrics.values()) / max(len(metrics), 1) * 100
    return f"mean={mean:.2f} | " + " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Unified checkpoint evaluation")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions", help="Optional per-image NPZ for case analysis")
    args = parser.parse_args()
    from dataset import PA100KDataset, build_transform
    from model import PAResNet
    from torch.utils.data import DataLoader
    import hashlib
    import numpy as np
    import platform
    from importlib.metadata import version

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    checkpoint_path = Path(args.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec, config = checkpoint["spec"], checkpoint["args"]
    model = PAResNet(spec, hidden=config.get("hidden", 256),
                     drop=config.get("drop", 0.3), attention=config.get("attention", "none"),
                     se_reduction=config.get("se_reduction", 16), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    root = Path(args.data_root)
    source = f"{args.split}.csv" if (root / f"{args.split}.csv").is_file() else f"{args.split}_list.txt"
    ds = PA100KDataset(str(root), source, spec,
                       build_transform(False, config.get("image_size", [256, 128])))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"Evaluating {len(ds)} {args.split} images on {device}", flush=True)
    saved = []
    def collect(pred, probability, target):
        saved.append((pred.numpy(), probability.numpy(), target.numpy()))
    _, _, results = evaluate(model, loader, spec, device, return_details=True,
                             prediction_sink=collect if args.predictions else None,
                             progress_every=20)
    results["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    results["checkpoint_args"] = config
    results["training_provenance"] = checkpoint.get("provenance")
    results["training_data_hashes"] = checkpoint.get("data_hashes")
    results["best_epoch"] = checkpoint.get("best_epoch")
    results["split"] = args.split
    results["evaluation_environment"] = {"torch": str(torch.__version__),
        "python": platform.python_version(), "device": device,
        "torchvision": version("torchvision"), "Pillow": version("Pillow"),
        "batch_size": args.batch_size, "workers": args.workers}
    results["label_file_sha256"] = hashlib.sha256((root / source).read_bytes()).hexdigest()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, paths=np.array([p for p, _ in ds.samples]),
                            attributes=np.array(sorted(spec["attr2idx"], key=spec["attr2idx"].get)),
                            predictions=np.concatenate([s[0] for s in saved]),
                            probabilities=np.concatenate([s[1] for s in saved]),
                            targets=np.concatenate([s[2] for s in saved]))
    print(json.dumps({k: results[k] for k in ("samples", "macro", "instance", "legacy_mean_group_AttrAcc")}, indent=2))


if __name__ == "__main__":
    main()
