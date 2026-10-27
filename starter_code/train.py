"""PA100K 训练入口：E0 baseline 与 E2 SE 使用同一套训练配置。

用法:
  python train.py                       # 默认训练
  python train.py --epochs 20 --batch-size 64
  python train.py --attention se        # E2：只启用 SE
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (PA100KDataset, build_transform, compute_pos_weight,
                     compute_sample_weights, load_spec)
from evaluate import evaluate, format_metrics
from losses import group_losses, total_loss
from model import PAResNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attributes", default="attributes.yaml")
    p.add_argument("--data-root", default="data/pa100k")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 128], help="高 宽")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64, help="显存不足可调小到 32")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--attention", choices=("none", "se"), default="none")
    p.add_argument("--se-reduction", type=int, default=16)
    p.add_argument("--pos-weight", action="store_true", help="选做：启用类别不平衡加权")
    p.add_argument("--sampler", choices=("shuffle", "weighted"), default="shuffle",
                   help="E3：weighted 只改变训练集采样策略")
    p.add_argument("--sampler-power", type=float, default=0.5,
                   help="稀有属性权重指数；0.5 表示平方根逆频率")
    p.add_argument("--sampler-max-weight", type=float, default=10.0,
                   help="单个属性的最大稀有度权重")
    p.add_argument("--loss", choices=("bce", "focal"), default="bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None,
                   help="Positive alpha in [0,1]; omitted isolates focal focusing")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        p.error("epochs and batch size must be positive; workers must be >=0")
    if args.loss == "focal" and args.pos_weight:
        p.error("Use focal and pos_weight in separate single-factor experiments")
    if args.focal_gamma < 0 or (args.focal_alpha is not None and not 0 <= args.focal_alpha <= 1):
        p.error("focal gamma must be >=0 and alpha in [0,1]")
    if args.loss != "focal" and (args.focal_gamma != 2.0 or args.focal_alpha is not None):
        p.error("Focal options require --loss focal")
    if args.sampler_power <= 0 or args.sampler_max_weight < 1:
        p.error("sampler power must be >0 and sampler max weight must be >=1")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join("runs", time.strftime("%m%d_%H%M"))
    if os.path.exists(os.path.join(out, "config.json")):
        raise FileExistsError(f"Training output already exists: {out}")
    os.makedirs(out, exist_ok=True)

    spec = load_spec(args.attributes)
    csv_data = os.path.isfile(os.path.join(args.data_root, "train.csv"))
    train_file = "train.csv" if csv_data else "train_list.txt"
    val_file = "val.csv" if csv_data else "val_list.txt"
    train_ds = PA100KDataset(args.data_root, train_file, spec,
                             build_transform(True, args.image_size))
    val_ds = PA100KDataset(args.data_root, val_file, spec,
                           build_transform(False, args.image_size))
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    sampler = None
    sampler_summary = None
    if args.sampler == "weighted":
        sample_weights, positive_rate, rarity = compute_sample_weights(
            train_ds, power=args.sampler_power, max_weight=args.sampler_max_weight)
        sampler_generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True,
            generator=sampler_generator)
        sampler_summary = {
            "formula": "max_positive(clamp(((1-p)/p)^power, 1, max_weight))",
            "weight_min": sample_weights.min().item(),
            "weight_mean": sample_weights.mean().item(),
            "weight_max": sample_weights.max().item(),
            "positive_rate": positive_rate.tolist(),
            "attribute_rarity": rarity.tolist(),
        }
        print("weighted sampler: "
              f"min={sample_weights.min():.3f} mean={sample_weights.mean():.3f} "
              f"max={sample_weights.max():.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    pos_weight = None
    if args.pos_weight:
        pos_weight = compute_pos_weight(train_ds, spec).to(device)

    model = PAResNet(spec, hidden=args.hidden, drop=args.drop,
                     attention=args.attention,
                     se_reduction=args.se_reduction).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.attention} parameters={parameters:,} "
          f"trainable={trainable_parameters:,} device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        if not os.path.exists(os.path.join(source_root, ".git")):
            raise FileNotFoundError("Source snapshot has no .git metadata")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    provenance = {"git_commit": commit, "git_dirty": dirty,
                  "python": platform.python_version(), "torch": str(torch.__version__),
                  "device": device, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
    source_manifest = os.path.join(source_root, "SOURCE_MANIFEST.json")
    if os.path.isfile(source_manifest):
        with open(source_manifest, "rb") as f:
            provenance["source_manifest_sha256"] = hashlib.sha256(f.read()).hexdigest()
    data_hashes = {}
    for name in (train_file, val_file):
        with open(os.path.join(args.data_root, name), "rb") as f:
            data_hashes[name] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "data_format": "csv" if csv_data else "list",
                   "parameters": parameters, "trainable_parameters": trainable_parameters,
                   "selection_metric": "mean_group_AttrAcc",
                   "sampler_summary": sampler_summary,
                   "provenance": provenance,
                   "data_hashes": data_hashes,
                   "pos_weight_values": pos_weight.cpu().tolist() if pos_weight is not None else None},
                  f, indent=2, ensure_ascii=False)

    best = -1.0
    best_epoch = None
    best_groups = None
    best_attributes = None
    best_complete = None
    experiment_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        train_started = time.perf_counter()
        run = {}
        for imgs, target in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outputs = model(imgs)
                losses = group_losses(outputs, target, spec, pos_weight,
                                      loss_mode=args.loss, focal_gamma=args.focal_gamma,
                                      focal_alpha=args.focal_alpha)
                loss = total_loss(losses)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + v.item()
        n = steps_per_epoch
        train_str = " ".join(f"{k}={v / n:.4f}" for k, v in sorted(run.items()))

        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        m, attr_detail, complete = evaluate(model, val_loader, spec, device, return_details=True)
        if device == "cuda":
            torch.cuda.synchronize()
        val_seconds = time.perf_counter() - val_started
        msg = (f"epoch {epoch + 1}/{args.epochs} "
               f"(train={train_seconds:.1f}s val={val_seconds:.1f}s) train: {train_str}")
        msg += f"\n  val: {format_metrics(m)}"
        score = sum(m.values()) / max(len(m), 1)
        improved = score > best
        if improved:
            best = score
            best_epoch = epoch + 1
            best_groups = m
            best_attributes = attr_detail
            best_complete = complete
            torch.save({"model": model.state_dict(), "spec": spec,
                        "args": vars(args), "val_metrics": score,
                        "best_epoch": best_epoch, "group_metrics": m,
                        "attr_metrics": attr_detail, "complete_metrics": complete,
                        "provenance": provenance, "data_hashes": data_hashes},
                       os.path.join(out, "best.pt"))
        record = {"epoch": epoch + 1, "train_seconds": train_seconds,
                  "val_seconds": val_seconds, "score": score,
                  "group_metrics": m, "attr_metrics": attr_detail,
                  "complete_metrics": complete, "best": improved}
        with open(os.path.join(out, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(msg, flush=True)
        with open(os.path.join(out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    torch.save({"model": model.state_dict(), "spec": spec, "args": vars(args)},
               os.path.join(out, "last.pt"))
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"attention": args.attention, "parameters": parameters,
                   "trainable_parameters": trainable_parameters,
                   "best_epoch": best_epoch, "best_score": best,
                   "best_group_metrics": best_groups,
                   "best_attr_metrics": best_attributes,
                   "loss_mode": args.loss, "pos_weight": args.pos_weight,
                   "best_complete_metrics": best_complete,
                   "total_seconds": time.perf_counter() - experiment_started},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({**best_complete, "checkpoint_args": vars(args), "best_epoch": best_epoch,
                   "split": "val", "provenance": provenance},
                  f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"done. best val mean acc = {best * 100:.2f}, checkpoints in {out}")


if __name__ == "__main__":
    main()
