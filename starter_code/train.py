"""PA100K 训练入口：E0 baseline 与 E2 SE 使用同一套训练配置。

用法:
  python train.py                       # 默认训练
  python train.py --epochs 20 --batch-size 64
  python train.py --attention se        # E2：只启用 SE
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import PA100KDataset, build_transform, compute_pos_weight, load_spec
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
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
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

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "data_format": "csv" if csv_data else "list",
                   "parameters": parameters, "trainable_parameters": trainable_parameters,
                   "selection_metric": "mean_group_AttrAcc"},
                  f, indent=2, ensure_ascii=False)

    best = -1.0
    best_epoch = None
    best_groups = None
    best_attributes = None
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
                losses = group_losses(outputs, target, spec, pos_weight)
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
        m, attr_detail = evaluate(model, val_loader, spec, device)
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
            torch.save({"model": model.state_dict(), "spec": spec,
                        "args": vars(args), "val_metrics": score,
                        "best_epoch": best_epoch, "group_metrics": m,
                        "attr_metrics": attr_detail},
                       os.path.join(out, "best.pt"))
        record = {"epoch": epoch + 1, "train_seconds": train_seconds,
                  "val_seconds": val_seconds, "score": score,
                  "group_metrics": m, "attr_metrics": attr_detail,
                  "best": improved}
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
                   "total_seconds": time.perf_counter() - experiment_started},
                  f, indent=2, ensure_ascii=False)
    print(f"done. best val mean acc = {best * 100:.2f}, checkpoints in {out}")


if __name__ == "__main__":
    main()
