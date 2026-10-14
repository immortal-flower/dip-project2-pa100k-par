"""训练入口（完整版，无需修改；理解流程即可）。

用法:
  python train.py                       # 默认训练
  python train.py --epochs 20 --batch-size 64
  python train.py --pos-weight          # 选做：启用类别不平衡加权
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
    p.add_argument("--pos-weight", action="store_true", help="选做：启用类别不平衡加权")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or os.path.join("runs", time.strftime("%m%d_%H%M"))
    os.makedirs(out, exist_ok=True)

    spec = load_spec(args.attributes)
    train_ds = PA100KDataset(args.data_root, "train_list.txt", spec,
                             build_transform(True, args.image_size))
    val_ds = PA100KDataset(args.data_root, "val_list.txt", spec,
                           build_transform(False, args.image_size))
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    pos_weight = None
    if args.pos_weight:
        pos_weight = compute_pos_weight(train_ds, spec).to(device)

    model = PAResNet(spec, hidden=args.hidden, drop=args.drop).to(device)
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
        json.dump({"args": vars(args)}, f, indent=2, ensure_ascii=False)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
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

        msg = f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s) train: {train_str}"
        m, _ = evaluate(model, val_loader, spec, device)
        msg += f"\n  val: {format_metrics(m)}"
        score = sum(m.values()) / max(len(m), 1)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "spec": spec,
                        "args": vars(args), "val_metrics": score},
                       os.path.join(out, "best.pt"))
        print(msg, flush=True)
        with open(os.path.join(out, "log.txt"), "a") as f:
            f.write(msg + "\n")

    torch.save({"model": model.state_dict(), "spec": spec, "args": vars(args)},
               os.path.join(out, "last.pt"))
    print(f"done. best val mean acc = {best * 100:.2f}, checkpoints in {out}")


if __name__ == "__main__":
    main()
