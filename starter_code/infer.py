"""单张/批量图片推理（完整版，无需修改）。

用法:
  python infer.py --ckpt runs/xxx/best.pt --img xxx.jpg
  python infer.py --ckpt runs/xxx/best.pt --img 图片目录/ --threshold 0.6
"""
import argparse
import glob
import json
import os

import torch
from PIL import Image

from dataset import build_transform
from model import PAResNet


def sigmoid(x):
    return torch.sigmoid(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img", required=True, help="图片路径或目录")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--image-size", type=int, nargs=2, default=None,
                   help="默认使用 checkpoint 保存的训练尺寸")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    spec = ckpt["spec"]
    model_args = ckpt.get("args", {})
    model = PAResNet(spec, hidden=model_args.get("hidden", 256),
                     drop=model_args.get("drop", 0.3),
                     attention=model_args.get("attention", "none"),
                     se_reduction=model_args.get("se_reduction", 16),
                     pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = build_transform(False, args.image_size or
                                model_args.get("image_size", [256, 128]))

    if os.path.isdir(args.img):
        paths = sorted(sum((glob.glob(os.path.join(args.img, f"*.{e}"))
                            for e in ("jpg", "jpeg", "png", "bmp")), []))
    else:
        paths = [args.img]

    for path in paths:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(x)
        result = {}
        for g in spec["groups"]:
            name, gtype = g["name"], g["type"]
            logits = outputs[name][0]
            if gtype == "softmax":
                prob = torch.softmax(logits, dim=0)
                k = int(prob.argmax())
                result[name] = {"label": g["attrs"][k], "prob": round(float(prob[k]), 4)}
            else:
                prob = sigmoid(logits)
                result[name] = {a: round(float(p), 4)
                                for a, p in zip(g["attrs"], prob)
                                if p >= args.threshold}
        print(json.dumps({path: result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
