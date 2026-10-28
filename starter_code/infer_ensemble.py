"""Run image inference with a frozen probability soft-voting ensemble."""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image

from dataset import build_transform
from model import PAResNet


def named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        result[name] = path
    return result


def spec_signature(spec):
    return [(group["name"], group["type"], tuple(group["attrs"]),
             tuple(group["indices"])) for group in spec["groups"]]


def load_model(path, device):
    # These are team-produced release checkpoints.  ``weights_only=False`` is
    # kept for compatibility with the older PyTorch environment used in class.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec, config = checkpoint["spec"], checkpoint.get("args", {})
    model = PAResNet(spec, hidden=config.get("hidden", 256),
                     drop=config.get("drop", 0.3),
                     attention=config.get("attention", "none"),
                     se_reduction=config.get("se_reduction", 16), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, spec, config


def group_probabilities(outputs, spec):
    probabilities = {}
    for group in spec["groups"]:
        logits = outputs[group["name"]]
        probabilities[group["name"]] = (
            logits.softmax(1) if group["type"] == "softmax" else logits.sigmoid())
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", action="append", required=True,
                        help="Repeat NAME=PATH for every model in the config")
    parser.add_argument("--img", required=True, help="Image path or directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoints = named_paths(args.ckpt)
    if set(checkpoints) != set(config["models"]):
        raise ValueError(f"Config requires checkpoints for {config['models']}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {name: load_model(checkpoints[name], device) for name in config["models"]}
    first_model, spec, first_args = loaded[config["models"][0]]
    del first_model
    for name, (_, candidate_spec, candidate_args) in loaded.items():
        if spec_signature(candidate_spec) != spec_signature(spec):
            raise ValueError(f"{name}: checkpoint attribute specification differs")
        if candidate_args.get("image_size", [256, 128]) != first_args.get(
                "image_size", [256, 128]):
            raise ValueError(f"{name}: checkpoint image size differs")
    transform = build_transform(False, first_args.get("image_size", [256, 128]))
    threshold = config.get("threshold", 0.5) if args.threshold is None else args.threshold

    if os.path.isdir(args.img):
        paths = sorted(sum((glob.glob(os.path.join(args.img, f"*.{extension}"))
                            for extension in ("jpg", "jpeg", "png", "bmp")), []))
    else:
        paths = [args.img]

    for path in paths:
        with Image.open(path) as source:
            image = transform(source.convert("RGB")).unsqueeze(0).to(device)
        blended = {}
        with torch.no_grad():
            for name, (model, _, _) in loaded.items():
                current = group_probabilities(model(image), spec)
                for group_name, probability in current.items():
                    weights = (config["group_weights"][group_name]
                               if "group_weights" in config else config["weights"])
                    blended[group_name] = (blended.get(group_name, 0)
                                           + weights[name] * probability)
        result = {}
        for group in spec["groups"]:
            probability = blended[group["name"]][0].cpu()
            if group["type"] == "softmax":
                winner = int(probability.argmax())
                result[group["name"]] = {
                    "label": group["attrs"][winner],
                    "prob": round(float(probability[winner]), 4),
                }
            else:
                result[group["name"]] = {
                    attr: round(float(value), 4)
                    for attr, value in zip(group["attrs"], probability)
                    if value >= threshold
                }
        print(json.dumps({path: result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
