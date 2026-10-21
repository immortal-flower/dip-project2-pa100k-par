"""Create a five-image prediction demo for the C/E3 deliverable."""
import argparse
import json
import os
import random
import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import PA100KDataset, build_transform, load_spec
from model import PAResNet


def font(size, bold=False):
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="data/pa100k")
    parser.add_argument("--attributes", default="attributes.yaml")
    parser.add_argument("--out", default="experiments/E3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    spec = load_spec(args.attributes)
    model = PAResNet(
        spec, hidden=model_args.get("hidden", 256), drop=model_args.get("drop", 0.3),
        attention=model_args.get("attention", "none"),
        se_reduction=model_args.get("se_reduction", 16), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    test_file = "test.csv" if os.path.isfile(os.path.join(args.data_root, "test.csv")) else "test_list.txt"
    dataset = PA100KDataset(
        args.data_root, test_file, spec,
        build_transform(False, model_args.get("image_size", [256, 128])))
    indices = random.Random(args.seed).sample(range(len(dataset)), 5)
    results = []
    for index in indices:
        tensor, _ = dataset[index]
        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0).to(device))
        labels = []
        for group in spec["groups"]:
            logits = outputs[group["name"]][0].cpu()
            if group["type"] == "softmax":
                probability = torch.softmax(logits, dim=0)
                selected = int(probability.argmax())
                labels.append({"attribute": group["attrs"][selected],
                               "probability": float(probability[selected])})
            else:
                probability = torch.sigmoid(logits)
                labels.extend({"attribute": name, "probability": float(value)}
                              for name, value in zip(group["attrs"], probability)
                              if value >= args.threshold)
        results.append({"dataset_index": index, "image": dataset.samples[index][0],
                        "predictions": labels})

    with open(os.path.join(args.out, "test_demo_predictions.json"), "w",
              encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    cell_w, image_h, caption_h = 390, 460, 170
    margin, gap, columns = 24, 18, 3
    rows = 2
    canvas = Image.new("RGB", (margin * 2 + columns * cell_w + gap * 2,
                               margin * 2 + rows * (image_h + caption_h) + gap),
                       "#edf0f5")
    draw = ImageDraw.Draw(canvas)
    head, body = font(21, True), font(18)
    for slot, result in enumerate(results):
        row, column = divmod(slot, columns)
        x = margin + column * (cell_w + gap)
        y = margin + row * (image_h + caption_h + gap)
        path = os.path.join(dataset.root, result["image"])
        with Image.open(path) as source:
            pedestrian = source.convert("RGB")
        pedestrian.thumbnail((cell_w, image_h), Image.Resampling.LANCZOS)
        draw.rectangle((x, y, x + cell_w, y + image_h), fill="white")
        canvas.paste(pedestrian, (x + (cell_w - pedestrian.width) // 2,
                                  y + (image_h - pedestrian.height) // 2))
        draw.text((x, y + image_h + 8), os.path.basename(result["image"]),
                  fill="#172033", font=head)
        label_text = ", ".join(
            f"{item['attribute']} {item['probability']:.2f}"
            for item in result["predictions"])
        draw.multiline_text((x, y + image_h + 42),
                            "\n".join(textwrap.wrap(label_text, width=43)[:5]),
                            fill="#455064", font=body, spacing=4)
    canvas.save(os.path.join(args.out, "test_demo_5.png"))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
