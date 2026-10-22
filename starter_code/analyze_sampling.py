"""Create the data checks and sampling diagnostics required by E3."""
import argparse
import csv
import json
import os
import random
import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import (PA100KDataset, compute_sample_weights, label_matrix,
                     load_spec)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", default="attributes.yaml")
    parser.add_argument("--data-root", default="data/pa100k")
    parser.add_argument("--out", default="experiments/E3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--sampler-max-weight", type=float, default=10.0)
    return parser.parse_args()


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_distribution(rows, output_path):
    width, row_h = 1500, 48
    top, left, right = 145, 250, 90
    height = top + row_h * len(rows) + 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, label_font = font(34, True), font(21), font(22)
    draw.text((left, 18), "PA-100K attribute frequency: original vs weighted sampling",
              fill="#172033", font=title_font)
    plot_width = width - left - right
    max_rate = max(max(row["original_rate"], row["sampled_rate"]) for row in rows)
    axis_max = max(0.1, min(1.0, (int(max_rate * 10) + 1) / 10))
    for tick in range(6):
        value = axis_max * tick / 5
        x = left + plot_width * tick / 5
        draw.line((x, top - 10, x, height - 65), fill="#d7dce5", width=1)
        draw.text((x - 20, height - 55), f"{value * 100:.0f}%", fill="#455064",
                  font=axis_font)
    for index, row in enumerate(rows):
        y = top + index * row_h
        draw.text((20, y + 8), row["attribute"], fill="#172033", font=label_font)
        original = row["original_rate"] / axis_max * plot_width
        sampled = row["sampled_rate"] / axis_max * plot_width
        draw.rectangle((left, y + 5, left + original, y + 20), fill="#5178c8")
        draw.rectangle((left, y + 25, left + sampled, y + 40), fill="#df7b4c")
    draw.rectangle((left, 78, left + 30, 98), fill="#5178c8")
    draw.text((left + 40, 73), "Original", fill="#172033", font=axis_font)
    draw.rectangle((left + 175, 78, left + 205, 98), fill="#df7b4c")
    draw.text((left + 215, 73), "Weighted", fill="#172033", font=axis_font)
    image.save(output_path)


def draw_sample_grid(dataset, labels, names, indices, output_path):
    cell_w, image_h, caption_h = 360, 430, 138
    margin, gap = 24, 18
    canvas = Image.new("RGB", (margin * 2 + cell_w * 3 + gap * 2,
                               margin * 2 + (image_h + caption_h) * 3 + gap * 2),
                       "#edf0f5")
    draw = ImageDraw.Draw(canvas)
    head_font, text_font = font(21, True), font(18)
    for slot, index in enumerate(indices):
        row, col = divmod(slot, 3)
        x = margin + col * (cell_w + gap)
        y = margin + row * (image_h + caption_h + gap)
        rel_path = dataset.samples[index][0]
        with Image.open(os.path.join(dataset.root, rel_path)) as source:
            pedestrian = source.convert("RGB")
        pedestrian.thumbnail((cell_w, image_h), Image.Resampling.LANCZOS)
        px = x + (cell_w - pedestrian.width) // 2
        py = y + (image_h - pedestrian.height) // 2
        draw.rectangle((x, y, x + cell_w, y + image_h), fill="white")
        canvas.paste(pedestrian, (px, py))
        positives = [names[i] for i, value in enumerate(labels[index]) if value > 0]
        caption = ", ".join(positives)
        draw.text((x, y + image_h + 8), f"#{index}  {os.path.basename(rel_path)}",
                  fill="#172033", font=head_font)
        wrapped = textwrap.wrap(caption, width=42)[:4]
        draw.multiline_text((x, y + image_h + 40), "\n".join(wrapped),
                            fill="#455064", font=text_font, spacing=3)
    canvas.save(output_path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    spec = load_spec(args.attributes)
    train_file = "train.csv" if os.path.isfile(os.path.join(args.data_root, "train.csv")) else "train_list.txt"
    dataset = PA100KDataset(args.data_root, train_file, spec)
    labels = label_matrix(dataset)
    names = [attribute for group in spec["groups"] for attribute in group["attrs"]]
    weights, positive_rate, rarity = compute_sample_weights(
        dataset, power=args.sampler_power, max_weight=args.sampler_max_weight)

    generator = torch.Generator().manual_seed(args.seed)
    sampled_indices = torch.multinomial(weights, len(dataset), replacement=True,
                                        generator=generator)
    sampled_rate = labels[sampled_indices].mean(dim=0)
    counts = torch.bincount(sampled_indices, minlength=len(dataset))
    expected_rate = (labels * weights.float().unsqueeze(1)).sum(0) / weights.float().sum()

    rows = []
    for index, name in enumerate(names):
        rows.append({
            "attribute": name,
            "original_count": int(labels[:, index].sum().item()),
            "original_rate": positive_rate[index].item(),
            "sampled_count": int(labels[sampled_indices, index].sum().item()),
            "sampled_rate": sampled_rate[index].item(),
            "expected_sampled_rate": expected_rate[index].item(),
            "rarity_factor": rarity[index].item(),
        })

    csv_path = os.path.join(args.out, "sampling_distribution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seed": args.seed,
        "num_samples": len(dataset),
        "formula": "max_positive(clamp(((1-p)/p)^power, 1, max_weight))",
        "sampler_power": args.sampler_power,
        "sampler_max_weight": args.sampler_max_weight,
        "normalized_weight": {
            "min": weights.min().item(), "mean": weights.mean().item(),
            "max": weights.max().item(),
        },
        "epoch_sampling": {
            "unique_images": int((counts > 0).sum().item()),
            "unseen_images": int((counts == 0).sum().item()),
            "repeated_images": int((counts > 1).sum().item()),
            "maximum_repetitions": int(counts.max().item()),
        },
    }
    with open(os.path.join(args.out, "sampling_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    draw_distribution(rows, os.path.join(args.out, "sampling_distribution.png"))
    sample_rng = random.Random(args.seed)
    grid_indices = sample_rng.sample(range(len(dataset)), 9)
    draw_sample_grid(dataset, labels, names, grid_indices,
                     os.path.join(args.out, "sample_labels_9.png"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote E3 diagnostics to {args.out}")


if __name__ == "__main__":
    main()
