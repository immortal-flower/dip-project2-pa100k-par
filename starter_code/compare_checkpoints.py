"""Detailed E0/E3 metrics and automatically selected error cases."""
import argparse
import csv
import json
import os
import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataset import PA100KDataset, build_transform, label_matrix, load_spec
from model import PAResNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--data-root", default="data/pa100k")
    parser.add_argument("--attributes", default="attributes.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="experiments/E3")
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = PAResNet(
        checkpoint["spec"], hidden=args.get("hidden", 256),
        drop=args.get("drop", 0.3), attention=args.get("attention", "none"),
        se_reduction=args.get("se_reduction", 16), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, spec, device):
    probabilities, predictions = [], []
    for images, _ in loader:
        outputs = model(images.to(device, non_blocking=True))
        batch_prob = torch.zeros((images.size(0), spec["num_attrs"]))
        batch_pred = torch.zeros_like(batch_prob)
        for group in spec["groups"]:
            indices = group["indices"]
            logits = outputs[group["name"]].cpu()
            if group["type"] == "softmax":
                prob = torch.softmax(logits, dim=1)
                chosen = prob.argmax(dim=1)
                pred = torch.zeros_like(prob).scatter_(1, chosen.unsqueeze(1), 1.0)
            else:
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).float()
            batch_prob[:, indices] = prob
            batch_pred[:, indices] = pred
        probabilities.append(batch_prob)
        predictions.append(batch_pred)
    return torch.cat(probabilities), torch.cat(predictions)


def detailed_metrics(target, prediction, names):
    rows = []
    for index, name in enumerate(names):
        truth = target[:, index].bool()
        pred = prediction[:, index].bool()
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        tn = int((~truth & ~pred).sum())
        positive_acc = tp / max(tp + fn, 1)
        negative_acc = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        recall = positive_acc
        rows.append({
            "attribute": name,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "mA": (positive_acc + negative_acc) / 2,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    sample_tp = (target.bool() & prediction.bool()).sum(1).float()
    sample_fp = (~target.bool() & prediction.bool()).sum(1).float()
    sample_fn = (target.bool() & ~prediction.bool()).sum(1).float()
    precision = sample_tp / (sample_tp + sample_fp).clamp(min=1)
    recall = sample_tp / (sample_tp + sample_fn).clamp(min=1)
    instance = {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": (2 * precision * recall / (precision + recall).clamp(min=1e-12)).mean().item(),
    }
    return rows, instance


def choose_cases(target, base_pred, candidate_pred, base_prob, candidate_prob, names):
    focus = ["boots", "holding", "lower_stripe", "age_lt18", "long_coat",
             "upper_splice", "hat"]
    cases, used = [], set()
    for status in ("improved_positive", "regressed_false_positive"):
        for name in focus:
            attr = names.index(name)
            truth = target[:, attr].bool()
            if status == "improved_positive":
                mask = truth & ~base_pred[:, attr].bool() & candidate_pred[:, attr].bool()
                confidence = candidate_prob[:, attr]
            else:
                mask = ~truth & ~base_pred[:, attr].bool() & candidate_pred[:, attr].bool()
                confidence = candidate_prob[:, attr]
            indices = torch.where(mask)[0]
            if len(indices):
                ordered = indices[confidence[indices].argsort(descending=True)]
                selected = next((int(i) for i in ordered if int(i) not in used), None)
                if selected is not None:
                    used.add(selected)
                    cases.append({
                        "dataset_index": selected, "attribute": name, "status": status,
                        "truth": int(target[selected, attr]),
                        "baseline_probability": base_prob[selected, attr].item(),
                        "candidate_probability": candidate_prob[selected, attr].item(),
                    })
            if len(cases) >= 8:
                return cases
    return cases


def font(size, bold=False):
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_cases(dataset, cases, output_path):
    cell_w, image_h, caption_h = 360, 430, 130
    margin, gap, columns = 24, 18, 2
    rows = (len(cases) + columns - 1) // columns
    canvas = Image.new("RGB", (margin * 2 + columns * cell_w + gap,
                               margin * 2 + rows * (image_h + caption_h) + gap * (rows - 1)),
                       "#edf0f5")
    draw = ImageDraw.Draw(canvas)
    head, body = font(20, True), font(18)
    for slot, case in enumerate(cases):
        row, column = divmod(slot, columns)
        x = margin + column * (cell_w + gap)
        y = margin + row * (image_h + caption_h + gap)
        rel_path = dataset.samples[case["dataset_index"]][0]
        with Image.open(os.path.join(dataset.root, rel_path)) as source:
            image = source.convert("RGB")
        image.thumbnail((cell_w, image_h), Image.Resampling.LANCZOS)
        draw.rectangle((x, y, x + cell_w, y + image_h), fill="white")
        canvas.paste(image, (x + (cell_w - image.width) // 2,
                             y + (image_h - image.height) // 2))
        draw.text((x, y + image_h + 8),
                  f"{case['attribute']} · {case['status'].replace('_', ' ')}",
                  fill="#172033", font=head)
        detail = (f"{os.path.basename(rel_path)} | truth={case['truth']} | "
                  f"E0={case['baseline_probability']:.3f} | "
                  f"E3={case['candidate_probability']:.3f}")
        draw.multiline_text((x, y + image_h + 42), "\n".join(textwrap.wrap(detail, 43)),
                            fill="#455064", font=body, spacing=4)
    canvas.save(output_path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = load_spec(args.attributes)
    source = f"{args.split}.csv" if os.path.isfile(os.path.join(args.data_root, f"{args.split}.csv")) else f"{args.split}_list.txt"
    candidate_model, candidate_checkpoint = load_model(args.candidate_ckpt, device)
    image_size = candidate_checkpoint.get("args", {}).get("image_size", [256, 128])
    dataset = PA100KDataset(args.data_root, source, spec, build_transform(False, image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    target = label_matrix(dataset)
    names = [attribute for group in spec["groups"] for attribute in group["attrs"]]

    baseline_model, _ = load_model(args.baseline_ckpt, device)
    baseline_prob, baseline_pred = predict(baseline_model, loader, spec, device)
    del baseline_model
    candidate_prob, candidate_pred = predict(candidate_model, loader, spec, device)
    baseline_rows, baseline_instance = detailed_metrics(target, baseline_pred, names)
    candidate_rows, candidate_instance = detailed_metrics(target, candidate_pred, names)

    comparison = []
    for baseline, candidate in zip(baseline_rows, candidate_rows):
        row = {"attribute": baseline["attribute"]}
        for metric in ("accuracy", "mA", "precision", "recall", "f1"):
            row[f"E0_{metric}"] = baseline[metric]
            row[f"E3_{metric}"] = candidate[metric]
            row[f"delta_{metric}"] = candidate[metric] - baseline[metric]
        comparison.append(row)
    with open(os.path.join(args.out, "e0_e3_attribute_comparison.csv"), "w",
              newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    cases = choose_cases(target, baseline_pred, candidate_pred,
                         baseline_prob, candidate_prob, names)
    for case in cases:
        case["image"] = dataset.samples[case["dataset_index"]][0]
    result = {
        "split": args.split,
        "baseline_checkpoint": args.baseline_ckpt,
        "candidate_checkpoint": args.candidate_ckpt,
        "baseline_instance_metrics": baseline_instance,
        "candidate_instance_metrics": candidate_instance,
        "baseline_attributes": baseline_rows,
        "candidate_attributes": candidate_rows,
        "selected_cases": cases,
    }
    with open(os.path.join(args.out, "e0_e3_detailed_metrics.json"), "w",
              encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    if cases:
        draw_cases(dataset, cases, os.path.join(args.out, "e0_e3_error_cases.png"))
    print(json.dumps({"E0_instance": baseline_instance,
                      "E3_instance": candidate_instance,
                      "selected_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
