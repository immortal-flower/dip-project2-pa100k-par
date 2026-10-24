"""Evaluate the residual-SE checkpoint on exactly the archived paired cases.

The reference JSONL contains the same 10,000 validation images predicted by
E0, standard group SE, and centered group SE. This script adds the new model's
scores without changing any threshold or training setting.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PA100KDataset, build_transform
from model import PAResNet


FOCUS = ("glasses", "short_sleeve", "long_sleeve", "backpack",
         "shoulder_bag", "handbag")


def load_reference(path):
    by_image = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            attr = row["attribute"]
            if attr not in FOCUS:
                continue
            image = row["image"].replace("\\", "/")
            if attr in by_image[image]:
                raise ValueError(f"Duplicate reference: {image}/{attr}")
            by_image[image][attr] = row
    if len(by_image) != 10000 or any(set(rows) != set(FOCUS)
                                     for rows in by_image.values()):
        raise ValueError("Reference must cover six attributes on 10,000 images")
    return by_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    if model_args["attention"] != "se_group_residual025":
        raise ValueError("Checkpoint is not the residual025 experiment")
    spec = checkpoint["spec"]
    model = PAResNet(spec, hidden=model_args["hidden"],
                     drop=model_args["drop"], attention=model_args["attention"],
                     se_reduction=model_args["se_reduction"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = PA100KDataset(str(args.data_root), "val.csv", spec,
                            build_transform(False, model_args["image_size"]))
    if len(dataset) != 10000:
        raise ValueError(f"Expected 10,000 validation images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    args.out.mkdir(parents=True, exist_ok=False)

    counts = {attr: Counter() for attr in FOCUS}
    focus_index = {attr: (group["name"], offset, group["indices"][offset])
                   for group in spec["groups"]
                   for offset, attr in enumerate(group["attrs"])
                   if attr in FOCUS}
    sleeve_pairs = []
    glasses_new_fns = []
    seen = set()
    position = 0
    rows_out = args.out / "paired_new_predictions.jsonl"
    with rows_out.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for images, targets in loader:
            outputs = model(images.to(device, non_blocking=True))
            scores = {attr: torch.sigmoid(outputs[group][:, offset]).cpu().tolist()
                      for attr, (group, offset, _) in focus_index.items()}
            for j in range(len(images)):
                filename = dataset.samples[position + j][0].replace("\\", "/")
                if filename not in reference or filename in seen:
                    raise ValueError(f"Missing/duplicate image in reference: {filename}")
                seen.add(filename)
                image_rows = {}
                for attr in FOCUS:
                    archived = reference[filename][attr]
                    _, _, global_index = focus_index[attr]
                    truth = int(targets[j, global_index])
                    if truth != archived["truth"]:
                        raise ValueError(f"Label mismatch: {filename}/{attr}")
                    score = float(scores[attr][j])
                    pred = int(score > 0.5)
                    row = {"image": filename, "attribute": attr,
                           "truth": truth, "e0_pred": archived["e0_pred"],
                           "e0_prob": archived["e0_prob"],
                           "group_pred": archived["group_pred"],
                           "group_prob": archived["group_prob"],
                           "center_pred": archived["center_pred"],
                           "center_prob": archived["center_prob"],
                           "residual_pred": pred,
                           "residual_prob": round(score, 6)}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    image_rows[attr] = row

                    e0_right = archived["e0_pred"] == truth
                    new_right = pred == truth
                    key = ("both_correct" if e0_right and new_right else
                           "both_wrong" if not e0_right and not new_right else
                           "fixed_fn" if new_right and truth else
                           "fixed_fp" if new_right else
                           "new_fn" if truth else "new_fp")
                    counts[attr][key] += 1

                short = image_rows["short_sleeve"]
                long = image_rows["long_sleeve"]
                if (short["truth"] == 1 and short["e0_pred"] == 1 and
                        short["center_pred"] == 0 and long["truth"] == 0 and
                        long["e0_pred"] == 0 and long["center_pred"] == 1):
                    sleeve_pairs.append((short, long))
                glasses = image_rows["glasses"]
                if (glasses["truth"] == 1 and glasses["e0_pred"] == 1 and
                        glasses["center_pred"] == 0):
                    glasses_new_fns.append(glasses)
            position += len(images)

    if seen != set(reference) or len(sleeve_pairs) != 459 or len(glasses_new_fns) != 347:
        raise ValueError("Paired validation cases do not match prior experiment")

    sleeve_recovered = sum(s["residual_pred"] == 1 and l["residual_pred"] == 0
                           for s, l in sleeve_pairs)
    sleeve_still_both_wrong = sum(s["residual_pred"] == 0 and l["residual_pred"] == 1
                                  for s, l in sleeve_pairs)
    glasses_recovered = sum(row["residual_pred"] == 1
                            for row in glasses_new_fns)
    summary = {
        "model": "se_group_residual025", "samples": len(seen),
        "threshold": 0.5, "best_epoch": checkpoint["best_epoch"],
        "sleeve_pair": {"original_count": 459,
                        "both_recovered": sleeve_recovered,
                        "still_both_wrong": sleeve_still_both_wrong,
                        "other_outcomes": 459 - sleeve_recovered - sleeve_still_both_wrong},
        "glasses_new_fn": {"original_count": 347,
                           "recovered": glasses_recovered,
                           "still_missed": 347 - glasses_recovered},
        "versus_E0": {attr: dict(counts[attr]) for attr in FOCUS},
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.out / "target_cases.csv").open("w", encoding="utf-8-sig",
                                               newline="") as stream:
        fields = ("cohort", "image", "attribute", "truth", "e0_prob",
                  "group_prob", "center_prob", "residual_prob",
                  "residual_pred")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cohort, items in (("sleeve_pair", [row for pair in sleeve_pairs
                                               for row in pair]),
                              ("glasses_new_fn", glasses_new_fns)):
            for row in items:
                writer.writerow({key: (cohort if key == "cohort" else row[key])
                                 for key in fields})

    print("459 sleeve pairs: both recovered", sleeve_recovered,
          "still both wrong", sleeve_still_both_wrong, flush=True)
    print("347 glasses misses: recovered", glasses_recovered,
          "still missed", 347 - glasses_recovered, flush=True)
    print("Saved", args.out, flush=True)


if __name__ == "__main__":
    main()
