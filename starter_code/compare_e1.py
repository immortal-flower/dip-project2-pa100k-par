"""Compare unified val evaluations; reject confounded E1 configurations."""
import argparse
import json
from pathlib import Path


FIXED = ("image_size", "epochs", "batch_size", "lr", "weight_decay",
         "warmup_epochs", "workers", "hidden", "drop", "attention", "seed", "se_reduction")
FOCUS = ("boots", "holding", "lower_stripe", "age_lt18")


def validate(records):
    base = records["E0"]["checkpoint_args"]
    for name, record in records.items():
        if record.get("review_status") == "provisional_cpu":
            raise ValueError(f"{name}: provisional CPU review cannot enter the final comparison")
        runtime = record.get("evaluation_environment")
        if not runtime:
            raise ValueError(f"{name}: evaluation environment is missing; rerun evaluate.py")
        for key in ("device", "torch", "torchvision", "Pillow", "batch_size"):
            if runtime.get(key) != records["E0"]["evaluation_environment"].get(key):
                raise ValueError(f"{name}: evaluation environment differs: {key}")
        if record["split"] != "val" or record["samples"] != 10000:
            raise ValueError(f"{name}: expected the complete 10,000-image validation split")
        args = record["checkpoint_args"]
        for key in FIXED:
            if args.get(key) != base.get(key):
                raise ValueError(f"{name}: confounded setting {key}: {args.get(key)} != {base.get(key)}")
        if base.get("attention", "none") != "none":
            raise ValueError("E0 must use attention=none")
        if record.get("label_file_sha256") != records["E0"].get("label_file_sha256"):
            raise ValueError(f"{name}: validation annotation hashes differ")
        if not record.get("label_file_sha256"):
            raise ValueError("Run evaluate.py to produce hashed, unified evaluations")
        training_hashes = record.get("training_data_hashes")
        if training_hashes and training_hashes.get("val.csv", training_hashes.get("val_list.txt")) != record["label_file_sha256"]:
            raise ValueError(f"{name}: evaluation labels differ from training-time validation labels")
        expected_loss = "focal" if name == "E1b" else "bce"
        if args.get("loss", "bce") != expected_loss or bool(args.get("pos_weight")) != (name == "E1a"):
            raise ValueError(f"{name}: wrong loss or pos_weight configuration")
        if name == "E1b" and (args.get("focal_gamma") != 2 or args.get("focal_alpha") is not None):
            raise ValueError("Primary E1b is predeclared as gamma=2, alpha=None")
    e1a_hashes = records.get("E1a", {}).get("training_data_hashes")
    e1b_hashes = records.get("E1b", {}).get("training_data_hashes")
    if e1a_hashes and e1b_hashes and e1a_hashes != e1b_hashes:
        raise ValueError("E1a and E1b training-time annotation hashes differ")


def render(records):
    validate(records)
    lines = ["# E0 / E1a / E1b unified validation results", "",
             "All percentages; best checkpoints selected by legacy mean group AttrAcc.", "",
             "| Experiment | Best epoch | Legacy group AttrAcc | Macro mA | Macro P | Macro R | Macro F1 | Instance P | Instance R | Instance F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in records.items():
        values = [record["legacy_mean_group_AttrAcc"]]
        values += [record["macro"][k] for k in ("mA", "precision", "recall", "f1")]
        values += [record["instance"][k] for k in ("precision", "recall", "f1")]
        formatted = ["undefined" if v is None else f"{100*v:.2f}" for v in values]
        lines.append(f"| {name} | {record.get('best_epoch')} | " + " | ".join(formatted) + " |")
    lines += ["", "## Rare attributes and age control", "",
              "| Attribute | Experiment | Positives | AttrAcc | mA | P | R | F1 | TP | FP | FN | TN |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for attr in FOCUS:
        for name, record in records.items():
            row = record["attributes"][attr]
            values = ["undefined" if row[k] is None else f"{100*row[k]:.2f}"
                      for k in ("AttrAcc", "mA", "precision", "recall", "f1")]
            counts = [str(row[k]) for k in ("TP", "FP", "FN", "TN")]
            lines.append(f"| {attr} | {name} | {row['positive_support']} | " + " | ".join(values + counts) + " |")
    lines += ["", "## Provenance", ""]
    for name, record in records.items():
        lines.append(f"- {name}: checkpoint SHA-256 `{record.get('checkpoint_sha256')}`; labels `{record['label_file_sha256']}`.")
    lines += ["", "A single seed is descriptive evidence, not statistical significance. Examine both recall and false positives.",
              "age_lt18 uses unchanged CE; differences can arise indirectly through the shared backbone.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", required=True)
    parser.add_argument("--e1a", required=True)
    parser.add_argument("--e1b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = {name: json.loads(Path(path).read_text(encoding="utf-8"))
               for name, path in (("E0", args.e0), ("E1a", args.e1a), ("E1b", args.e1b))}
    content = render(records)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
