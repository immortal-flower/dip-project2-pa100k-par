"""Check a unified E0 evaluation against the archived baseline evidence."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", required=True)
    p.add_argument("--archive", default="experiments/E0")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", help="Independently recompute counts from exported NPZ")
    args = p.parse_args()
    result = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    archive = Path(args.archive)
    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    old_counts = {}
    for line in (archive / "e0_diagnostic.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 12 and parts[0] in result["attributes"]:
            old_counts[parts[0]] = [int(x) for x in parts[-4:]]
    if len(old_counts) != 26:
        raise ValueError("Expected 26 archived attribute confusion rows")
    if result["samples"] != 10000 or result["split"] != "val":
        raise ValueError("Expected complete validation evaluation")
    if result["checkpoint_sha256"] != "991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806":
        raise ValueError("Not the shared E0 checkpoint")
    checked_npz = False
    if args.predictions:
        import numpy as np
        with np.load(args.predictions, allow_pickle=False) as evidence:
            pred = evidence["predictions"].astype(bool)
            target = evidence["targets"].astype(bool)
            if pred.shape != (10000, 26) or target.shape != pred.shape:
                raise ValueError("Unexpected prediction or target shape")
            counts = np.stack([(pred & target).sum(0), (pred & ~target).sum(0),
                               (~pred & target).sum(0), (~pred & ~target).sum(0)])
            for index, attr in enumerate(evidence["attributes"].tolist()):
                expected = [result["attributes"][attr][k] for k in ("TP", "FP", "FN", "TN")]
                if counts[:, index].tolist() != expected:
                    raise ValueError(f"JSON and independent NPZ count disagree: {attr}")
        checked_npz = True
    differences = {}
    for attr, expected in old_counts.items():
        actual = [result["attributes"][attr][k] for k in ("TP", "FP", "FN", "TN")]
        if actual != expected:
            differences[attr] = {"archive": expected, "unified": actual}
    group_diff = max(abs(result["legacy_group_AttrAcc"][g] - old)
                     for g, old in summary["best_group_metrics"].items())
    lines = ["# E0 unified evaluation review", "",
             f"Validation images: {result['samples']}; checkpoint best epoch: {result['best_epoch']}.",
             f"Matching attribute confusion rows: {26-len(differences)}/26.",
             f"Maximum legacy group AttrAcc absolute difference: {group_diff:.10f}.",
             f"Independent NumPy recomputation of exported prediction counts: {'passed, all 26 attributes' if checked_npz else 'not requested'}.",
             "", "| Metric | Value (%) |", "| --- | ---: |",
             f"| Legacy mean group AttrAcc | {100*result['legacy_mean_group_AttrAcc']:.4f} |"]
    for name in ("mA", "precision", "recall", "f1"):
        lines.append(f"| Macro {name} | {100*result['macro'][name]:.4f} |")
    for name in ("precision", "recall", "f1"):
        lines.append(f"| Instance {name} | {100*result['instance'][name]:.4f} |")
    lines += ["", "| Attribute | Positive support | mA (%) | P (%) | R (%) | F1 (%) | TP | FP | FN | TN |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for attr in ("boots", "holding", "lower_stripe", "age_lt18"):
        row = result["attributes"][attr]
        metrics = [f"{100*row[k]:.4f}" for k in ("mA", "precision", "recall", "f1")]
        counts = [str(row[k]) for k in ("TP", "FP", "FN", "TN")]
        lines.append(f"| {attr} | {row['positive_support']} | " + " | ".join(metrics+counts) + " |")
    lines += ["", "## Verification boundary", "",
              "The archived E0 used CUDA; local review environment is recorded in verification.md (and in JSON for current evaluator versions).",
              "Matching counts verify predictions and metric interpretation; they do not establish multi-seed robustness.",
              "E0 has no saved training-data hash; current annotation audit documents the locally supplied files.",
              "The historical checkpoint selection remains legacy group AttrAcc, not macro mA.", ""]
    if differences or group_diff > 1e-7:
        lines += ["## Differences requiring review", "",
                  "This CPU review is provisional. Do not mix these numbers with GPU E1 runs as the final comparison.",
                  "Re-evaluate E0/E1a/E1b on the same server/software/preprocessing stack; E0 need not be retrained.",
                  "The current evidence does not identify the cause of prediction differences.", "", "```json",
                  json.dumps(differences, indent=2), "```", ""]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"matching_attributes": 26-len(differences),
                      "max_group_difference": group_diff, "differences": differences}, indent=2))


if __name__ == "__main__":
    main()
