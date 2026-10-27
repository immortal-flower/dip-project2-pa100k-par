"""Independently audit returned E1 NPZ/JSON evidence (NumPy only, no torch).

Usage: python starter_code/audit_e1_results.py --suite runs/E1_suite --out experiments/E1/results_audit.json
Raises on inconsistency; never modifies input results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_e1 import validate


GROUPS = {"gender": (0, 1), "age": (1, 4), "orientation": (4, 7),
          "accessory": (7, 9), "upper": (9, 15), "lower": (15, 21),
          "bag": (21, 24), "boots": (24, 25), "holding": (25, 26)}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def audit(suite):
    folders = {"E0": "E0_review", "E1a": "E1a_pos_weight", "E1b": "E1b_focal"}
    records = {n: load(suite / f / "evaluation.json") for n, f in folders.items()}
    validate(records)
    reference = None
    output = {"status": "passed", "samples_per_model": 10000, "models": {}}
    manifest_hash = hashlib.sha256((suite / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    for name, folder in folders.items():
        r = records[name]
        with np.load(suite / folder / "predictions.npz", allow_pickle=False) as data:
            p = data["predictions"].copy()
            y = data["targets"].astype(bool)
            attrs = data["attributes"].tolist()
            paths = data["paths"].copy()
            raw_y = data["targets"]
            assert np.isin(raw_y, [0, 1]).all(), "non-binary labels"
            assert p.shape == y.shape == (10000, 26)
            assert len(set(paths)) == 10000 and len(set(attrs)) == 26
            probs = data["probabilities"]
            assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
            for lo, hi in [(1, 4), (4, 7)]:
                assert (y[:, lo:hi].sum(1) == 1).all()
                assert (p[:, lo:hi].sum(1) == 1).all()
                np.testing.assert_array_equal(p[:, lo:hi].argmax(1), probs[:, lo:hi].argmax(1))
            idx = [i for i in range(26) if i not in range(1, 7)]
            np.testing.assert_array_equal(p[:, idx], probs[:, idx] > .5)
        if reference is None:
            reference = (paths, attrs, y)
        else:
            np.testing.assert_array_equal(paths, reference[0])
            assert attrs == reference[1]
            np.testing.assert_array_equal(y, reference[2])
        tp, fp = (p & y).sum(0), (p & ~y).sum(0)
        fn, tn = (~p & y).sum(0), (~p & ~y).sum(0)
        div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
        stats = {"AttrAcc": (tp + tn) / len(y), "mA": .5 * (div(tp, tp + fn) + div(tn, tn + fp)),
                 "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "f1": div(2 * tp, 2 * tp + fp + fn)}
        for i, attr in enumerate(attrs):
            for key, values in {"TP": tp, "FP": fp, "FN": fn, "TN": tn}.items():
                assert int(values[i]) == r["attributes"][attr][key]
            for key, values in stats.items():
                close(values[i], r["attributes"][attr][key])
        for key, values in stats.items():
            close(values.mean(), r["macro"][key])
            for group, (lo, hi) in GROUPS.items():
                close(values[lo:hi].mean(), r["groups"][group][key])
        itp = (p & y).sum(1)
        for key, values in {"precision": div(itp, p.sum(1)), "recall": div(itp, y.sum(1)),
                            "f1": div(2 * itp, p.sum(1) + y.sum(1))}.items():
            close(values.mean(), r["instance"][key])
        legacy = []
        for group, (lo, hi) in GROUPS.items():
            score = (p[:, lo:hi] == y[:, lo:hi]).all(1).mean() if group in ("age", "orientation") else stats["AttrAcc"][lo:hi].mean()
            np.testing.assert_allclose(score, r["legacy_group_AttrAcc"][group], rtol=0, atol=1e-7)
            legacy.append(score)
        np.testing.assert_allclose(np.mean(legacy), r["legacy_mean_group_AttrAcc"], rtol=0, atol=1e-7)
        result = {"confusion_rows_verified": 26, "metric_tolerance": 1e-10,
                  "legacy_float32_tolerance": 1e-7, "best_epoch": r["best_epoch"],
                  "checkpoint_sha256_reported_by_server": r["checkpoint_sha256"]}
        if name != "E0":
            rows = [json.loads(line) for line in (suite / folder / "metrics.jsonl").read_text().splitlines()]
            assert [row["epoch"] for row in rows] == list(range(1, 16))
            best = max(rows, key=lambda row: row["score"])
            summary = load(suite / folder / "summary.json")
            assert best["epoch"] == r["best_epoch"] == summary["best_epoch"]
            close(best["score"], summary["best_score"])
            assert best["complete_metrics"]["attributes"] == r["attributes"]
            config = load(suite / folder / "config.json")
            assert config["args"] == r["checkpoint_args"]
            assert config["provenance"]["source_manifest_sha256"] == manifest_hash
            if name == "E1a":
                inventory = load(suite / "data_audit.json")
                expected_weights = [inventory["attributes"][attr]["pos_weight"] for attr in attrs]
                np.testing.assert_allclose(config["pos_weight_values"], expected_weights, rtol=1e-6)
            result.update(epochs_verified=15, total_seconds=summary["total_seconds"], source_manifest_sha256=manifest_hash)
        output["models"][name] = result
    output["deltas_percentage_points"] = {
        name: {key: 100 * (records[name]["macro"][key] - records["E0"]["macro"][key]) for key in stats}
        for name in ("E1a", "E1b")}
    output["scope"] = "Audits returned predictions and metadata, not E1 checkpoint bytes or a fresh model forward pass."
    return output


def attribute_table(suite):
    lines = ["# All 26 validation attributes", "", "Percentages; P/R/F1 are per-attribute, not instance metrics.", "",
             "| Attribute | Model | Positives | AttrAcc | mA | P | R | F1 | TP | FP | FN | TN |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    records = {n: load(suite / f / "evaluation.json") for n, f in
               [("E0", "E0_review"), ("E1a", "E1a_pos_weight"), ("E1b", "E1b_focal")]}
    for attr in records["E0"]["attributes"]:
        for name, record in records.items():
            row = record["attributes"][attr]
            values = [f"{100 * row[k]:.2f}" for k in ("AttrAcc", "mA", "precision", "recall", "f1")]
            values += [str(row[k]) for k in ("TP", "FP", "FN", "TN")]
            lines.append(f"| {attr} | {name} | {row['positive_support']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def selected_cases(suite):
    """Scores for the six manually inspected report figures, all three models."""
    cases = [("boots", "081407"), ("boots", "080188"), ("holding", "080790"),
             ("lower_stripe", "085565"), ("lower_stripe", "087064"), ("boots", "080051")]
    result = []
    for attr, stem in cases:
        row = {"path": f"data/{stem}.jpg", "attribute": attr, "models": {}}
        for name, folder in [("E0", "E0_review"), ("E1a", "E1a_pos_weight"), ("E1b", "E1b_focal")]:
            with np.load(suite / folder / "predictions.npz", allow_pickle=False) as data:
                i = data["paths"].tolist().index(row["path"])
                j = data["attributes"].tolist().index(attr)
                row["target"] = int(data["targets"][i, j])
                row["models"][name] = {"prediction": int(data["predictions"][i, j]),
                                        "probability": float(data["probabilities"][i, j])}
        result.append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.with_name("all_attributes.md").write_text(attribute_table(args.suite), encoding="utf-8")
    args.out.with_name("selected_cases.json").write_text(
        json.dumps(selected_cases(args.suite), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
