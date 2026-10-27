"""Validation-only soft-voting search and frozen ensemble evaluation.

The optimizer consumes NPZ files written by ``evaluate.py --predictions``.
It never loads images or checkpoints, so calibration and final evaluation are
explicitly separated: run ``optimize`` on validation predictions, then run
``apply`` with the saved config on test predictions.
"""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

SCORE_KEYS = (
    "legacy_mean_group_AttrAcc",
    "macro.mA",
    "macro.f1",
    "instance.f1",
)


def load_spec(path="attributes.yaml"):
    """Load only the label layout, without importing the PyTorch dataset."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    groups, offset = [], 0
    for name, group in config["groups"].items():
        attrs = list(group["attrs"])
        indices = list(range(offset, offset + len(attrs)))
        groups.append({"name": name, "type": group["type"],
                       "attrs": attrs, "indices": indices})
        offset += len(attrs)
    return {"groups": groups,
            "attr2idx": {attr: index for group in groups
                         for attr, index in zip(group["attrs"], group["indices"])},
            "num_attrs": offset}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Missing or duplicate model name: {name!r}")
        result[name] = Path(path)
    return result


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"paths", "attributes", "probabilities", "targets"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        return {key: data[key].copy() for key in required}


def assert_aligned(named):
    items = list(named.items())
    if not items:
        raise ValueError("At least one prediction file is required")
    reference_name, reference = items[0]
    for name, current in items[1:]:
        for key in ("paths", "attributes", "targets"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(
                    f"{name} is not aligned with {reference_name}: {key} differs"
                )
        if current["probabilities"].shape != reference["probabilities"].shape:
            raise ValueError(f"{name}: probability shape differs")
    return reference


def decode_probabilities(probabilities, spec, threshold=0.5):
    probabilities = np.asarray(probabilities)
    predictions = np.zeros_like(probabilities, dtype=bool)
    for group in spec["groups"]:
        indices = group["indices"]
        if group["type"] == "softmax":
            winner = probabilities[:, indices].argmax(axis=1)
            predictions[np.arange(len(predictions)), np.asarray(indices)[winner]] = True
        else:
            predictions[:, indices] = probabilities[:, indices] >= threshold
    return predictions


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def calculate_metrics(probabilities, targets, spec, threshold=0.5):
    pred = decode_probabilities(probabilities, spec, threshold)
    target = np.asarray(targets).astype(bool)
    if pred.shape != target.shape:
        raise ValueError("Predictions and targets must have identical shapes")
    samples = len(target)
    attributes = {}
    for attr, index in spec["attr2idx"].items():
        p, t = pred[:, index], target[:, index]
        tp = int(np.sum(p & t))
        fp = int(np.sum(p & ~t))
        fn = int(np.sum(~p & t))
        tn = int(np.sum(~p & ~t))
        attributes[attr] = {
            "AttrAcc": (tp + tn) / samples,
            "mA": (0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))
                   if tp + fn and tn + fp else None),
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "positive_support": tp + fn,
        }

    def macro(names):
        output = {}
        for key in ("AttrAcc", "mA", "precision", "recall", "f1"):
            values = [attributes[name][key] for name in names
                      if attributes[name][key] is not None]
            output[key] = float(np.mean(values)) if values else None
        return output

    group_metrics = {}
    for group in spec["groups"]:
        idx = group["indices"]
        if group["type"] == "softmax":
            group_metrics[group["name"]] = float(np.mean(
                pred[:, idx].argmax(1) == target[:, idx].argmax(1)))
        else:
            group_metrics[group["name"]] = float(np.mean(pred[:, idx] == target[:, idx]))

    tp_per_sample = np.sum(pred & target, axis=1)
    pred_count = np.sum(pred, axis=1)
    target_count = np.sum(target, axis=1)
    instance = {
        "precision": float(np.mean(np.divide(tp_per_sample, pred_count,
            out=np.zeros(samples, dtype=float), where=pred_count != 0))),
        "recall": float(np.mean(np.divide(tp_per_sample, target_count,
            out=np.zeros(samples, dtype=float), where=target_count != 0))),
        "f1": float(np.mean(np.divide(2 * tp_per_sample, pred_count + target_count,
            out=np.zeros(samples, dtype=float), where=(pred_count + target_count) != 0))),
    }
    return {
        "schema_version": 1,
        "samples": samples,
        "threshold": threshold,
        "macro": macro(list(attributes)),
        "groups": {g["name"]: macro(g["attrs"]) for g in spec["groups"]},
        "attributes": attributes,
        "instance": instance,
        "legacy_group_AttrAcc": group_metrics,
        "legacy_mean_group_AttrAcc": float(np.mean(list(group_metrics.values()))),
    }


def metric_value(metrics, dotted_key):
    value = metrics
    for part in dotted_key.split("."):
        value = value[part]
    return value


def composite_score(metrics, baseline_metrics):
    ratios = [metric_value(metrics, key) / metric_value(baseline_metrics, key)
              for key in SCORE_KEYS]
    return float(np.mean(ratios))


def blend(probabilities, weights):
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ensemble weights must sum to 1, got {total}")
    return sum(probabilities[name] * weight for name, weight in weights.items())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


def optimize(args):
    if args.split.lower() != "val":
        raise ValueError("Weight optimization is restricted to the validation split")
    spec = load_spec(args.attributes)
    sources = parse_named_paths(args.prediction)
    if args.baseline not in sources:
        raise ValueError("--baseline must match one --prediction name")
    loaded = {name: load_prediction(path) for name, path in sources.items()}
    reference = assert_aligned(loaded)
    expected_attrs = np.array(sorted(spec["attr2idx"], key=spec["attr2idx"].get))
    if not np.array_equal(reference["attributes"], expected_attrs):
        raise ValueError("Prediction attributes do not match attributes.yaml")
    probabilities = {name: item["probabilities"] for name, item in loaded.items()}
    singles = {name: calculate_metrics(prob, reference["targets"], spec, args.threshold)
               for name, prob in probabilities.items()}
    baseline_metrics = singles[args.baseline]
    rows, best = [], None
    grid = np.linspace(0.0, 1.0, round(1.0 / args.step) + 1)
    for first, second in combinations(sources, 2):
        for first_weight in grid:
            weights = {first: float(first_weight), second: float(1 - first_weight)}
            metrics = calculate_metrics(blend(probabilities, weights),
                                        reference["targets"], spec, args.threshold)
            row = {"models": [first, second], "weights": weights,
                   "composite_score": composite_score(metrics, baseline_metrics),
                   **{key: metric_value(metrics, key) for key in SCORE_KEYS}}
            rows.append(row)
            if best is None or row["composite_score"] > best["composite_score"]:
                best = dict(row)
                best["metrics"] = metrics

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "method": "probability_soft_voting",
        "selection_split": "val",
        "test_split_used_for_selection": False,
        "threshold": args.threshold,
        "score": {"name": "equal_mean_relative_to_baseline", "keys": list(SCORE_KEYS),
                  "baseline": args.baseline},
        "models": best["models"],
        "weights": best["weights"],
        "prediction_sources": {name: {"path": str(path), "sha256": sha256(path)}
                               for name, path in sources.items()},
    }
    save_json(output / "ensemble_config.json", config)
    save_json(output / "val_evaluation.json", {
        "split": "val", "best": best, "single_models": singles,
    })
    save_json(output / "weight_search.json", {"step": args.step, "rows": rows})
    selected = blend(probabilities, best["weights"])
    if args.save_predictions:
        np.savez_compressed(output / "val_ensemble_predictions.npz",
                            paths=reference["paths"], attributes=reference["attributes"],
                            probabilities=selected,
                            predictions=decode_probabilities(selected, spec, args.threshold),
                            targets=reference["targets"])
    print(json.dumps({"config": str(output / "ensemble_config.json"),
                      "best": best}, indent=2, ensure_ascii=False))


def apply(args):
    spec = load_spec(args.attributes)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sources = parse_named_paths(args.prediction)
    expected = set(config["models"])
    if set(sources) != expected:
        raise ValueError(f"Frozen config requires predictions for {sorted(expected)}")
    loaded = {name: load_prediction(path) for name, path in sources.items()}
    reference = assert_aligned(loaded)
    references = parse_named_paths(args.reference or [])
    reference_loaded = {name: load_prediction(path) for name, path in references.items()}
    if reference_loaded:
        assert_aligned(dict(loaded, **reference_loaded))
    combined = blend({name: item["probabilities"] for name, item in loaded.items()},
                     config["weights"])
    metrics = calculate_metrics(combined, reference["targets"], spec,
                                config.get("threshold", 0.5))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    single_metrics = {
        name: calculate_metrics(item["probabilities"], reference["targets"], spec,
                                config.get("threshold", 0.5))
        for name, item in dict(loaded, **reference_loaded).items()
    }
    save_json(output / f"{args.split}_evaluation.json", {
        "split": args.split,
        "selection_split": config["selection_split"],
        "config_sha256": sha256(args.config),
        "prediction_sources": {name: {"path": str(path), "sha256": sha256(path)}
                               for name, path in sources.items()},
        "reference_sources": {name: {"path": str(path), "sha256": sha256(path)}
                              for name, path in references.items()},
        "single_models": single_metrics,
        "metrics": metrics,
    })
    baseline_name = config["score"]["baseline"]
    baseline = single_metrics.get(baseline_name)
    with (output / f"{args.split}_attribute_comparison.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        fields = ["attribute", "positive_support"]
        for name in single_metrics:
            fields.extend([f"{name}_mA", f"{name}_f1"])
        fields.extend(["ensemble_mA", "ensemble_f1",
                       f"delta_f1_vs_{baseline_name}"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attr in reference["attributes"]:
            row = {"attribute": str(attr),
                   "positive_support": metrics["attributes"][str(attr)]["positive_support"]}
            for name, result in single_metrics.items():
                row[f"{name}_mA"] = result["attributes"][str(attr)]["mA"]
                row[f"{name}_f1"] = result["attributes"][str(attr)]["f1"]
            row["ensemble_mA"] = metrics["attributes"][str(attr)]["mA"]
            row["ensemble_f1"] = metrics["attributes"][str(attr)]["f1"]
            row[f"delta_f1_vs_{baseline_name}"] = (
                metrics["attributes"][str(attr)]["f1"]
                - baseline["attributes"][str(attr)]["f1"] if baseline else "")
            writer.writerow(row)
    if args.save_predictions:
        np.savez_compressed(output / f"{args.split}_ensemble_predictions.npz",
                            paths=reference["paths"], attributes=reference["attributes"],
                            probabilities=combined,
                            predictions=decode_probabilities(combined, spec,
                                                             config.get("threshold", 0.5)),
                            targets=reference["targets"])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--attributes", default="attributes.yaml")
    common.add_argument("--prediction", action="append", required=True,
                        help="Repeat NAME=PATH for aligned evaluate.py NPZ files")
    common.add_argument("--out-dir", required=True)
    common.add_argument("--save-predictions", action="store_true",
                        help="Also save the combined per-image NPZ (large, local artifact)")

    search = sub.add_parser("optimize", parents=[common])
    search.add_argument("--split", default="val")
    search.add_argument("--baseline", required=True)
    search.add_argument("--step", type=float, default=0.05)
    search.add_argument("--threshold", type=float, default=0.5)
    search.set_defaults(func=optimize)

    frozen = sub.add_parser("apply", parents=[common])
    frozen.add_argument("--config", required=True)
    frozen.add_argument("--split", default="test")
    frozen.add_argument("--reference", action="append",
                        help="Optional NAME=PATH single-model reference (not blended)")
    frozen.set_defaults(func=apply)
    args = parser.parse_args()
    if getattr(args, "step", 0.05) <= 0 or getattr(args, "step", 0.05) > 1:
        parser.error("--step must be in (0, 1]")
    args.func(args)


if __name__ == "__main__":
    main()
