"""Audit mapped CSV labels and record training-only positive weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset import PA100KDataset, load_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--attributes", default="attributes.yaml")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    spec = load_spec(args.attributes)
    results = {"splits": {}, "attributes": {}, "source": "raw CSV through PA100KDataset mapping"}
    split_paths = {}
    for split, expected in (("train", 80000), ("val", 10000)):
        ds = PA100KDataset(args.data_root, f"{split}.csv", spec)
        if len(ds) != expected:
            raise ValueError(f"{split}: expected {expected} samples, got {len(ds)}")
        source_order = np.asarray([values for _, values in ds.samples], dtype=np.int64)
        global_order = np.zeros_like(source_order)
        global_order[:, ds.col2global] = source_order
        paths = [relative for relative, _ in ds.samples]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{split}: duplicate image paths")
        split_paths[split] = set(paths)
        for group in spec["groups"]:
            if group["type"] == "softmax" and not np.all(global_order[:, group["indices"]].sum(1) == 1):
                raise ValueError(f"{split}: invalid one-hot {group['name']} labels")
        results["splits"][split] = {
            "samples": len(ds), "csv_sha256": hashlib.sha256(
                (Path(args.data_root) / f"{split}.csv").read_bytes()).hexdigest()}
        positives = global_order.sum(0)
        for attr, idx in spec["attr2idx"].items():
            results["attributes"].setdefault(attr, {})[split] = {
                "positives": int(positives[idx]), "positive_rate": float(positives[idx] / len(ds))}
            if split == "train":
                results["attributes"][attr]["pos_weight"] = float((len(ds) - positives[idx]) / max(positives[idx], 1))
    if split_paths["train"] & split_paths["val"]:
        raise ValueError("Training/validation paths overlap")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({attr: results["attributes"][attr]
                      for attr in ("boots", "holding", "lower_stripe", "age_lt18")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
