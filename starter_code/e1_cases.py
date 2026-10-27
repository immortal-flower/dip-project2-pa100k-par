"""Export deterministic improvement/regression examples from matched NPZ files."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e0", required=True)
    p.add_argument("--e1", required=True)
    p.add_argument("--name", choices=("E1a", "E1b"), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with np.load(args.e0, allow_pickle=False) as base, np.load(args.e1, allow_pickle=False) as improved:
        for key in ("paths", "attributes", "targets"):
            if not np.array_equal(base[key], improved[key]):
                raise ValueError(f"Unmatched prediction files: {key}")
        attributes = base["attributes"].tolist()
        targets = base["targets"]
        for attr in ("boots", "holding", "lower_stripe", "age_lt18"):
            col = attributes.index(attr)
            old_correct = base["predictions"][:, col] == targets[:, col]
            new_correct = improved["predictions"][:, col] == targets[:, col]
            for kind, selected in (("improvement", ~old_correct & new_correct),
                                   ("regression", old_correct & ~new_correct)):
                # Fixed filename order, not cherry-picked by confidence.
                indices = sorted(np.flatnonzero(selected), key=lambda i: str(base["paths"][i]))[:2]
                for index in indices:
                    relative = str(base["paths"][index])
                    image_path = (Path(args.data_root) / relative).resolve()
                    image_path.relative_to(Path(args.data_root).resolve())
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    image.thumbnail((190, 330))
                    panel = Image.new("RGB", (760, 390), "white")
                    panel.paste(image, (15, 40))
                    draw = ImageDraw.Draw(panel)
                    font = ImageFont.load_default()
                    metadata = {"path": relative, "attribute": attr, "change": kind,
                                "target": int(targets[index, col]),
                                "E0": {"prediction": int(base['predictions'][index, col]),
                                       "probability": float(base['probabilities'][index, col])},
                                args.name: {"prediction": int(improved['predictions'][index, col]),
                                            "probability": float(improved['probabilities'][index, col])}}
                    text = (f"{attr}: {kind}\nImage: {Path(relative).name}\n"
                            f"Target: {metadata['target']}\n"
                            f"E0: pred={metadata['E0']['prediction']} p={metadata['E0']['probability']:.4f}\n"
                            f"{args.name}: pred={metadata[args.name]['prediction']} p={metadata[args.name]['probability']:.4f}\n\n"
                            "Visual cause: pending human inspection\n"
                            "Filename-order selection; 2 per change/attribute.")
                    draw.multiline_text((225, 50), text, fill="black", font=font, spacing=12)
                    filename = f"{attr}_{kind}_{Path(relative).stem}.png"
                    panel.save(out / filename)
                    metadata["figure"] = filename
                    records.append(metadata)
    (out / "cases.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} cases in {out}; inspect images before writing causal explanations.")


if __name__ == "__main__":
    main()
