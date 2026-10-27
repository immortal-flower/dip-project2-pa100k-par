"""Plan or execute A's two single-factor experiments and unified evaluation.

Default is plan-only. --execute requires CUDA with bfloat16 support, the shared
E0 checkpoint and fresh output folders. No shell interpolation is used.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E0_SHA256 = "991891f5b367a2a51cefaffd1d647e5e9861b225884ac78122f89d1da778c806"


def build_commands(data_root, e0_ckpt, out_root, python=sys.executable):
    config = json.loads((ROOT / "experiments/E0/config.json").read_text(encoding="utf-8"))["args"]
    common = []
    for key in ("epochs", "batch_size", "lr", "weight_decay", "warmup_epochs", "workers",
                "hidden", "drop", "attention", "se_reduction", "seed"):
        common += ["--" + key.replace("_", "-"), str(config[key])]
    common += ["--image-size", *map(str, config["image_size"]),
               "--attributes", str(ROOT / "attributes.yaml"), "--data-root", str(data_root)]
    out_root = Path(out_root)
    e0 = out_root / "E0_review"
    e1a, e1b = out_root / "E1a_pos_weight", out_root / "E1b_focal"
    commands = [[python, str(ROOT / "starter_code/audit_e1_data.py"), "--data-root", str(data_root),
                 "--attributes", str(ROOT / "attributes.yaml"), "--out", str(out_root / "data_audit.json")]]
    for name, folder, loss in (("E1a", e1a, ["--loss", "bce", "--pos-weight"]),
                               ("E1b", e1b, ["--loss", "focal", "--focal-gamma", "2"])):
        commands.append([python, str(ROOT / "starter_code/train.py"), *common,
                         *loss, "--out", str(folder)])
    for checkpoint, folder in ((e0_ckpt, e0), (e1a / "best.pt", e1a), (e1b / "best.pt", e1b)):
        commands.append([python, str(ROOT / "starter_code/evaluate.py"), "--ckpt", str(checkpoint),
                         "--data-root", str(data_root), "--device", "cuda", "--batch-size", "64",
                         "--workers", "4", "--out", str(folder / "evaluation.json"),
                         "--predictions", str(folder / "predictions.npz")])
    commands.append([python, str(ROOT / "starter_code/verify_e0.py"),
                     "--evaluation", str(e0 / "evaluation.json"), "--predictions", str(e0 / "predictions.npz"),
                     "--archive", str(ROOT / "experiments/E0"), "--out", str(e0 / "review.md")])
    commands.append([python, str(ROOT / "starter_code/compare_e1.py"),
                     "--e0", str(e0 / "evaluation.json"), "--e1a", str(e1a / "evaluation.json"),
                     "--e1b", str(e1b / "evaluation.json"), "--out", str(out_root / "E1_comparison.md")])
    for name, folder in (("E1a", e1a), ("E1b", e1b)):
        commands.append([python, str(ROOT / "starter_code/e1_cases.py"),
                         "--e0", str(e0 / "predictions.npz"), "--e1", str(folder / "predictions.npz"),
                         "--name", name, "--data-root", str(data_root), "--out", str(out_root / f"{name}_cases")])
    return commands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--e0-ckpt", required=True)
    p.add_argument("--out-root", default="runs/E1_suite")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    data, checkpoint, out = Path(args.data_root).resolve(), Path(args.e0_ckpt).resolve(), Path(args.out_root).resolve()
    commands = build_commands(data, checkpoint, out)
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
    if not args.execute:
        print("Plan only. Add --execute on the prepared GPU server to run.")
        return
    import torch
    if sys.version_info < (3, 10):
        p.error("Formal experiments require Python 3.10+; local Python 3.9 is smoke-test only")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("Formal E1 execution requires CUDA with bfloat16 support, matching E0 precision")
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != E0_SHA256:
        p.error("Supply the shared E0 Release checkpoint with the documented SHA-256")
    if out.exists() and any(out.iterdir()):
        p.error("Use a fresh output directory; partial/completed experiments are never overwritten")
    for source in ("train.csv", "val.csv"):
        if not (data / source).is_file():
            p.error(f"Missing {source} in raw dataset root")
        expected = json.loads((ROOT / "experiments/E1/data_audit.json").read_text(encoding="utf-8"))
        split = source.split(".")[0]
        if hashlib.sha256((data / source).read_bytes()).hexdigest() != expected["splits"][split]["csv_sha256"]:
            p.error(f"{source} differs from the audited dataset; inspect annotations before formal training")
    # Source ZIP snapshots carry this manifest when .git is unavailable.
    snapshot = ROOT / "SOURCE_MANIFEST.json"
    if snapshot.is_file():
        import shutil
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, out / "SOURCE_MANIFEST.json")
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Finished. Review {out / 'E1_comparison.md'} and inspect exported cases before writing conclusions.")


if __name__ == "__main__":
    main()
