#!/usr/bin/env bash
# 独立对照：只将九组中心 SE 的门控幅度由 alpha=1 改为 alpha=0.25。
# E0 的数据、损失、采样、增强、训练参数和最佳轮次选择规则全部不变。
set -euo pipefail
cd "$(dirname "$0")/.."

for required in \
  PA-100K/train.csv PA-100K/val.csv attributes.yaml requirements.txt \
  reference/paired_predictions.jsonl \
  starter_code/model.py starter_code/train.py starter_code/infer.py \
  starter_code/dataset.py starter_code/evaluate.py starter_code/losses.py \
  starter_code/compare_se.py starter_code/evaluate_attributes.py \
  starter_code/evaluate_residual_cases.py \
  docs/e2_group_residual025_experiment.md \
  runs/E0_baseline/best.pt runs/E0_baseline/config.json \
  runs/E0_baseline/summary.json runs/E0_baseline/metrics.jsonl \
  runs/E2_se/best.pt runs/E2_se/config.json \
  runs/E2_se/summary.json runs/E2_se/metrics.jsonl \
  runs/E2_se_group/best.pt runs/E2_se_group/config.json \
  runs/E2_se_group/summary.json runs/E2_se_group/metrics.jsonl \
  runs/E2_se_group_center/best.pt runs/E2_se_group_center/config.json \
  runs/E2_se_group_center/summary.json \
  runs/E2_se_group_center/metrics.jsonl; do
  if test ! -s "$required"; then
    echo "缺少必需文件：$required" >&2
    exit 1
  fi
done
test -d PA-100K/data || { echo "缺少图片目录：PA-100K/data" >&2; exit 1; }
if test -e runs/E2_se_group_residual025 || \
   test -e pa100k_e2_group_residual025_record.tar.gz; then
  echo "本实验已有输出；为保护结果拒绝覆盖。" >&2
  exit 1
fi

python - <<'PY'
import json
import hashlib
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path("starter_code").resolve()))
from model import PAResNet

e0_source_hashes = {
    "starter_code/losses.py": "4fb0218eb081885cf4e70c390565262008836db5c0dfe186ca27403fa9d07cec",
    "starter_code/dataset.py": "9a50c1d9897b43402e2148590f078bd296d4e6fb443c6add42c2d890d91ee76d",
    "starter_code/evaluate.py": "2fb2a3cb11c7b2a37305c66e8efe15b13103bc1ea6b930383a91477d1ba627cd",
    "attributes.yaml": "a2a8d386f064cec810b18a1f6641407c342a184e8b47bb7b904cfc1b5d552b6a",
}
for path, expected in e0_source_hashes.items():
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert actual == expected, f"{path} 已偏离 E0；拒绝把 A/C 改动混入结构实验"

assert torch.cuda.is_available(), "CUDA 不可用；请检查环境和 GPU"
print("CUDA GPU:", torch.cuda.get_device_name(0), flush=True)
names = ("E0_baseline", "E2_se", "E2_se_group", "E2_se_group_center")
configs = [json.loads((Path("runs") / name / "config.json").read_text(encoding="utf-8"))
           for name in names]
expected_modes = ("none", "se", "se_group", "se_group_center")
for name, config, mode in zip(names, configs, expected_modes):
    assert config["args"]["attention"] == mode, f"{name} attention 不符"
    assert config["data_format"] == "csv", f"{name} 不是原始 CSV 数据"
    assert config["selection_metric"] == "mean_group_AttrAcc", name
    ref = {k: v for k, v in configs[0]["args"].items()
           if k not in ("attention", "out")}
    current = {k: v for k, v in config["args"].items()
               if k not in ("attention", "out")}
    assert current == ref, f"{name} 与 E0 训练配置不同"

required = {"attributes": "./attributes.yaml", "data_root": "./PA-100K",
            "image_size": [256, 128], "epochs": 15, "batch_size": 64,
            "lr": 0.0003, "weight_decay": 0.0005, "warmup_epochs": 1,
            "workers": 4, "hidden": 256, "drop": 0.3,
            "se_reduction": 16, "pos_weight": False, "seed": 42}
for key, expected in required.items():
    assert ref[key] == expected, f"E0 {key}={ref[key]}，与运行脚本不一致"

checkpoint = torch.load("runs/E2_se_group_center/best.pt", map_location="cpu",
                        weights_only=False)
assert checkpoint["args"]["attention"] == "se_group_center"
kw = dict(hidden=checkpoint["args"]["hidden"],
          drop=checkpoint["args"]["drop"],
          se_reduction=checkpoint["args"]["se_reduction"], pretrained=False)
old = PAResNet(checkpoint["spec"], attention="se_group_center", **kw)
new = PAResNet(checkpoint["spec"], attention="se_group_residual025", **kw)
old.load_state_dict(checkpoint["model"], strict=True)
new.load_state_dict(checkpoint["model"], strict=True)
assert list(old.state_dict()) == list(new.state_dict())
probe = torch.randn(3, 512)
for name in old.group_attention:
    block = old.group_attention[name]
    expected = probe * (1 + .25 * (2 * block.gate(probe) - 1))
    actual = new.group_attention[name](probe)
    assert torch.allclose(actual, expected, atol=1e-6), name
print("预检通过：唯一模型改动为 SE 系数范围 0.75～1.25；训练配置与 E0 相同。",
      flush=True)
PY

python starter_code/train.py \
  --attributes ./attributes.yaml \
  --data-root ./PA-100K \
  --image-size 256 128 \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.0003 \
  --weight-decay 0.0005 \
  --warmup-epochs 1 \
  --workers 4 \
  --hidden 256 \
  --drop 0.3 \
  --attention se_group_residual025 \
  --se-reduction 16 \
  --seed 42 \
  --out ./runs/E2_se_group_residual025

python starter_code/compare_se.py \
  runs/E0_baseline runs/E2_se_group_residual025 \
  --out runs/E0_E2_residual025_comparison.md
python starter_code/compare_se.py \
  runs/E2_se_group_center runs/E2_se_group_residual025 \
  --out runs/E2_center_residual025_comparison.md

python starter_code/evaluate_attributes.py \
  --data-root ./PA-100K \
  --checkpoints runs/E0_baseline/best.pt runs/E2_se/best.pt \
                runs/E2_se_group/best.pt runs/E2_se_group_center/best.pt \
                runs/E2_se_group_residual025/best.pt \
  --batch-size 64 --workers 4 \
  --out runs/E0_E2_residual025_attribute_metrics.json

python starter_code/evaluate_residual_cases.py \
  --data-root ./PA-100K \
  --checkpoint runs/E2_se_group_residual025/best.pt \
  --reference reference/paired_predictions.jsonl \
  --batch-size 64 --workers 4 \
  --out runs/E2_se_group_residual025/paired_cases

tar -czf pa100k_e2_group_residual025_record.tar.gz \
  attributes.yaml requirements.txt \
  starter_code/model.py starter_code/train.py starter_code/infer.py \
  starter_code/dataset.py starter_code/evaluate.py starter_code/losses.py \
  starter_code/compare_se.py starter_code/evaluate_attributes.py \
  starter_code/evaluate_residual_cases.py \
  scripts/run_e2_group_residual025.sh \
  docs/e2_group_residual025_experiment.md \
  runs/E0_baseline/config.json runs/E0_baseline/summary.json \
  runs/E0_baseline/metrics.jsonl \
  runs/E2_se/config.json runs/E2_se/summary.json runs/E2_se/metrics.jsonl \
  runs/E2_se_group/config.json runs/E2_se_group/summary.json \
  runs/E2_se_group/metrics.jsonl \
  runs/E2_se_group_center/config.json runs/E2_se_group_center/summary.json \
  runs/E2_se_group_center/metrics.jsonl \
  runs/E2_se_group_residual025/config.json \
  runs/E2_se_group_residual025/summary.json \
  runs/E2_se_group_residual025/metrics.jsonl \
  runs/E2_se_group_residual025/log.txt \
  runs/E2_se_group_residual025/best.pt \
  runs/E2_se_group_residual025/paired_cases \
  runs/E0_E2_residual025_comparison.md \
  runs/E2_center_residual025_comparison.md \
  runs/E0_E2_residual025_attribute_metrics.json
sha256sum pa100k_e2_group_residual025_record.tar.gz
echo E2_RESIDUAL025_FINISHED_AND_PACKAGED
