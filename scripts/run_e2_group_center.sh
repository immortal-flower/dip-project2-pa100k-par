#!/usr/bin/env bash
# 独立实验：在 E2_se_group 基础上只将 SE 权重由 sigmoid(z) 改为 2*sigmoid(z)。
# 使用与 E0/E2 相同的训练设置；保留全部旧结果。
set -euo pipefail
cd "$(dirname "$0")/.."

for required in \
  PA-100K/train.csv PA-100K/val.csv \
  runs/E0_baseline/best.pt runs/E0_baseline/config.json \
  runs/E2_se/best.pt runs/E2_se/config.json \
  runs/E2_se_group/best.pt runs/E2_se_group/config.json; do
  if test ! -s "$required"; then
    echo "缺少必需文件：$required" >&2
    exit 1
  fi
done
test -d PA-100K/data || { echo "缺少图片目录：PA-100K/data" >&2; exit 1; }
if test -e runs/E2_se_group_center || test -e pa100k_e2_group_center_record.tar.gz; then
  echo "新实验目录或压缩包已存在；为保护已有结果，拒绝覆盖。" >&2
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path("starter_code").resolve()))
from model import PAResNet

assert torch.cuda.is_available(), "CUDA 不可用，请先激活 pa100k_e0 环境并检查 GPU"
print("CUDA GPU:", torch.cuda.get_device_name(0), flush=True)
dirs = ("E0_baseline", "E2_se", "E2_se_group")
configs = [json.loads((Path("runs") / name / "config.json").read_text(encoding="utf-8"))
           for name in dirs]
reference = {k: v for k, v in configs[0]["args"].items()
             if k not in ("attention", "out")}
for name, config in zip(dirs, configs):
    current = {k: v for k, v in config["args"].items()
               if k not in ("attention", "out")}
    assert current == reference, f"{name} 与 E0 的训练参数不一致"
    assert config["data_format"] == "csv", f"{name} 未使用 CSV 数据"
assert [x["args"]["attention"] for x in configs] == ["none", "se", "se_group"]
assert not reference["pos_weight"], "E0 启用了 pos_weight，需重新核对本脚本设置"
checkpoint = torch.load("runs/E2_se_group/best.pt", map_location="cpu", weights_only=False)
model_args = checkpoint["args"]
kwargs = dict(hidden=model_args["hidden"], drop=model_args["drop"],
              se_reduction=model_args["se_reduction"], pretrained=False)
old = PAResNet(checkpoint["spec"], attention="se_group", **kwargs)
new = PAResNet(checkpoint["spec"], attention="se_group_center", **kwargs)
old.load_state_dict(checkpoint["model"], strict=True)
new.load_state_dict(checkpoint["model"], strict=True)
assert list(old.state_dict()) == list(new.state_dict()), "新旧 SE 参数结构不一致"
probe = torch.randn(2, 512)
for name in old.group_attention:
    assert torch.allclose(new.group_attention[name](probe),
                          2 * old.group_attention[name](probe), atol=1e-6), name
print("预检通过：分组 SE 参数完全兼容，唯一计算差异为通道系数乘 2。", flush=True)
print("预检通过：旧实验除注意力与输出目录外，训练配置相同。", flush=True)
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
  --attention se_group_center \
  --se-reduction 16 \
  --seed 42 \
  --out ./runs/E2_se_group_center

python starter_code/compare_se.py \
  runs/E0_baseline runs/E2_se_group_center \
  --out runs/E0_E2_group_center_comparison.md
python starter_code/compare_se.py \
  runs/E2_se_group runs/E2_se_group_center \
  --out runs/E2_group_center_comparison.md

python starter_code/evaluate_attributes.py \
  --data-root ./PA-100K \
  --checkpoints runs/E0_baseline/best.pt runs/E2_se/best.pt \
                runs/E2_se_group/best.pt runs/E2_se_group_center/best.pt \
  --batch-size 64 --workers 4 \
  --out runs/E0_E2_group_center_attribute_metrics.json

tar -czf pa100k_e2_group_center_record.tar.gz \
  attributes.yaml requirements.txt \
  starter_code/model.py starter_code/train.py starter_code/infer.py \
  starter_code/dataset.py starter_code/evaluate.py starter_code/losses.py \
  starter_code/compare_se.py starter_code/evaluate_attributes.py \
  scripts/run_e2_group_center.sh docs/e2_group_center_experiment.md \
  runs/E0_baseline/config.json runs/E0_baseline/summary.json \
  runs/E0_baseline/metrics.jsonl \
  runs/E2_se/config.json runs/E2_se/summary.json \
  runs/E2_se/metrics.jsonl \
  runs/E2_se_group/config.json runs/E2_se_group/summary.json \
  runs/E2_se_group/metrics.jsonl \
  runs/E2_se_group_center/config.json \
  runs/E2_se_group_center/summary.json \
  runs/E2_se_group_center/metrics.jsonl \
  runs/E2_se_group_center/log.txt \
  runs/E2_se_group_center/best.pt \
  runs/E0_E2_group_center_comparison.md \
  runs/E2_group_center_comparison.md \
  runs/E0_E2_group_center_attribute_metrics.json
sha256sum pa100k_e2_group_center_record.tar.gz
echo E2_GROUP_CENTER_FINISHED_AND_PACKAGED
