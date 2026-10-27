#!/usr/bin/env bash
# 在已激活训练环境的服务器上运行：保留旧 E2，不覆盖任何已有结果。
set -euo pipefail
cd "$(dirname "$0")/.."

test -f PA-100K/train.csv
test -f PA-100K/val.csv
test -s runs/E0_baseline/best.pt
test -s runs/E2_se/best.pt
if test -e runs/E2_se_group; then
  echo "Refusing to overwrite runs/E2_se_group" >&2
  exit 1
fi

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
  --attention se_group \
  --se-reduction 16 \
  --seed 42 \
  --out ./runs/E2_se_group

python starter_code/compare_se.py \
  runs/E0_baseline runs/E2_se_group \
  --out runs/E0_E2_group_comparison.md

python starter_code/evaluate_attributes.py \
  --data-root ./PA-100K \
  --checkpoints runs/E0_baseline/best.pt runs/E2_se/best.pt \
                runs/E2_se_group/best.pt \
  --batch-size 64 --workers 4 \
  --out runs/E0_E2_group_attribute_metrics.json

tar -czf pa100k_e2_group_record.tar.gz \
  starter_code/model.py \
  starter_code/train.py \
  starter_code/infer.py \
  starter_code/compare_se.py \
  starter_code/evaluate_attributes.py \
  docs/se_group_architecture.svg \
  runs/E2_se_group/config.json \
  runs/E2_se_group/summary.json \
  runs/E2_se_group/metrics.jsonl \
  runs/E2_se_group/log.txt \
  runs/E2_se_group/best.pt \
  runs/E0_E2_group_comparison.md \
  runs/E0_E2_group_attribute_metrics.json
sha256sum pa100k_e2_group_record.tar.gz
echo E2_GROUP_FINISHED_AND_PACKAGED
