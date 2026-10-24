#!/usr/bin/env bash
# 不训练：复用 E0 逐图预测，补算九组 SE 与中心门控 SE 的同图预测。
set -euo pipefail
cd "$(dirname "$0")/.."

for required in \
  PA-100K/val.csv attributes.yaml \
  runs/E0_baseline/best.pt \
  runs/E2_se_group/best.pt \
  runs/E2_se_group_center/best.pt \
  runs/E0_E2_group_center_attribute_metrics.json; do
  if test ! -s "$required"; then
    echo "缺少必需文件：$required" >&2
    exit 1
  fi
done
test -d PA-100K/data || { echo "缺少图片目录：PA-100K/data" >&2; exit 1; }
if test -e runs/E0_E2_paired_errors || test -e pa100k_paired_errors_record.tar.gz; then
  echo "分析目录或结果包已存在；拒绝覆盖。" >&2
  exit 1
fi

python starter_code/paired_error_analysis.py \
  --data-root ./PA-100K \
  --attributes ./attributes.yaml \
  --e0-pred runs/E0_baseline/e0_val_predictions.pt \
  --e0-ckpt runs/E0_baseline/best.pt \
  --group-ckpt runs/E2_se_group/best.pt \
  --center-ckpt runs/E2_se_group_center/best.pt \
  --reference-metrics runs/E0_E2_group_center_attribute_metrics.json \
  --batch-size 64 --workers 4 --images-per-case 8 \
  --out runs/E0_E2_paired_errors

tar -czf pa100k_paired_errors_record.tar.gz \
  starter_code/paired_error_analysis.py \
  starter_code/dataset.py starter_code/model.py \
  scripts/run_paired_error_analysis.sh \
  docs/paired_error_analysis_readme.md \
  attributes.yaml \
  runs/E0_E2_group_center_attribute_metrics.json \
  runs/E0_E2_paired_errors
sha256sum pa100k_paired_errors_record.tar.gz
echo PAIRED_ERRORS_FINISHED_AND_PACKAGED
