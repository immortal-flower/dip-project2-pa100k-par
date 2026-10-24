#!/usr/bin/env bash
# 从独立解压目录安装补丁；先验证旧代码版本并备份，避免覆盖他人修改。
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd -P)"
target_root="${1:-$HOME/pa100k_b}"
target_root="$(cd "$target_root" && pwd -P)"
cd "$target_root"

for relative in starter_code/model.py starter_code/train.py \
                starter_code/compare_se.py; do
  test -s "$relative" || {
    echo "目标目录缺少 $relative" >&2; exit 1;
  }
done

declare -A expected=(
  [starter_code/model.py]=53a35796882e5f04401df9fa9c456af6cec3c7f1c0b8cd0e3a657ce798f3aebd
  [starter_code/train.py]=baefa3f310406d72edc69dedbe37870ddeab5e59fa4e5145bcda22b102a6dccb
  [starter_code/compare_se.py]=ca8f5ccf107ed70f86f54798cc64b9b31b3ab19b6323d2c558de4cad601de575
)
for relative in "${!expected[@]}"; do
  actual="$(sha256sum "$relative")"
  actual="${actual%% *}"
  if test "$actual" != "${expected[$relative]}"; then
    echo "现有 $relative 与已知 E2 版本不同；为保护服务器修改，停止安装。" >&2
    exit 1
  fi
done

for relative in \
  starter_code/evaluate_residual_cases.py \
  scripts/run_e2_group_residual025.sh \
  docs/e2_group_residual025_experiment.md \
  reference/paired_predictions.jsonl; do
  if test -e "$relative"; then
    echo "目标文件已存在：$relative；拒绝覆盖。" >&2
    exit 1
  fi
  test -s "$package_root/$relative" || {
    echo "补丁包缺少 $relative" >&2; exit 1;
  }
done

backup="se_residual025_backup_$(date +%Y%m%d_%H%M%S)"
if test -e "$backup"; then
  echo "备份目录已存在：$backup；请稍后重试。" >&2
  exit 1
fi
mkdir -p "$backup/starter_code" reference
for relative in starter_code/model.py starter_code/train.py \
                starter_code/compare_se.py; do
  cp -p "$relative" "$backup/$relative"
done

for relative in \
  starter_code/model.py starter_code/train.py \
  starter_code/compare_se.py starter_code/evaluate_residual_cases.py \
  scripts/run_e2_group_residual025.sh \
  docs/e2_group_residual025_experiment.md \
  reference/paired_predictions.jsonl; do
  cp -p "$package_root/$relative" "$relative"
done
echo "安装完成；原有三个代码文件已备份到 $target_root/$backup"
echo "没有改动损失函数、数据加载器、增强或任何旧实验结果。"
