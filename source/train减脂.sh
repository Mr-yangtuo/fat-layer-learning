#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export LINEDET_UNIFY_MODE="unify"
export LINEDET_UNIFY_SIZE="516,594"

exec "$PROJECT_ROOT/run_dual_split.sh" \
  "减脂" \
  "/home/zlg/0.减脂数据集/无垫/" \
  "$PROJECT_ROOT/splits/grouped_v2/减脂无垫_20261009" \
  -- \
  --backbone "mobilenet_v3_large" \
  --num_lines 2 \
  --img_size 256 \
  --epochs 100 \
  --batch_size 48 \
  --lr 1e-3 \
  --num_workers "${NUM_WORKERS:-16}"
