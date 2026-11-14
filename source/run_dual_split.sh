#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 TASK_NAME DATA_ROOT MANIFEST_DIR -- [main.py arguments...]" >&2
  exit 2
fi

TASK_NAME=$1
DATA_ROOT=$2
MANIFEST_DIR=$3
shift 3
if [[ ${1:-} != "--" ]]; then
  echo "Expected -- before main.py arguments" >&2
  exit 2
fi
shift

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAFE_TASK=${TASK_NAME//\//_}
RUN_ROOT=${SAVE_DIR:-"$PROJECT_ROOT/runs/compare_${SAFE_TASK}_${TIMESTAMP}"}
TRAIN100_DIR="$RUN_ROOT/train100"
TRAIN90_DIR="$RUN_ROOT/train90"
BACKUP_DIR="$RUN_ROOT/code_backup"
SNAPSHOT_DIR="$RUN_ROOT/split_manifest_snapshot"
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-20260925}

show_split_update_warning() {
  local manifest_name
  local red=""
  local yellow=""
  local bold=""
  local reset=""

  manifest_name=$(basename "$MANIFEST_DIR")
  if [[ -t 1 ]]; then
    red=$'\033[1;31m'
    yellow=$'\033[1;33m'
    bold=$'\033[1m'
    reset=$'\033[0m'
  fi

  printf '%b\n' "${red}${bold}"
  printf '%s\n' \
    '################################################################################' \
    '#                                                                              #' \
    '#             !!!  训练前必须检查并更新固定划分清单  !!!                     #' \
    '#                                                                              #' \
    '#   如果训练数据有新增、删除或替换：请立即停止本次训练，先重新生成划分文件！   #' \
    '#   必须更新：train.jsonl、val.jsonl、meta.json，并确认最终 overlap=0。       #' \
    '#                                                                              #' \
    '################################################################################'
  printf '%b\n' "${reset}${yellow}${bold}当前任务：${TASK_NAME}${reset}"
  printf '数据目录：%s\n' "$DATA_ROOT"
  printf '划分目录：%s\n' "$MANIFEST_DIR"
  printf '%b\n' "${yellow}${bold}数据发生变化时，先运行下面的命令：${reset}"
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf '%q ' "$PYTHON_BIN" generate_group_split.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$MANIFEST_DIR" \
    --name "$manifest_name" \
    --val-ratio 0.1 \
    --seed "$SEED" \
    --force
  printf '\n'
  printf '%b\n' "${red}${bold}确认数据没有变化，或已完成上述更新后，才可继续训练。${reset}"
  printf '%s\n' '################################################################################'

  if [[ -t 1 ]]; then
    for seconds in 5 4 3 2 1; do
      printf '\r提醒停留中，%s 秒后继续启动训练……' "$seconds"
      sleep 1
    done
    printf '\r%*s\r' 70 ''
  fi
}

show_split_update_warning

if [[ ! -f "$MANIFEST_DIR/meta.json" || ! -f "$MANIFEST_DIR/train.jsonl" || ! -f "$MANIFEST_DIR/val.jsonl" ]]; then
  echo "Incomplete split manifest: $MANIFEST_DIR" >&2
  exit 2
fi

mkdir -p "$TRAIN100_DIR" "$TRAIN90_DIR" "$BACKUP_DIR" "$SNAPSHOT_DIR"
cp -a main.py ./*.sh ./*.py "$BACKUP_DIR/" 2>/dev/null || true
cp -a LineDet "$BACKUP_DIR/"
cp -a "$MANIFEST_DIR/." "$SNAPSHOT_DIR/"

{
  echo "comparison_started_at=$(date --iso-8601=seconds)"
  echo "task_name=$TASK_NAME"
  echo "data_root=$DATA_ROOT"
  echo "manifest_dir=$MANIFEST_DIR"
  echo "seed=$SEED"
  printf "main_args="
  printf "%q " "$@"
  printf "\n"
} > "$RUN_ROOT/training_info.txt"

run_version() {
  local version=$1
  local save_dir=$2
  shift 2
  echo "[$(date --iso-8601=seconds)] Starting $version"
  PYTHONPATH=. "$PYTHON_BIN" main.py \
    --save_dir "$save_dir" \
    --data_root "$DATA_ROOT" \
    --seed "$SEED" \
    --split_mode "$version" \
    "$@" \
    2>&1 | tee >(grep -vE '^(Training|Evaluating)\s*\|' > "$save_dir/training.log")
  echo "[$(date --iso-8601=seconds)] Finished $version"
}

run_version train100 "$TRAIN100_DIR" "$@"
run_version train90 "$TRAIN90_DIR" --split_manifest "$MANIFEST_DIR" "$@"

"$PYTHON_BIN" "$PROJECT_ROOT/summarize_comparison.py" \
  --run-root "$RUN_ROOT" \
  --task-name "$TASK_NAME"

echo "Both split versions completed: $RUN_ROOT"
