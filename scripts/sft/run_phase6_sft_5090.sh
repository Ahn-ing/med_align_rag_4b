#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <local-qwen3-4b-path> [micro-batch: 4|2] [run-name]" >&2
  exit 2
fi

MODEL_PATH="$1"
MICRO_BATCH="${2:-4}"
RUN_NAME="${3:-qwen3_4b_lora_2epoch}"
if [[ "$MICRO_BATCH" != "4" && "$MICRO_BATCH" != "2" ]]; then
  echo "micro-batch must be 4 or 2" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="artifacts/runs/phase_6/${RUN_NAME}"
cd "$ROOT_DIR"

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-type formal_train \
  --config configs/sft/qwen3_4b_lora_phase6.yaml \
  --epochs 2 \
  --micro-batch "$MICRO_BATCH" \
  --eval-data data/evaluation/validation_v2/cases.jsonl \
  --data-release data/sft/released/v2_plus_repair_r2 \
  --wandb-mode online

echo "Phase 6 两轮 SFT 已完成；请传回完整输出目录后运行只读验收器。"
