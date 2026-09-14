#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <local-qwen3-4b-path> <run-name> [micro-batch: 4|2]" >&2
  exit 2
fi

MODEL_PATH="$1"
RUN_NAME="$2"
MICRO_BATCH="${3:-4}"
if [[ "$MICRO_BATCH" != "4" && "$MICRO_BATCH" != "2" ]]; then
  echo "micro-batch must be 4 or 2" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="artifacts/runs/phase_5/${RUN_NAME}"
cd "$ROOT_DIR"

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" --output-dir "artifacts/runs/phase_5/${RUN_NAME}_preflight" \
  --experiment-type smoke --max-steps 1 --micro-batch "$MICRO_BATCH" \
  --wandb-mode disabled --preflight-only

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" --output-dir "artifacts/runs/phase_5/${RUN_NAME}_one_step" \
  --experiment-type smoke --max-steps 1 --micro-batch "$MICRO_BATCH" --wandb-mode online

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --experiment-type smoke --max-steps 8 --micro-batch "$MICRO_BATCH" --wandb-mode online

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" \
  --experiment-type smoke --max-steps 16 --micro-batch "$MICRO_BATCH" --wandb-mode online \
  --resume-from-checkpoint "$ROOT_DIR/$OUTPUT_DIR/checkpoint-8"

uv run --extra sft python scripts/sft/validate_adapter.py \
  --model-path "$MODEL_PATH" --adapter-path "$OUTPUT_DIR/final_adapter"

echo "Smoke 8 -> 16 与 adapter reload 已完成。short run 需另行提供获批 eval-data。"
