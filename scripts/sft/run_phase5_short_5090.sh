#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <local-qwen3-4b-path> [micro-batch: 4|2] [run-name]" >&2
  exit 2
fi

MODEL_PATH="$1"
MICRO_BATCH="${2:-4}"
RUN_NAME="${3:-short_run_qwen3_4b_lora}"
if [[ "$MICRO_BATCH" != "4" && "$MICRO_BATCH" != "2" ]]; then
  echo "micro-batch must be 4 or 2" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="artifacts/runs/phase_5/${RUN_NAME}"
cd "$ROOT_DIR"

uv run --extra sft python scripts/sft/run_sft.py \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-type short_run \
  --max-steps 50 \
  --micro-batch "$MICRO_BATCH" \
  --eval-data data/evaluation/phase5_loss_v1.jsonl \
  --wandb-mode online

uv run --extra sft python scripts/sft/validate_adapter.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OUTPUT_DIR/final_adapter" \
  --report-path "$OUTPUT_DIR/adapter_validation.json" \
  --run-fixtures

echo "Phase 5 short run 与工程 fixture 验收已完成；请下载完整输出目录进行本地最终审计。"
