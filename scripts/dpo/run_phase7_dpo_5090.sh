#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the merged SFT directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new run directory under artifacts/runs/phase_7}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-smoke}"
CONFIG="${CONFIG:-configs/dpo/qwen3_4b_lora_phase7.yaml}"
DATA_RELEASE="${DATA_RELEASE:-data/dpo/released/phase7_v1}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-artifacts/reports/phase_7/preflight.json}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTHON="${PYTHON:-.venv/bin/python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ARGS=(
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --config "$CONFIG"
  --experiment-type "$EXPERIMENT_TYPE"
  --data-release "$DATA_RELEASE"
  --preflight-report "$PREFLIGHT_REPORT"
  --micro-batch "$MICRO_BATCH"
  --wandb-mode "$WANDB_MODE"
)

if [[ "$EXPERIMENT_TYPE" == "smoke" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON" scripts/dpo/run_dpo.py "${ARGS[@]}"