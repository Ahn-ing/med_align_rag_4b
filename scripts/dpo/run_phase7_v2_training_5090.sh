#!/usr/bin/env bash
set -euo pipefail

: "${MERGED_MODEL:?set MERGED_MODEL to the merged SFT directory}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/phase_7/dpo_formal_v2}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-artifacts/reports/phase_7/preflight_v2.json}"
CONFIG="${CONFIG:-configs/dpo/qwen3_4b_lora_phase7.yaml}"
RELEASE_DIR="${RELEASE_DIR:-data/dpo/released/phase7_v2}"
SAMPLES="${SAMPLES:-4}"
PREFLIGHT_WANDB_MODE="${PREFLIGHT_WANDB_MODE:-offline}"
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-online}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

test ! -e "$OUTPUT_DIR"
test ! -e "$PREFLIGHT_REPORT"

WANDB_MODE="$PREFLIGHT_WANDB_MODE" "$PYTHON" scripts/dpo/preflight_phase7.py \
  --merged-model "$MERGED_MODEL" \
  --release-dir "$RELEASE_DIR" \
  --config "$CONFIG" \
  --output "$PREFLIGHT_REPORT" \
  --samples "$SAMPLES" \
  --device "$DEVICE"

WANDB_MODE="$TRAIN_WANDB_MODE" "$PYTHON" scripts/dpo/run_dpo.py \
  --model-path "$MERGED_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --config "$CONFIG" \
  --experiment-type formal_dpo \
  --data-release "$RELEASE_DIR" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --micro-batch 1 \
  --wandb-mode "$TRAIN_WANDB_MODE"
