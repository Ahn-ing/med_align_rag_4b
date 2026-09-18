#!/usr/bin/env bash
set -euo pipefail

: "${MERGED_MODEL:?set MERGED_MODEL to the merged SFT directory}"
: "${RUN_DIR:?set RUN_DIR to the completed formal DPO run directory}"
OUTPUT="${OUTPUT:-artifacts/reports/phase_7/dpo_formal_evaluation.json}"
CONFIG="${CONFIG:-configs/dpo/qwen3_4b_lora_phase7.yaml}"
RELEASE_DIR="${RELEASE_DIR:-data/dpo/released/phase7_v1}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

"$PYTHON" scripts/dpo/evaluate_phase7.py \
  --merged-model "$MERGED_MODEL" \
  --run-dir "$RUN_DIR" \
  --release-dir "$RELEASE_DIR" \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  "$@"
