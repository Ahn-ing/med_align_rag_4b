#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL to the local Qwen3-4B directory}"
: "${ADAPTER:?set ADAPTER to the Phase 6 final adapter directory}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/phase_7/merge_sft}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

"$PYTHON" scripts/dpo/merge_phase6_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"