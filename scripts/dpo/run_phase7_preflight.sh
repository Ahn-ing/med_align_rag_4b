#!/usr/bin/env bash
set -euo pipefail

: "${MERGED_MODEL:?set MERGED_MODEL to the merged SFT directory}"
RELEASE_DIR="${RELEASE_DIR:-data/dpo/released/phase7_v1}"
OUTPUT="${OUTPUT:-artifacts/reports/phase_7/preflight.json}"
SAMPLES="${SAMPLES:-4}"
PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

"$PYTHON" scripts/dpo/preflight_phase7.py \
  --merged-model "$MERGED_MODEL" \
  --release-dir "$RELEASE_DIR" \
  --output "$OUTPUT" \
  --samples "$SAMPLES" \
  --device "$DEVICE"