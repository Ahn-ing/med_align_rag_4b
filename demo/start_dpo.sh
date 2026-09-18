#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec uv run python demo/cli_chat.py \
  --model-path artifacts/runs/phase_7/merge_sft \
  --adapter-path artifacts/runs/phase_7/dpo_formal_v2/final_adapter \
  "$@"
