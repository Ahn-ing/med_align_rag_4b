#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${MODEL_PATH:?请通过 MODEL_PATH 指定本地 Qwen3-4B 目录}"
: "${BGE_M3_PATH:?请通过 BGE_M3_PATH 指定本地 BGE-M3 目录}"
MERGED_SFT_PATH="${MERGED_SFT_PATH:-artifacts/runs/phase_7/merge_sft}"
DEVICE="${DEVICE:-cuda}"
RAG_DEVICE="${RAG_DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/evaluations/phase_8_rag/formal_v4}"

for path in "$MODEL_PATH" "$MERGED_SFT_PATH" "$BGE_M3_PATH"; do
  if [[ ! -d "$path" ]]; then
    echo "目录不存在: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"

run_arm() {
  local arm="$1"
  shift
  echo "[Phase 8 RAG] 开始 $arm ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  uv run python scripts/evaluation/run_rag_ablation.py generate \
    --config configs/evaluation/phase8_rag.yaml \
    --arm "$arm" \
    --device "$DEVICE" \
    --resume \
    --output "$OUTPUT_ROOT/$arm.jsonl" \
    "$@"
}

run_arm R0 --base-model-path "$MODEL_PATH"
run_arm R1 --base-model-path "$MODEL_PATH" \
  --rag-embedding-model-path "$BGE_M3_PATH" \
  --rag-device "$RAG_DEVICE"
run_arm R2 --base-model-path "$MODEL_PATH"
run_arm R3 --base-model-path "$MODEL_PATH" \
  --rag-embedding-model-path "$BGE_M3_PATH" \
  --rag-device "$RAG_DEVICE"
run_arm R4 --merged-sft-path "$MERGED_SFT_PATH"
run_arm R5 --merged-sft-path "$MERGED_SFT_PATH" \
  --rag-embedding-model-path "$BGE_M3_PATH" \
  --rag-device "$RAG_DEVICE"

echo "[Phase 8 RAG] R0-R5 完成，产物位于 $OUTPUT_ROOT"
ls -lh "$OUTPUT_ROOT"