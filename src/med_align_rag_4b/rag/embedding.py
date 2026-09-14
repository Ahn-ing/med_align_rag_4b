"""生成 float32、L2 归一化的 BGE-M3 dense embedding。"""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic


def normalize_embeddings(vectors: Any, first_row: int = 0) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not np.isfinite(array).all():
        raise ValueError(f"row {first_row} 附近的 embedding 无效")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"row {first_row} 附近存在零向量")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def load_embedding_model(model_path: str | Path, *, device: str, use_fp16: bool) -> Any:
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(str(model_path), device=device, use_fp16=use_fp16)


def _validated_records(path: str | Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for record in iter_jsonl(path):
        chunk_id, text = record.get("chunk_id"), record.get("text")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ValueError("chunk_id 必须非空且唯一")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{chunk_id} 缺少非空 text")
        seen.add(chunk_id)
        yield record


def generate_embeddings(input_path: str | Path, output_dir: str | Path, model: Any, *, batch_size: int = 32, max_length: int = 1024, overwrite: bool = False) -> tuple[int, int]:
    """分批编码；三个产物全部成功后才发布。"""
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size 和 max_length 必须为正数")
    count = sum(1 for _ in _validated_records(input_path))
    if not count:
        raise ValueError("chunk 输入为空")
    destination = Path(output_dir)
    outputs = {"embeddings": destination / "embeddings.npy", "metadata": destination / "metadata.jsonl"}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("embedding 输出已存在")
    destination.mkdir(parents=True, exist_ok=True)
    marker = uuid.uuid4().hex
    temporary_npy = destination / f".embeddings.{marker}.npy"
    temporary_metadata = destination / f".metadata.{marker}.jsonl"
    try:
        write_jsonl_atomic(temporary_metadata, _validated_records(input_path))
        matrix = None
        batch: list[dict[str, Any]] = []
        written = 0
        for record in _validated_records(input_path):
            batch.append(record)
            if len(batch) < batch_size and written + len(batch) < count:
                continue
            encoded = model.encode([item["text"] for item in batch], batch_size=len(batch), max_length=max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            vectors = normalize_embeddings(encoded["dense_vecs"], written)
            if matrix is None:
                matrix = np.lib.format.open_memmap(temporary_npy, mode="w+", dtype=np.float32, shape=(count, vectors.shape[1]))
            matrix[written : written + len(batch)] = vectors
            written += len(batch)
            batch = []
        if matrix is None:
            raise RuntimeError("未生成 embedding")
        matrix.flush()
        shape = (count, int(matrix.shape[1]))
        del matrix
        os.replace(temporary_npy, outputs["embeddings"])
        os.replace(temporary_metadata, outputs["metadata"])
    except Exception:
        temporary_npy.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise
    return shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model = load_embedding_model(args.model_path, device=args.device, use_fp16=args.use_fp16)
    print(generate_embeddings(args.input, args.output_dir, model, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
