"""校验 embedding/metadata 并原子构建 FAISS Inner Product 索引。"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from med_align_rag_4b.common.jsonl import iter_jsonl


def load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("请安装 rag 可选依赖 faiss-cpu") from exc
    return faiss


def validate_embeddings(path: str | Path, *, expected_count: int | None = None, expected_dimension: int | None = None, batch_size: int = 8192) -> np.ndarray:
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise ValueError("embedding 必须是二维 float32 数组")
    if expected_count is not None and vectors.shape[0] != expected_count:
        raise ValueError("embedding count 与 manifest 不一致")
    if expected_dimension is not None and vectors.shape[1] != expected_dimension:
        raise ValueError("embedding dimension 与 manifest 不一致")
    for start in range(0, vectors.shape[0], batch_size):
        batch = np.asarray(vectors[start : start + batch_size])
        if not np.isfinite(batch).all() or not np.all(np.abs(np.linalg.norm(batch, axis=1) - 1) <= 1e-3):
            raise ValueError(f"row {start} 附近存在非有限或未 L2 归一化 embedding")
    return vectors


def validate_metadata(path: str | Path, expected_count: int) -> int:
    seen: set[str] = set()
    count = 0
    for record in iter_jsonl(path):
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ValueError("metadata chunk_id 必须非空且唯一")
        seen.add(chunk_id)
        count += 1
    if count != expected_count:
        raise ValueError(f"metadata/vector 数量不一致: {count} != {expected_count}")
    return count


def build_index(embedding_path: str | Path, metadata_path: str | Path, output_path: str | Path, *, batch_size: int = 8192, overwrite: bool = False) -> tuple[int, int]:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"索引已存在: {output}")
    vectors = validate_embeddings(embedding_path, batch_size=batch_size)
    validate_metadata(metadata_path, vectors.shape[0])
    faiss = load_faiss()
    index = faiss.IndexFlatIP(vectors.shape[1])
    for start in range(0, vectors.shape[0], batch_size):
        index.add(np.ascontiguousarray(vectors[start : start + batch_size]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        faiss.write_index(index, str(temporary))
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return int(index.ntotal), int(index.d)


def validate_faiss_index(path: str | Path, *, count: int, dimension: int) -> Any:
    faiss = load_faiss()
    index = faiss.read_index(str(path))
    if type(index).__name__ != "IndexFlatIP" or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index 必须是 IndexFlatIP/Inner Product")
    if index.ntotal != count or index.d != dimension:
        raise ValueError("FAISS index 的 ntotal/dimension 不一致")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_path", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(build_index(args.embedding_path, args.metadata_path, args.output_path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
