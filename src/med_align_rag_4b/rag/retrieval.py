"""加载经完整契约校验的 RAG 产物并执行 BGE-M3 检索。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.rag.index import (
    validate_embeddings,
    validate_faiss_index,
    validate_metadata,
)


@dataclass(frozen=True)
class RetrievalArtifacts:
    index: Any
    metadata_path: Path
    metadata_offsets: tuple[int, ...]
    model_id: str
    model_revision: str | None
    max_length: int


def load_artifact_manifest(path: str | Path, project_root: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2 or manifest.get("kind") != "rag_artifacts":
        raise ValueError("RAG artifact manifest 必须是 format_version 2")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.artifacts 必须是对象")
    paths: dict[str, Path] = {}
    for name in ("embeddings", "metadata", "index"):
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"manifest 缺少 {name}")
        candidate = resolve_project_path(project_root, descriptor.get("path", ""))
        if not candidate.is_file():
            raise FileNotFoundError(f"{name} 文件不存在")
        if candidate.stat().st_size != descriptor.get("bytes"):
            raise ValueError(f"{name} 文件大小不匹配")
        paths[name] = candidate
    return manifest, paths


def _metadata_offsets(path: Path, expected_count: int) -> tuple[int, ...]:
    offsets: list[int] = []
    with path.open("rb") as stream:
        while line := stream.readline():
            if line.strip():
                offsets.append(stream.tell() - len(line))
    if len(offsets) != expected_count:
        raise ValueError("metadata 与 FAISS row 数量不一致")
    return tuple(offsets)


def load_artifacts(manifest_path: str | Path, project_root: str | Path) -> RetrievalArtifacts:
    manifest, paths = load_artifact_manifest(manifest_path, project_root)
    embedding = manifest.get("embedding", {})
    count, dimension = embedding.get("count"), embedding.get("dimension")
    if embedding.get("dtype") != "float32" or embedding.get("normalized") is not True or manifest.get("similarity") != "inner_product":
        raise ValueError("embedding manifest 契约必须为 float32、normalized、inner_product")
    if not isinstance(count, int) or not isinstance(dimension, int):
        raise ValueError("embedding count/dimension 无效")
    vectors = validate_embeddings(paths["embeddings"], expected_count=count, expected_dimension=dimension)
    validate_metadata(paths["metadata"], vectors.shape[0])
    index = validate_faiss_index(paths["index"], count=count, dimension=dimension)
    return RetrievalArtifacts(index, paths["metadata"], _metadata_offsets(paths["metadata"], count), embedding["model_id"], embedding.get("model_revision"), embedding["max_length"])


def normalize_query_embedding(vectors: Any, expected_dimension: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (1, expected_dimension) or not np.isfinite(array).all():
        raise ValueError(f"query embedding 必须是有限的 (1, {expected_dimension}) float32 数组")
    norm = float(np.linalg.norm(array[0]))
    if norm == 0:
        raise ValueError("query embedding 不得为零向量")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def read_metadata_record(path: Path, offset: int, row_id: int) -> dict[str, Any]:
    with path.open("rb") as stream:
        stream.seek(offset)
        record = json.loads(stream.readline().decode("utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError(f"FAISS row {row_id} 对应 metadata 无效")
    return record


class RAGRetriever:
    def __init__(self, artifacts: RetrievalArtifacts, model: Any, model_path: str | Path | None = None) -> None:
        self.artifacts, self.model, self.model_path = artifacts, model, model_path

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query or top_k <= 0 or top_k > self.artifacts.index.ntotal:
            raise ValueError("query 必须非空，且 0 < top_k <= index.ntotal")
        encoded = self.model.encode([query], batch_size=1, max_length=self.artifacts.max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vector = normalize_query_embedding(encoded["dense_vecs"], self.artifacts.index.d)
        scores, ids = self.artifacts.index.search(vector, top_k)
        results = []
        for score, row_id in zip(scores[0], ids[0], strict=True):
            row = int(row_id)
            metadata = read_metadata_record(self.artifacts.metadata_path, self.artifacts.metadata_offsets[row], row)
            results.append({"index": row, "score": float(score), "text": metadata["text"], "metadata": metadata})
        return {"query": query, "similarity": "inner_product", "model_id": self.artifacts.model_id, "results": results}


def create_retriever(manifest_path: str | Path, project_root: str | Path, model_path: str | Path, *, device: str = "cpu", use_fp16: bool = False) -> RAGRetriever:
    from med_align_rag_4b.rag.embedding import load_embedding_model
    artifacts = load_artifacts(manifest_path, project_root)
    return RAGRetriever(artifacts, load_embedding_model(model_path, device=device, use_fp16=use_fp16), model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(create_retriever(args.manifest, args.project_root, args.model_path).retrieve(args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
