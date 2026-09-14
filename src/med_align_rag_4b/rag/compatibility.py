"""Phase 2 新旧 RAG 检索兼容性验收。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import portable_relative_path
from med_align_rag_4b.common.utils import utc_now_iso
from med_align_rag_4b.rag.embedding import load_embedding_model
from med_align_rag_4b.rag.retrieval import (
    RAGRetriever,
    load_artifacts,
    normalize_query_embedding,
)

TOP_K_VALUES = (1, 3, 5, 10)
SCORE_TOLERANCE = 1e-5
TIE_TOLERANCE = 1e-7
MODEL_FINGERPRINT_FILES = (
    "config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "sparse_linear.pt",
    "colbert_linear.pt",
)


@dataclass(frozen=True)
class CompatibilityQuery:
    query_id: str
    query: str
    source_row_id: int
    source_chunk_id: str
    length_bucket: str


def load_queries(path: str | Path, expected_count: int = 32) -> list[CompatibilityQuery]:
    queries: list[CompatibilityQuery] = []
    for value in iter_jsonl(path):
        try:
            query = CompatibilityQuery(**value)
        except TypeError as exc:
            raise ValueError("query 记录字段不符合 Phase 2 契约") from exc
        if not query.query.strip() or query.source_row_id < 0:
            raise ValueError("query 文本必须非空且 source_row_id 非负")
        queries.append(query)
    if len(queries) != expected_count:
        raise ValueError(f"query 集必须包含 {expected_count} 条记录")
    if len({item.query_id for item in queries}) != len(queries):
        raise ValueError("query_id 必须唯一")
    if len({item.query for item in queries}) != len(queries):
        raise ValueError("query 文本必须唯一")
    return queries


def fingerprint_model(model_path: str | Path) -> dict[str, Any]:
    """用模型权重、配置与 tokenizer 的文件哈希锁定本地模型。"""
    root = Path(model_path)
    files = []
    for relative in MODEL_FINGERPRINT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"模型指纹文件缺失: {relative}")
        files.append({"path": str(PurePosixPath(relative)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"upstream_revision": "unknown", "files": files, "fingerprint_sha256": sha256_record(files)}


def _load_old_retrieval(old_project_root: Path) -> Any:
    module_path = old_project_root / "rag" / "rag_retriever.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"旧 retriever 不存在: {module_path}")
    old_root = str(old_project_root.resolve())
    if old_root not in sys.path:
        sys.path.insert(0, old_root)
    spec = importlib.util.spec_from_file_location("phase2_old_rag_retriever", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载旧 retriever")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare_ranked_results(old_results: list[dict[str, Any]], new_results: list[dict[str, Any]], *, score_tolerance: float = SCORE_TOLERANCE, tie_tolerance: float = TIE_TOLERANCE) -> dict[str, Any]:
    if len(old_results) != len(new_results):
        raise AssertionError("新旧结果数量不同")
    maximum_error = 0.0
    tie_risk = False
    for position, (old, new) in enumerate(zip(old_results, new_results, strict=True)):
        if old["index"] != new["index"]:
            raise AssertionError(f"rank {position + 1} row id 不一致")
        if old["metadata"].get("chunk_id") != new["metadata"].get("chunk_id"):
            raise AssertionError(f"rank {position + 1} chunk id 不一致")
        if old["text"] != new["text"] or old["metadata"] != new["metadata"]:
            raise AssertionError(f"rank {position + 1} metadata/text 不一致")
        error = abs(float(old["score"]) - float(new["score"]))
        maximum_error = max(maximum_error, error)
        if error > score_tolerance:
            raise AssertionError(f"rank {position + 1} score 误差 {error} 超限")
    for results in (old_results, new_results):
        tie_risk = tie_risk or any(abs(float(left["score"]) - float(right["score"])) <= tie_tolerance for left, right in zip(results, results[1:], strict=False))
    return {"max_score_abs_error": maximum_error, "tie_risk": tie_risk}


def _encode(model: Any, texts: list[str], max_length: int) -> np.ndarray:
    encoded = model.encode(texts, batch_size=len(texts), max_length=max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return np.asarray(encoded["dense_vecs"], dtype=np.float32)


def _normalize_rows(vectors: np.ndarray, normalize_one: Any) -> np.ndarray:
    return np.concatenate([normalize_one(vectors[row : row + 1], vectors.shape[1]) for row in range(vectors.shape[0])])


def _result(index: Any, metadata_path: Path, offsets: tuple[int, ...], vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    scores, row_ids = index.search(vector, top_k)
    results = []
    with metadata_path.open("rb") as stream:
        for score, row_id in zip(scores[0], row_ids[0], strict=True):
            row = int(row_id)
            stream.seek(offsets[row])
            metadata = json.loads(stream.readline().decode("utf-8"))
            results.append({"index": row, "score": float(score), "text": metadata["text"], "metadata": metadata})
    return results


def _offsets(path: Path) -> tuple[int, ...]:
    result = []
    with path.open("rb") as stream:
        while line := stream.readline():
            if line.strip():
                result.append(stream.tell() - len(line))
    return tuple(result)


def validate_entry_gate(project_root: Path, old_project_root: Path, artifact_manifest: Path) -> dict[str, Any]:
    plans = (project_root / "PLANS.md").read_text(encoding="utf-8")
    if "### Phase 1：RAG 迁移\n\n状态：**COMPLETED**" not in plans:
        raise RuntimeError("Phase 1 未完成")
    migration = json.loads((project_root / "manifests/phase_1/migration.json").read_text(encoding="utf-8"))
    if migration.get("review_status") != "approved_and_verified":
        raise RuntimeError("Phase 1 migration manifest 未通过")
    manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    for entry in migration["entries"]:
        source = old_project_root / entry["source_path"]
        destination = project_root / entry["destination_path"]
        if sha256_file(source) != entry["source_sha256"] or sha256_file(destination) != entry["destination_sha256"]:
            raise RuntimeError(f"Phase 1 迁移哈希不一致: {entry['destination_path']}")
    return manifest


def _package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "FlagEmbedding", "faiss-cpu", "torch", "transformers"):
        try:
            result[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


def write_json_atomic(path: str | Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"输出已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_compatibility(*, project_root: Path, old_project_root: Path, artifact_manifest: Path, query_path: Path, output_path: Path, embedding_model_path: Path) -> dict[str, Any]:
    validate_entry_gate(project_root, old_project_root, artifact_manifest)
    queries = load_queries(query_path)
    old_module = _load_old_retrieval(old_project_root)
    old_dir = old_project_root / "data/rag/huatuo_encyclopedia/embeddings"
    old_artifacts = old_module.load_artifacts(old_dir)
    new_artifacts = load_artifacts(artifact_manifest, project_root)
    old_metadata = old_dir / "metadata.jsonl"
    if sha256_file(old_metadata) != sha256_file(new_artifacts.metadata_path):
        raise RuntimeError("新旧 metadata 字节不一致")
    if len(_offsets(old_metadata)) != len(new_artifacts.metadata_offsets):
        raise RuntimeError("新旧 metadata offset 语义不一致")
    model_fingerprint = fingerprint_model(embedding_model_path)
    model = load_embedding_model(embedding_model_path, device="cpu", use_fp16=False)
    texts = [item.query for item in queries]
    raw_old = _encode(model, texts, 1024)
    raw_new = _encode(model, texts, 1024)
    old_vectors = _normalize_rows(raw_old, old_module.normalize_query_embedding)
    new_vectors = _normalize_rows(raw_new, normalize_query_embedding)
    vector_error = float(np.max(np.abs(old_vectors - new_vectors)))
    per_query = []
    max_score_error = 0.0
    ties = 0
    old_offsets = _offsets(old_metadata)
    for row, query in enumerate(queries):
        checks = []
        for top_k in TOP_K_VALUES:
            old_results = _result(old_artifacts.index, old_metadata, old_offsets, old_vectors[row : row + 1], top_k)
            new_results = _result(new_artifacts.index, new_artifacts.metadata_path, new_artifacts.metadata_offsets, new_vectors[row : row + 1], top_k)
            comparison = compare_ranked_results(old_results, new_results)
            max_score_error = max(max_score_error, comparison["max_score_abs_error"])
            ties += int(comparison["tie_risk"])
            checks.append({"top_k": top_k, **comparison, "row_ids": [item["index"] for item in new_results], "chunk_ids": [item["metadata"]["chunk_id"] for item in new_results]})
        per_query.append({"query_id": query.query_id, "checks": checks})
    old_public = old_module.RAGRetriever(old_artifacts, model, embedding_model_path)
    new_public = RAGRetriever(new_artifacts, model, embedding_model_path)
    smoke = []
    for query in queries[:3]:
        comparison = compare_ranked_results(old_public.retrieve(query.query, 5)["results"], new_public.retrieve(query.query, 5)["results"])
        smoke.append({"query_id": query.query_id, **comparison})
    report = {
        "format_version": 1,
        "kind": "rag_compatibility",
        "created_at_utc": utc_now_iso(),
        "status": "passed",
        "inputs": {
            "artifact_manifest": portable_relative_path(artifact_manifest, project_root),
            "artifact_manifest_sha256": sha256_file(artifact_manifest),
            "query_set": portable_relative_path(query_path, project_root),
            "query_set_sha256": sha256_file(query_path),
            "old_retriever_sha256": sha256_file(old_project_root / "rag/rag_retriever.py"),
            "new_retriever": "src/med_align_rag_4b/rag/retrieval.py",
            "new_retriever_sha256": sha256_file(project_root / "src/med_align_rag_4b/rag/retrieval.py"),
            "compatibility_runner": "src/med_align_rag_4b/rag/compatibility.py",
            "compatibility_runner_sha256": sha256_file(project_root / "src/med_align_rag_4b/rag/compatibility.py"),
        },
        "model": model_fingerprint,
        "environment": {"device": "cpu", "use_fp16": False, "max_length": 1024, "dense_only": True, "versions": _package_versions()},
        "criteria": {"query_count": 32, "top_k": list(TOP_K_VALUES), "score_abs_tolerance": SCORE_TOLERANCE, "tie_tolerance": TIE_TOLERANCE},
        "results": {"static_artifacts": "passed", "query_vector_max_abs_error": vector_error, "max_score_abs_error": max_score_error, "tie_checks_flagged": ties, "queries": per_query, "public_api_smoke": smoke},
        "risks": ["BAAI/bge-m3 upstream revision unknown; this run is locked by local file fingerprint."],
    }
    write_json_atomic(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--old-project-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_compatibility(project_root=args.project_root.resolve(), old_project_root=args.old_project_root.resolve(), artifact_manifest=args.artifact_manifest.resolve(), query_path=args.query_set.resolve(), output_path=args.output.resolve(), embedding_model_path=args.embedding_model_path.resolve())
    summary = {"status": report["status"], **report["results"]}
    summary["queries"] = len(report["results"]["queries"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
