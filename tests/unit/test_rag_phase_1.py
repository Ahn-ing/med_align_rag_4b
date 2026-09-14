import json

import numpy as np
import pytest

from med_align_rag_4b.rag.chunking import ChunkGenerator, inspect_input
from med_align_rag_4b.rag.cleaning import clean_dataset
from med_align_rag_4b.rag.embedding import normalize_embeddings
from med_align_rag_4b.rag.generation import GenerationConfig, QwenGenerator
from med_align_rag_4b.rag.index import (
    validate_embeddings,
    validate_faiss_index,
    validate_metadata,
)
from med_align_rag_4b.rag.pipeline import RAGPipeline
from med_align_rag_4b.rag.prompting import build_prompt
from med_align_rag_4b.rag.retrieval import (
    load_artifact_manifest,
    normalize_query_embedding,
)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False, truncation=False):
        return list(text)

    def decode(self, ids, **kwargs):
        return "".join(ids)


def test_cleaning_handles_invalid_empty_duplicate_and_overwrite(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text('{"question":" 问题 ","answer":" 答案 "}\n坏 JSON\n{"question":"问题","answer":"答案"}\n{"question":"","answer":"答案"}\n', encoding="utf-8")
    output = tmp_path / "cleaned.jsonl"
    stats = clean_dataset(source, output)
    assert (stats.cleaned_records, stats.invalid_json, stats.duplicate_qa, stats.empty_questions) == (1, 1, 1, 1)
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "huatuo_000001"
    with pytest.raises(FileExistsError):
        clean_dataset(source, output)


def test_chunk_ids_order_limit_and_cleaned_contract(tmp_path):
    source = tmp_path / "cleaned.jsonl"
    source.write_text('{"id":"a","question":"问","answer":"甲甲。乙乙。丙丙。"}\n', encoding="utf-8")
    assert inspect_input(source) == 1
    generator = ChunkGenerator(CharacterTokenizer(), chunk_size=12, chunk_overlap=3)
    chunks = generator.generate_for_record(json.loads(source.read_text(encoding="utf-8")), 0)
    assert [item["chunk_id"] for item in chunks] == [f"a_chunk_{i:04d}" for i in range(len(chunks))]
    assert all(item["token_length"] <= 12 for item in chunks)


def test_embedding_and_query_normalization_are_float32():
    vectors = normalize_embeddings([[3.0, 4.0], [0.0, 2.0]])
    query = normalize_query_embedding([[0.0, 5.0]], 2)
    assert vectors.dtype == query.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1])
    assert np.linalg.norm(query[0]) == pytest.approx(1)


def test_artifact_validators_reject_count_dtype_and_norm(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"chunk_id":"a","text":"A"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="数量"):
        validate_metadata(metadata, 2)
    bad_dtype = tmp_path / "dtype.npy"
    np.save(bad_dtype, np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        validate_embeddings(bad_dtype)
    bad_norm = tmp_path / "norm.npy"
    np.save(bad_norm, np.array([[2, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="L2"):
        validate_embeddings(bad_norm)


def test_manifest_rejects_absolute_path_and_wrong_hash(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")
    descriptor = {"path": str(artifact), "sha256": "0" * 64, "bytes": 7}
    manifest = {"format_version": 2, "kind": "rag_artifacts", "artifacts": {name: descriptor for name in ("embeddings", "metadata", "index")}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="相对路径"):
        load_artifact_manifest(path, tmp_path)
    descriptor["path"] = "artifact.bin"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_artifact_manifest(path, tmp_path)


def test_faiss_validator_rejects_metric_and_dimension(tmp_path):
    faiss = pytest.importorskip("faiss")
    index_path = tmp_path / "index.faiss"
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1, 0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))
    with pytest.raises(ValueError, match="IndexFlatIP"):
        validate_faiss_index(index_path, count=1, dimension=2)
    ip_index = faiss.IndexFlatIP(2)
    ip_index.add(np.array([[1, 0]], dtype=np.float32))
    faiss.write_index(ip_index, str(index_path))
    with pytest.raises(ValueError, match="ntotal/dimension"):
        validate_faiss_index(index_path, count=1, dimension=3)


def test_prompt_boundaries_and_limit():
    result = {"score": 0.9, "text": "内容" * 100, "metadata": {"chunk_id": "c1", "question": "来源"}}
    payload = build_prompt("用户问题", [result], max_context_chars=150)
    assert len(payload.context) <= 150
    assert "<medical_knowledge_context>" in payload.messages[1]["content"]
    assert "<user_question>\n用户问题" in payload.messages[1]["content"]


def test_generator_decodes_new_tokens_only(monkeypatch):
    torch = pytest.importorskip("torch")
    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=torch.tensor([[1, 2]]))
            self.input_ids = self["input_ids"]

        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Inputs()

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [3, 4]
            return "回答"

    class Model:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])
    result = QwenGenerator(Model(), Tokenizer()).generate([{"role": "user", "content": "问"}], GenerationConfig(do_sample=False))
    assert result["answer"] == "回答"
    assert result["generation_config"]["enable_thinking"] is False


def test_pipeline_preserves_evidence():
    class Retriever:
        def retrieve(self, query, top_k):
            return {"query": query, "results": [{"index": 7, "score": 0.8, "text": "证据", "metadata": {"chunk_id": "c7"}}]}
    class Generator:
        def generate(self, messages, config):
            return {"answer": "回答", "prompt": "p", "model": "Qwen/Qwen3-4B", "generation_config": {}}
    result = RAGPipeline(Retriever(), Generator()).run("问题")
    assert result["evidence"] == [{"row_id": 7, "score": 0.8, "text": "证据", "metadata": {"chunk_id": "c7"}}]
