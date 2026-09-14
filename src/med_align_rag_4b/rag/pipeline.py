"""组合检索、prompt 与 generation，并保留完整证据引用。"""

from __future__ import annotations

from typing import Any

from med_align_rag_4b.rag.generation import GenerationConfig
from med_align_rag_4b.rag.prompting import build_prompt


class RAGPipeline:
    def __init__(self, retriever: Any, generator: Any) -> None:
        self.retriever, self.generator = retriever, generator

    def run(self, query: str, *, top_k: int = 5, max_context_chars: int = 12000, generation_config: GenerationConfig | None = None) -> dict[str, Any]:
        retrieval = self.retriever.retrieve(query, top_k)
        results = retrieval["results"]
        payload = build_prompt(retrieval["query"], results, max_context_chars)
        generated = self.generator.generate(payload.messages, generation_config or GenerationConfig())
        return {"query": retrieval["query"], "answer": generated["answer"], "evidence": [{"row_id": item["index"], "score": item["score"], "text": item["text"], "metadata": item["metadata"]} for item in results], "prompt": generated["prompt"], "context": payload.context, "model": generated["model"], "generation_config": generated["generation_config"]}
