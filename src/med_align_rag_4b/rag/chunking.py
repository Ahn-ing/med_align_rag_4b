"""按 BGE-M3 token 预算生成稳定、句子感知的检索 chunk。"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic

CHUNK_SIZE = 768
CHUNK_OVERLAP = 100
SENTENCE_END_RE = re.compile(r"(?<=[。！？!?])")


def inspect_input(path: str | Path) -> int:
    """校验 cleaning 输出；不再错误要求不存在的 ``text`` 字段。"""
    seen: set[str] = set()
    count = 0
    for record in iter_jsonl(path):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("cleaned record 缺少非空 id")
        if record_id in seen:
            raise ValueError(f"重复 record id: {record_id}")
        if not all(isinstance(record.get(key), str) and record[key].strip() for key in ("question", "answer")):
            raise ValueError(f"{record_id} 缺少非空 question/answer")
        seen.add(record_id)
        count += 1
    if not count:
        raise ValueError("cleaned 输入没有记录")
    return count


class ChunkGenerator:
    def __init__(self, tokenizer: Any, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("要求 chunk_size > chunk_overlap >= 0")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def token_len(self, text: str) -> int:
        return len(self.encode(text))

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [part.strip() for part in SENTENCE_END_RE.split(text) if part.strip()]

    def build_prefix(self, question: str) -> str:
        return f"问题：{question}\n答案："

    def build_chunk_text(self, question: str, answer: str) -> str:
        return self.build_prefix(question) + answer

    def _split_long(self, text: str, limit: int) -> list[str]:
        ids = self.encode(text)
        result: list[str] = []
        start = 0
        while start < len(ids):
            end = min(start + limit, len(ids))
            while end > start and self.token_len(piece := self.decode(ids[start:end])) > limit:
                end -= 1
            if end == start or not piece:
                raise RuntimeError("无法在 token 预算内切分文本")
            result.append(piece)
            start = end
        return result

    def generate_for_record(self, record: dict[str, Any], source_index: int) -> list[dict[str, Any]]:
        source_id = str(record.get("id") or f"record_{source_index:06d}")
        question = str(record.get("question") or "").strip()
        answer = str(record.get("answer") or "").strip()
        if not question or not answer:
            return []
        budget = self.chunk_size - self.token_len(self.build_prefix(question))
        if budget <= 0:
            raise ValueError("question 已占满 chunk token 预算")
        units: list[str] = []
        for sentence in self.split_sentences(answer):
            units.extend([sentence] if self.token_len(sentence) <= budget else self._split_long(sentence, budget))
        answer_chunks: list[str] = []
        start = 0
        while start < len(units):
            end = start
            while end < len(units) and self.token_len(self.build_chunk_text(question, "".join(units[start : end + 1]))) <= self.chunk_size:
                end += 1
            if end == start:
                raise RuntimeError("已切分单元仍超过 chunk token 预算")
            answer_chunks.append("".join(units[start:end]))
            if end == len(units):
                break
            capacity = min(self.chunk_overlap, budget - self.token_len(units[end]))
            overlap_start = end
            used = 0
            for index in range(end - 1, start - 1, -1):
                length = self.token_len(units[index])
                if used + length > capacity:
                    break
                used += length
                overlap_start = index
                if used >= self.chunk_overlap:
                    break
            start = overlap_start if overlap_start < end else end
        chunks = []
        for chunk_index, answer_chunk in enumerate(answer_chunks):
            text = self.build_chunk_text(question, answer_chunk)
            chunks.append({"chunk_id": f"{source_id}_chunk_{chunk_index:04d}", "source_id": source_id, "question": question, "text": text, "token_length": self.token_len(text), "chunk_index": chunk_index})
        return chunks


def generate_dataset(input_path: str | Path, output_path: str | Path, tokenizer: Any, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, overwrite: bool = False) -> tuple[int, int]:
    inspect_input(input_path)
    generator = ChunkGenerator(tokenizer, chunk_size, chunk_overlap)
    seen: set[str] = set()
    record_count = 0
    chunk_count = 0
    def chunks():
        nonlocal chunk_count, record_count
        for source_index, record in enumerate(iter_jsonl(input_path)):
            record_count += 1
            for chunk in generator.generate_for_record(record, source_index):
                if chunk["chunk_id"] in seen:
                    raise ValueError(f"重复 chunk_id: {chunk['chunk_id']}")
                seen.add(chunk["chunk_id"])
                chunk_count += 1
                yield chunk
    write_jsonl_atomic(output_path, chunks(), overwrite=overwrite)
    return record_count, chunk_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    print(generate_dataset(args.input, args.output, tokenizer, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
