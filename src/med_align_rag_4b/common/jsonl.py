"""JSONL 流式读取和原子写入。"""

import json
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐条读取 JSON 对象，并在错误中保留物理行号。"""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON: {source}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象: {source}")
            yield value


def count_jsonl(path: str | Path) -> int:
    """统计通过解析的非空 JSONL 记录。"""
    return sum(1 for _ in iter_jsonl(path))


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> int:
    """先写入同目录临时文件，再原子发布 JSONL。"""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"输出已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count
