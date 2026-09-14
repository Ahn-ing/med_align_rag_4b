"""用于来源追踪的确定性 SHA256 函数。"""

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免将大型产物整体载入内存。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """计算 UTF-8 文本的 SHA256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_record(record: Any) -> str:
    """使用稳定键顺序计算 JSON 兼容对象的 SHA256。"""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)
