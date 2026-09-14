"""Phase 0 使用的最小数据契约。"""

from dataclasses import dataclass
from typing import Any

from med_align_rag_4b.common.utils import require_non_empty_string


@dataclass(frozen=True)
class ArtifactDescriptor:
    """一个不可变产物的可迁移描述。"""

    path: str
    sha256: str
    bytes: int
    records: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactDescriptor":
        path = require_non_empty_string(value.get("path"), "artifact.path")
        if path.startswith(("/", "\\")) or (len(path) > 1 and path[1] == ":"):
            raise ValueError("artifact.path 必须是可迁移的相对路径")
        digest = require_non_empty_string(value.get("sha256"), "artifact.sha256")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact.sha256 必须是小写 SHA256 十六进制字符串")
        byte_count = value.get("bytes")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("artifact.bytes 必须是非负整数")
        records = value.get("records")
        if records is not None and (not isinstance(records, int) or records < 0):
            raise ValueError("artifact.records 必须是非负整数或 null")
        return cls(path, digest, byte_count, records)


@dataclass(frozen=True)
class SFTReview:
    """未来 SFT 正式数据使用的审核门禁。"""

    status: str
    reviewer_role: str
    rubric_version: str

    def is_training_approved(self) -> bool:
        """仅允许审核通过的记录进入正式训练。"""
        return self.status == "approved"
