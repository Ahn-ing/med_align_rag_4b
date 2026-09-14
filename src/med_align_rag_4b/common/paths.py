"""保证持久化路径可迁移且位于项目内。"""

from pathlib import Path, PurePosixPath


def resolve_project_path(project_root: str | Path, value: str | Path) -> Path:
    """解析项目相对路径，并拒绝绝对路径和目录穿越。"""
    root = Path(project_root).resolve()
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError("项目路径必须是相对路径")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("项目路径必须位于项目根目录内") from exc
    return candidate


def portable_relative_path(path: str | Path, project_root: str | Path) -> str:
    """生成可写入 manifest 的 POSIX 风格项目相对路径。"""
    root = Path(project_root).resolve()
    try:
        relative = Path(path).resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest 路径必须位于项目根目录内") from exc
    return str(PurePosixPath(relative))
