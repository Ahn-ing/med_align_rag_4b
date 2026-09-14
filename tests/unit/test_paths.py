from pathlib import Path

import pytest

from med_align_rag_4b.common.paths import portable_relative_path, resolve_project_path


def test_resolve_project_path_stays_inside_root(tmp_path):
    expected = (tmp_path / "data/file.jsonl").resolve()
    assert resolve_project_path(tmp_path, "data/file.jsonl") == expected


def test_resolve_project_path_rejects_absolute_and_traversal(tmp_path):
    with pytest.raises(ValueError, match="相对路径"):
        resolve_project_path(tmp_path, Path(tmp_path / "absolute.json"))
    with pytest.raises(ValueError, match="根目录内"):
        resolve_project_path(tmp_path, "../outside.json")


def test_portable_relative_path_uses_forward_slashes(tmp_path):
    path = tmp_path / "data" / "file.json"
    path.parent.mkdir()
    path.touch()
    assert portable_relative_path(path, tmp_path) == "data/file.json"
