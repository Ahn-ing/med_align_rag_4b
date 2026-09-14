import json

import pytest

from med_align_rag_4b.common.manifests import (
    build_manifest,
    describe_artifact,
    load_manifest,
)
from med_align_rag_4b.common.schemas import ArtifactDescriptor, SFTReview


def test_manifest_describes_relative_hashed_artifact(tmp_path):
    artifact = tmp_path / "artifacts" / "sample.txt"
    artifact.parent.mkdir()
    artifact.write_text("sample", encoding="utf-8")
    descriptor = describe_artifact(artifact, tmp_path, records=1)
    manifest = build_manifest("phase_0_test", [descriptor])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded["artifacts"][0]["path"] == "artifacts/sample.txt"
    assert len(loaded["artifacts"][0]["sha256"]) == 64


def test_artifact_descriptor_rejects_absolute_path():
    with pytest.raises(ValueError, match="相对路径"):
        ArtifactDescriptor.from_dict(
            {"path": "C:/models/file.bin", "sha256": "0" * 64, "bytes": 1}
        )


def test_only_approved_sft_review_allows_training():
    assert SFTReview("approved", "medical_reviewer", "v1").is_training_approved()
    assert not SFTReview("unreviewed", "", "v1").is_training_approved()
