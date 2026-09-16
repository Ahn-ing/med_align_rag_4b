from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data/sft/sources/evidence_registry_v5_scale.json"

def test_v5_registry_reaches_target_scale_with_unique_facts() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = registry["sources"]
    claims = [(source["evidence_id"], claim) for source in sources for claim in source["supports"]]

    assert registry["status"] == "registered_for_internal_sft_v5_authoring"
    assert registry["release_allowed"] is False
    assert len(sources) == registry["source_count"] == 198
    assert len(claims) == registry["claim_count"] == 1965
    assert len({source["evidence_id"] for source in sources}) == len(sources)
    assert len({claim for _, claim in claims}) == len(claims)

    routed_sources = [source for source in sources if source.get("topic_terms")]
    assert len(routed_sources) == 169
    for source in routed_sources:
        assert source["topic_terms"]
        assert all(term.strip() == term and term for term in source["topic_terms"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", source["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", source["retrieved_html_sha256"])
        assert source["url"].startswith("https://www.who.int/zh/")
        assert source["supports"]
