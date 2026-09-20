from med_align_rag_4b.evaluation.rag_scoring import score_record


def _case(required_facts: list[str]) -> dict:
    return {
        "id": "case",
        "task": "rag_faithfulness",
        "primary_category": "rag_faithfulness",
        "risk_level": "medium",
        "required_facts": required_facts,
        "rag_requirement": "required",
    }


def _row(output: str) -> dict:
    return {
        "arm": "R3",
        "case_id": "case",
        "output": output,
        "truncated": False,
        "rag": {"enabled": True, "context": "[E1] 证据", "evidence": []},
    }


def test_answerable_rag_does_not_require_citation():
    score = score_record(_case(["事实"]), _row("事实"))
    assert score["hard_fail"] is False
    assert "citation_valid" not in score


def test_citation_marker_is_not_scored():
    score = score_record(_case(["事实"]), _row("事实 [E99]"))
    assert score["hard_fail"] is False
    assert "citation_valid" not in score
