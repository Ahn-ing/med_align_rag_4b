from med_align_rag_4b.sft.evidence_routing import evidence_id_for_question


def test_registered_topic_terms_take_precedence() -> None:
    sources = [{"evidence_id": "registered", "topic_terms": ["哮喘"]}]

    assert evidence_id_for_question("儿童哮喘如何管理", sources) == "registered"


def test_curated_topic_fallback_is_available() -> None:
    assert evidence_id_for_question("急性脑梗需要做什么") == "who_stroke_2025"


def test_unregistered_question_returns_none() -> None:
    assert evidence_id_for_question("如何整理桌面文件") is None
