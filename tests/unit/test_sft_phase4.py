from med_align_rag_4b.sft.preprocessing import build_candidate, clean_text
from med_align_rag_4b.sft.safety import unresolved_safety_flags
from med_align_rag_4b.sft.selection import select_nearest


def test_cleaning_preserves_paragraphs_numbers_and_negation():
    assert clean_text(" 不使用  10 mg\r\n\r\n不要停药 ") == "不使用 10 mg\n\n不要停药"


def test_safety_specific_fields_and_deterministic_selection():
    record = build_candidate(
        source_id="s", source_record_id="1", source_revision="r", user="轻微咳嗽",
        assistant="可以观察。如果出现呼吸困难应就医。", domain="medical_safety",
        ability="triage", primary_category="home_observation", risk_level="low",
        evidence_ids=["e"], template_family="f",
    )
    assert unresolved_safety_flags(record) == []
    rows = []
    for index, tokens in enumerate((4, 6, 7)):
        row = dict(record, id=str(index), assistant_supervised_tokens=tokens)
        rows.append(row)
    assert select_nearest(rows, 10, 7) == select_nearest(list(reversed(rows)), 10, 7)
