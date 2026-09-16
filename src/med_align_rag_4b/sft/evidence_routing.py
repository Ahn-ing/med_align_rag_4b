"""把旧规则池问题路由到已登记权威证据来源。"""

from __future__ import annotations

import re

TOPIC_PATTERNS = (
    ("nhs_anaphylaxis_2026", re.compile(r"严重过敏|过敏性休克|肾上腺素")),
    ("nhs_poisoning_2025", re.compile(r"中毒|误服|有害物|一氧化碳")),
    ("who_amr_2026", re.compile(r"抗微生物药物耐药|抗菌药物耐药|抗生素耐药|耐药性")),
    (
        "fda_pregnancy_2026",
        re.compile(
            r"(?:孕期|妊娠|怀孕).{0,12}(?:药|用药)|(?:药|用药).{0,12}(?:孕期|妊娠|怀孕)"
        ),
    ),
    ("fda_otc_2024", re.compile(r"非处方药|OTC|药品标签|药物相互作用|用药|吃药|服药|药物")),
    ("cdc_antibiotics_2025", re.compile(r"抗生素|抗菌药|消炎药")),
    ("cdc_cold_2026", re.compile(r"感冒|上呼吸道感染")),
    ("who_stroke_2025", re.compile(r"卒中|中风|脑梗|脑出血")),
    ("who_cvd_2025", re.compile(r"心肌梗|心梗|心血管|冠心|胸痛")),
    ("who_hypertension_2025", re.compile(r"高血压|血压")),
    ("who_diabetes_2024", re.compile(r"糖尿病|血糖|胰岛素")),
    ("who_asthma_2026", re.compile(r"哮喘|喘鸣")),
    ("who_copd_2026", re.compile(r"慢阻肺|慢性阻塞性肺|COPD")),
    ("who_sepsis_2024", re.compile(r"败血症|脓毒症|脓毒血症|感染性休克")),
    ("who_dengue_2025", re.compile(r"登革热")),
    ("who_influenza_2025", re.compile(r"流感|流行性感冒")),
    ("who_cervical_cancer_2026", re.compile(r"宫颈癌|子宫颈癌")),
    ("who_breast_cancer_2026", re.compile(r"乳腺癌")),
    ("who_colorectal_cancer_2026", re.compile(r"结直肠癌|结肠癌|直肠癌")),
    ("who_cancer_2026", re.compile(r"癌|肿瘤|恶性")),
    ("who_depression_2026", re.compile(r"抑郁|抑郁症|抑郁障碍")),
    ("who_anxiety_2025", re.compile(r"焦虑|惊恐|恐惧")),
    ("who_tuberculosis_2026", re.compile(r"结核|肺痨")),
    ("who_obesity_2025", re.compile(r"肥胖|超重|体重管理")),
    ("who_hepatitis_b_2026", re.compile(r"乙肝|乙型肝炎|HBV")),
    ("who_hiv_2026", re.compile(r"艾滋|HIV|艾滋病")),
    ("who_malaria_2025", re.compile(r"疟疾")),
    ("who_food_safety_2026", re.compile(r"食品安全|食源性|食物中毒")),
    ("who_maternal_mortality_2025", re.compile(r"孕产妇|产后出血|分娩|妊娠|怀孕")),
)


def evidence_id_for_question(
    question: str, registered_sources: list[dict[str, object]] | None = None
) -> str | None:
    """优先使用来源 topic_terms，再回退到精选主题规则。"""
    if registered_sources is not None:
        folded = question.casefold()
        for source in registered_sources:
            for term in source.get("topic_terms", []):
                if str(term).casefold() in folded:
                    return str(source["evidence_id"])
    for evidence_id, pattern in TOPIC_PATTERNS:
        if pattern.search(question):
            return evidence_id
    return None
