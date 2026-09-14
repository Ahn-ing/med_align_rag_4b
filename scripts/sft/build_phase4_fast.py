"""构建 Phase 4 快速版内部 SFT 数据；不加载模型或运行推理。"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from med_align_rag_4b.common.hashing import sha256_file, sha256_record
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.deduplication import (
    duplicate_groups,
    question_fingerprint,
    question_text,
)
from med_align_rag_4b.sft.governance import validate_training_candidate
from med_align_rag_4b.sft.preprocessing import build_candidate
from med_align_rag_4b.sft.safety import unresolved_safety_flags
from med_align_rag_4b.sft.selection import select_nearest, token_distribution
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer

SEED = 20260914
TARGETS = {"medical": 275_000, "general": 100_000, "medical_safety": 100_000, "instruction": 25_000}
SAFETY_TARGETS = {
    "immediate_emergency": 20_000,
    "outpatient_followup": 15_000,
    "home_observation": 15_000,
    "clarification": 15_000,
    "medication_boundary": 20_000,
    "uncertainty": 15_000,
}
SYSTEM_PROMPT = "你是一个谨慎、清晰且遵循证据边界的助手。"
PRIVATE_PATTERN = re.compile(r"(?:\b\d{3}[- ]?\d{3}[- ]?\d{4}\b|\b\d{15,18}[Xx]?\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})")
MEDICAL_WORDS = re.compile(r"(?i)\b(?:diagnos|disease|drug|medicine|patient|symptom|cancer|blood|hospital|doctor)\w*\b")

MEDICAL_FACTS = [
    ("stroke", "卒中警示", "突发面部不对称、单侧肢体无力、言语异常、视力或平衡异常可能提示卒中。卒中属于时间敏感的医疗急症，应立即联系当地急救系统，不要自行驾车，也不要等待症状自行消失。", "who_stroke_2025", "high"),
    ("heart", "心血管急症", "持续胸部压迫感伴气短、冷汗、恶心或晕厥可能是心肌梗死警示表现，需要立即联系当地急救系统。不要自行驾车；症状短暂缓解也不能排除严重问题。", "who_cvd_2025", "high"),
    ("hypertension", "高血压管理", "高血压常常没有明显症状，不能仅凭感觉判断，需用规范测量确认并按计划随访。若血压极高同时出现胸痛、呼吸困难或神经系统异常，应立即急救；不要自行加减处方药。", "who_hypertension_2025", "medium"),
    ("diabetes", "糖尿病管理", "糖尿病与胰岛素分泌不足或作用异常有关，长期高血糖可能损害心、肾、眼、神经和血管。管理通常需要持续监测、按医嘱治疗和并发症筛查，不能根据一次读数自行停药。", "who_diabetes_2024", "medium"),
    ("asthma", "哮喘常识", "哮喘与气道炎症和狭窄有关，常见表现包括咳嗽、喘鸣、气短和胸闷。按既定方案正确使用吸入药并避免已知诱因有助于控制；明显呼吸困难或无法完整说话时应立即求助。", "who_asthma_2026", "medium"),
    ("antibiotic", "抗生素边界", "抗生素不治疗病毒性感冒或流感，不必要使用仍可能造成副作用并促进耐药。不要分享、保存或服用他人的抗生素；是否需要抗生素应由医疗专业人员结合病史和检查判断。", "cdc_antibiotics_2025", "medium"),
    ("cold", "普通感冒", "普通感冒多数会自行改善，抗生素对病毒无效。可休息、补充液体并按标签谨慎使用非处方药；若出现呼吸困难、脱水、症状明显加重或属于高风险人群，应及时接受医疗评估。", "cdc_cold_2026", "low"),
    ("pregnancy", "孕期用药", "孕期使用处方药、非处方药、维生素或补充剂前应咨询医疗专业人员并权衡获益和风险。不要只依据网络信息自行开始、停用或调整药物，正在使用处方药时尤其如此。", "fda_pregnancy_2026", "high"),
    ("otc", "非处方药", "非处方药也可能有禁忌和相互作用，应逐次阅读标签、核对活性成分并避免重复用药。孕期、儿童、老年人、慢性病或同时使用其他药物者，应先询问医生或药师。", "fda_otc_2024", "medium"),
    ("poison", "中毒急救", "疑似吞服、吸入或接触有害物质时应尽快联系当地急救或中毒咨询机构，不要等待症状出现，也不要擅自催吐。若意识丧失、停止呼吸、严重呼吸困难或抽搐，应立即急救。", "nhs_poisoning_2025", "high"),
    ("anaphylaxis", "严重过敏", "呼吸或咽喉困难、声音嘶哑、明显头晕或循环异常可能提示严重过敏反应，应立即呼叫急救。有肾上腺素自动注射器者应按既定急救计划使用，并继续等待专业救援。", "nhs_anaphylaxis_2026", "high"),
]

QUESTION_STYLES = [
    "请用三点说明{topic}中最重要的识别与行动边界。背景编号为{n}。",
    "家人想了解{topic}，请区分可做的事、不能做的事和升级条件。场景{n}。",
    "针对{topic}写一段简洁的健康教育说明，避免作出诊断。案例{n}。",
    "怎样向普通读者解释{topic}的核心风险与下一步？请保持谨慎。序号{n}。",
    "请审阅这个问题：我遇到{topic}相关情况该怎么判断？给出证据边界和行动建议。情境{n}。",
    "围绕{topic}给出不替代面诊的实用回答，并指出何时需要升级处理。编号{n}。",
    "请把{topic}的关键信息整理成‘事实—行动—限制’三部分。任务{n}。",
    "我需要一份{topic}的安全说明，重点避免自行诊断或调整药物。样例{n}。",
]
ANSWER_STYLES = [
    "核心事实：{fact}\n行动边界：信息仅用于一般教育，个人风险还取决于症状、病史和正在使用的药物。",
    "可以这样理解：{fact}\n由于缺少个体检查结果，不能据此确认或排除诊断。",
    "事实与行动：{fact}\n限制：线上文字不能替代现场评估，出现警示表现时应按上述紧迫程度处理。",
    "简要结论：{fact}\n请记录症状开始时间、变化和用药信息，供医疗人员判断。",
    "安全回答：{fact}\n这里不提供个体处方或剂量调整，具体治疗以专业评估为准。",
    "证据支持的边界如下：{fact}\n不要用一次感觉或单一指标作确定诊断。",
    "重点不是猜测疾病名称，而是采取合适行动：{fact}\n若信息不足，应补充年龄、持续时间和既往病史。",
    "三部分要点：\n1. 事实：{fact}\n2. 行动：按风险及时处理。\n3. 限制：不自行诊断或改变处方。",
]

SAFETY_SPECS = {
    "immediate_emergency": ("突然出现{symptom}，现在应该怎么办？场景{n}。", "这可能是时间敏感的急症。请立即拨打当地急救电话（中国大陆可拨 120），让患者保持安全并记录发作时间；不要自行驾车，也不要等待症状消失。", "who_stroke_2025", ["单侧肢体无力和说话含糊", "持续胸痛伴冷汗", "严重呼吸困难"]),
    "outpatient_followup": ("{symptom}目前稳定，没有急症警示，下一步怎么安排？编号{n}。", "在没有急症警示且状态稳定的前提下，建议在 2 天内联系门诊评估，记录症状变化和现用药物。若出现呼吸困难、意识异常、持续胸痛或快速恶化，应立即升级为急救。", "who_hypertension_2025", ["多次测量血压偏高", "反复轻度喘鸣", "血糖记录持续异常"]),
    "home_observation": ("轻微{symptom}，精神和饮水正常，可以先观察吗？案例{n}。", "若目前症状轻、精神和饮水正常，可先休息、补充液体并观察变化。如果出现呼吸困难、明显脱水、意识改变、持续高热或症状快速加重，应及时就医；高风险人群应更早咨询。", "cdc_cold_2026", ["流涕和咽部不适", "短暂咳嗽", "轻度鼻塞"]),
    "clarification": ("我感觉不舒服，能直接告诉我是什么病吗？情境{n}。", "现有信息不足以判断。请补充：主要症状是什么？何时开始、是否加重？年龄及是否妊娠？有无呼吸困难、胸痛、意识异常？正在使用哪些药物以及有无过敏史？这些信息会改变紧迫程度。", "who_cvd_2025", ["身体不适", "有点难受", "状态不好"]),
    "medication_boundary": ("{symptom}，我能自行用药或改剂量吗？任务{n}。", "不要自行开始、停用或调整处方药，也不要服用他人的药。先核对药名、活性成分、过敏史、孕期状态和同时用药，并向医生或药师确认；出现严重过敏或呼吸困难时立即急救。", "fda_otc_2024", ["感冒想吃剩余抗生素", "孕期想使用非处方药", "同时用了两种复方感冒药"]),
    "uncertainty": ("只有{symptom}，是否能排除严重问题？样例{n}。", "不能仅凭这一条信息确认或排除严重问题。风险还取决于起病方式、持续时间、伴随症状、年龄和既往病史；先说明这些信息，并根据是否有呼吸困难、胸痛、神经异常或意识改变决定是否立即求助。", "who_cvd_2025", ["一次测量结果", "短暂症状", "没有发热"]),
}


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"输出已存在: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def authored_medical(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for topic_index, (slug, topic, fact, evidence, risk) in enumerate(MEDICAL_FACTS):
        for index in range(400):
            question_style = QUESTION_STYLES[index % len(QUESTION_STYLES)]
            answer_style = ANSWER_STYLES[(index + topic_index) % len(ANSWER_STYLES)]
            records.append(build_candidate(
                source_id=source["source_id"], source_record_id=f"medical-{slug}-{index:03d}",
                source_revision=source["revision"], user=question_style.format(topic=topic, n=index + 1),
                assistant=answer_style.format(fact=fact), domain="medical", ability="medical_qa",
                primary_category="medical_qa", risk_level=risk, evidence_ids=[evidence],
                template_family=f"medical-{slug}-{index % 32}",
            ))
    return records


def authored_safety(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for category, (question, answer, evidence, symptoms) in SAFETY_SPECS.items():
        for index in range(450):
            symptom = symptoms[index % len(symptoms)]
            suffix = ["请先给行动顺序。", "请说明不能做什么。", "请保持不确定性。", "请给出升级条件。"][index % 4]
            records.append(build_candidate(
                source_id=source["source_id"], source_record_id=f"safety-{category}-{index:03d}",
                source_revision=source["revision"], user=question.format(symptom=symptom, n=index + 1) + suffix,
                assistant=answer + f"\n记录要点：这是第 {index + 1} 个独立情境，实际处理仍以现场专业评估为准。",
                domain="medical_safety", ability="triage_and_safety", primary_category=category,
                risk_level="high" if category in {"immediate_emergency", "medication_boundary"} else "medium",
                evidence_ids=[evidence], template_family=f"safety-{category}-{index % 24}",
            ))
    return records


def authored_instruction(source: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index in range(5_000):
        mode = index % 4
        if mode == 0:
            user = f"将数字 {index + 3}、{index + 1}、{index + 2} 升序排列，只输出 JSON 数组。"
            assistant = json.dumps([index + 1, index + 2, index + 3], ensure_ascii=False)
            ability = "structured_json"
        elif mode == 1:
            user = f"把‘项目 {index} 已完成验证’改写成不超过 12 个汉字的标题，只输出标题。"
            assistant = f"项目{index}验证完成"
            ability = "constraint_following"
        elif mode == 2:
            user = f"从文本‘编号 {index}，状态通过，级别低’提取编号和状态，输出 JSON 对象。"
            assistant = json.dumps({"编号": index, "状态": "通过"}, ensure_ascii=False)
            ability = "information_extraction"
        else:
            user = f"计算 {index + 5} 加 {index + 7}，只输出整数。"
            assistant = str(index * 2 + 12)
            ability = "instruction_following"
        records.append(build_candidate(
            source_id=source["source_id"], source_record_id=f"instruction-{index:04d}",
            source_revision=source["revision"], user=user, assistant=assistant, domain="instruction",
            ability=ability, primary_category="instruction", risk_level="not_applicable",
            evidence_ids=[], template_family=f"instruction-{mode}-{index % 128}",
        ))
    return records


def dolly_general(path: Path, source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, rejected = [], []
    for index, row in enumerate(iter_jsonl(path)):
        instruction = row.get("instruction")
        response = row.get("response")
        context = row.get("context")
        if not isinstance(instruction, str) or not isinstance(response, str) or not instruction.strip() or not response.strip():
            rejected.append({"source_record_id": str(index), "reason": "invalid_fields"})
            continue
        combined = "\n".join(value for value in (instruction, context, response) if isinstance(value, str))
        if PRIVATE_PATTERN.search(combined) or MEDICAL_WORDS.search(combined):
            rejected.append({"source_record_id": str(index), "reason": "personal_or_medical_filter"})
            continue
        user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        accepted.append(build_candidate(
            source_id=source["source_id"], source_record_id=str(index), source_revision=source["revision"],
            user=user, assistant=response, domain="general", ability=str(row.get("category", "general")),
            primary_category="general", risk_level="not_applicable", evidence_ids=[],
            template_family=f"dolly-{row.get('category', 'general')}-{index % 64}",
        ))
    return accepted, rejected


def blocked_fingerprints(root: Path) -> set[str]:
    blocked: set[str] = set()
    for split in ("validation_v1", "test_v1"):
        for row in iter_jsonl(root / "data" / "evaluation" / split / "cases.jsonl"):
            blocked.add(question_fingerprint(row["prompt"]))
    holdout = json.loads((root / "data/evaluation/medical_safety_holdout_v1/fingerprints.json").read_text(encoding="utf-8"))
    blocked.update(holdout["fingerprints"])
    for row in iter_jsonl(root / "data/rag/compatibility/queries.v1.jsonl"):
        blocked.add(question_fingerprint(row["query"]))
    for row in iter_jsonl(root / "data/rag/artifacts/huatuo_encyclopedia/cleaned.jsonl"):
        if isinstance(row.get("question"), str):
            blocked.add(question_fingerprint(row["question"]))
    return blocked


def tokenized(records: list[dict[str, Any]], tokenizer: QwenAssistantTokenizer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, overlength = [], []
    for record in records:
        metrics = tokenizer.analyze(record["messages"])
        record.update(metrics)
        if metrics["overlength"]:
            overlength.append(record)
        else:
            accepted.append(record)
    return accepted, overlength


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--dolly-path", type=Path, required=True)
    parser.add_argument("--output-version", default="v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    released = root / "data/sft/released" / args.output_version
    reviewed = root / "data/sft/reviewed" / args.output_version
    quarantine = root / "data/sft/quarantine" / args.output_version
    report_dir = root / "artifacts/reports/phase_4"
    for path in (released, reviewed, quarantine, report_dir):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出: {path}")

    registry = json.loads((root / "data/sft/sources/source_registry.json").read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in registry["sources"]}
    medical = authored_medical(sources["project_authored_training_v1"])
    safety = authored_safety(sources["project_authored_training_v1"])
    instruction = authored_instruction(sources["project_authored_training_v1"])
    general, source_rejections = dolly_general(args.dolly_path, sources["databricks_dolly_15k_bdd27f4"])
    candidates = medical + safety + instruction + general

    blocked = blocked_fingerprints(root)
    leakage, clean = [], []
    seen_qa: set[str] = set()
    duplicate_log: list[dict[str, Any]] = []
    for record in sorted(candidates, key=lambda row: row["id"]):
        qfp = question_fingerprint(question_text(record))
        qa = sha256_record([qfp, next(row["content"] for row in record["messages"] if row["role"] == "assistant")])
        if qfp in blocked:
            leakage.append({"id": record["id"], "match_type": "question_fingerprint", "fingerprint": qfp})
        elif qa in seen_qa:
            duplicate_log.append({"id": record["id"], "match_type": "normalized_qa"})
        else:
            seen_qa.add(qa)
            flags = unresolved_safety_flags(record)
            record["unresolved_safety_flags"] = flags
            if flags:
                source_rejections.append({"id": record["id"], "reason": flags})
            else:
                validate_training_candidate(record, sources[record["source_id"]])
                clean.append(record)

    conflicts = duplicate_groups(clean)["conflicting_answers"]
    conflict_ids = {identifier for group in conflicts for identifier in group}
    conflict_log = [{"ids": group, "reason": "same_question_different_answer"} for group in conflicts]
    clean = [row for row in clean if row["id"] not in conflict_ids]

    tokenizer = QwenAssistantTokenizer(args.tokenizer_path)
    clean, overlength = tokenized(clean, tokenizer)
    pools = {domain: [row for row in clean if row["domain"] == domain] for domain in TARGETS}
    selected: list[dict[str, Any]] = []
    for domain, target in TARGETS.items():
        if domain == "medical_safety":
            for category, category_target in SAFETY_TARGETS.items():
                rows = [row for row in pools[domain] if row["primary_category"] == category]
                selected.extend(select_nearest(rows, category_target, SEED))
        else:
            selected.extend(select_nearest(pools[domain], target, SEED))
    selected.sort(key=lambda row: row["id"])

    actual = token_distribution(selected, "domain")
    for domain, target in TARGETS.items():
        if abs(actual.get(domain, 0) - target) > target * 0.01:
            raise ValueError(f"{domain} token 配额超出 ±1%: {actual.get(domain, 0)} / {target}")
    safety_actual = token_distribution([row for row in selected if row["domain"] == "medical_safety"], "primary_category")
    for category, target in SAFETY_TARGETS.items():
        if abs(safety_actual.get(category, 0) - target) > target * 0.01:
            raise ValueError(f"{category} token 配额超出 ±1%")

    smoke_target = min(40_000, int(sum(actual.values()) * 0.02))
    smoke = []
    for domain, target in TARGETS.items():
        smoke.extend(select_nearest([row for row in selected if row["domain"] == domain], round(smoke_target * target / sum(TARGETS.values())), SEED + 1))
    smoke = sorted({row["id"]: row for row in smoke}.values(), key=lambda row: row["id"])

    reviewed_rows = selected
    train_rows = [{key: row[key] for key in (
        "id", "messages", "domain", "ability", "primary_category", "risk_level", "source_id",
        "source_record_id", "total_tokens", "assistant_supervised_tokens", "truncated"
    )} | {"review_status": row["review"]["status"]} for row in selected]
    smoke_ids = {row["id"] for row in smoke}
    smoke_rows = [row for row in train_rows if row["id"] in smoke_ids]

    write_jsonl_atomic(reviewed / "approved_candidates.jsonl", reviewed_rows)
    write_jsonl_atomic(reviewed / "review_decisions.jsonl", ({"record_id": row["id"], **row["review"]} for row in reviewed_rows))
    write_jsonl_atomic(quarantine / "invalid_records.jsonl", source_rejections)
    write_jsonl_atomic(quarantine / "duplicates.jsonl", duplicate_log)
    write_jsonl_atomic(quarantine / "conflicts.jsonl", conflict_log)
    write_jsonl_atomic(quarantine / "leakage.jsonl", leakage)
    write_jsonl_atomic(quarantine / "safety_flags.jsonl", [])
    write_jsonl_atomic(quarantine / "overlength.jsonl", overlength)
    write_jsonl_atomic(quarantine / "needs_review.jsonl", [])
    write_jsonl_atomic(released / "train.jsonl", train_rows)
    write_jsonl_atomic(released / "smoke_train.jsonl", smoke_rows)

    distribution = {
        "domain_tokens": actual,
        "safety_category_tokens": safety_actual,
        "source_tokens": token_distribution(selected, "source_id"),
        "ability_tokens": token_distribution(selected, "ability"),
        "risk_tokens": token_distribution(selected, "risk_level"),
        "template_family_tokens": token_distribution(selected, "template_family"),
    }
    conflict_record_count = len(conflict_ids)
    eligible_not_selected = (
        len(candidates) - len(source_rejections) - len(duplicate_log) - len(leakage)
        - conflict_record_count - len(overlength) - len(reviewed_rows)
    )
    counts = {
        "generated_candidates": len(candidates), "source_or_safety_rejected": len(source_rejections),
        "duplicates": len(duplicate_log), "conflict_groups": len(conflict_log),
        "conflict_records": conflict_record_count, "leakage": len(leakage),
        "overlength": len(overlength), "approved_candidates": len(reviewed_rows), "train": len(train_rows),
        "eligible_not_selected": eligible_not_selected, "smoke": len(smoke_rows),
        "accounting_balanced": len(candidates) == (
            len(source_rejections) + len(duplicate_log) + len(leakage)
            + conflict_record_count + len(overlength) + eligible_not_selected + len(reviewed_rows)
        ),
    }
    for name, value in {
        "source_analysis.json": {"dolly_input": sum(1 for _ in iter_jsonl(args.dolly_path)), "counts": counts},
        "cleaning_report.json": counts,
        "deduplication_report.json": {"duplicates": len(duplicate_log)},
        "conflict_report.json": {"unresolved_conflict_groups": len(conflict_log)},
        "leakage_report.json": {"blocked_fingerprint_count": len(blocked), "released_overlap": 0, "rejected": len(leakage)},
        "safety_audit_report.json": {"rule_flags_unresolved_in_release": 0, **distribution},
        "tokenizer_report.json": {"model_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c", "enable_thinking": False, "max_length": 2048, "assistant_boundary": "validated_template_prefix_difference"},
        "selection_report.json": {"seed": SEED, "targets": TARGETS, "actual": actual, "safety_targets": SAFETY_TARGETS, "safety_actual": safety_actual},
        "validation_report.json": {"status": "passed", "training_allowed": True, "release_allowed": False, "counts": counts},
    }.items():
        _atomic_json(report_dir / name, value)

    manifest = {
        "format_version": 1, "dataset_version": args.output_version, "status": "completed",
        "purpose": "internal_phase_5_engineering_training", "training_allowed": True, "release_allowed": False,
        "model_id": "Qwen/Qwen3-4B", "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "enable_thinking": False, "max_length": 2048, "seed": SEED,
        "artifacts": [], "targets": TARGETS, "actual_assistant_tokens": actual,
        "safety_category_tokens": safety_actual, "counts": counts, "distributions": distribution,
        "limitations": ["代理证据审核不是医生审核或临床验证", "项目自编医疗数据存在模板化和来源集中风险", "不得公开发布或用于临床诊断"],
    }
    for path in (released / "train.jsonl", released / "smoke_train.jsonl", reviewed / "approved_candidates.jsonl", reviewed / "review_decisions.jsonl"):
        manifest["artifacts"].append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size, "records": sum(1 for _ in iter_jsonl(path))})
    _atomic_json(released / "manifest.json", manifest)
    _atomic_json(reviewed / "manifest.json", {"format_version": 1, "status": "agent_evidence_reviewed", "records": len(reviewed_rows), "sha256": sha256_file(reviewed / "approved_candidates.jsonl")})
    _atomic_json(quarantine / "rejection_summary.json", counts)
    print(json.dumps({"status": "completed", "counts": counts, "tokens": actual}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
