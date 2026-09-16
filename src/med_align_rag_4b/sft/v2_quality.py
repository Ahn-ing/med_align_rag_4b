"""SFT v2 自然问答的流式读取、质量门禁与风险分层。"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from med_align_rag_4b.common.hashing import sha256_record, sha256_text
from med_align_rag_4b.sft.deduplication import qa_fingerprint, question_fingerprint
from med_align_rag_4b.sft.preprocessing import clean_text

MEDICAL_CATEGORIES = (
    "knowledge_explanation",
    "symptom_risk_clarification",
    "medication_boundary",
    "action_followup",
    "emergency_escalation",
    "summary_synthesis",
)
MEDICAL_TOKEN_TARGETS = {
    "knowledge_explanation": 900_000,
    "symptom_risk_clarification": 600_000,
    "medication_boundary": 450_000,
    "action_followup": 450_000,
    "emergency_escalation": 240_000,
    "summary_synthesis": 360_000,
}
API_TOKEN_TARGETS = {
    "knowledge_explanation": 22_500,
    "symptom_risk_clarification": 56_250,
    "medication_boundary": 45_000,
    "action_followup": 33_750,
    "emergency_escalation": 45_000,
    "summary_synthesis": 22_500,
}
API_MEDICAL_TOKEN_MINIMUM = 150_000
API_MEDICAL_TOKEN_TARGET = 225_000
API_MEDICAL_TOKEN_MAXIMUM = 300_000

_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_META = re.compile(
    r"记录要点|第\s*\d+\s*个(?:独立)?(?:情境|场景|任务|样例)|"
    r"(?:背景编号|案例编号|序号|任务)\s*(?:为|是|[:：])?\s*\d+",
    re.IGNORECASE,
)
_PRIVATE = re.compile(
    r"(?:\b1[3-9]\d{9}\b|\b\d{15,18}[Xx]?\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})"
)
_AD = re.compile(
    r"加(?:微信|QQ)|公众号|咨询热线|包治|根治|立即购买|"
    r"(?:点击|扫码).{0,12}(?:链接|购买)"
)
_EMERGENCY = re.compile(
    r"胸痛|呼吸困难|意识(?:不清|丧失|改变)|单侧.*无力|言语不清|抽搐|"
    r"严重过敏|大出血|自杀|中毒|急救|120"
)
_MEDICATION = re.compile(
    r"药|剂量|处方|服用|用药|副作用|禁忌|相互作用|抗生素|疫苗|注射"
)
_SYMPTOM = re.compile(r"症状|是否|判断|诊断|可能|原因|怎么回事|怎么办|检查")
_ACTION = re.compile(r"注意|护理|生活|饮食|运动|复诊|随访|观察|预防|管理")
_SUMMARY = re.compile(r"总结|概括|归纳|整理|区别|比较|要点")
_UNSAFE_CERTAINTY = re.compile(r"一定是|肯定是|可以确诊|无需检查|不用就医")
_CONSULTATION_TEMPLATE = re.compile(
    r"(?:根据|结合)你的描述.{0,12}(?:考虑|判断)|这种情况(?:考虑|可能是).{0,20}引起"
)
_SELF_MEDICATION = re.compile(
    r"(?:可以|建议)(?:自行)?(?:服用|使用|口服).{0,30}(?:试试|观察)"
)
_EMPTY_CLOSING = re.compile(r"祝(?:你)?早日康复|如有问题欢迎(?:继续)?咨询")
_FIELD_LABEL = re.compile(r"(?:病因|症状|检查|诊断|治疗|预防|护理)\s*[:：]")
_DOSAGE = re.compile(
    r"(?:每日|每次|一天)\s*\d|\d+(?:\.\d+)?\s*(?:mg|毫克|克|片|粒)", re.I
)


def iter_json_array(
    path: str | Path, *, chunk_size: int = 1024 * 1024
) -> Iterator[dict[str, Any]]:
    """以有界内存读取 JSONL 或顶层 JSON 数组。"""
    source = Path(path)
    with source.open(encoding="utf-8") as probe:
        first = ""
        while True:
            character = probe.read(1)
            if not character or not character.isspace():
                first = character
                break
    if first == "{":
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"医疗源 JSONL 第 {line_number} 行无效") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"医疗源 JSONL 第 {line_number} 行必须是对象")
                yield value
        return
    if first != "[":
        raise ValueError("医疗源文件必须是 JSONL 或顶层 JSON 数组")

    decoder = json.JSONDecoder()
    with source.open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        started = False
        finished = False
        while not finished:
            chunk = stream.read(chunk_size)
            if chunk:
                buffer = buffer[position:] + chunk
                position = 0
            elif position >= len(buffer):
                break
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("医疗源文件必须是顶层 JSON 数组")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    position += 1
                    finished = True
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if chunk:
                        break
                    raise ValueError("医疗源 JSON 数组不完整") from None
                if not isinstance(value, dict):
                    raise ValueError("医疗源数组元素必须是对象")
                yield value
                position = end
            if not chunk and not finished:
                raise ValueError("医疗源 JSON 数组缺少结束符")
        if not started or not finished:
            raise ValueError("医疗源 JSON 数组格式不完整")


def normalize_match(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", value).strip()


def template_skeleton(text: str) -> str:
    """移除数字差异，揭示编号或数值替换形成的伪多样性。"""
    value = _DIGITS.sub("<num>", normalize_match(text))
    return re.sub(r"[^\w\u4e00-\u9fff<>]+", "", value)


def medical_category(question: str, answer: str) -> str:
    text = f"{question} {answer}"
    if _EMERGENCY.search(text):
        return "emergency_escalation"
    if _MEDICATION.search(text):
        return "medication_boundary"
    if _SUMMARY.search(question):
        return "summary_synthesis"
    if _ACTION.search(question):
        return "action_followup"
    if _SYMPTOM.search(question):
        return "symptom_risk_clarification"
    return "knowledge_explanation"


def medical_risk(category: str, question: str, answer: str) -> str:
    text = f"{question} {answer}"
    if category == "emergency_escalation" or _DOSAGE.search(text):
        return "high"
    if category in {
        "medication_boundary",
        "symptom_risk_clarification",
        "action_followup",
    }:
        return "medium"
    return "low"


def automatic_quality_flags(question: str, answer: str) -> list[str]:
    """只做可确定的机械筛查，不冒充医学正确性审核。"""
    flags: list[str] = []
    if not question or not answer:
        flags.append("empty_text")
    if _PRIVATE.search(f"{question} {answer}"):
        flags.append("possible_personal_data")
    if _META.search(answer):
        flags.append("generator_meta_language")
    if _AD.search(answer):
        flags.append("advertising_or_unverifiable_claim")
    if _UNSAFE_CERTAINTY.search(answer):
        flags.append("unsafe_diagnostic_certainty")
    if _CONSULTATION_TEMPLATE.search(answer):
        flags.append("online_consultation_template")
    if _SELF_MEDICATION.search(answer):
        flags.append("unsupported_self_medication_trial")
    if _EMPTY_CLOSING.search(answer):
        flags.append("generic_consultation_closing")
    labels = _FIELD_LABEL.findall(answer)
    if len(labels) >= 4:
        flags.append("dense_fixed_medical_schema")
    if len(labels) != len(set(labels)):
        flags.append("repeated_medical_field")
    if len(answer) < 12:
        flags.append("answer_too_short_for_medical_review")
    if len(answer) > 12_000:
        flags.append("answer_extreme_length")
    if question == answer:
        flags.append("question_copy")
    return flags


def raw_medical_record(raw: dict[str, Any], line_number: int) -> dict[str, Any]:
    """把旧项目原始记录转换为无 system prompt 的 v2 审核候选。"""
    if not all(
        isinstance(raw.get(key), str) for key in ("instruction", "input", "output")
    ):
        raise ValueError("schema_invalid")
    question = clean_text(
        "\n".join(part for part in (raw["instruction"], raw["input"]) if part.strip())
    )
    answer = clean_text(raw["output"])
    if not question or not answer:
        raise ValueError("empty_text")
    category = medical_category(question, answer)
    risk = medical_risk(category, question, answer)
    source_record_id = str(line_number)
    identifier = (
        "sftv2_"
        + sha256_record(
            ["shibing624_medical_6e219f1", source_record_id, question, answer]
        )[:24]
    )
    return {
        "id": identifier,
        "source_id": "shibing624_medical_6e219f1",
        "source_record_id": source_record_id,
        "source_revision": "6e219f1a14856833ee436063d3b73c5f1ab9cfb9",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "domain": "medical",
        "ability": category,
        "primary_category": category,
        "risk_level": risk,
        "license_status": "approved_for_internal_candidate_only",
        "release_allowed": False,
        "quality_flags": automatic_quality_flags(question, answer),
        "review": {
            "status": "ready_for_review",
            "reviewer_role": None,
            "evidence_checked": False,
        },
        "medical_evidence": [],
        "fingerprints": {
            "question": question_fingerprint(question),
            "qa": qa_fingerprint(question, answer),
            "question_skeleton": sha256_text(template_skeleton(question)),
            "answer_skeleton": sha256_text(template_skeleton(answer)),
        },
        "provenance": {"generation_method": "upstream_original"},
    }


def write_json_line(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
