"""完整 RAG 消融的确定性规则评分。"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
INSUFFICIENCY_PHRASES = (
    "信息不足",
    "无法确定",
    "无法回答",
    "无法根据",
    "没有足够",
    "证据不足",
    "上下文未提供",
)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _schema_errors(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(instance, dict):
            return ["expected_object"]
        errors: list[str] = []
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"missing:{key}")
        if schema.get("additionalProperties") is False:
            errors.extend(f"extra:{key}" for key in sorted(set(instance) - set(properties)))
        for key, value in instance.items():
            if key in properties:
                errors.extend(f"{key}:{error}" for error in _schema_errors(value, properties[key]))
        return errors
    if expected == "array":
        if not isinstance(instance, list):
            return ["expected_array"]
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            return []
        errors: list[str] = []
        for index, item in enumerate(instance):
            errors.extend(f"{index}:{error}" for error in _schema_errors(item, item_schema))
        return errors
    if expected == "string":
        return [] if isinstance(instance, str) else ["expected_string"]
    if expected == "integer":
        return [] if isinstance(instance, int) and not isinstance(instance, bool) else ["expected_integer"]
    if expected == "number":
        return [] if isinstance(instance, (int, float)) and not isinstance(instance, bool) else ["expected_number"]
    if expected == "boolean":
        return [] if isinstance(instance, bool) else ["expected_boolean"]
    if expected == "null":
        return [] if instance is None else ["expected_null"]
    return [f"unsupported_schema_type:{expected}"] if expected else []


def score_record(case: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    output = row.get("output")
    if not isinstance(output, str):
        raise ValueError(f"{row.get('case_id')} output 无效")
    normalized = _normalize(output)
    required_facts = [str(item) for item in case.get("required_facts", [])]
    literal_hits = [item for item in required_facts if _normalize(item) in normalized]
    hard_fail_reasons: list[str] = []
    if not output.strip():
        hard_fail_reasons.append("empty")
    if any(token in output for token in CONTROL_TOKENS):
        hard_fail_reasons.append("control_token")
    if "<think>" in output.lower() or "</think>" in output.lower():
        hard_fail_reasons.append("think_block")
    if row.get("truncated"):
        hard_fail_reasons.append("truncated")

    json_parse_ok: bool | None = None
    json_schema_ok: bool | None = None
    json_errors: list[str] = []
    if case.get("task") == "structured_output":
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            json_parse_ok = False
            json_schema_ok = False
            json_errors = ["invalid_json"]
        else:
            json_parse_ok = True
            schema = case.get("output_schema")
            if isinstance(schema, Mapping):
                json_errors = _schema_errors(parsed, schema)
                json_schema_ok = not json_errors
            else:
                json_schema_ok = False
                json_errors = ["missing_schema"]
        if not json_schema_ok:
            hard_fail_reasons.append("structured_output_invalid")

    context = row.get("rag", {}).get("context")

    return {
        "arm": row["arm"],
        "case_id": case["id"],
        "task": case["task"],
        "primary_category": case["primary_category"],
        "risk_level": case["risk_level"],
        "hard_fail": bool(hard_fail_reasons),
        "hard_fail_reasons": sorted(set(hard_fail_reasons)),
        "required_fact_count": len(required_facts),
        "literal_fact_hits": len(literal_hits),
        "literal_fact_hit_rate": len(literal_hits) / len(required_facts) if required_facts else None,
        "json_parse_ok": json_parse_ok,
        "json_schema_ok": json_schema_ok,
        "json_errors": json_errors,
        "rag_enabled": bool(row.get("rag", {}).get("enabled")),
        "uses_rag_system_prompt": bool(row.get("rag", {}).get("uses_rag_system_prompt")),
        "context_available": bool(context),
        "context_source": row.get("rag", {}).get("context_source"),
        "evidence_count": len(row.get("rag", {}).get("evidence", [])),
        "insufficient_evidence_statement": any(phrase in output for phrase in INSUFFICIENCY_PHRASES),
        "generated_tokens": row.get("generated_tokens"),
        "finish_reason": row.get("finish_reason"),
        "truncated": bool(row.get("truncated")),
        "latency_seconds": row.get("runtime", {}).get("latency_seconds"),
        "output_chars": len(output),
    }


def _mean(values: Iterable[float | int]) -> float | None:
    rows = list(values)
    return statistics.fmean(rows) if rows else None


def aggregate_scores(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(scores)
    if not rows:
        raise ValueError("评分记录为空")
    structured = [row for row in rows if row["json_schema_ok"] is not None]
    fact_rows = [row for row in rows if row["required_fact_count"]]
    return {
        "records": len(rows),
        "hard_fail_count": sum(bool(row["hard_fail"]) for row in rows),
        "truncated_count": sum(bool(row["truncated"]) for row in rows),
        "uses_rag_system_prompt_count": sum(bool(row["uses_rag_system_prompt"]) for row in rows),
        "context_available_count": sum(bool(row["context_available"]) for row in rows),
        "structured_schema_pass_rate": _mean(bool(row["json_schema_ok"]) for row in structured),
        "literal_fact_hit_rate": _mean(row["literal_fact_hit_rate"] for row in fact_rows),
        "insufficient_evidence_statement_rate": _mean(
            bool(row["insufficient_evidence_statement"]) for row in rows
        ),
        "mean_generated_tokens": _mean(row["generated_tokens"] for row in rows),
        "mean_output_chars": _mean(row["output_chars"] for row in rows),
        "mean_latency_seconds": _mean(row["latency_seconds"] for row in rows),
    }