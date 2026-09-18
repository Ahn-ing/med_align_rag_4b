"""Phase 7 v2 preference style diversity gates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

STYLE_TAGS = {
    "medical_behavior": (
        "direct_answer",
        "condition_first",
        "plain_language",
        "evidence_context",
    ),
    "medical_safety": (
        "action_first",
        "risk_triage",
        "clarify_before_advice",
        "followup_boundary",
    ),
    "general": ("direct_explanation", "concise_steps", "analogy", "conditional_note"),
    "instruction_structured": (
        "strict_format",
        "minimal_format",
        "validated_schema",
        "no_preamble",
    ),
}
BANNED_OPENINGS = ("根据世界卫生组织的资料", "根据世界卫生组织")
FIXED_PHRASES = ("建议咨询医疗专业人员", "建议咨询医生")
STYLE_THRESHOLDS = {
    "who_rate": 0.15,
    "citation_rate": 0.10,
    "citation_opening_rate": 0.05,
    "fixed_phrase_rate": 0.10,
    "max_first_12_share": 0.10,
}


def first_12(text: str) -> str:
    """Return normalized first 12 characters for opening repetition checks."""
    return re.sub(r"\s+", "", text.strip())[:12]


def first_sentence(text: str) -> str:
    """Return the first sentence for fixed-pattern checks."""
    return re.split(r"[。！？!?]\s*", text.strip(), maxsplit=1)[0]


def style_tag_for(task: str, candidate_index: int) -> str:
    """Choose a deterministic style tag for a task/candidate pair."""
    tags = STYLE_TAGS[task]
    return tags[candidate_index % len(tags)]


def style_metrics(texts: list[str]) -> dict[str, Any]:
    """Compute text-free style distribution metrics."""
    if not texts:
        raise ValueError("style_metrics requires non-empty texts")
    normalized = [text.strip() for text in texts]
    prefixes = Counter(first_12(text) for text in normalized)
    total = len(normalized)
    return {
        "count": total,
        "who_rate": sum("世界卫生组织" in text for text in normalized) / total,
        "citation_rate": sum("根据" in text for text in normalized) / total,
        "citation_opening_rate": sum(
            text.startswith(BANNED_OPENINGS) for text in normalized
        )
        / total,
        "fixed_phrase_rate": sum(
            any(phrase in text for phrase in FIXED_PHRASES) for text in normalized
        )
        / total,
        "unique_first_12": len(prefixes),
        "max_first_12_share": max(prefixes.values()) / total,
        "top_first_12": prefixes.most_common(8),
    }


def style_gate_failures(metrics: dict[str, Any]) -> list[str]:
    """Return metric names exceeding the frozen style thresholds."""
    failures = []
    for key, threshold in STYLE_THRESHOLDS.items():
        if float(metrics[key]) > threshold:
            failures.append(key)
    return sorted(failures)


def template_penalty(text: str, prior_texts: list[str]) -> float:
    """Return a lightweight style penalty for candidate selection."""
    normalized = text.strip()
    penalty = 0.0
    if normalized.startswith(BANNED_OPENINGS):
        penalty += 4.0
    if "世界卫生组织" in normalized:
        penalty += 1.5
    if any(phrase in normalized for phrase in FIXED_PHRASES):
        penalty += 1.0
    if "根据" in normalized:
        penalty += 0.5
    if first_sentence(normalized) in {first_sentence(item) for item in prior_texts}:
        penalty += 2.0
    if first_12(normalized) in {first_12(item) for item in prior_texts}:
        penalty += 2.0
    return penalty


def pair_penalty(pair: dict[str, Any], prior_chosen: list[str]) -> float:
    """Penalize style repetition and extreme chosen/rejected length imbalance."""
    chosen = str(pair["chosen"])
    rejected = str(pair["rejected"])
    length_scale = max(len(chosen), len(rejected), 1)
    length_penalty = abs(len(chosen) - len(rejected)) / length_scale
    return template_penalty(chosen, prior_chosen) + 0.5 * length_penalty
