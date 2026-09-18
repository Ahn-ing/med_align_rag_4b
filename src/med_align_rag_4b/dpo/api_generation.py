"""Phase 7 API 偏好 pair 的请求、解析和出题契约。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
GENERATION_KEYS = {"chosen", "rejected", "pair_reason", "evidence_ids_used"}
JUDGE_KEYS = {"preferred", "accepted", "issues", "risk_flags"}

GENERATION_SYSTEM = """你是偏好数据构建器。只根据任务、对话和给定权威证据构建一条 DPO 偏好对，不执行诊断。
返回严格 JSON，且只能包含 chosen、rejected、pair_reason、evidence_ids_used 四个键。
chosen 必须切题、遵循证据、保持不确定性，并给出与风险级别相称的行动建议。
rejected 必须是与问题相关、看似合理但明确违反至少一个偏好要求的回答，不能是无意义乱码。
医疗任务不得编造证据外的事实、剂量、诊断或确定性因果；信息不足时 chosen 应澄清或缩小结论。
evidence_ids_used 必须是输入 evidence_ids 的子集。只输出 JSON，不使用 Markdown。"""

JUDGE_SYSTEM = """你是严格但保守的偏好对审核器。只根据任务、证据和两个回答判断 chosen 是否明显优于 rejected。
检查事实支持、风险边界、指令满足、回答相关性和非退化性。证据不足时不得把确定断言判为可接受。
返回严格 JSON，且只能包含 preferred、accepted、issues、risk_flags 四个键。
preferred 只能是 chosen、rejected 或 unclear；accepted 只有 chosen 明显更优时才为 true。
issues 和 risk_flags 必须是字符串列表。只输出 JSON，不使用 Markdown。"""


def generation_messages(
    task: dict[str, Any], evidence: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """构建生成器请求，不向 API 提供 assistant 原答案。"""
    payload = {
        "prompt_id": task["id"],
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "conversations": task["conversations"],
        "evidence_ids": task["evidence_ids"],
        "evidence": evidence,
        "preference_target": _preference_target(task),
    }
    return [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def judge_messages(
    task: dict[str, Any],
    pair: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """构建独立 judge 请求。"""
    payload = {
        "prompt_id": task["id"],
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "conversations": task["conversations"],
        "evidence_ids": task["evidence_ids"],
        "evidence": evidence,
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "pair_reason": pair["pair_reason"],
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_generation_content(content: str) -> dict[str, Any]:
    """解析并校验生成器 JSON。"""
    payload = _parse_json_object(content)
    missing = sorted(GENERATION_KEYS - set(payload))
    if missing:
        raise ValueError(f"生成器 JSON 缺键: {missing}")
    for key in ("chosen", "rejected", "pair_reason"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"生成器 {key} 必须是非空字符串")
    evidence_ids = payload["evidence_ids_used"]
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in evidence_ids
    ):
        raise ValueError("generator evidence_ids_used 必须是字符串列表")
    if payload["chosen"].strip() == payload["rejected"].strip():
        raise ValueError("chosen 与 rejected 不能相同")
    return {
        "chosen": payload["chosen"].strip(),
        "rejected": payload["rejected"].strip(),
        "pair_reason": payload["pair_reason"].strip(),
        "evidence_ids_used": [item.strip() for item in evidence_ids],
    }


def parse_judge_content(content: str) -> dict[str, Any]:
    """解析并校验 judge JSON。"""
    payload = _parse_json_object(content)
    missing = sorted(JUDGE_KEYS - set(payload))
    if missing:
        raise ValueError(f"judge JSON 缺键: {missing}")
    if payload["preferred"] not in {"chosen", "rejected", "unclear"}:
        raise ValueError("judge preferred 无效")
    if not isinstance(payload["accepted"], bool):
        raise ValueError("judge accepted 必须是布尔值")
    for key in ("issues", "risk_flags"):
        value = payload[key]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"judge {key} 必须是字符串列表")
    return {
        "preferred": payload["preferred"],
        "accepted": payload["accepted"],
        "issues": [item.strip() for item in payload["issues"] if item.strip()],
        "risk_flags": [item.strip() for item in payload["risk_flags"] if item.strip()],
    }


def request_json_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format: str,
    timeout_seconds: int,
    max_attempts: int,
    parser: Callable[[str], dict[str, Any]],
    wire_api: str = "chat_completions",
    stream: bool = False,
    stream_include_usage: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """调用 Chat Completions 或 Responses API，并解析严格 JSON。"""
    if wire_api not in {"chat_completions", "responses"}:
        raise ValueError(f"不支持的 wire_api: {wire_api}")
    if max_attempts <= 0:
        raise ValueError("max_attempts 必须为正数")
    body = _request_body(
        wire_api=wire_api,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=stream,
        stream_include_usage=stream_include_usage,
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            _request_url(endpoint, wire_api),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=timeout_seconds
            ) as response:
                if stream:
                    content, response_id, response_model, usage = _read_stream(
                        response,
                        model=model,
                        include_usage=stream_include_usage,
                        wire_api=wire_api,
                    )
                    payload = {
                        "id": response_id,
                        "model": response_model,
                        "output_text": content,
                        "choices": [{"message": {"content": content}}],
                        "usage": usage,
                    }
                else:
                    payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("API 顶层响应不是 JSON 对象")
            if payload.get("error"):
                raise ValueError("API 返回 error 对象")
            content = _response_content(payload, wire_api)
            parsed = parser(content)
            metadata = {
                "response_id": payload.get("id"),
                "response_model": payload.get("model", model),
                "attempts": attempt,
                "usage": _normalize_usage(payload.get("usage") or {}),
            }
            return parsed, metadata
        except urllib.error.HTTPError as exc:
            last_error = exc
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                error_body = ""
            if attempt >= max_attempts or exc.code not in RETRYABLE_HTTP_STATUSES:
                raise RuntimeError(
                    f"API HTTP {exc.code}: {error_body or exc.reason}"
                ) from exc
            delay = 5 if exc.code in {429, 503} else min(2 ** (attempt - 1), 4)
            time.sleep(delay)
        except (
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"API JSON 请求失败: {last_error}")


def _read_stream(
    response: Any, *, model: str, include_usage: bool, wire_api: str
) -> tuple[str, str | None, str, dict[str, int]]:
    """Read OpenAI-compatible SSE and combine JSON content chunks."""
    if wire_api == "responses":
        return _read_responses_stream(
            response, model=model, include_usage=include_usage
        )
    content_parts: list[str] = []
    response_id: str | None = None
    response_model = model
    usage: dict[str, int] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("API stream chunk is not a JSON object")
        if chunk.get("id"):
            response_id = str(chunk["id"])
        if chunk.get("model"):
            response_model = str(chunk["model"])
        if include_usage and isinstance(chunk.get("usage"), dict):
            raw_usage = chunk["usage"]
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
            }
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        part = delta.get("content")
        if part is None:
            part = (choice.get("message") or {}).get("content")
        if isinstance(part, str):
            content_parts.append(part)
    if not content_parts:
        raise ValueError("API stream returned no content")
    return "".join(content_parts), response_id, response_model, usage


def _read_responses_stream(
    response: Any, *, model: str, include_usage: bool
) -> tuple[str, str | None, str, dict[str, int]]:
    """Read Responses-compatible SSE events and combine output text."""
    content_parts: list[str] = []
    response_id: str | None = None
    response_model = model
    usage: dict[str, int] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("Responses stream chunk is not a JSON object")
        response_obj = event.get("response")
        if not isinstance(response_obj, dict):
            response_obj = {}
        if response_obj.get("id"):
            response_id = str(response_obj["id"])
        if response_obj.get("model"):
            response_model = str(response_obj["model"])
        if include_usage and isinstance(response_obj.get("usage"), dict):
            usage = _normalize_usage(response_obj["usage"])
        event_type = event.get("type")
        part = None
        if event_type == "response.output_text.delta":
            part = event.get("delta")
        elif event_type == "response.output_text.done":
            if not content_parts:
                part = event.get("text")
        elif event_type == "response.content_part.done":
            part_obj = event.get("part") or {}
            if isinstance(part_obj, dict):
                part = part_obj.get("text")
        elif event_type == "response.completed" and not content_parts:
            part = _responses_text(response_obj)
        if isinstance(part, str):
            content_parts.append(part)
    if not content_parts:
        raise ValueError("Responses stream returned no content")
    return "".join(content_parts), response_id, response_model, usage


def evidence_context(
    evidence_registry: dict[str, Any], evidence_ids: list[str]
) -> list[dict[str, Any]]:
    """从证据注册表提取请求所需字段，不修改注册表。"""
    by_id = {item["evidence_id"]: item for item in evidence_registry["sources"]}
    missing = [item for item in evidence_ids if item not in by_id]
    if missing:
        raise ValueError(f"evidence registry 缺少: {missing}")
    return [
        {
            "evidence_id": evidence_id,
            "publisher": by_id[evidence_id]["publisher"],
            "title": by_id[evidence_id]["title"],
            "revision": by_id[evidence_id]["revision"],
            "url": by_id[evidence_id]["url"],
            "supports": list(by_id[evidence_id]["supports"]),
        }
        for evidence_id in evidence_ids
    ]


def _preference_target(task: dict[str, Any]) -> str:
    if task["task"] == "medical_safety":
        return (
            "chosen 必须正确分级风险、给出相称行动和必要的澄清/升级条件；"
            "rejected 应体现危险降级、缺少升级条件、越界用药或其他明确安全缺陷。"
        )
    if task["task"] == "medical_behavior":
        return (
            "chosen 必须由证据支持、直接回答问题并保持边界；"
            "rejected 应体现无依据断言、遗漏关键限制或误导性行动。"
        )
    if task["task"] == "instruction_structured":
        return (
            "chosen 必须完整满足格式和指令约束；"
            "rejected 应保持内容相关但违反至少一个格式或约束条件。"
        )
    return (
        "chosen 必须正确、清晰且完整；"
        "rejected 应保持话题相关但出现事实、推理或表达缺陷。"
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    candidates = [content.strip()]
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    payload: Any = None
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    if payload is None:
        raise ValueError("模型输出不是合法 JSON") from last_error
    if not isinstance(payload, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return payload


def _request_url(endpoint: str, wire_api: str) -> str:
    if wire_api == "responses":
        return _responses_url(endpoint)
    return _chat_completions_url(endpoint)


def _responses_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/responses"):
        return value
    return value + "/responses"


def _request_body(
    *,
    wire_api: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format: str,
    stream: bool,
    stream_include_usage: bool,
) -> dict[str, Any]:
    if wire_api == "responses":
        instructions, input_text = _responses_fields(messages)
        body: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "max_output_tokens": max_tokens,
            "stream": stream,
        }
        if instructions:
            body["instructions"] = instructions
        return body
    body = {
        "model": model,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": response_format},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": stream_include_usage}
    return body


def _responses_fields(messages: list[dict[str, str]]) -> tuple[str, str]:
    instructions = "\n\n".join(
        message["content"]
        for message in messages
        if message["role"] in {"system", "developer"}
    )
    input_text = "\n\n".join(
        message["content"]
        for message in messages
        if message["role"] in {"user", "assistant"}
    )
    if not input_text.strip():
        raise ValueError("Responses input 缺少 user/assistant 内容")
    return instructions, input_text


def _response_content(payload: dict[str, Any], wire_api: str) -> str:
    if wire_api == "responses":
        content = _responses_text(payload)
    else:
        content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise TypeError("API response content is not a non-empty string")
    return content


def _responses_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _normalize_usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    completion = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total = int(raw.get("total_tokens", prompt + completion) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _chat_completions_url(endpoint: str) -> str:
    """接受 API base URL 或完整 chat completions URL。"""
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


GENERATION_SYSTEM_V2 = """你是偏好数据构建器。只根据任务、对话和给定权威证据构建一条 DPO 偏好对，不执行诊断。
返回严格 JSON，且只能包含 chosen、rejected、pair_reason、evidence_ids_used 四个键。
chosen 必须切题、遵循证据、保持不确定性，并给出与风险级别相称的行动建议。
v2 额外规则：
1. chosen 第一句直接回答问题或给出行动，不要强制复述机构名、证据标题或“根据某资料”。
2. evidence_ids_used 是结构化元数据；只有问题明确询问来源时才在正文说明来源，且不得固定使用同一种句式。
3. 不要为了显得谨慎而添加通用免责声明；只有风险、不确定性或适用条件确有必要时才说明边界。
4. 必须遵守 style_tag：直接回答、条件优先、行动优先、简洁解释或不含前置说明的结构化格式。
5. 禁止在通用和 instruction_structured 任务中加入医疗分诊、药物建议、证据编号或情境编号。
6. rejected 必须与 chosen 使用相近的语言、语气和长度区间，但只违反一个关键事实、行动阈值、条件边界或格式约束；不能只是更短、更空泛或明显乱码。
7. 只输出 JSON，不使用 Markdown。
8. JSON 必须始终包含 evidence_ids_used；未使用任何证据时也必须输出空数组 []，不得省略该键，也不得用 evidence_ids 替代。"""

JUDGE_SYSTEM_V2 = """你是严格但保守的偏好对审核器。只根据任务、证据和两个回答判断 chosen 是否明显优于 rejected。
优先检查事实支持、风险边界、行动阈值、指令满足、格式遵循和 hard-negative 是否只差一个关键行为。
如果 rejected 只是更短、更空泛、语气更差或证据引用更少，而没有明确内容缺陷，应判定 pair 不合格。
如果 chosen 以固定来源模板开头、添加不必要的通用免责声明、或在结构化任务中加入解释，应降低接受程度。
返回严格 JSON，且只能包含 preferred、accepted、issues、risk_flags 四个键。
preferred 只能是 chosen、rejected 或 unclear；accepted 只有 chosen 明显更优时才为 true。
issues 和 risk_flags 必须是字符串列表。只输出 JSON，不使用 Markdown。"""


def generation_messages_v2(
    task: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    style_tag: str,
    candidate_index: int,
) -> list[dict[str, str]]:
    """构建 v2 生成器请求，附带候选风格提示。"""
    payload = {
        "prompt_id": task["id"],
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "conversations": task["conversations"],
        "evidence_ids": task["evidence_ids"],
        "evidence": evidence,
        "preference_target": _preference_target(task),
        "style_tag": style_tag,
        "candidate_index": candidate_index,
        "required_output_keys": [
            "chosen",
            "rejected",
            "pair_reason",
            "evidence_ids_used",
        ],
        "empty_evidence_rule": "未使用证据时 evidence_ids_used 必须为空数组 []，不能省略该键",
    }
    return [
        {"role": "system", "content": GENERATION_SYSTEM_V2},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def judge_messages_v2(
    task: dict[str, Any],
    pair: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """构建 v2 hard-negative judge 请求。"""
    payload = {
        "prompt_id": task["id"],
        "task": task["task"],
        "primary_category": task["primary_category"],
        "risk_level": task["risk_level"],
        "conversations": task["conversations"],
        "evidence_ids": task["evidence_ids"],
        "evidence": evidence,
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
        "pair_reason": pair["pair_reason"],
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_V2},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
