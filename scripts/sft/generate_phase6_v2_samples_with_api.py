"""用 API 从权威证据任务生成不超过 10% 的 SFT v2 补充候选。"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from med_align_rag_4b.common.hashing import sha256_file, sha256_text
from med_align_rag_4b.common.jsonl import iter_jsonl, write_jsonl_atomic
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer
from med_align_rag_4b.sft.v2_quality import (
    API_MEDICAL_TOKEN_MAXIMUM,
    API_MEDICAL_TOKEN_MINIMUM,
    API_MEDICAL_TOKEN_TARGET,
    automatic_quality_flags,
    medical_category,
    medical_risk,
    normalize_match,
    template_skeleton,
)

RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


SYSTEM = """根据任务给出的 supported_claims 构建一条自然中文医疗 SFT 候选，并重点回答 focus_claim。
可以围绕证据作合理扩充，包括通俗解释、必要上下文和与风险相称的简短行动建议；扩充内容必须
与证据一致、不过度确定，也不能改变证据原意。具体药物剂量、个体诊断、精确时间窗、额外治疗
方案或具体因果机制必须有明确证据，不能凭常识补写。信息不足时缩小问题或明确不确定性。
回答必须聚焦用户问题和 focus_claim；不得为了增加长度而附加同一来源中与问题无关的其他事实。
避免固定字段、固定开场、固定结尾和模板化免责声明。只输出严格 JSON。"""


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.offset < 0:
        raise ValueError("--offset 不得为负数")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正数")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts 必须为正数")
    if not args.tasks.is_file():
        raise FileNotFoundError(args.tasks)
    task_count = args.limit
    if task_count is None:
        tasks_manifest_path = args.tasks.with_suffix(".manifest.json")
        if not tasks_manifest_path.is_file():
            raise FileNotFoundError(
                f"未传入 --limit 且任务 manifest 不存在: {tasks_manifest_path}"
            )
        tasks_manifest = json.loads(tasks_manifest_path.read_text(encoding="utf-8"))
        task_count = tasks_manifest.get("records")
        if not isinstance(task_count, int) or task_count < 0:
            raise ValueError(
                f"任务 manifest 的 records 必须为非负整数: {tasks_manifest_path}"
            )
        task_count -= args.offset
        if task_count < 0:
            raise ValueError(
                f"任务 manifest 的 records 减去 offset 后必须为非负整数: {tasks_manifest_path}"
            )
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖已有 API 候选: {args.output}")
    manifest_path = args.output.with_suffix(".manifest.json")
    rejection_path = args.output.with_suffix(".rejections.jsonl")
    if manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 API manifest: {manifest_path}")
    if rejection_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 API 拒绝记录: {rejection_path}")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"环境变量 {args.api_key_env} 未设置")
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(args.tokenizer_path)
    tokenizer = QwenAssistantTokenizer(args.tokenizer_path, max_length=2048)
    accepted_rows: list[dict[str, object]] = []
    rejection_rows: list[dict[str, object]] = []
    rejected = processed = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    with tqdm(
        total=task_count,
        desc="生成 API 候选",
        unit="task",
    ) as progress:
        for task_index, task in enumerate(iter_jsonl(args.tasks)):
            if task_index < args.offset:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            progress.update(1)
            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
                ],
                "thinking": {"type": "disabled"},
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            }
            request = urllib.request.Request(
                args.endpoint.rstrip("/") + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            payload = None
            last_error: Exception | None = None
            last_http_status: int | None = None
            last_error_body: str | None = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
                        candidate_payload = json.loads(response.read().decode("utf-8"))
                    if not isinstance(candidate_payload, dict):
                        raise TypeError("API 顶层响应不是 JSON 对象")
                    content = candidate_payload["choices"][0]["message"]["content"]
                    if not isinstance(content, str):
                        raise TypeError("API message.content 不是字符串")
                    payload = candidate_payload
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    last_http_status = exc.code
                    try:
                        last_error_body = exc.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        last_error_body = None
                    if attempt < args.max_attempts and exc.code in RETRYABLE_HTTP_STATUSES:
                        time.sleep(min(2 ** (attempt - 1), 4))
                    else:
                        break
                except (
                    urllib.error.URLError,
                    ConnectionError,
                    TimeoutError,
                    json.JSONDecodeError,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as exc:
                    last_error = exc
                    if attempt < args.max_attempts:
                        time.sleep(min(2 ** (attempt - 1), 4))
            if payload is None:
                rejected += 1
                rejection_rows.append(
                    {
                        "task_id": task["task_id"],
                        "reasons": ["api_response_failed_after_retries"],
                        "error_type": type(last_error).__name__,
                        "http_status": last_http_status,
                        "error_detail": last_error_body,
                    }
                )
                continue
            api_usage = payload.get("usage") or {}
            if not isinstance(api_usage, dict):
                api_usage = {}
            for name in usage:
                usage[name] += int(api_usage.get(name, 0) or 0)
            generated = json.loads(payload["choices"][0]["message"]["content"])
            question, answer = generated.get("question"), generated.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                rejected += 1
                rejection_rows.append(
                    {"task_id": task["task_id"], "reasons": ["invalid_json_fields"]}
                )
                continue
            question, answer = question.strip(), answer.strip()
            flags = automatic_quality_flags(question, answer)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            token_report = tokenizer.analyze(messages)
            assistant_tokens = token_report["assistant_supervised_tokens"]
            question_key = normalize_match(question)
            answer_key = template_skeleton(answer)
            reasons = list(flags)
            if token_report["overlength"]:
                reasons.append("conversation_over_2048_tokens")
            if not 24 <= assistant_tokens <= 768:
                reasons.append("assistant_tokens_outside_24_768")
            if reasons:
                rejected += 1
                rejection_rows.append(
                    {
                        "task_id": task["task_id"],
                        "question": question,
                        "answer": answer,
                        "assistant_tokens": assistant_tokens,
                        "reasons": sorted(set(reasons)),
                    }
                )
                continue
            if question_key in seen_questions or answer_key in seen_answers:
                rejected += 1
                rejection_rows.append(
                    {
                        "task_id": task["task_id"],
                        "question": question,
                        "answer": answer,
                        "assistant_tokens": assistant_tokens,
                        "reasons": ["duplicate_generated_question_or_answer_skeleton"],
                    }
                )
                continue
            current_tokens = sum(int(row["assistant_tokens"]) for row in accepted_rows)
            if current_tokens + assistant_tokens > API_MEDICAL_TOKEN_MAXIMUM:
                rejection_rows.append(
                    {
                        "task_id": task["task_id"],
                        "assistant_tokens": assistant_tokens,
                        "reasons": ["api_medical_token_maximum_would_be_exceeded"],
                    }
                )
                rejected += 1
                break
            seen_questions.add(question_key)
            seen_answers.add(answer_key)
            row = {
                "id": task["task_id"],
                "messages": messages,
                "domain": "medical",
                "primary_category": task["primary_category"],
                "risk_level": medical_risk(
                    medical_category(question, answer), question, answer
                ),
                "assistant_tokens": assistant_tokens,
                "medical_evidence": [task["evidence_id"]],
                "review": {
                    "status": "api_generated_pending_evidence_audit",
                    "evidence_checked": False,
                },
                "provenance": {
                    "generation_method": "evidence_constrained_api",
                    "model": args.model,
                    "reference_record_id": task.get("reference_record_id"),
                },
                "release_allowed": False,
            }
            accepted_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output, accepted_rows)
    write_jsonl_atomic(rejection_path, rejection_rows)
    manifest = {
        "format_version": 1,
        "status": "generated_pending_evidence_audit",
        "training_allowed": False,
        "model": args.model,
        "generation_prompt_sha256": sha256_text(SYSTEM),
        "tasks_sha256": sha256_file(args.tasks),
        "records_processed": processed,
        "task_offset": args.offset,
        "mechanically_passed": len(accepted_rows),
        "mechanically_rejected": rejected,
        "assistant_tokens": sum(int(row["assistant_tokens"]) for row in accepted_rows),
        "api_medical_token_budget": {
            "minimum": API_MEDICAL_TOKEN_MINIMUM,
            "target": API_MEDICAL_TOKEN_TARGET,
            "maximum": API_MEDICAL_TOKEN_MAXIMUM,
        },
        "usage": usage,
        "output_sha256": sha256_file(args.output),
        "rejections_sha256": sha256_file(rejection_path),
        "tokenizer": {"model_id": "Qwen/Qwen3-4B", "local_path_persisted": False},
        "endpoint_persisted": False,
    }
    _write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "mechanically_passed": len(accepted_rows),
                "rejected": rejected,
            }
        )
    )


if __name__ == "__main__":
    main()
