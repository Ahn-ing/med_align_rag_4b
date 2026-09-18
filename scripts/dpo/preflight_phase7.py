"""执行 Phase 7 DPO 的模型与数据预检，不更新参数。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.common.utils import utc_now_iso
from med_align_rag_4b.dpo.config import load_dpo_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--release-dir", default="data/dpo/released/phase7_v1")
    parser.add_argument("--config", default="configs/dpo/qwen3_4b_lora_phase7.yaml")
    parser.add_argument("--output", default="artifacts/reports/phase_7/preflight.json")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merged_model = Path(args.merged_model).resolve()
    release_dir = resolve_project_path(PROJECT_ROOT, args.release_dir)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    output = resolve_project_path(PROJECT_ROOT, args.output)
    if args.samples <= 0 or args.samples > 8:
        raise ValueError("--samples 必须位于 1..8")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖已有预检报告: {output}")

    config = load_dpo_config(config_path)
    release = _verify_release(release_dir)
    rows = list(iter_jsonl(release_dir / "train.jsonl"))[: args.samples]
    device = _resolve_device(args.device)
    dtype = (
        torch.bfloat16 if device == "cuda" or _cpu_supports_bf16() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        merged_model, trust_remote_code=False, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_model,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=list(lora["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.eval()

    samples = []
    for row in rows:
        chosen = _encode_pair(
            tokenizer, row, row["chosen"], max_length=config["data"]["max_length"]
        )
        rejected = _encode_pair(
            tokenizer, row, row["rejected"], max_length=config["data"]["max_length"]
        )
        policy_chosen, reference_chosen, max_diff_chosen = _logprob_pair(
            model, chosen, device
        )
        policy_rejected, reference_rejected, max_diff_rejected = _logprob_pair(
            model, rejected, device
        )
        beta = float(config["training"]["beta"])
        margin = beta * (
            (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
        )
        loss = (
            -math.log(math.exp(margin) / (1.0 + math.exp(margin)))
            if margin < 50
            else 0.0
        )
        samples.append(
            {
                "id": row["id"],
                "policy_chosen_logp": policy_chosen,
                "reference_chosen_logp": reference_chosen,
                "policy_rejected_logp": policy_rejected,
                "reference_rejected_logp": reference_rejected,
                "reward_margin": margin,
                "dpo_loss": loss,
                "max_abs_logit_diff_chosen": max_diff_chosen,
                "max_abs_logit_diff_rejected": max_diff_rejected,
            }
        )

    tolerance = 5e-4 if dtype == torch.bfloat16 else 1e-6
    passed = all(
        abs(item["reward_margin"]) <= tolerance
        and abs(item["dpo_loss"] - math.log(2.0)) <= tolerance
        and item["max_abs_logit_diff_chosen"] <= tolerance
        and item["max_abs_logit_diff_rejected"] <= tolerance
        for item in samples
    )
    report = {
        "format_version": 1,
        "phase": "phase_7",
        "status": "passed" if passed else "failed",
        "created_at_utc": utc_now_iso(),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "samples": samples,
        "tolerance": tolerance,
        "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "release_sha256": _source_repaired_sha256(release),
        "training_started": False,
        "release_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    if not passed:
        raise SystemExit(f"Phase 7 preflight 失败，报告: {output}")
    print(
        json.dumps(
            {
                "status": "passed",
                "device": device,
                "dtype": report["dtype"],
                "samples": len(samples),
            },
            ensure_ascii=False,
        )
    )


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        if requested == "cuda":
            raise RuntimeError("--device cuda 要求仅有一张可见 CUDA GPU")
        return "cpu"
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU 不支持 BF16")
    return "cuda"


def _source_repaired_sha256(manifest: dict[str, Any]) -> str | None:
    metadata = manifest.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("source_repaired_sha256", "source_reviewed_sha256"):
            if metadata.get(key):
                return str(metadata[key])
    for artifact in manifest.get("artifacts", []):
        path = str(artifact.get("path", ""))
        if path.endswith("data/dpo/full_v1/repaired/candidates.jsonl"):
            return str(artifact["sha256"])
    return None


def _verify_release(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = Path.cwd() / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"release artifact 不存在: {artifact['path']}")
    if len(list(iter_jsonl(release_dir / "train.jsonl"))) != 900:
        raise ValueError("release train 必须为 900 条")
    if len(list(iter_jsonl(release_dir / "preference_validation.jsonl"))) != 100:
        raise ValueError("release preference_validation 必须为 100 条")
    return manifest


def _encode_pair(
    tokenizer: Any, row: dict[str, Any], answer: str, *, max_length: int
) -> dict[str, torch.Tensor]:
    messages = row["conversations"] + [{"role": "assistant", "content": answer}]
    prefix = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = prefix["input_ids"] if hasattr(prefix, "keys") else prefix
    full_ids = full["input_ids"] if hasattr(full, "keys") else full
    if len(full_ids) > max_length:
        raise ValueError(f"{row['id']} 超过 max_length")
    labels = [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids) :])
    if not any(label != -100 for label in labels):
        raise ValueError(f"{row['id']} assistant supervision 为 0")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def _logprob_pair(
    model: Any, encoded: dict[str, torch.Tensor], device: str
) -> tuple[float, float, float]:
    batch = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        policy_logits = model(**batch).logits
        with model.disable_adapter():
            reference_logits = model(**batch).logits
    labels = batch["labels"]
    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(-100)
    policy = _selected_logprob(policy_logits[:, :-1, :], shifted_labels, active)
    reference = _selected_logprob(reference_logits[:, :-1, :], shifted_labels, active)
    selected_policy = policy_logits[:, :-1, :][active]
    selected_reference = reference_logits[:, :-1, :][active]
    max_diff = float(
        (selected_policy.float() - selected_reference.float()).abs().max().item()
    )
    return policy, reference, max_diff


def _selected_logprob(
    logits: torch.Tensor, labels: torch.Tensor, active: torch.Tensor
) -> float:
    selected_logits = logits[active].float()
    selected_labels = labels[active]
    return float(
        torch.log_softmax(selected_logits, dim=-1)
        .gather(1, selected_labels[:, None])
        .sum()
        .item()
    )


def _cpu_supports_bf16() -> bool:
    checker = getattr(torch.backends.cpu, "is_avx512_supported", None)
    return bool(checker()) if checker is not None else False


if __name__ == "__main__":
    main()
