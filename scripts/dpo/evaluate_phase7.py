"""在 preference validation 上评测已训练 Phase 7 DPO adapter，不更新参数。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.common.utils import utc_now_iso
from med_align_rag_4b.dpo.config import load_dpo_config
from med_align_rag_4b.dpo.loss import dpo_log_sigmoid_loss
from med_align_rag_4b.sft.checkpoints import validate_checkpoint
from med_align_rag_4b.sft.run_manifest import write_manifest_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", default="data/dpo/released/phase7_v1")
    parser.add_argument("--config", default="configs/dpo/qwen3_4b_lora_phase7.yaml")
    parser.add_argument(
        "--output", default="artifacts/reports/phase_7/dpo_formal_evaluation.json"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    merged_model = args.merged_model.resolve()
    run_dir = resolve_project_path(PROJECT_ROOT, args.run_dir)
    release_dir = resolve_project_path(PROJECT_ROOT, args.release_dir)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    output = resolve_project_path(PROJECT_ROOT, args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖已有评测报告: {output}")

    config = load_dpo_config(config_path)
    manifest, adapter_dir = _verify_run(run_dir, release_dir, config_path)
    rows = list(iter_jsonl(release_dir / "preference_validation.jsonl"))
    if len(rows) != config["data"]["counts"]["preference_validation"]:
        raise ValueError("preference validation 数量与冻结配置不一致")

    device = _resolve_device(args.device)
    dtype = (
        torch.bfloat16 if device == "cuda" or _cpu_supports_bf16() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir, trust_remote_code=False, local_files_only=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        merged_model,
        dtype=dtype,
        attn_implementation=config["model"]["attn_implementation"],
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
    ).to(device)
    model.config.use_cache = False
    model.eval()

    beta = float(config["training"]["beta"])
    samples = []
    for row in rows:
        chosen = _encode_answer(
            tokenizer, row, row["chosen"], max_length=config["data"]["max_length"]
        )
        rejected = _encode_answer(
            tokenizer, row, row["rejected"], max_length=config["data"]["max_length"]
        )
        policy_chosen = _sequence_logprob(model, chosen, device=device, reference=False)
        reference_chosen = _sequence_logprob(
            model, chosen, device=device, reference=True
        )
        policy_rejected = _sequence_logprob(
            model, rejected, device=device, reference=False
        )
        reference_rejected = _sequence_logprob(
            model, rejected, device=device, reference=True
        )
        losses, chosen_reward, rejected_reward, margins = dpo_log_sigmoid_loss(
            torch.tensor([policy_chosen], dtype=torch.float32),
            torch.tensor([policy_rejected], dtype=torch.float32),
            torch.tensor([reference_chosen], dtype=torch.float32),
            torch.tensor([reference_rejected], dtype=torch.float32),
            beta=beta,
        )
        samples.append(
            {
                "id": row["id"],
                "policy_chosen_logp": policy_chosen,
                "reference_chosen_logp": reference_chosen,
                "policy_rejected_logp": policy_rejected,
                "reference_rejected_logp": reference_rejected,
                "chosen_reward": float(chosen_reward[0]),
                "rejected_reward": float(rejected_reward[0]),
                "reward_margin": float(margins[0]),
                "dpo_loss": float(losses[0]),
            }
        )

    metrics = _summarize(samples)
    report = {
        "format_version": 1,
        "phase": "phase_7",
        "kind": "phase_7_dpo_preference_evaluation",
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "run": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "adapter_weight_sha256": manifest["artifacts"]["final_adapter_weight_sha256"],
        "adapter_config_sha256": manifest["artifacts"]["final_adapter_config_sha256"],
        "config_sha256": sha256_file(config_path),
        "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "beta": beta,
        "counts": {"preference_validation": len(samples)},
        "metrics": metrics,
        "samples": samples,
        "limitations": [
            "preference validation 仅覆盖本项目构建的偏好对，不代表临床效果或安全性",
            "reference 为 merged SFT 的 adapter-disabled 路径，不读取冻结 test 或 medical safety holdout",
        ],
    }
    if not all(
        math.isfinite(value)
        for key, value in metrics.items()
        if key != "preference_accuracy"
    ):
        raise ValueError("评测指标出现非有限值")
    write_manifest_atomic(output, report)
    print(json.dumps({"status": "completed", "metrics": metrics}, ensure_ascii=False))


def _verify_run(
    run_dir: Path, release_dir: Path, config_path: Path
) -> tuple[dict[str, Any], Path]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("run manifest 不是 completed")
    if manifest.get("experiment_type") != "formal_dpo":
        raise ValueError("run manifest 不是 formal_dpo")
    contract = manifest.get("checkpoint_contract") or {}
    if contract.get("expected_steps") != manifest.get("last_completed_step"):
        raise ValueError("完成步数与 checkpoint contract 不一致")
    if not config_path.is_file():
        raise FileNotFoundError("当前冻结配置不存在")
    if not (release_dir / "manifest.json").is_file():
        raise FileNotFoundError("release manifest 不存在")
    if not (release_dir / "train.jsonl").is_file():
        raise FileNotFoundError("train 数据不存在")
    if not (release_dir / "preference_validation.jsonl").is_file():
        raise FileNotFoundError("preference validation 数据不存在")

    checkpoints = sorted(
        run_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if not checkpoints:
        raise ValueError("formal run 缺少 checkpoint")
    for checkpoint in checkpoints:
        validate_checkpoint(checkpoint, contract)

    adapter_dir = run_dir / "final_adapter"
    artifact = manifest.get("artifacts") or {}
    weight = adapter_dir / str(artifact.get("final_adapter_weight", ""))
    if not weight.is_file():
        raise FileNotFoundError("final adapter 权重不存在")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError("final adapter 配置不存在")
    return manifest, adapter_dir


def _encode_answer(
    tokenizer: Any, row: dict[str, Any], answer: str, *, max_length: int
) -> dict[str, torch.Tensor]:
    prompt_messages = row["conversations"]
    answer_messages = prompt_messages + [{"role": "assistant", "content": answer}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        answer_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        enable_thinking=False,
    )
    prompt_ids = _flat_ids(prompt["input_ids"])
    full_ids = _flat_ids(full["input_ids"])
    if len(full_ids) > max_length:
        raise ValueError(f"{row['id']} 超过 max_length")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{row['id']} 的 prompt/answer token 前缀不一致")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError(f"{row['id']} assistant supervision 为 0")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def _flat_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _sequence_logprob(
    model: PeftModel, encoded: dict[str, torch.Tensor], *, device: str, reference: bool
) -> float:
    batch = {
        name: value.to(device)
        for name, value in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }
    context = model.disable_adapter() if reference else _null_context()
    with torch.no_grad(), context:
        logits = model(**batch).logits
    labels = encoded["labels"].to(device)
    shifted_labels = labels[:, 1:]
    active = shifted_labels.ne(-100)
    selected_logits = logits[:, :-1, :][active].float()
    selected_labels = shifted_labels[active]
    value = (
        torch.log_softmax(selected_logits, dim=-1)
        .gather(1, selected_labels[:, None])
        .sum()
        .item()
    )
    if not math.isfinite(float(value)):
        raise ValueError("sequence log-probability 非有限")
    return float(value)


def _null_context() -> Any:
    from contextlib import nullcontext

    return nullcontext()


def _summarize(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("评测样本为空")
    margins = [float(item["reward_margin"]) for item in samples]
    losses = [float(item["dpo_loss"]) for item in samples]
    chosen_rewards = [float(item["chosen_reward"]) for item in samples]
    rejected_rewards = [float(item["rejected_reward"]) for item in samples]
    positive = sum(margin > 0 for margin in margins)
    return {
        "mean_dpo_loss": statistics.fmean(losses),
        "mean_reward_margin": statistics.fmean(margins),
        "median_reward_margin": statistics.median(margins),
        "min_reward_margin": min(margins),
        "max_reward_margin": max(margins),
        "mean_chosen_reward": statistics.fmean(chosen_rewards),
        "mean_rejected_reward": statistics.fmean(rejected_rewards),
        "positive_margin_count": positive,
        "preference_accuracy": positive / len(margins),
    }


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


def _cpu_supports_bf16() -> bool:
    checker = getattr(torch.backends.cpu, "is_avx512_supported", None)
    return bool(checker()) if checker is not None else False


if __name__ == "__main__":
    main()
