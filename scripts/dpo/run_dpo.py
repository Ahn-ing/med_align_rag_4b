"""运行 Phase 7 DPO 训练，支持 smoke 与 formal_dpo 两种模式。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import iter_jsonl
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.config import (
    load_dpo_config,
    require_frozen_training_parameters,
)
from med_align_rag_4b.dpo.data import to_trl_record
from med_align_rag_4b.dpo.manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)
from med_align_rag_4b.sft.checkpoints import write_checkpoint_contract
from med_align_rag_4b.sft.training import CudaMemoryCallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/dpo/qwen3_4b_lora_phase7.yaml")
    parser.add_argument(
        "--experiment-type", choices=("smoke", "formal_dpo"), required=True
    )
    parser.add_argument("--data-release", default="data/dpo/released/phase7_v1")
    parser.add_argument(
        "--preflight-report", default="artifacts/reports/phase_7/preflight.json"
    )
    parser.add_argument("--micro-batch", type=int, choices=(1, 2, 4))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="online"
    )
    args = parser.parse_args(argv)
    if args.experiment_type == "smoke":
        if args.max_steps is None or args.max_steps <= 0:
            parser.error("smoke 必须提供正数 --max-steps")
    elif args.max_steps is not None:
        parser.error("formal_dpo 必须使用配置中的 epochs，不能传 --max-steps")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_path = args.model_path.resolve()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    release_dir = resolve_project_path(PROJECT_ROOT, args.data_release)
    preflight_path = resolve_project_path(PROJECT_ROOT, args.preflight_report)
    output = _project_output(args.output_dir)
    if output.exists() and args.resume_from_checkpoint is None:
        raise FileExistsError(f"输出目录已存在，不覆盖: {output}")

    config = load_dpo_config(config_path)
    require_frozen_training_parameters(config)
    release = _verify_release(release_dir)
    _verify_preflight(preflight_path, release_dir, release)
    train_rows = list(iter_jsonl(release_dir / "train.jsonl"))
    eval_rows = list(iter_jsonl(release_dir / "preference_validation.jsonl"))
    train_dataset = Dataset.from_list([to_trl_record(row) for row in train_rows])
    eval_dataset = Dataset.from_list([to_trl_record(row) for row in eval_rows])

    device = _require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=config["model"]["attn_implementation"],
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=list(lora["target_modules"]),
    )

    training = config["training"]
    micro_batch = int(args.micro_batch or training["per_device_train_batch_size"])
    accumulation = _gradient_accumulation(training, micro_batch)
    expected_steps = (
        args.max_steps
        if args.experiment_type == "smoke"
        else math.ceil(len(train_rows) / training["global_batch_size"])
        * training["epochs"]
    )
    warmup_steps = math.ceil(expected_steps * training["warmup_ratio"])
    output.mkdir(parents=True, exist_ok=args.resume_from_checkpoint is not None)
    contract = {
        "phase": "phase_7",
        "experiment_type": args.experiment_type,
        "model_path_source": "runtime_local_path",
        "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
        "train_sha256": sha256_file(release_dir / "train.jsonl"),
        "validation_sha256": sha256_file(release_dir / "preference_validation.jsonl"),
        "config_sha256": sha256_file(config_path),
        "expected_steps": expected_steps,
    }
    manifest = initial_manifest(
        experiment_type=args.experiment_type,
        payload={
            "model": {"source": "runtime_local_path", "dtype": "bfloat16"},
            "data": {"train": len(train_rows), "preference_validation": len(eval_rows)},
            "reference": {
                "strategy": "fresh_lora_on_merged_sft_with_disabled_reference_adapter"
            },
            "lora": lora,
            "training": training,
            "runtime": _runtime_info(),
            "checkpoint_contract": contract,
            "wandb_mode": args.wandb_mode,
        },
    )
    manifest_path = output / "run_manifest.json"
    write_manifest_atomic(manifest_path, manifest)

    dpo_config = DPOConfig(
        output_dir=output.relative_to(PROJECT_ROOT).as_posix(),
        max_length=config["data"]["max_length"],
        beta=training["beta"],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["epochs"]
        if args.experiment_type == "formal_dpo"
        else 1.0,
        max_steps=args.max_steps if args.experiment_type == "smoke" else -1,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=training["bf16"],
        tf32=training["tf32"],
        optim="adamw_torch_fused",
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        logging_steps=training["logging_steps"],
        eval_strategy="steps",
        eval_steps=training["evaluation_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        seed=training["seed"],
        data_seed=training["data_seed"],
        report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
        run_name=f"{args.experiment_type}-{output.name}",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        precompute_ref_log_probs=False,
        loss_type=["sigmoid"],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[CudaMemoryCallback(), _CheckpointContractCallback(output, contract)],
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    for checkpoint in output.glob("checkpoint-*"):
        _sanitize_adapter_config(checkpoint)
        write_checkpoint_contract(checkpoint, contract)
    final_adapter = output / "final_adapter"
    trainer.save_model(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    _sanitize_adapter_config(final_adapter)
    adapter_weight = next(
        path
        for path in (
            final_adapter / "adapter_model.safetensors",
            final_adapter / "adapter_model.bin",
        )
        if path.is_file()
    )
    completed = finish_manifest(
        manifest,
        "completed",
        last_completed_step=int(trainer.state.global_step),
        completed_epochs=float(trainer.state.epoch or 0.0),
        final_metrics=dict(result.metrics),
        periodic_evaluations=[
            {"step": int(item["step"]), "eval_loss": float(item["eval_loss"])}
            for item in trainer.state.log_history
            if "step" in item
            and "eval_loss" in item
            and int(item["step"]) < expected_steps
        ],
        artifacts={
            "final_adapter_weight": adapter_weight.name,
            "final_adapter_weight_sha256": sha256_file(adapter_weight),
            "final_adapter_config_sha256": sha256_file(
                final_adapter / "adapter_config.json"
            ),
        },
    )
    write_manifest_atomic(manifest_path, completed)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": output.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )


class _CheckpointContractCallback(TrainerCallback):
    def __init__(self, output_dir: Path, contract: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.contract = contract

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        checkpoint = self.output_dir / f"checkpoint-{int(state.global_step)}"
        write_checkpoint_contract(checkpoint, self.contract)


def _project_output(value: Path) -> Path:
    output = (
        value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()
    )
    allowed = (PROJECT_ROOT / "artifacts/runs/phase_7").resolve()
    try:
        relative = output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("output-dir 必须位于 artifacts/runs/phase_7 内") from exc
    if relative == Path("."):
        raise ValueError("output-dir 必须包含独立 run name")
    return output


def _require_cuda() -> str:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("DPO 训练要求 CUDA 可见且仅有一张 GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16")
    return "cuda"


def _verify_release(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"release artifact 不存在: {artifact['path']}")
    metadata = manifest.get("metadata") or {}
    if metadata.get("training_allowed") is not True:
        raise ValueError("release 未标记 training_allowed=true")
    if metadata.get("release_allowed") is not False:
        raise ValueError("release_allowed 必须为 false")
    if len(list(iter_jsonl(release_dir / "train.jsonl"))) != 900:
        raise ValueError("release train 必须为 900 条")
    if len(list(iter_jsonl(release_dir / "preference_validation.jsonl"))) != 100:
        raise ValueError("release preference_validation 必须为 100 条")
    return manifest


def _verify_preflight(
    preflight_path: Path, release_dir: Path, release: dict[str, Any]
) -> None:
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("preflight report 状态不是 passed")
    if not (release_dir / "manifest.json").is_file():
        raise FileNotFoundError("release manifest 不存在")
    metadata = release.get("metadata") or {}
    source_sha = metadata.get("source_repaired_sha256") or metadata.get("source_reviewed_sha256")
    if source_sha and report.get("release_sha256") not in {source_sha, None}:
        raise ValueError("preflight report 与 release 数据哈希不匹配")


def _gradient_accumulation(training: dict[str, Any], micro_batch: int) -> int:
    global_batch = int(training["global_batch_size"])
    if global_batch % micro_batch:
        raise ValueError("global_batch_size 必须整除 micro-batch")
    return global_batch // micro_batch


def _runtime_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"python": platform.python_version(), "torch": torch.__version__}
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "unknown",
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "transformers",
                "trl",
                "peft",
                "accelerate",
                "datasets",
                "wandb",
            )
        },
    }


def _sanitize_adapter_config(directory: Path) -> None:
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "Qwen/Qwen3-4B"
    config["revision"] = None
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = directory / "README.md"
    if readme.is_file():
        lines = []
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.startswith("- base_model:"):
                line = "- base_model: Qwen/Qwen3-4B"
            elif line.startswith("- base_model:adapter:"):
                line = "- base_model:adapter: Qwen/Qwen3-4B"
            lines.append(line)
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
