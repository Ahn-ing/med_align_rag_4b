"""运行 Phase 5 工程训练或 Phase 6 两轮正式 SFT。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
import yaml

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.sft.adapter_validation import (
    validate_target_modules,
    validate_trainable_parameters,
)
from med_align_rag_4b.sft.checkpoints import (
    validate_checkpoint,
    write_checkpoint_contract,
)
from med_align_rag_4b.sft.collator import AssistantOnlyCollator
from med_align_rag_4b.sft.config import (
    derive_gradient_accumulation,
    derive_optimizer_steps,
    derive_warmup_steps,
    load_sft_config,
    sanitized_config,
    validate_phase6_config,
)
from med_align_rag_4b.sft.data import (
    load_and_encode,
    load_eval_and_encode,
    validate_phase4_release,
    validate_phase5_eval,
    validate_phase6_release,
    validate_phase6_validation,
)
from med_align_rag_4b.sft.run_manifest import (
    finish_manifest,
    initial_manifest,
    write_manifest_atomic,
)
from med_align_rag_4b.sft.tokenization import QwenAssistantTokenizer
from med_align_rag_4b.sft.training import (
    CheckpointContractCallback,
    CudaMemoryCallback,
    EncodedDataset,
    TokenNormalizedTrainer,
    evaluate_active_token_loss,
    inspect_local_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-type", choices=("smoke", "short_run", "formal_train"), required=True
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch", type=int, choices=(2, 4), default=4)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--data-release", type=Path)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="online")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.experiment_type != "formal_train" and args.data_release is not None:
        parser.error("--data-release 只适用于 formal_train")
    if args.experiment_type == "formal_train":
        if args.epochs != 2 or args.max_steps is not None:
            parser.error("formal_train 必须且只能使用 --epochs 2")
        if args.eval_data is None:
            parser.error("formal_train 必须提供冻结的完整 --eval-data")
    else:
        if args.max_steps is None or args.max_steps <= 0 or args.epochs is not None:
            parser.error("Phase 5 必须且只能使用正数 --max-steps")
    return args


def project_output(root: Path, value: Path, experiment_type: str) -> Path:
    output = (root / value).resolve() if not value.is_absolute() else value.resolve()
    phase = "phase_6" if experiment_type == "formal_train" else "phase_5"
    allowed = (root / f"artifacts/runs/{phase}").resolve()
    try:
        relative = output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"output-dir 必须位于 artifacts/runs/{phase} 内") from exc
    if relative == Path("."):
        raise ValueError("output-dir 必须包含独立 run name")
    return output


def validate_runtime(phase: str = "Phase 5") -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"{phase} 要求 CUDA 可见且仅有一张 GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda or "unknown", "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "peft", "accelerate", "wandb")
        },
    }


def verify_tokenizer_files(model_path: Path, release: dict[str, Any]) -> None:
    """确认运行 tokenizer 与 Phase 4 冻结 tokenizer 完全一致。"""
    for name, expected in release["tokenizer_files"].items():
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"本地 tokenizer 文件不存在: {name}")


def checkpoint_contract(
    config: dict[str, Any], data_sha256: str, tokenizer_revision: str,
    model_revision: str, tokenizer_files: dict[str, str], *, phase: str = "phase_5",
) -> dict[str, Any]:
    result = {
        "model_id": config["model"]["model_id"], "tokenizer_revision": tokenizer_revision,
        "model_revision": model_revision, "tokenizer_files": tokenizer_files,
        "data_sha256": data_sha256, "lora": config["lora"],
        "max_length": config["data"]["max_length"],
        "global_batch_size": config["training"]["global_batch_size"],
    }
    if phase == "phase_6":
        result["phase"] = phase
    return result


def sanitize_adapter_config(directory: Path) -> None:
    path = directory / "adapter_config.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["base_model_name_or_path"] = "Qwen/Qwen3-4B"
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    readme = directory / "README.md"
    if readme.is_file():
        lines = readme.read_text(encoding="utf-8").splitlines()
        sanitized = []
        for line in lines:
            if line.startswith("base_model: "):
                line = "base_model: Qwen/Qwen3-4B"
            elif line.startswith("- base_model:adapter:"):
                line = "- base_model:adapter:Qwen/Qwen3-4B"
            sanitized.append(line)
        readme.write_text("\n".join(sanitized) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    phase = "phase_6" if args.experiment_type == "formal_train" else "phase_5"
    output = project_output(root, args.output_dir, args.experiment_type)
    selected_config = args.config or Path(
        "configs/sft/qwen3_4b_lora_phase6.yaml"
        if phase == "phase_6" else "configs/sft/qwen3_4b_lora.yaml"
    )
    config_path = root / selected_config if not selected_config.is_absolute() else selected_config
    config = load_sft_config(config_path)
    if phase == "phase_6":
        validate_phase6_config(config)
    config["training"]["per_device_train_batch_size"] = args.micro_batch
    accumulation = derive_gradient_accumulation(config)
    if args.experiment_type == "short_run" and args.eval_data is None:
        raise ValueError("short_run 必须显式提供获批的 --eval-data")
    if args.eval_data:
        eval_manifest = (
            validate_phase6_validation(root, args.eval_data)
            if phase == "phase_6"
            else validate_phase5_eval(root, args.eval_data)
        )
    else:
        eval_manifest = None
    if phase == "phase_6":
        release_dir = args.data_release or Path("data/sft/released/v2_plus_repair_r2")
        released = (root / release_dir).resolve() if not release_dir.is_absolute() else release_dir.resolve()
        train_path = released / ("smoke_train.jsonl" if args.experiment_type == "smoke" else "train.jsonl")
        release = validate_phase6_release(root, train_path)
    else:
        train_path = root / "data/sft/released/v1" / ("smoke_train.jsonl" if args.experiment_type == "smoke" else "train.jsonl")
        release = validate_phase4_release(root, train_path)
    model_info = inspect_local_model(args.model_path)
    verify_tokenizer_files(args.model_path.resolve(), release)
    if model_info["model_revision"] not in {"unknown", release["tokenizer_revision"]}:
        raise ValueError("本地模型 revision 与 Phase 4 tokenizer revision 不一致")
    runtime = validate_runtime("Phase 6" if phase == "phase_6" else "Phase 5")
    resume = args.resume_from_checkpoint is not None
    if output.exists() and not resume:
        raise FileExistsError(f"输出目录已存在: {output}")
    if resume and not output.is_dir():
        raise ValueError("resume 要求原 run 输出目录已存在")
    output.mkdir(parents=True, exist_ok=resume)
    data_sha = sha256_file(train_path)
    artifact_items = (
        release["artifacts"].values()
        if isinstance(release["artifacts"], dict)
        else release["artifacts"]
    )
    artifact = next(
        item for item in artifact_items
        if item["path"] == train_path.relative_to(root).as_posix()
    )
    expected_steps = (
        derive_optimizer_steps(
            artifact["records"], config["training"]["global_batch_size"], args.epochs
        )
        if phase == "phase_6" else args.max_steps
    )
    contract = checkpoint_contract(
        config, data_sha, release["tokenizer_revision"], model_info["model_revision"],
        release["tokenizer_files"], phase=phase,
    )
    if phase == "phase_6":
        contract["training_schedule"] = {
            "epochs": args.epochs, "expected_optimizer_steps": expected_steps,
        }
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        try:
            checkpoint.relative_to(output)
        except ValueError as exc:
            raise ValueError("resume checkpoint 必须属于当前 output-dir") from exc
        validate_checkpoint(checkpoint, contract)
    resolved = sanitized_config(config)
    resolved["training"]["gradient_accumulation_steps"] = accumulation
    if phase == "phase_6":
        resolved["training"]["epochs"] = args.epochs
        resolved["training"]["expected_optimizer_steps"] = expected_steps
    else:
        resolved["training"]["max_steps"] = args.max_steps
    resolved["tracking"]["wandb_mode"] = args.wandb_mode
    resolved_path = output / "resolved_config.yaml"
    if not resume:
        resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest_path = output / "run_manifest.json"
    payload = {
            "model": {"model_id": "Qwen/Qwen3-4B", "model_revision": model_info["model_revision"],
                      "source": "runtime_local_path", "file_sha256": model_info["file_sha256"],
                      "weight_shards": model_info["weight_shards"]},
            "data": {
                "path": train_path.relative_to(root).as_posix(),
                "sha256": data_sha,
                "records": artifact["records"],
            },
            "evaluation": None if args.eval_data is None else {
                "path": args.eval_data.resolve().relative_to(root).as_posix(),
                "sha256": eval_manifest["data"]["sha256"],
                "manifest_sha256": sha256_file(
                    args.eval_data.parent / "manifest.json"
                    if phase == "phase_6" else args.eval_data.with_suffix(".manifest.json")
                ),
                "records": eval_manifest["records"],
                "generation_allowed": False,
                "gradient_update_allowed": False,
            },
            "tokenizer": {"revision": release["tokenizer_revision"], "enable_thinking": False,
                          "file_sha256": release["tokenizer_files"]},
            "lora": config["lora"],
            "batching": {"micro_batch": args.micro_batch, "gradient_accumulation_steps": accumulation,
                         "world_size": 1, "global_batch_size": 64},
            "training_schedule": {
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "expected_optimizer_steps": expected_steps,
                "warmup_ratio": config["training"]["warmup_ratio"],
                "warmup_steps": derive_warmup_steps(
                    expected_steps, config["training"]["warmup_ratio"]
                ),
                "loss_normalization": "active_assistant_tokens",
                "periodic_eval_aggregation": "trainer_batch_aggregation",
                "final_eval_aggregation": "all_active_assistant_tokens",
            },
            "configuration": {"sha256": sha256_file(config_path)},
            "runtime_environment": runtime,
            "tracking": {"wandb_mode": args.wandb_mode, "project": config["tracking"]["wandb_project"]},
            "resume": {"enabled": resume, "parent_checkpoint": args.resume_from_checkpoint.name if resume else None},
        }
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") not in {"completed", "failed"}:
            raise ValueError("resume 前原 run manifest 必须为 completed 或 failed")
        history = list(previous.get("resume_history", []))
        history.append({
            "status": previous["status"],
            "last_completed_step": previous.get("last_completed_step"),
            "finished_at_utc": previous.get("finished_at_utc"),
        })
        manifest = {
            **previous, **payload, "status": "running", "resume_history": history,
        }
        manifest.pop("finished_at_utc", None)
        target = f"epoch_{args.epochs}" if phase == "phase_6" else f"step_{args.max_steps}"
        resume_config = output / f"resolved_config_resume_to_{target}.yaml"
        resume_config.write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    else:
        manifest = initial_manifest(
            experiment_type=args.experiment_type, payload=payload, phase=phase
        )
    write_manifest_atomic(manifest_path, manifest)
    try:
        tokenizer = QwenAssistantTokenizer(args.model_path, max_length=config["data"]["max_length"])
        rows = load_and_encode(train_path, tokenizer)
        eval_rows = load_eval_and_encode(args.eval_data, tokenizer) if args.eval_data else None
        if args.preflight_only:
            write_manifest_atomic(manifest_path, finish_manifest(manifest, "completed", preflight_only=True))
            print(json.dumps({"status": "passed", "records": len(rows), "preflight_only": True}))
            return
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, TrainingArguments

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model_parameters = model.num_parameters()
        if not 3_500_000_000 <= model_parameters <= 4_500_000_000:
            raise ValueError(f"本地模型参数规模不是预期 4B: {model_parameters}")
        model.config.use_cache = False
        lora = config["lora"]
        validate_target_modules(model, lora["target_modules"])
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], bias=lora["bias"], target_modules=lora["target_modules"],
        ))
        parameter_stats = validate_trainable_parameters(model)
        os.environ["WANDB_MODE"] = args.wandb_mode
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_PROJECT"] = config["tracking"]["wandb_project"]
        training = config["training"]
        collator = AssistantOnlyCollator(tokenizer.tokenizer.pad_token_id)
        memory_callback = CudaMemoryCallback()
        trainer = TokenNormalizedTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=output.relative_to(root).as_posix(),
                max_steps=args.max_steps if args.max_steps is not None else -1,
                num_train_epochs=float(args.epochs) if args.epochs is not None else 3.0,
                per_device_train_batch_size=args.micro_batch,
                gradient_accumulation_steps=accumulation,
                learning_rate=training["learning_rate"], lr_scheduler_type=training["lr_scheduler_type"],
                warmup_steps=derive_warmup_steps(expected_steps, training["warmup_ratio"]),
                weight_decay=training["weight_decay"],
                max_grad_norm=training["max_grad_norm"], bf16=True, tf32=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=training["logging_steps"], save_steps=training["save_steps"],
                save_total_limit=training["save_total_limit"], seed=training["seed"],
                data_seed=training["data_seed"], report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
                run_name=f"{phase.replace('_', '')}-{args.experiment_type}-{output.name}",
                remove_unused_columns=False,
                eval_strategy="steps" if eval_rows else "no",
                eval_steps=training["evaluation_steps"] if eval_rows else None,
            ),
            train_dataset=EncodedDataset(rows),
            eval_dataset=EncodedDataset(eval_rows) if eval_rows else None,
            data_collator=collator,
            callbacks=[memory_callback, CheckpointContractCallback(output, contract)],
        )
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if resume else None)
        train_counts = (collator.samples, collator.input_tokens, collator.assistant_tokens)
        eval_metrics = trainer.evaluate() if eval_rows else {}
        if eval_rows:
            eval_metrics["eval_active_assistant_token_loss"] = evaluate_active_token_loss(
                model, EncodedDataset(eval_rows), collator,
                batch_size=training["per_device_eval_batch_size"],
            )
        for checkpoint in output.glob("checkpoint-*"):
            sanitize_adapter_config(checkpoint)
            write_checkpoint_contract(checkpoint, contract)
        final_adapter = output / "final_adapter"
        model.save_pretrained(final_adapter, safe_serialization=True)
        sanitize_adapter_config(final_adapter)
        metrics = dict(result.metrics)
        metrics.update(eval_metrics)
        metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        runtime_seconds = float(metrics.get("train_runtime", 0.0))
        if runtime_seconds > 0:
            metrics["observed_samples_per_second"] = train_counts[0] / runtime_seconds
            metrics["input_tokens_per_second"] = train_counts[1] / runtime_seconds
            metrics["assistant_tokens_per_second"] = train_counts[2] / runtime_seconds
        adapter_weight = next(
            path for path in (final_adapter / "adapter_model.safetensors", final_adapter / "adapter_model.bin")
            if path.is_file()
        )
        wandb_run_id = None
        wandb_project = manifest["tracking"]["project"]
        if args.wandb_mode != "disabled":
            import wandb

            if wandb.run is not None:
                wandb_run_id = wandb.run.id
                wandb_project = wandb.run.project
        completed = finish_manifest(
            manifest, "completed", trainable_parameters=parameter_stats,
            final_metrics=metrics, last_completed_step=trainer.state.global_step,
            completed_epochs=float(trainer.state.epoch or 0.0),
            periodic_evaluations=[
                {"step": int(item["step"]), "eval_loss": float(item["eval_loss"])}
                for item in trainer.state.log_history
                if "step" in item and "eval_loss" in item and int(item["step"]) < expected_steps
            ],
            gpu_memory_samples=memory_callback.samples,
            tracking={
                **manifest["tracking"], "project": wandb_project,
                "run_id": wandb_run_id,
            },
            artifacts={
                "final_adapter_config_sha256": sha256_file(final_adapter / "adapter_config.json"),
                "final_adapter_weight": adapter_weight.name,
                "final_adapter_weight_sha256": sha256_file(adapter_weight),
            },
        )
        write_manifest_atomic(manifest_path, completed)
    except BaseException as exc:
        failed = finish_manifest(manifest, "failed", failure_type=type(exc).__name__)
        write_manifest_atomic(manifest_path, failed)
        raise


if __name__ == "__main__":
    main()
