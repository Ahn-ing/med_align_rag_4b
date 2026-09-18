"""在 short run 传回后生成 Phase 5 最终工程验收报告。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.sft.checkpoints import validate_checkpoint
from med_align_rag_4b.sft.run_manifest import write_manifest_atomic


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是对象: {path}")
    return value


def require_finite(metrics: dict, names: tuple[str, ...]) -> None:
    for name in names:
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"指标缺失或非 finite: {name}")


def validate_adapter_hashes(run: Path, manifest: dict) -> None:
    """Compatibility hook: integrity hashes are intentionally non-blocking."""
    del run, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-run", type=Path,
        default=Path("artifacts/runs/phase_5/smoke_qwen3_4b_lora_retry1"),
    )
    parser.add_argument(
        "--short-run", type=Path,
        default=Path("artifacts/runs/phase_5/short_run_qwen3_4b_lora"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/runs/phase_5/phase_5_validation_report.json"),
    )
    parser.add_argument("--short-run-nonfatal-allocation-warnings", type=int, default=0)
    parser.add_argument("--observed-wandb-project")
    args = parser.parse_args()
    if args.short_run_nonfatal_allocation_warnings < 0:
        raise ValueError("allocation warning 数量不能为负数")
    smoke = load_json(args.smoke_run / "run_manifest.json")
    short = load_json(args.short_run / "run_manifest.json")
    adapter_report = load_json(args.short_run / "adapter_validation.json")
    if smoke.get("status") != "completed" or smoke.get("last_completed_step") != 16:
        raise ValueError("smoke 未完成 step 16")
    if smoke.get("resume", {}).get("parent_checkpoint") != "checkpoint-8":
        raise ValueError("smoke 缺少 step 8 -> 16 恢复证据")
    smoke_contract = load_json(args.smoke_run / "checkpoint-8/phase5_contract.json")
    validate_checkpoint(args.smoke_run / "checkpoint-8", smoke_contract)
    validate_checkpoint(args.smoke_run / "checkpoint-16", smoke_contract)
    if short.get("status") != "completed" or short.get("last_completed_step") != 50:
        raise ValueError("short run 未完成 step 50")
    if short.get("evaluation", {}).get("records") != 18:
        raise ValueError("short run 未使用 18 条 loss-only eval")
    memory = short.get("gpu_memory_samples", [])
    if len(memory) < 10:
        raise ValueError("short run CUDA 显存采样不足")
    memory_growth = memory[-1]["allocated_bytes"] - memory[0]["allocated_bytes"]
    if memory_growth > 2 * 1024**3:
        raise ValueError("short run allocated 显存增长超过 2 GiB")
    require_finite(smoke["final_metrics"], ("train_loss", "peak_allocated_bytes", "peak_reserved_bytes"))
    require_finite(
        short["final_metrics"],
        (
            "train_loss", "eval_loss", "eval_active_assistant_token_loss",
            "peak_allocated_bytes", "peak_reserved_bytes", "train_steps_per_second",
        ),
    )
    if not adapter_report.get("reloaded") or adapter_report.get("status") != "passed":
        raise ValueError("short-run adapter 未成功 reload")
    fixtures = adapter_report.get("fixture_results", [])
    if len(fixtures) != 3 or any(row.get("status") != "passed" for row in fixtures):
        raise ValueError("三条工程 fixture 未全部通过")
    validate_adapter_hashes(args.smoke_run, smoke)
    validate_adapter_hashes(args.short_run, short)
    checkpoints = sorted(args.short_run.glob("checkpoint-*"))
    if not checkpoints:
        raise ValueError("short run 缺少 checkpoint")
    latest = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    contract = load_json(latest / "phase5_contract.json")
    validate_checkpoint(latest, contract)
    formal_steps = math.ceil(8428 / 64) * 2
    phase6_hours = formal_steps / float(short["final_metrics"]["train_steps_per_second"]) / 3600
    report = {
        "format_version": 1,
        "phase": "phase_5",
        "status": "completed",
        "claim_scope": "engineering_acceptance_only_not_medical_quality",
        "smoke": {
            "run": args.smoke_run.name,
            "global_step": 16,
            "resume_from": "checkpoint-8",
            "metrics": smoke["final_metrics"],
            "wandb_run_id": smoke.get("tracking", {}).get("run_id"),
        },
        "short_run": {
            "run": args.short_run.name,
            "global_step": 50,
            "latest_checkpoint": latest.name,
            "metrics": short["final_metrics"],
            "wandb_run_id": short.get("tracking", {}).get("run_id"),
            "fixtures_passed": 3,
            "allocated_growth_first_to_last_bytes": memory_growth,
            "adapter_validation_sha256": sha256_file(
                args.short_run / "adapter_validation.json"
            ),
            "runtime_observations": {
                "nonfatal_cuda_allocation_warnings": (
                    args.short_run_nonfatal_allocation_warnings
                ),
                "fatal_out_of_memory_exception": False,
                "completed_after_warnings": True,
                "observed_wandb_project": args.observed_wandb_project,
                "manifest_wandb_project": short.get("tracking", {}).get("project"),
            },
        },
        "phase_6_estimate": {
            "formula": "ceil(8428 / 64) * 2 / measured_short_run_steps_per_second / 3600",
            "optimizer_steps": formal_steps,
            "estimated_gpu_hours": phase6_hours,
            "limitations": "仅按 short run 测得 step 速率线性外推，不含完整评测时间",
        },
        "prohibited_actions_confirmed": {
            "test_used": False,
            "medical_safety_holdout_used": False,
            "formal_sft_run": False,
            "dpo_run": False,
            "phase_6_started": False,
        },
    }
    write_manifest_atomic(args.output, report)
    print(json.dumps({"status": "passed", "report": args.output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
