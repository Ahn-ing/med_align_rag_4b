"""只读验收传回的 Phase 6 两轮 SFT 训练产物。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.sft.checkpoints import validate_checkpoint


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是对象: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("artifacts/runs/phase_6/qwen3_4b_lora_2epoch"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    allowed = (root / "artifacts/runs/phase_6").resolve()
    try:
        run.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("run-dir 必须位于 artifacts/runs/phase_6 内") from exc

    manifest = load_object(run / "run_manifest.json")
    schedule = manifest.get("training_schedule", {})
    if manifest.get("phase") != "phase_6" or manifest.get("experiment_type") != "formal_train":
        raise ValueError("不是 Phase 6 formal_train manifest")
    if manifest.get("status") != "completed":
        raise ValueError("Phase 6 训练尚未完成")
    if schedule.get("epochs") != 2 or schedule.get("expected_optimizer_steps") != 298:
        raise ValueError("Phase 6 训练预算不是冻结的 2 epochs / 298 steps")
    if manifest.get("data", {}).get("records") != 9495:
        raise ValueError("Phase 6 训练集记录数不是 9495")
    if manifest.get("data", {}).get("path") != "data/sft/released/v2_plus_repair_r2/train.jsonl":
        raise ValueError("Phase 6 训练数据路径不是 v2_plus_repair")
    evaluation = manifest.get("evaluation", {})
    if (
        evaluation.get("records") != 60
        or evaluation.get("generation_allowed") is not False
        or evaluation.get("gradient_update_allowed") is not False
    ):
        raise ValueError("Phase 6 validation 使用契约不匹配")
    if evaluation.get("path") != "data/evaluation/validation_v2/cases.jsonl":
        raise ValueError("Phase 6 未使用 validation_v2")
    if manifest.get("last_completed_step") != 298:
        raise ValueError("Phase 6 未完成 298 optimizer steps")
    if not math.isclose(float(manifest.get("completed_epochs", 0.0)), 2.0, abs_tol=1e-6):
        raise ValueError("Phase 6 未完成 2 epochs")

    metrics = manifest.get("final_metrics", {})
    for name in ("train_loss", "eval_loss", "eval_active_assistant_token_loss"):
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise ValueError(f"指标缺失或非 finite: {name}")
    periodic = manifest.get("periodic_evaluations", [])
    if [item.get("step") for item in periodic] != [100, 200]:
        raise ValueError("训练内 validation loss 必须记录在 step 100 和 200")
    if any(not math.isfinite(float(item.get("eval_loss", math.nan))) for item in periodic):
        raise ValueError("周期 validation loss 存在非 finite 值")

    checkpoints = sorted(run.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    if not checkpoints:
        raise ValueError("Phase 6 缺少 checkpoint")
    expected_contract = {
        "phase": "phase_6",
        "model_id": manifest["model"]["model_id"],
        "tokenizer_revision": manifest["tokenizer"]["revision"],
        "model_revision": manifest["model"]["model_revision"],
        "tokenizer_files": manifest["tokenizer"]["file_sha256"],
        "data_sha256": manifest["data"]["sha256"],
        "lora": manifest["lora"],
        "max_length": 2048,
        "global_batch_size": manifest["batching"]["global_batch_size"],
        "training_schedule": {
            "epochs": schedule["epochs"],
            "expected_optimizer_steps": schedule["expected_optimizer_steps"],
        },
    }
    validate_checkpoint(checkpoints[-1], expected_contract)

    artifacts = manifest.get("artifacts", {})
    adapter = run / "final_adapter"
    weight = adapter / artifacts.get("final_adapter_weight", "")
    if not weight.is_file() or sha256_file(weight) != artifacts.get("final_adapter_weight_sha256"):
        raise ValueError("final adapter 权重缺失或 SHA256 不匹配")
    config = adapter / "adapter_config.json"
    if not config.is_file() or sha256_file(config) != artifacts.get("final_adapter_config_sha256"):
        raise ValueError("final adapter config 缺失或 SHA256 不匹配")
    print(json.dumps({"status": "passed", "run": run.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
