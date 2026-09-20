"""运行完整 RAG 组件消融 R0-R5；RAG ON 使用 RAGPipeline prompt。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.jsonl import count_jsonl, write_jsonl_atomic
from med_align_rag_4b.common.paths import portable_relative_path, resolve_project_path
from med_align_rag_4b.evaluation.rag_ablation import (
    ARM_ORDER,
    generate_case_record,
    get_arm,
    limit_cases,
    load_rag_ablation_config,
    load_validation_cases,
)
from med_align_rag_4b.rag.generation import GenerationConfig, QwenGenerator


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有 manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else resolve_project_path(root, path)


def _source_metadata(path: Path, run_manifest: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not run_manifest.is_file():
        raise FileNotFoundError(run_manifest)
    return {
        "path": portable_relative_path(path, root),
        "run_manifest": portable_relative_path(run_manifest, root),
        "run_manifest_sha256": sha256_file(run_manifest),
    }


def _load_generator(model_path: Path, adapter_path: Path | None, device: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("--device cuda 要求仅有一张可见 CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA GPU 不支持 BF16")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model = model.to(device).eval()
    return QwenGenerator(model, tokenizer), torch, model
def _plan(args: argparse.Namespace, root: Path) -> None:
    config_path = _resolve(root, args.config)
    config = load_rag_ablation_config(config_path, root)
    cases, cases_sha256 = load_validation_cases(root, config["dataset"])
    print(json.dumps({
        "status": "ready",
        "phase": 8,
        "protocol": "full_rag_component",
        "dataset": {"path": config["dataset"]["cases"], "records": len(cases), "sha256": cases_sha256},
        "arms": [{"arm": arm_id, **config["arms"][arm_id]} for arm_id in ARM_ORDER],
        "rag": config["rag"],
        "locked_splits": ["test", "medical_safety_holdout"],
    }, ensure_ascii=False, indent=2))


def _generate(args: argparse.Namespace, root: Path) -> None:
    config_path = _resolve(root, args.config)
    config = load_rag_ablation_config(config_path, root)
    arm = get_arm(config, args.arm)
    cases, cases_sha256 = load_validation_cases(root, config["dataset"])
    cases = limit_cases(cases, args.limit)
    output = _resolve(root, args.output)
    manifest_path = output.with_suffix(".manifest.json")
    portable_relative_path(output, root)
    if args.resume and output.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if count_jsonl(output) != existing.get("records") or sha256_file(output) != existing.get("output_sha256"):
            raise ValueError(f"{arm.arm_id} 已有产物不完整，拒绝覆盖")
        print(json.dumps({"status": "skipped_complete", "arm": arm.arm_id}, ensure_ascii=False))
        return
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {output}")

    if arm.model_source == "base":
        if args.base_model_path is None:
            raise ValueError(f"{arm.arm_id} 必须传入 --base-model-path")
        model_path = args.base_model_path.resolve()
        model_meta = {"role": "base", "revision": config["model_revisions"]["base"]}
    else:
        configured = resolve_project_path(root, config["artifacts"]["merged_sft"])
        model_path = args.merged_sft_path.resolve() if args.merged_sft_path else configured
        model_meta = _source_metadata(model_path, resolve_project_path(root, config["artifacts"]["merged_sft_manifest"]), root)
        model_meta["role"] = "merged_sft"
        model_meta["revision"] = config["model_revisions"]["merged_sft"]

    adapter_path = None
    adapter_meta: dict[str, Any] = {"role": None, "enabled": False}
    if arm.adapter is not None:
        adapter_path = resolve_project_path(root, config["artifacts"][f"{arm.adapter}_adapter"])
        run_manifest = resolve_project_path(root, config["artifacts"][f"{arm.adapter}_run_manifest"])
        adapter_meta = {"role": arm.adapter, "enabled": True, **_source_metadata(adapter_path, run_manifest, root)}

    retriever = None
    rag_meta: dict[str, Any] = {"enabled": False}
    if arm.rag:
        if args.rag_embedding_model_path is None:
            raise ValueError(f"{arm.arm_id} 必须传入 --rag-embedding-model-path")
        from med_align_rag_4b.rag.retrieval import create_retriever

        rag_manifest = resolve_project_path(root, config["rag"]["manifest"])
        retriever = create_retriever(rag_manifest, root, args.rag_embedding_model_path, device=args.rag_device)
        rag_meta = {
            "enabled": True,
            "manifest": portable_relative_path(rag_manifest, root),
            "manifest_sha256": sha256_file(rag_manifest),
            "embedding_model_id": retriever.artifacts.model_id,
            "top_k": config["rag"]["top_k"],
            "max_context_chars": config["rag"]["max_context_chars"],
        }

    generator, torch, model = _load_generator(model_path, adapter_path, args.device)
    generation_config = GenerationConfig(**config["generation"])
    records: list[dict[str, Any]] = []
    for case in tqdm(cases, desc=f"Phase 8 RAG {arm.arm_id}", unit="case", dynamic_ncols=True):
        torch.manual_seed(config["generation"]["seed"])
        record = generate_case_record(
            case,
            arm=arm,
            base_system_prompt=config["base_system_prompt"],
            generation_config=generation_config,
            generate=generator.generate,
            retrieve=retriever.retrieve if retriever is not None else None,
            rag_top_k=config["rag"]["top_k"],
            max_context_chars=config["rag"]["max_context_chars"],
            controlled_context_for_required=config["rag"]["controlled_context_for_required"],
        )
        record["model"] = {**model_meta, "revision": getattr(model.config, "_commit_hash", None) or model_meta.get("revision")}
        record["adapter"] = adapter_meta
        record["runtime"].update({"device": args.device, "dtype": str(next(model.parameters()).dtype)})
        records.append(record)
    write_jsonl_atomic(output, records)
    manifest = {
        "format_version": 1,
        "phase": 8,
        "kind": "phase8_rag_ablation_generation",
        "status": "completed_pending_scoring",
        "arm": arm.arm_id,
        "protocol": "full_rag_component",
        "records": len(records),
        "output": portable_relative_path(output, root),
        "output_sha256": sha256_file(output),
        "config": portable_relative_path(config_path, root),
        "config_sha256": sha256_file(config_path),
        "cases_sha256": cases_sha256,
        "model": model_meta,
        "adapter": adapter_meta,
        "rag": rag_meta,
        "generation": config["generation"],
        "release_allowed": False,
        "limitations": ["仅生成原始输出，尚未运行规则评分、模型 judge 或代理审核", "test 与 medical safety holdout 未读取"],
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "arm": arm.arm_id, "records": len(records)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=Path("configs/evaluation/phase8_rag.yaml"))
    generate = sub.add_parser("generate")
    generate.add_argument("--config", type=Path, default=Path("configs/evaluation/phase8_rag.yaml"))
    generate.add_argument("--arm", choices=ARM_ORDER, required=True)
    generate.add_argument("--base-model-path", type=Path)
    generate.add_argument("--merged-sft-path", type=Path)
    generate.add_argument("--rag-embedding-model-path", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    generate.add_argument("--rag-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "plan":
        _plan(args, root)
    else:
        _generate(args, root)


if __name__ == "__main__":
    main()