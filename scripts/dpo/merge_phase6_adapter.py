"""将 Phase 6 LoRA adapter 合并为可复现的 merged SFT 模型。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.common.manifests import build_manifest, describe_artifact
from med_align_rag_4b.common.paths import resolve_project_path
from med_align_rag_4b.dpo.manifest import write_manifest_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input-freeze", default="manifests/phase_7/input_freeze.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_model = Path(args.base_model).resolve()
    adapter = Path(args.adapter).resolve()
    input_freeze_path = resolve_project_path(PROJECT_ROOT, args.input_freeze)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (PROJECT_ROOT / args.output_dir).resolve()
    )
    allowed = (PROJECT_ROOT / "artifacts/runs/phase_7").resolve()
    try:
        output_dir.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("output-dir 必须位于 artifacts/runs/phase_7 内") from exc
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖: {output_dir}")

    frozen = json.loads(input_freeze_path.read_text(encoding="utf-8"))
    _verify_frozen_inputs(frozen, base_model, adapter)
    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=False,
        local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
        local_files_only=True,
    ).to(device)
    peft_model = PeftModel.from_pretrained(base, adapter)
    merged = peft_model.merge_and_unload(safe_merge=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".merged_sft.", dir=output_dir.parent))
    try:
        merged.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(staging)
        files = _describe_directory(staging)
        manifest = build_manifest(
            "phase_7_merged_sft",
            files,
            metadata={
                "phase": "phase_7",
                "base_model": _portable_input_label(base_model),
                "adapter": _portable_input_label(adapter),
                "input_freeze_sha256": sha256_file(input_freeze_path),
                "dtype": "bfloat16",
                "device": device,
                "training_started": False,
                "release_allowed": False,
            },
        )
        write_manifest_atomic(staging / "merge_manifest.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {"output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix()},
            ensure_ascii=False,
        )
    )


def _portable_input_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


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


def _verify_frozen_inputs(
    frozen: dict[str, Any], base_model: Path, adapter: Path
) -> None:
    model = frozen["model"]
    if model["model_id"] != "Qwen/Qwen3-4B":
        raise ValueError("input freeze model_id 无效")
    for item in model["files"]:
        path = base_model / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"base model 文件不存在: {item['path']}")
    for item in frozen["sft_starting_point"]["adapter"]["files"]:
        path = adapter / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"adapter 文件不存在: {item['path']}")


def _describe_directory(directory: Path) -> list[Any]:
    return [
        describe_artifact(path, directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


if __name__ == "__main__":
    main()
