"""校验 Phase 5 LoRA adapter 文件与配置；可选在 GPU 新进程中重新加载。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from med_align_rag_4b.common.hashing import sha256_file
from med_align_rag_4b.sft.adapter_validation import validate_adapter_directory
from med_align_rag_4b.sft.config import load_sft_config
from med_align_rag_4b.sft.evaluation import load_fixtures, validate_fixture_output
from med_align_rag_4b.sft.run_manifest import write_manifest_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/sft/qwen3_4b_lora.yaml"))
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--run-fixtures", action="store_true")
    parser.add_argument(
        "--fixtures", type=Path,
        default=Path("data/evaluation/phase5_engineering_fixtures_v1.json"),
    )
    args = parser.parse_args()
    config = load_sft_config(args.config)
    adapter_config = validate_adapter_directory(args.adapter_path, config["lora"])
    reloaded = False
    fixture_results = []
    if args.model_path:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base, args.adapter_path, local_files_only=True
        )
        reloaded = True
        if args.run_fixtures:
            if not torch.cuda.is_available():
                raise RuntimeError("fixture 生成要求 CUDA")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, local_files_only=True, trust_remote_code=False,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = model.to("cuda").eval()
            for fixture in load_fixtures(args.fixtures):
                rendered = tokenizer.apply_chat_template(
                    fixture["messages"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt",
                    return_dict=True,
                )
                model_inputs = {
                    name: tensor.to(model.device) for name, tensor in rendered.items()
                }
                prompt_length = model_inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    generated = model.generate(
                        **model_inputs, max_new_tokens=fixture["max_new_tokens"],
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                output = tokenizer.decode(
                    generated[0, prompt_length:], skip_special_tokens=True
                )
                fixture_results.append({
                    "id": fixture["id"], "kind": fixture["kind"], "output": output,
                    **validate_fixture_output(fixture, output),
                })
    elif args.run_fixtures:
        raise ValueError("--run-fixtures 必须同时提供 --model-path")
    report = {
        "status": "passed", "reloaded": reloaded,
        "engineering_fixtures_only": True,
        "fixtures_sha256": sha256_file(args.fixtures) if args.run_fixtures else None,
        "fixture_results": fixture_results, "adapter": adapter_config,
    }
    report_path = args.report_path or args.adapter_path.parent / "adapter_validation.json"
    write_manifest_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
