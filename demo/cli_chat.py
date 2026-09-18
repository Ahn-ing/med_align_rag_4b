"""加载 phase_6 最终 LoRA 模型，在命令行中进行交互式问答。

用法（在项目根目录运行）::

    uv run python demo/cli_chat.py
    uv run python demo/cli_chat.py --max-new-tokens 512 --temperature 0.6
    uv run python demo/cli_chat.py --base-only
    uv run python demo/cli_chat.py --adapter-path artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_r2_formal/checkpoint-298
    uv run python demo/cli_chat.py --model-path artifacts/runs/phase_7/merge_sft --adapter-path artifacts/runs/phase_7/dpo_formal_v2/final_adapter

退出：输入 ``quit`` / ``exit`` / ``q``；清空多轮历史：输入 ``clear``。
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TextIteratorStreamer,
)

# 推理统一使用的 system prompt；与 data/sft/released/v2_plus_repair_r2
# manifest 中的 system_prompt 一致（SHA256 92f31efc...）。
SYSTEM_PROMPT = "你是一个谨慎、清晰且遵循证据边界的助手。"

# 项目根目录（本脚本位于 <root>/demo 下）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "Qwen3-4B"
DEFAULT_ADAPTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "phase_6"
    / "qwen3_4b_lora_v2_plus_repair_r2_formal"
    / "final_adapter"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 Qwen3-4B 基础模型和 LoRA 适配器并进入交互式问答。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="基础模型（Qwen3-4B 或 merged SFT）目录")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH,
                        help="LoRA 适配器目录（SFT 或 DPO 的 final_adapter/checkpoint-<step>）")
    parser.add_argument("--base-only", action="store_true",
                        help="只加载基础模型（Qwen3-4B），不加载 LoRA 适配器")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="单次回答最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="采样温度；需要确定性输出时改用 --greedy")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--greedy", action="store_true",
                        help="关闭采样，使用贪心解码（确定性）")
    parser.add_argument("--no-stream", action="store_true",
                        help="关闭流式输出，一次性打印完整回答")
    parser.add_argument("--multi-turn", action="store_true",
                        help="保留多轮对话历史；默认每轮只发送当前问题")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU（默认优先 CUDA）")
    return parser.parse_args(argv)


def build_model_and_tokenizer(args: argparse.Namespace):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    # 统一 bf16，与 5090 训练/推理配置一致。
    dtype = torch.bfloat16
    print(f"[加载] 基础模型: {args.model_path}")
    print(f"[加载] 设备: {device}，dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=False,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.base_only:
        model: PeftModel | PreTrainedModel = base
        print("[加载] 未加载 LoRA 适配器（仅基础模型）。")
    else:
        print(f"[加载] LoRA 适配器: {args.adapter_path}")
        model = PeftModel.from_pretrained(base, args.adapter_path, local_files_only=True)
    model = model.to(device).eval()

    print("[加载] 完成。输入问题开始对话，输入 quit 退出，输入 clear 清空历史。\n")
    return model, tokenizer, device


def generate_answer(
    model: PeftModel | PreTrainedModel,
    tokenizer: AutoTokenizer,
    device: str,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {name: tensor.to(device) for name, tensor in rendered.items()}
    prompt_length = model_inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=not args.greedy,
    )
    if not args.greedy:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    if args.no_stream:
        with torch.no_grad():
            generated = model.generate(**model_inputs, **gen_kwargs)
        answer = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        print(answer)
        return answer

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs["streamer"] = streamer
    thread = threading.Thread(target=model.generate, kwargs={**model_inputs, **gen_kwargs})
    thread.start()

    answer_parts: list[str] = []
    for piece in streamer:
        print(piece, end="", flush=True)
        answer_parts.append(piece)
    thread.join()
    print()
    return "".join(answer_parts).strip()


def main() -> None:
    args = parse_args()
    paths_to_check = [args.model_path] if args.base_only else [args.model_path, args.adapter_path]
    for path in paths_to_check:
        if not Path(path).exists():
            print(f"[错误] 路径不存在: {path}", file=sys.stderr)
            sys.exit(1)

    model, tokenizer, device = build_model_and_tokenizer(args)
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            try:
                user_input = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("再见。")
                break
            if user_input.lower() in {"clear", "reset"}:
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("[已清空多轮历史]")
                continue

            if args.multi_turn:
                history.append({"role": "user", "content": user_input})
                messages = history
            else:
                messages = history + [{"role": "user", "content": user_input}]

            print("\n助手 > ", end="", flush=True)
            answer = generate_answer(model, tokenizer, device, messages, args)

            if args.multi_turn and answer:
                history.append({"role": "assistant", "content": answer})
    finally:
        # 释放显存，便于脚本结束后干净退出。
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
