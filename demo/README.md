# 交互式问答 Demo

`demo/cli_chat.py` 在本地加载 `Qwen/Qwen3-4B` 基座和 Phase 6 LoRA 适配器，提供一个最小可用的命令行对话界面。它用于人工观察模型行为，不是评测脚本，输出不构成评测成绩或医疗建议。

## 前置条件

- 基础模型：`model/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- 适配器：默认 `artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_r2_formal/final_adapter`。
- 依赖：在项目根目录执行过 `uv sync --extra sft`（脚本用到 `torch`、`transformers`、`peft`）。

模型尚未下载时，在 cmd 中执行：

```bat
set HF_ENDPOINT=https://hf-mirror.com
uv run hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir model/Qwen3-4B
```

## 快速开始

```powershell
uv run python demo/cli_chat.py
```

加载完成后直接输入问题回车。默认每轮只发送 `system prompt + 当前问题`，不携带历史；需要多轮上下文时加 `--multi-turn`。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `model/Qwen3-4B` | 基础模型目录 |
| `--adapter-path` | `.../qwen3_4b_lora_v2_plus_repair_r2_formal/final_adapter` | LoRA 适配器目录，可换成同一 run 的 `checkpoint-298` 或上一版 adapter |
| `--base-only` | 关闭 | 只加载基座，不加载 LoRA，用于对比 |
| `--max-new-tokens` | `1024` | 单次回答最大生成 token 数 |
| `--temperature` | `0.3` | 采样温度，偏保守；更自然的通用对话可调到 `0.6` |
| `--top-p` | `0.95` | 核采样阈值 |
| `--top-k` | `20` | top-k 采样 |
| `--greedy` | 关闭 | 关闭采样，使用贪心解码（确定性） |
| `--no-stream` | 关闭 | 关闭流式输出，一次性打印完整回答 |
| `--multi-turn` | 关闭 | 保留多轮对话历史 |
| `--cpu` | 关闭 | 强制使用 CPU（默认优先 CUDA） |

需要确定性输出时用 `--greedy`，不要用 `--temperature 0`：后者仍是采样路径，且部分版本会直接报错。

## 常用场景

```powershell
# 默认采样（temperature 0.3，偏保守）
uv run python demo/cli_chat.py --max-new-tokens 1024

# 更自然但更发散
uv run python demo/cli_chat.py --temperature 0.6 --top-p 0.95

# 贪心解码，逐条对比时结果可复现
uv run python demo/cli_chat.py --greedy

# 只跑基座，对比 SFT 前后
uv run python demo/cli_chat.py --base-only

# 指定其它 run 或中间 checkpoint（含上一版 adapter 用于对比）
uv run python demo/cli_chat.py --adapter-path artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_r2_formal/checkpoint-298
uv run python demo/cli_chat.py --adapter-path artifacts/runs/phase_6/qwen3_4b_lora_v2_plus_repair_2epoch/final_adapter

# 多轮对话
uv run python demo/cli_chat.py --multi-turn

# CPU 回退（仅小规模验证，速度远慢于 GPU）
uv run python demo/cli_chat.py --cpu --max-new-tokens 64
```

## 交互命令

| 输入 | 行为 |
| --- | --- |
| `quit` / `exit` / `q` | 退出 |
| `clear` / `reset` | 清空多轮历史 |
| `Ctrl+C` | 中断当前等待并退出 |

## 数据流

1. 每轮把 `system + 当前问题`（`--multi-turn` 时为完整历史）交给 `tokenizer.apply_chat_template`，固定 `enable_thinking=False`。
2. 生成参数：`--greedy` 时 `do_sample=False`；否则使用 `temperature` / `top_p` / `top_k`。
3. 只解码新增 token，跳过 special token；默认流式打印。
4. `--multi-turn` 时把助手回答追加进历史，`clear` 会重置为只剩 system prompt。

## 与训练、评测协议的关系

- 推理 system prompt 固定为 `你是一个谨慎、清晰且遵循证据边界的助手。`，与 `data/sft/released/v2_plus_repair_r2/manifest.json` 的 `system_prompt` 一致。
- 默认适配器的训练数据是 `v2_plus_repair_r2`，训练记录统一为 `system/user/assistant`；上一版 `v2_plus_repair` adapter 只作为对比保留。
- 公开 regression 审核记录见 `artifacts/evaluations/phase_6_public_regression/public_regression_audit_r2.json`：医疗表现相较 base 改善、通用能力无明显退化、两套协议 forbidden 零命中，按该标准验收通过。
- 需要写入报告或做逐条对比时使用 `scripts/sft/evaluate_public_regression.py`；本 demo 只做人工观察，不生成审计产物。

## 常见问题

- `[错误] 路径不存在`：确认模型或适配器路径，用 `--adapter-path` 指向实际存在的 `final_adapter` 或 `checkpoint-*`。
- PowerShell 下 `uv run hf ...` 报 `Failed to canonicalize script path`：这是 uv 入口脚本在本机的问题，模型下载请在 cmd 中执行，或改用 `uv run python -c "from huggingface_hub import snapshot_download; ..."`。
- CPU 很慢：只用于小规模回退验证，正常观察请在 5090 等 GPU 上运行。
- 每次回答不一样：采样解码本身随机，需要复现时用 `--greedy`，或走 `evaluate_public_regression.py` 的固定协议。

## 限制

- 训练数据为内部工程数据，审核状态为 `agent_evidence_reviewed`，不是医生审核或临床验证。
- `release_allowed=false`，本 demo 不得用于临床诊断，也不得据此宣称医疗能力。
- 输出只用于工程观察，不足以作为模型能力结论。
