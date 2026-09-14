"""Qwen3-4B RAG 生成适配器；模型仅在显式调用时加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from med_align_rag_4b.rag import GENERATION_MODEL_ID


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    enable_thinking: bool = False


class QwenGenerator:
    def __init__(self, model: Any, tokenizer: Any, model_id: str = GENERATION_MODEL_ID) -> None:
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id

    @classmethod
    def from_pretrained(cls, model_path: str, *, model_id: str = GENERATION_MODEL_ID, device: str = "cpu", torch_dtype: str = "float32") -> QwenGenerator:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False, local_files_only=True, torch_dtype=getattr(torch, torch_dtype)).to(device)
        model.eval()
        return cls(model, tokenizer, model_id)

    def generate(self, messages: list[dict[str, str]], config: GenerationConfig) -> dict[str, Any]:
        import torch
        if config.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须为正数")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=config.enable_thinking)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(next(self.model.parameters()).device)
        kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample, "repetition_penalty": config.repetition_penalty}
        if config.do_sample:
            kwargs.update(temperature=config.temperature, top_p=config.top_p, top_k=config.top_k)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        new_ids = generated[0][inputs.input_ids.shape[1] :]
        return {"answer": self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "prompt": prompt, "model": self.model_id, "generation_config": asdict(config)}
