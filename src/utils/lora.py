"""LoRA configuration helpers.

Defaults: rank 16, alpha 32, dropout 0.05, attention proj + MLP proj.
Per-method configs (config/{dpo,kto,grpo}.yaml) may override. GRPO sets
dropout to 0.0 because TRL keeps the model in train() mode during rollouts.

We deliberately do NOT target `embed_tokens` or `lm_head`: OuteTTS extends
Llama's vocabulary with thousands of audio-codec tokens whose embeddings are
already trained, and we don't want LoRA messing with them on a small dataset.
"""
from __future__ import annotations

from peft import LoraConfig, TaskType


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention projections
    "gate_proj", "up_proj", "down_proj",     # SwiGLU MLP projections
]


def make_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
