"""Map an HF model's `config.architectures[0]` to the llama.cpp arch string used
in the sidecar's `abl.arch` KV.

The engine compares `abl.arch` (string) against the GGUF's `general.architecture`
(also string), which is set by `convert_hf_to_gguf.py`. So we need to replicate
its mapping. The table below covers our v1 targets and common neighbors.

If the user's target isn't in the table, they can override via CLI `--arch`.
"""
from __future__ import annotations

# HF config architecture class name → llama.cpp/GGUF general.architecture
HF_TO_GGUF_ARCH: dict[str, str] = {
    # Gemma family
    "Gemma2ForCausalLM": "gemma2",
    "Gemma3ForCausalLM": "gemma3",
    "Gemma3ForConditionalGeneration": "gemma3",
    "Gemma4ForCausalLM": "gemma4-iswa",
    "Gemma4ForConditionalGeneration": "gemma4-iswa",
    # Qwen family
    "Qwen2ForCausalLM": "qwen2",
    "Qwen2MoeForCausalLM": "qwen2moe",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3moe",
    # Qwen 3.5 (Jackrong fork uses the same arch class)
    "Qwen35ForCausalLM": "qwen35",
    "Qwen3_5ForCausalLM": "qwen35",
    # GLM family
    "Glm4ForCausalLM": "glm4",
    "Glm4MoeForCausalLM": "glm4-moe",
    "Glm4MoeLiteForCausalLM": "glm4-moe",  # GLM-4.7-Flash (30B-A3B, 64 experts)
    "ChatGLMForConditionalGeneration": "chatglm",
    # Llama / Mistral
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",  # Mistral arch maps to llama in convert_hf_to_gguf
    "Llama4ForCausalLM": "llama4",
    # DeepSeek
    "DeepseekV2ForCausalLM": "deepseek2",
    "DeepseekV3ForCausalLM": "deepseek2",
    # MiniMax
    "MiniMaxM2ForCausalLM": "minimax-m2",
}


def hf_to_gguf_arch(hf_arch: str) -> str | None:
    """Return the llama.cpp arch string for an HF architecture class name.

    Returns None if unknown — caller should ask user to supply via --arch.
    """
    return HF_TO_GGUF_ARCH.get(hf_arch)
