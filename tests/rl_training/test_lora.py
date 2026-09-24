from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from trl.trainer.utils import patch_chunked_lm_head

QWEN35_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def tiny_qwen35() -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            max_position_embeddings=64,
            layer_types=["linear_attention", "full_attention"],
        )
    )


def qwen35_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
        target_modules=QWEN35_LORA_TARGET_MODULES,
    )


def test_qwen35_lora_config_matches_recipe() -> None:
    config = qwen35_lora_config()

    assert config.r == 16
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.bias == "none"
    assert config.use_rslora is False
    assert set(config.target_modules) == set(QWEN35_LORA_TARGET_MODULES)


def test_qwen35_lora_wraps_all_language_projection_families() -> None:
    model = get_peft_model(tiny_qwen35(), qwen35_lora_config())

    adapted = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    }
    assert adapted == set(QWEN35_LORA_TARGET_MODULES)
    assert all(
        "lora_" in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_qwen35_lora_works_with_async_chunked_loss() -> None:
    base_model = tiny_qwen35()
    patch_chunked_lm_head(
        base_model, chunk_size=32, temperature=1.0, output_router_logits=False
    )
    model = get_peft_model(base_model, qwen35_lora_config())

    result = model(
        input_ids=torch.randint(0, 128, (1, 8)),
        labels=torch.randint(0, 128, (1, 8)),
        use_cache=False,
    )

    assert set(result) == {"log_probs", "entropy", "aux_loss"}
    assert result["log_probs"].shape == (1, 7)
