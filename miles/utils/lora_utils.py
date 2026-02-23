"""Shared LoRA utilities for both Megatron and FSDP backends."""

from collections.abc import Iterator

# Megatron module name -> HuggingFace module name(s)
MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "router": ["gate"],
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_up": ["up_proj"],
    "linear_fc1_gate": ["gate_proj"],
    # DeepSeek MLA
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_q_proj": ["q_proj"],
}

# SGLang/vLLM stacked params that need base_layer suffix for adapter-only sync
STACKED_PARAMS = [
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".mlp.gate.weight",
    ".mlp.gate.e_score_correction_bias",
    ".kv_a_proj_with_mqa.weight",
    ".kv_b_proj.weight",
    ".q_a_proj.weight",
    ".q_b_proj.weight",
]

# Default target modules for each backend
MEGATRON_DEFAULT_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
FSDP_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_lora_enabled(args) -> bool:
    """Check if LoRA training is enabled."""
    return getattr(args, "use_lora", False)


def is_lora_param(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter.

    Handles multiple naming conventions:
    - HuggingFace PEFT: "lora_A", "lora_B" in param name
    - Megatron Bridge (simple linear): "lora_a", "lora_b" in param name
    - Megatron Bridge (parallel linear): ".adapter." in param name or ends with ".adapters"
    """
    name_lower = name.lower()
    return "lora_a" in name_lower or "lora_b" in name_lower or ".adapter." in name or name.endswith(".adapters")


def filter_lora_params(named_params) -> Iterator:
    """Filter to only yield LoRA adapter parameters."""
    for name, param in named_params:
        if is_lora_param(name):
            yield name, param


def filter_base_params(named_params) -> Iterator:
    """Filter to only yield base model parameters (non-LoRA)."""
    for name, param in named_params:
        if not is_lora_param(name):
            yield name, param


def get_megatron_lora_config(args) -> dict:
    """Build Megatron Bridge LoRA config dict from CLI arguments."""
    return {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "type": getattr(args, "lora_type", "lora"),
        "target_modules": args.lora_target_modules,
        "exclude_modules": getattr(args, "lora_exclude_modules", None) or [],
        "dropout": getattr(args, "lora_dropout", 0.0),
        "lora_A_init_method": getattr(args, "lora_a_init_method", "kaiming"),
        "lora_B_init_method": getattr(args, "lora_b_init_method", "zero"),
    }


def get_peft_lora_config(args):
    """Build HuggingFace PEFT LoraConfig for FSDP backend."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        modules_to_save=getattr(args, "lora_modules_to_save", None),
        lora_dropout=getattr(args, "lora_dropout", 0.0),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def convert_megatron_to_hf_target_modules(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron module names to HuggingFace equivalents."""
    hf_modules = []
    seen = set()
    for m in megatron_modules:
        for hf_m in MEGATRON_TO_HF_MODULES.get(m, [m]):
            if hf_m not in seen:
                hf_modules.append(hf_m)
                seen.add(hf_m)
    return hf_modules


def build_peft_config_for_sglang(args) -> dict:
    """Build PEFT config dict that SGLang/vLLM can understand for adapter loading."""
    target_modules = args.lora_target_modules
    if getattr(args, "train_backend", None) == "megatron":
        target_modules = convert_megatron_to_hf_target_modules(target_modules)
    return {
        "task_type": "CAUSAL_LM",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "bias": "none",
        "lora_dropout": getattr(args, "lora_dropout", 0.0),
    }


def add_base_layer_suffix(named_params, model_type: str = None) -> Iterator:
    """Add .base_layer suffix to stacked params for SGLang/vLLM LoRA loading.

    When using adapter-only sync, SGLang/vLLM expects base weights to have
    a .base_layer prefix before the module name, e.g.:
        "model.layers.0.q_proj.weight" -> "model.layers.0.q_proj.base_layer.weight"
    """
    stacked = STACKED_PARAMS
    if model_type == "llama":
        stacked = [".embed_tokens.weight"] + stacked

    for name, param in named_params:
        matched_suffix = None
        for suffix in stacked:
            if name.endswith(suffix):
                matched_suffix = suffix
                break
        if matched_suffix:
            # Insert "base_layer." before the attribute name
            # e.g., "model.layers.0.q_proj.weight" -> "model.layers.0.q_proj.base_layer.weight"
            attr_parts = matched_suffix.lstrip(".")
            # Find where the module name ends and the attribute begins
            last_dot = attr_parts.rfind(".")
            if last_dot >= 0:
                module_part = attr_parts[:last_dot]  # e.g., "q_proj"
                attr_part = attr_parts[last_dot + 1 :]  # e.g., "weight"
                prefix = name[: -len(attr_parts)]  # e.g., "model.layers.0."
                name = f"{prefix}{module_part}.base_layer.{attr_part}"
        yield name, param


def count_trainable_params(model) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
