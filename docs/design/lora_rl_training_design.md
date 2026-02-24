# Miles LoRA RL Training Framework Design

## 终极目标（必须达成）

在 miles 框架内完成 LoRA RL 训练闭环，确保 **FSDP** 与 **Megatron bridge** 两个后端都可稳定运行、可复现，并在 AIME 2025 评测上相对 baseline 明确涨分且符合预期表现。

执行上以以下两个脚本为唯一主入口，持续调参直至目标达成：
- `/opt/tiger/miles/examples/lora/run-qwen3-8B-aime-fsdp-lora.sh`
- `/opt/tiger/miles/examples/lora/run-qwen3-8B-aime-megatron-lora.sh`

达标条件：
- 两个后端均能端到端完成训练与评估（无阻塞性报错，checkpoint 可保存/恢复）。
- 两个后端在同一评测口径下均较 baseline（约 15-20%）取得显著提升。
- FSDP 与 Megatron 的最终精度差异控制在可接受范围（目标 < 2%）。

## Context

Miles 框架当前只支持全参数 RL 训练，不支持 LoRA/PEFT 参数高效训练。参考 [verl 的 LoRA 设计](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3moe-30b_megatron_lora.sh)，为 miles 添加 LoRA RL 训练支持，覆盖 Megatron (bridge mode) 和 FSDP 两个训练后端。

LoRA 训练可以将训练参数量降低到 ~1%，大幅减少显存占用、通信开销和 checkpoint 大小。

---

## 总体架构

```
                      CLI Arguments (--lora-rank, --lora-alpha, ...)
                              |
                       miles/utils/lora_utils.py (共享工具)
                      /                          \
              Megatron Backend                  FSDP Backend
              (bridge mode only)                (HuggingFace PEFT)
              |                                  |
    megatron.bridge.peft.LoRA            peft.get_peft_model()
    applied via pre-wrap hook            applied before FSDP2
              |                                  |
    DDP wrapping (only LoRA grads)       FSDP wrapping (only LoRA grads)
              |                                  |
              +------------ 训练 ----------------+
              |          (只更新 LoRA 参数)         |
              +-------- Weight Sync to SGLang ----+
              |   Phase 1: base weights (一次)     |
              |   Phase 2: adapter only (每步)     |
              +------ Checkpoint (只存 adapter) ---+
```

---

## verl LoRA 设计参考分析

### verl 的 LoRA 配置 (HFModelConfig.lora)

```python
{
    "rank": 32,            # LoRA 维度
    "alpha": 64,           # 缩放因子
    "type": "lora",        # lora / canonical_lora / dora / vlm_lora
    "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    "exclude_modules": [],
    "dropout": 0.0,
    "dropout_position": "pre",
    "lora_A_init_method": "kaiming",
    "lora_B_init_method": "zero",
    "merge": False,        # 是否合并后再同步
    "adapter_path": None,  # 预训练 adapter 路径
}
```

### verl 的 LoRA 应用流程 (Megatron)

```
1. get_peft_cls(model_config, bridge, provider) → LoRA instance
2. provider.register_pre_wrap_hook(lambda model: peft_cls(model, training=True))
   - 冻结 base model 参数 (requires_grad=False)
   - 添加 LoRA A/B adapter 矩阵
   - 必须在 DDP wrapping 之前
3. provider.provide_distributed_model() → DDP 包装 (知道哪些参数需要梯度)
4. 训练: optimizer 只更新 LoRA 参数
5. 权重同步: 两阶段
   - Phase 1 (首次): export_hf_weights() + add_base_layer_suffix() → base 权重
   - Phase 2 (每步): export_adapter_weights() → 只有 adapter 权重 (~1%)
6. Ref 模型: 纯 base 模型，不应用 LoRA
```

### verl 的参数名映射 (Megatron → HuggingFace)

```python
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
```

---

## 文件修改清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `miles/utils/lora_utils.py` | LoRA 共享工具：配置构建、参数过滤、名称转换、PEFT config |
| `scripts/models/qwen3-30b-lora.sh` | LoRA RL 训练示例脚本 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `miles/utils/arguments.py` | 添加 LoRA CLI 参数和校验 |
| `miles/backends/megatron_utils/model_provider.py` | Bridge 模式下通过 pre-wrap hook 应用 LoRA |
| `miles/backends/megatron_utils/model.py` | LoRA 参数冻结 & optimizer 只跟踪 adapter 参数 |
| `miles/backends/megatron_utils/actor.py` | 两阶段 weight sync、ref 模型处理、checkpoint 调度 |
| `miles/backends/megatron_utils/update_weight/common.py` | 添加 LoRA 参数过滤的 iterator |
| `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | adapter-only 同步模式 |
| `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | adapter-only 同步模式 |
| `miles/backends/megatron_utils/checkpoint.py` | 保存/加载 LoRA adapter checkpoint |
| `miles/backends/fsdp_utils/actor.py` | PEFT 包装、LoRA optimizer、FSDP2 兼容 |
| `miles/backends/fsdp_utils/update_weight_utils.py` | adapter-only 同步模式 |
| `miles/backends/fsdp_utils/checkpoint.py` | LoRA adapter checkpoint |
| `miles/backends/sglang_utils/sglang_engine.py` | 添加 LoRA adapter 加载接口 |

---

## 详细实现方案

### Step 1: 共享基础设施

#### 1.1 添加 LoRA CLI 参数

**文件**: `miles/utils/arguments.py`

在 `add_miles_arguments` 中新增 `add_lora_arguments(parser)`，并在 parser 链中调用:

```python
def add_lora_arguments(parser):
    parser.add_argument("--lora-rank", type=int, default=None,
        help="LoRA rank. 设置后启用 LoRA 训练。")
    parser.add_argument("--lora-alpha", type=int, default=None,
        help="LoRA scaling alpha. 默认为 2 * lora_rank。")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None,
        help="LoRA 目标模块。"
             "Megatron bridge 默认: linear_qkv linear_proj linear_fc1 linear_fc2; "
             "FSDP 默认: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj")
    parser.add_argument("--lora-exclude-modules", type=str, nargs="+", default=None,
        help="从 LoRA 中排除的模块")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
        help="LoRA dropout 比率")
    parser.add_argument("--lora-type", type=str, default="lora",
        choices=["lora", "canonical_lora", "dora"],
        help="LoRA 变体类型")
    parser.add_argument("--lora-a-init-method", type=str, default="kaiming",
        choices=["kaiming", "xavier"],
        help="LoRA A 矩阵初始化方法")
    parser.add_argument("--lora-b-init-method", type=str, default="zero",
        choices=["zero", "random"],
        help="LoRA B 矩阵初始化方法")
    parser.add_argument("--lora-merge-on-sync", action="store_true", default=False,
        help="合并 LoRA 权重后同步到 rollout 引擎（不使用 adapter-only 同步）")
    parser.add_argument("--save-lora-only", action="store_true", default=True,
        help="Checkpoint 只保存 LoRA adapter 权重")
    return parser
```

在 `miles_validate_args` 中添加校验:

```python
# LoRA 校验
args.use_lora = args.lora_rank is not None
if args.use_lora:
    if args.train_backend == "megatron":
        assert args.megatron_to_hf_mode == "bridge", (
            "LoRA training with Megatron backend requires --megatron-to-hf-mode bridge. "
            "Raw mode does not support LoRA adapter handling."
        )
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    if args.lora_target_modules is None:
        if args.train_backend == "megatron":
            args.lora_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        else:
            args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"]
```

#### 1.2 创建共享 LoRA 工具模块

**新文件**: `miles/utils/lora_utils.py`

```python
"""Shared LoRA utilities for both Megatron and FSDP backends."""

from collections.abc import Iterator

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
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
}

# SGLang/vLLM stacked params that need base_layer suffix
STACKED_PARAMS = [
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
    ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias",
    ".kv_a_proj_with_mqa.weight", ".kv_b_proj.weight",
    ".q_a_proj.weight", ".q_b_proj.weight",
]


def is_lora_enabled(args) -> bool:
    return getattr(args, "use_lora", False)


def is_lora_param(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def filter_lora_params(named_params) -> Iterator:
    for name, param in named_params:
        if is_lora_param(name):
            yield name, param


def filter_base_params(named_params) -> Iterator:
    for name, param in named_params:
        if not is_lora_param(name):
            yield name, param


def get_megatron_lora_config(args) -> dict:
    """Build Megatron Bridge LoRA config dict."""
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
    """Build PEFT config dict that SGLang/vLLM can understand."""
    target_modules = args.lora_target_modules
    if args.train_backend == "megatron":
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
    """Add .base_layer suffix to stacked params for SGLang/vLLM LoRA loading."""
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
            # "model.layers.0.q_proj.weight" → "model.layers.0.base_layer.q_proj.weight"
            parts = matched_suffix.rsplit(".", 1)
            if len(parts) == 2:
                attr_name = parts[-1]  # "weight"
                prefix = name[:-len(attr_name)]  # "model.layers.0.q_proj."
                name = f"{prefix}base_layer.{attr_name}"
        yield name, param
```

---

### Step 2: Megatron Backend (Bridge Mode)

#### 2.1 Model Provider: 通过 Bridge Pre-Wrap Hook 应用 LoRA

**文件**: `miles/backends/megatron_utils/model_provider.py`

修改 `get_model_provider_func()` 中 `if args.megatron_to_hf_mode == "bridge":` 分支 (L84-96):

```python
if args.megatron_to_hf_mode == "bridge":
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.sequence_parallel = args.sequence_parallel
    provider.finalize()

    # ============ 新增: LoRA 支持 ============
    from miles.utils.lora_utils import is_lora_enabled
    if is_lora_enabled(args) and role == "actor":
        peft_cls = _get_megatron_peft_cls(args, bridge, provider)
        if peft_cls is not None:
            import logging
            _logger = logging.getLogger(__name__)

            def peft_pre_wrap_hook(model):
                """Apply LoRA before DDP wrapping. Must freeze base params first."""
                model = peft_cls(model, training=True)
                peft_cls.set_params_to_save(model)
                # Log trainable parameter count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                _logger.info(
                    f"LoRA applied: {trainable_params:,} trainable params / "
                    f"{total_params:,} total ({100*trainable_params/total_params:.2f}%)"
                )
                return model

            provider.register_pre_wrap_hook(peft_pre_wrap_hook)
    # ============ End LoRA ============

    return provider.provide
```

新增辅助函数:

```python
def _get_megatron_peft_cls(args, bridge, provider):
    """Create Megatron Bridge PEFT class instance (mirrors verl's get_peft_cls)."""
    from miles.utils.lora_utils import get_megatron_lora_config

    lora_config = get_megatron_lora_config(args)
    lora_type = lora_config.get("type", "lora")

    try:
        if lora_type == "lora":
            from megatron.bridge.peft.lora import LoRA
            return LoRA(
                target_modules=lora_config["target_modules"],
                dim=lora_config["rank"],
                alpha=lora_config["alpha"],
                dropout=lora_config.get("dropout", 0.0),
                lora_A_init_method=lora_config.get("lora_A_init_method", "kaiming"),
                lora_B_init_method=lora_config.get("lora_B_init_method", "zero"),
                exclude_modules=lora_config.get("exclude_modules", []),
            )
        elif lora_type == "canonical_lora":
            from megatron.bridge.peft.canonical_lora import CanonicalLoRA
            return CanonicalLoRA(
                target_modules=lora_config["target_modules"],
                dim=lora_config["rank"],
                alpha=lora_config["alpha"],
            )
        elif lora_type == "dora":
            from megatron.bridge.peft.dora import DoRA
            return DoRA(
                target_modules=lora_config["target_modules"],
                dim=lora_config["rank"],
                alpha=lora_config["alpha"],
            )
    except ImportError:
        raise ImportError(
            "LoRA training with Megatron backend requires megatron-bridge >= 0.2.0 "
            "with peft support. Install with: pip install megatron-bridge>=0.2.0"
        )
```

**关键设计点**:
- LoRA 必须在 DDP wrapping 之前应用 (`provider.register_pre_wrap_hook`)
- 只对 actor 角色应用 LoRA，critic 和 ref 模型不应用
- 依赖 `megatron.bridge.peft` 模块，需要 megatron-bridge >= 0.2.0

#### 2.2 Optimizer: 只跟踪 LoRA 参数

**文件**: `miles/backends/megatron_utils/model.py`

修改 `setup_model_and_optimizer()` (L86-126):

```python
def setup_model_and_optimizer(args, role="actor"):
    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # ============ 新增: LoRA 参数冻结 ============
    from miles.utils.lora_utils import is_lora_enabled, is_lora_param
    if is_lora_enabled(args) and role == "actor":
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if not is_lora_param(name):
                    param.requires_grad = False
    # ============ End LoRA ============

    # Optimizer (自动只跟踪 requires_grad=True 的参数)
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config, model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler
```

**注意**: 如果 `megatron.bridge.peft.LoRA` 已经在 pre-wrap hook 中冻结了 base 参数，这里的冻结是双重保险。Megatron 的 `get_megatron_optimizer` 会自动只收集 `requires_grad=True` 的参数到优化器中。

#### 2.3 Actor: 两阶段 Weight Sync 与 Ref 模型

**文件**: `miles/backends/megatron_utils/actor.py`

在 `MegatronTrainRayActor` 类中:

1. `init()` 方法中，weight_updater 初始化时传入 LoRA 配置:

```python
# 修改 weight_updater 初始化 (L134-141)
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
self.weight_updater = update_weight_cls(
    self.args,
    self.model,
    weights_getter=lambda: self.weights_backuper.get("actor"),
    model_name=...,
    quantization_config=...,
    # 新增: LoRA 配置
    lora_enabled=is_lora_enabled(self.args),
    lora_merge_on_sync=getattr(self.args, "lora_merge_on_sync", False),
)
```

2. `update_weights()` 方法保持不变，LoRA 逻辑在 weight_updater 内部处理。

3. **Ref 模型**: `load_other_checkpoint("ref", ...)` 加载 ref checkpoint 后，模型中 LoRA_B 权重为零（因为 ref checkpoint 没有 LoRA 训练），因此 ref forward 等效于纯 base 模型。无需额外修改。

#### 2.4 Weight Update: 两阶段 Adapter-Only 同步

**文件**: `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
**文件**: `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`

在两个 UpdateWeight 类中添加:

```python
class UpdateWeightFromTensor:
    def __init__(self, args, model, weights_getter, model_name, quantization_config,
                 lora_enabled=False, lora_merge_on_sync=False):
        # ... existing init ...
        self._lora_enabled = lora_enabled
        self._lora_merge_on_sync = lora_merge_on_sync
        self._base_sync_done = False

    def update_weights(self):
        self.weight_version += 1

        if self._lora_enabled and not self._lora_merge_on_sync:
            # === 两阶段 Adapter-Only 同步 ===
            if not self._base_sync_done:
                # Phase 1: 发送 base 权重 (只执行一次)
                self._sync_base_weights()
                self._base_sync_done = True
            # Phase 2: 只发送 adapter 权重 (每步)
            self._sync_adapter_weights()
        else:
            # 原有逻辑 (完整权重同步或 merge 模式)
            # 如果 lora_merge_on_sync=True，weights_getter 返回的权重中
            # LoRA 已被合并到 base 中
            self._original_update_weights()

    def _sync_base_weights(self):
        """Phase 1: 发送 base 模型权重，带 base_layer suffix"""
        megatron_weights = self.weights_getter()
        base_weights = filter_base_params(megatron_weights.items())
        hf_weights = self._hf_weight_iterator.get_hf_weight_chunks(dict(base_weights))
        # 添加 base_layer suffix
        hf_weights_with_suffix = add_base_layer_suffix(hf_weights, self._model_type)
        self._send_hf_params(hf_weights_with_suffix)

    def _sync_adapter_weights(self):
        """Phase 2: 只发送 LoRA adapter 权重"""
        megatron_weights = self.weights_getter()
        adapter_weights = dict(filter_lora_params(megatron_weights.items()))
        # 转换 Megatron adapter 名称为 HF 格式
        hf_adapter_weights = self._convert_adapter_names_to_hf(adapter_weights)
        # 保存到临时文件并通知 SGLang 加载
        self._load_adapter_to_engines(hf_adapter_weights)

    def _load_adapter_to_engines(self, adapter_weights):
        """通过 SGLang LoadLoRAAdapter 接口加载 adapter"""
        import tempfile, json, torch
        from miles.utils.lora_utils import build_peft_config_for_sglang

        # 保存 adapter 到临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(adapter_weights, f"{tmpdir}/adapter_model.bin")
            peft_config = build_peft_config_for_sglang(self.args)
            with open(f"{tmpdir}/adapter_config.json", "w") as f:
                json.dump(peft_config, f)

            # 通知所有 SGLang 引擎加载 adapter
            for engine in self._engines:
                engine.load_lora_adapter(tmpdir)
```

**注意**: 如果 SGLang 不支持从临时文件加载 LoRA (需要共享文件系统)，可以改用内存传输方式——将 adapter 权重序列化后通过现有的 `update_weights_from_tensor` 接口发送，但需要 SGLang 侧能识别这是 LoRA 权重。

**替代方案 (更简单)**: 首先实现 merge 模式 (`--lora-merge-on-sync`)，在 `weights_getter` 中返回合并后的权重。这不需要任何 SGLang 侧修改:

```python
# 在 MegatronTrainRayActor.init() 中修改 weights_getter
if is_lora_enabled(args) and args.lora_merge_on_sync:
    # Merge LoRA into base before returning
    def merged_weights_getter():
        weights = self.weights_backuper.get("actor")
        return merge_lora_into_base(weights, args)
    self.weight_updater = update_weight_cls(
        ..., weights_getter=merged_weights_getter, ...
    )
```

#### 2.5 SGLang 引擎: 添加 LoRA Adapter 加载接口

**文件**: `miles/backends/sglang_utils/sglang_engine.py`

新增方法:

```python
def load_lora_adapter(self, adapter_path: str, adapter_name: str = "default"):
    """通过 SGLang 的 LoadLoRAAdapterReqInput 加载 LoRA adapter"""
    return self._make_request("load_lora_adapter", {
        "lora_path": adapter_path,
        "lora_name": adapter_name,
    })

def unload_lora_adapter(self, adapter_name: str = "default"):
    """卸载已加载的 LoRA adapter"""
    return self._make_request("unload_lora_adapter", {
        "lora_name": adapter_name,
    })
```

SGLang 已有 `LoadLoRAAdapterReqInput` 支持（见 `docker/patch/latest/sglang.patch`）。

#### 2.6 Checkpoint: 只保存 Adapter

**文件**: `miles/backends/megatron_utils/checkpoint.py`

新增函数:

```python
def save_lora_checkpoint(iteration, model, args):
    """Save only LoRA adapter parameters."""
    import json
    from pathlib import Path
    from miles.utils.lora_utils import is_lora_param, get_megatron_lora_config

    save_dir = Path(args.save) / f"iter_{iteration:07d}" / "lora_adapter"
    if mpu.get_data_parallel_rank() == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 收集 LoRA 参数
    lora_state = {}
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if is_lora_param(name):
                lora_state[name] = param.detach().cpu()

    # 保存 (每个 TP/PP rank 分别保存)
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    filename = save_dir / f"adapter_tp{tp_rank:02d}_pp{pp_rank:03d}.pt"
    torch.save(lora_state, filename)

    # Rank 0 保存配置
    if mpu.get_data_parallel_rank() == 0 and tp_rank == 0 and pp_rank == 0:
        config = get_megatron_lora_config(args)
        with open(save_dir / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)


def load_lora_checkpoint(model, checkpoint_path, args):
    """Load LoRA adapter parameters from checkpoint."""
    from pathlib import Path

    lora_dir = Path(checkpoint_path) / "lora_adapter"
    if not lora_dir.exists():
        return False

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    filename = lora_dir / f"adapter_tp{tp_rank:02d}_pp{pp_rank:03d}.pt"

    if filename.exists():
        lora_state = torch.load(filename, map_location="cpu")
        for model_chunk in model:
            model_chunk.load_state_dict(lora_state, strict=False)
        return True
    return False
```

在 `save_checkpoint` 中添加 LoRA 分支:

```python
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
    if is_lora_enabled(get_args()) and get_args().save_lora_only:
        save_lora_checkpoint(iteration, model, get_args())
        # 也保存 optimizer state (已经只包含 LoRA 参数的状态)
        # ... 可选: 保存 optimizer
    else:
        # 原有 Megatron checkpoint 逻辑
        megatron_save_checkpoint(...)
```

---

### Step 3: FSDP Backend

#### 3.1 Actor: 在 FSDP2 之前应用 PEFT

**文件**: `miles/backends/fsdp_utils/actor.py`

修改 `FSDPTrainRayActor.init()` (L93-112):

```python
init_context = self._get_init_weight_context_manager()
with init_context():
    model = self.get_model_cls().from_pretrained(
        self.args.hf_checkpoint,
        trust_remote_code=True,
        attn_implementation=self.args.attn_implementation,
    )

# ============ 新增: LoRA 应用 ============
from miles.utils.lora_utils import is_lora_enabled
if is_lora_enabled(self.args):
    from peft import get_peft_model
    from miles.utils.lora_utils import get_peft_lora_config
    lora_config = get_peft_lora_config(self.args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
# ============ End LoRA ============

model.train()
full_state = model.state_dict()
model = apply_fsdp2(model, mesh=self.parallel_state.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)
model = self._fsdp2_load_full_state_dict(model, full_state, self.parallel_state.dp_mesh, ...)
self.model = model
```

修改 Optimizer 创建 (L117-126):

```python
if is_lora_enabled(self.args):
    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
    logger.info(f"LoRA optimizer: {len(trainable_params)} parameter groups")
    self.optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps, weight_decay=args.weight_decay,
    )
elif args.optimizer == "adam":
    self.optimizer = torch.optim.AdamW(self.model.parameters(), ...)
```

#### 3.2 FSDP2 兼容性

**文件**: `miles/backends/fsdp_utils/actor.py` 中的 `apply_fsdp2()` (L656-707)

修改 `_no_split_modules` 检测以兼容 PEFT 包装:

```python
def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # ============ 修改: 兼容 PEFT 模型 ============
    base_model = model
    if hasattr(model, 'base_model'):
        # PEFT wraps model: PeftModelForCausalLM.base_model.model = original model
        base_model = getattr(model.base_model, 'model', model.base_model)

    layer_cls_to_wrap = getattr(base_model, '_no_split_modules', None)
    if layer_cls_to_wrap is None:
        layer_cls_to_wrap = model._no_split_modules
    # ============ End 修改 ============

    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # ... rest unchanged ...
```

#### 3.3 Ref 模型

`_create_ref_model()` 无需修改。它已经创建纯 base 模型（不调用 `get_peft_model`），天然作为 LoRA 训练的参考模型。

#### 3.4 Weight Sync

**文件**: `miles/backends/fsdp_utils/update_weight_utils.py`

修改 `UpdateWeight` 基类:

```python
class UpdateWeight:
    def __init__(self, args, model):
        self.weight_version = 0
        self.args = args
        self.model = model
        self._lora_enabled = is_lora_enabled(args)
        self._lora_merge_on_sync = getattr(args, "lora_merge_on_sync", False)
        self._base_sync_done = False

    def update_weights(self):
        self.weight_version += 1

        if self._lora_enabled and not self._lora_merge_on_sync:
            if not self._base_sync_done:
                self._sync_base_weights()
                self._base_sync_done = True
            self._sync_adapter_weights()
        elif self._lora_enabled and self._lora_merge_on_sync:
            # 合并 LoRA 后发送完整权重
            self.model.merge_adapter()
            self._sync_all_weights()
            self.model.unmerge_adapter()
        else:
            self._sync_all_weights()
```

#### 3.5 Checkpoint

**文件**: `miles/backends/fsdp_utils/checkpoint.py`

修改 `save()`:

```python
def save(actor, iteration):
    if is_lora_enabled(actor.args) and actor.args.save_lora_only:
        adapter_dir = f"{actor.args.save}/iter_{iteration:07d}/lora_adapter"
        if dist.get_rank() == 0:
            os.makedirs(adapter_dir, exist_ok=True)
        dist.barrier()
        # 使用 PEFT save_pretrained
        if hasattr(actor.model, 'save_pretrained'):
            actor.model.save_pretrained(adapter_dir)
        # 保存 optimizer state
        opt_dir = f"{actor.args.save}/iter_{iteration:07d}/optimizer"
        # ... save optimizer ...
    else:
        # 原有 checkpoint 逻辑
```

修改 `load()`:

```python
def load(actor):
    lora_dir = f"{actor.args.load}/lora_adapter"
    if os.path.exists(lora_dir) and is_lora_enabled(actor.args):
        from peft import PeftModel
        # 加载 LoRA adapter
        if hasattr(actor.model, 'load_adapter'):
            actor.model.load_adapter(lora_dir, adapter_name="default")
    else:
        # 原有 checkpoint 加载逻辑
```

---

### Step 4: 示例脚本

**新文件**: `scripts/models/qwen3-30b-lora.sh`

```bash
#!/bin/bash
# LoRA RL Training for Qwen3-30B-A3B with Miles

set -xeuo pipefail

# LoRA 配置
LORA_ARGS=(
    --lora-rank 32
    --lora-alpha 64
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
    --lora-type lora
    --lora-a-init-method kaiming
    --save-lora-only
)

# Megatron Bridge 模式 (LoRA 必须使用 bridge)
TRAIN_ARGS=(
    --train-backend megatron
    --megatron-to-hf-mode bridge
    --hf-checkpoint Qwen/Qwen3-30B-A3B-Instruct
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --context-parallel-size 2
    --colocate
)

# 训练超参
ALGO_ARGS=(
    --lr 3e-6
    --rollout-batch-size 128
    --n-samples-per-prompt 4
    --num-rollout 200
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --ref-load /path/to/base/checkpoint
)

python -m miles.pipeline.run \
    "${TRAIN_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "$@"
```

---

## 关键设计决策总结

| 决策 | 选择 | 原因 |
|------|------|------|
| Megatron LoRA 入口 | Bridge `provider.register_pre_wrap_hook()` | LoRA 必须在 DDP 之前应用 |
| Megatron LoRA 实现 | `megatron.bridge.peft.LoRA` | verl 验证过的成熟方案 |
| FSDP LoRA 入口 | `peft.get_peft_model()` before `apply_fsdp2()` | HuggingFace PEFT 与 HF 模型无缝集成 |
| Megatron mode 要求 | 只支持 `--megatron-to-hf-mode bridge` | raw 模式手动 HF 转换太复杂 |
| Base 模型冻结 | `requires_grad = False` | 标准做法 |
| Ref 模型 | base 权重 + zero LoRA_B | 零额外开销 |
| Weight Sync 默认 | 两阶段 adapter-only | 通信量降低 ~99% |
| Weight Sync 备选 | `--lora-merge-on-sync` | 无需 SGLang LoRA 支持 |
| SGLang 加载 | `LoadLoRAAdapterReqInput` (文件路径) | SGLang 原生支持 |
| Checkpoint | 只保存 adapter 权重 | ~99% 体积缩减 |

---

## 风险与应对

| 风险 | 应对方案 |
|------|----------|
| `megatron.bridge.peft` 不可用 | 要求 megatron-bridge >= 0.2.0；若不可用提供清晰错误信息 |
| PEFT + FSDP2 兼容性 | 需要 peft >= 0.13.0；调整 `_no_split_modules` 检测 |
| SGLang adapter-only sync | 首先实现 `--lora-merge-on-sync` 作为 fallback |
| Megatron 分布式 optimizer | LoRA 参数少，shard 粒度变化需验证 |
| 临时文件共享 (多节点) | adapter-only sync 需要共享文件系统或改用内存传输 |

---

## 实现优先级

1. **P0**: `lora_utils.py` + CLI 参数 (共享基础)
2. **P0**: FSDP LoRA 训练 (最简单的端到端验证)
3. **P1**: Megatron bridge LoRA 训练
4. **P1**: `--lora-merge-on-sync` weight sync (简单但非最优)
5. **P2**: Adapter-only 两阶段 weight sync
6. **P2**: LoRA adapter checkpoint save/load
7. **P3**: SGLang 原生 LoRA adapter 加载优化

---

## 验证方案

1. **单元测试**: `lora_utils.py` 参数过滤、名称转换
2. **FSDP LoRA E2E**: Qwen2-0.5B + FSDP + LoRA → 验证 loss 下降
3. **Megatron LoRA E2E**: Qwen2-0.5B + bridge + LoRA → 验证 loss 下降
4. **Weight Sync**: adapter-only sync 后推理结果与合并权重一致
5. **Checkpoint**: 保存 → 加载 → 验证训练可继续
6. **Ref 模型**: KL divergence 计算正确

---

## 持续调参执行规范（AIME 2025）

调参过程中，默认只修改并迭代这两个脚本中的参数组合，不引入新的实验入口：
- `/opt/tiger/miles/examples/lora/run-qwen3-8B-aime-fsdp-lora.sh`
- `/opt/tiger/miles/examples/lora/run-qwen3-8B-aime-megatron-lora.sh`

每轮迭代至少记录以下指标并与上一轮对比：
1. AIME 2025 pass@1（主指标）
2. 训练 reward 曲线趋势（是否稳定上升）
3. 吞吐与显存（tokens/sec/GPU、峰值显存）
4. 训练稳定性（是否出现 NaN、OOM、发散、卡死）

调参优先级建议：
1. 批大小与采样相关：`rollout-batch-size`、`n-samples-per-prompt`、`global-batch-size`
2. RL 稳定性相关：`kl-loss-coef`、`eps-clip`、`eps-clip-high`、`entropy-coef`
3. LoRA 容量相关：`lora-rank`、`lora-alpha`、`lora-dropout`
4. 性能相关：`max-tokens-per-gpu`、`sglang-mem-fraction-static`、重计算与并行参数

---

## Qwen3-8B AIME 2025 性能测试计划

### 测试目标

验证 Miles LoRA RL 训练在竞赛数学场景 (AIME 2025) 下的效果，对比 FSDP 和 Megatron 两个后端。

### 测试矩阵

| 实验 | 模型 | 后端 | 训练数据 | 评估数据 | 脚本 | 状态 |
|------|------|------|----------|----------|------|------|
| Baseline | Qwen3-8B | - | - | AIME 2025 | - | 待测 |
| FSDP LoRA | Qwen3-8B | FSDP | MATH L3-5 | AIME 2025 | `run-qwen3-8B-aime-fsdp-lora.sh` | 待测 |
| Megatron LoRA | Qwen3-8B | Megatron bridge | MATH L3-5 | AIME 2025 | `run-qwen3-8B-aime-megatron-lora.sh` | 待测 |

### 实验配置

#### 统一超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank | 64 | 适中的 adapter 容量 |
| LoRA alpha | 32 | alpha / rank = 0.5 |
| 训练数据 | MATH level 3-5 | ~9K 竞赛数学题 |
| 评估数据 | AIME 2025 | 30 道整数答案题 (0-999) |
| Learning rate | 3e-6 | 恒定学习率 |
| Rollout batch size | 16 | 每批采样 16 个 prompt |
| Samples per prompt | 8 | 每个 prompt 采 8 条 |
| Max response length | 8192 (train) / 16384 (eval) | 允许长链式推理 |
| Global batch size | 128 | GRPO 训练批大小 |
| Advantage estimator | GRPO | Group Relative Policy Optimization |
| KL loss | low_var_kl, coef=0.001 | 防止策略偏移过大 |
| Reward model | math (boxed answer extraction) | SymPy 等价性检查 |
| Save interval | 20 rollouts | |
| Eval interval | 10 rollouts | |

#### 后端差异

| 参数 | FSDP | Megatron Bridge |
|------|------|-----------------|
| `--train-backend` | fsdp | megatron |
| `--megatron-to-hf-mode` | - | bridge |
| `--tensor-model-parallel-size` | - | 1 |
| `--pipeline-model-parallel-size` | - | 1 |
| `--seq-length` | - | 8192 |
| `--attention-backend` | flash_attention_2 | flash |
| Gradient checkpointing | `--gradient-checkpointing` | `--recompute-granularity full` |
| Model args | 自动 (HF config) | `scripts/models/qwen3-8B.sh` |
| LoRA target modules | q/k/v/o/gate/up/down_proj | linear_qkv/proj/fc1/fc2 |

### 数据准备

```bash
# 准备 MATH level 3-5 训练数据和 AIME 2025 评估数据
python examples/lora/prepare_math_aime_data.py ${DATA_DIR}
```

输出:
- `${DATA_DIR}/math_level3to5/train.jsonl` (~9K 条)
- `${DATA_DIR}/aime2025/aime2025.jsonl` (30 条)

### 运行步骤

```bash
# Step 1: 准备数据
export DATA_DIR="/root/data"
python examples/lora/prepare_math_aime_data.py ${DATA_DIR}

# Step 2a: FSDP LoRA 训练
NUM_GPUS=2 bash examples/lora/run-qwen3-8B-aime-fsdp-lora.sh

# Step 2b: Megatron LoRA 训练
NUM_GPUS=2 bash examples/lora/run-qwen3-8B-aime-megatron-lora.sh
```

### 预期结果

| 指标 | Baseline | FSDP LoRA | Megatron LoRA |
|------|----------|-----------|---------------|
| AIME 2025 (pass@1) | ~15-20% | ~25-35% | ~25-35% |
| 训练 reward 曲线 | - | 单调上升 | 单调上升 |
| FSDP vs Megatron 精度差异 | - | - | < 2% |

### 关注指标

1. **AIME 2025 pass@1**: 训练前后在 30 道 AIME 题上的正确率
2. **训练 reward 曲线**: MATH 数据集上的平均 reward 是否单调上升
3. **训练吞吐量**: tokens/sec/GPU，两个后端的对比
4. **显存占用**: 峰值 GPU 显存，LoRA vs full-param 的节省比例
5. **收敛速度**: 达到最优 AIME 性能所需的 rollout 数
6. **FSDP vs Megatron 一致性**: 两个后端训练后模型在同一评估集上的性能差异
