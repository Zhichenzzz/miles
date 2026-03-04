---
name: miles-agentic-rl-pipeline-zh
description: 用中文分析、讲解、排障和改造 Miles 的 agentic RL 训练全链路（数据源、rollout 生成、工具调用代理循环、奖励计算、rollout 缓冲区、训练数据转换、Actor/Critic 优化、权重同步）。当需求提到 Miles agentic 或 multi-turn RL、自定义 generate 或 rollout 函数、OpenAI session 路由、Miles Router、rollout 到训练数据流时使用。
---

# Miles Agentic RL Pipeline 中文版

## 目标
理解并可直接操作 Miles 从提示采样到优化更新的 agentic RL 训练流程。

## 起步顺序
1. 先确认运行模式。
   - `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` 会启用新路径 `InferenceRolloutFn` 与 `generate_hub/*`。
2. 确认 rollout 入口。
   - `train.py` -> `create_rollout_manager` -> `RolloutManager.generate`。
3. 确认 agentic 生成函数。
   - `--custom-generate-function-path`（refactor 模式下默认会回退到 `single_turn.generate`）。
4. 确认奖励来源。
   - `--custom-rm-path` 或 `--rm-type`。
5. 确认路由模式。
   - OpenAI session 流必须开启 `--use-miles-router`。

## 控制流总览（Control Plane）
`train.py` 主循环：
1. 创建 Ray placement groups（`miles/ray/placement_group.py`）。
2. 启动 `RolloutManager`（`miles/ray/rollout.py`）。
3. 初始化 actor 或 critic 训练组（`miles/ray/actor_group.py` + 后端 actor）。
4. 先把 actor 权重同步到 rollout 引擎（`actor_model.update_weights()`）。
5. 对每个 `rollout_id`：
   - `rollout_manager.generate(rollout_id)` 产出 rollout 数据并转成 train refs。
   - `actor_model.async_train(...)`（PPO 时还有 critic）。
   - 按配置执行保存和评估。
   - 把新权重同步回 rollout 引擎。
6. 训练结束后 `dispose` rollout manager。

## 数据流总览（Data Plane）
`RolloutManager.generate`：
1. 调用 rollout function。
   - refactor：`InferenceRolloutFn.__call__(RolloutFnTrainInput)`。
   - legacy：`sglang_rollout.generate_rollout`。
2. 收集分组 `Sample`。
3. 展平并按 batch 规则裁剪或重排。
4. 把 `Sample` 转为训练字典。
   - 常见键：`tokens`、`response_lengths`、`rewards`、`loss_masks`。
   - 可选键：`rollout_log_probs`、`rollout_routed_experts`、多模态输入与 metadata。
5. 按 DP 切分后通过 Ray refs 传给训练端。

## Refactor Rollout 细节
文件：`miles/rollout/inference_rollout/inference_rollout_common.py` 与 `miles/rollout/inference_rollout/inference_rollout_train.py`。

1. 构造 `GenerateState`。
   - tokenizer 或 processor
   - rollout 默认 sampling 参数
   - 并发 semaphore
   - generate 函数（`load_generate_function(custom_generate_path) or single_turn.generate`）
2. 采样主循环。
   - `data_source.get_samples(over_sampling_batch_size)` 拉取每组 `n_samples_per_prompt`。
   - 每组提交一个异步任务（`generate_and_rm_group`）。
   - 对已完成组应用动态过滤（`--dynamic-sampling-filter-path`）。
   - 保留到 `rollout_batch_size`。
3. 中止剩余 pending 请求。
   - 若启用 `--partial-rollout`，把半成品样本回收进 buffer。
4. 执行可选后处理 hook。
   - `--rollout-sample-filter-path`
   - `--rollout-all-samples-process-path`

## Agentic 生成路径
### A) OpenAI Session Agentic（`agentic_tool_call.generate`）
文件：
- `miles/rollout/generate_hub/agentic_tool_call.py`
- `miles/rollout/generate_utils/openai_endpoint_utils.py`
- `miles/router/session/sessions.py`

流程：
1. 创建 session（`POST /sessions`）并拿到 `session_id`。
2. 调用 `--custom-agent-function-path` 指向的 agent 函数。
3. agent 向 `base_url + /v1/chat/completions` 发一到多次请求。
4. Miles Router 会记录 session 内 request 或 response（含 token logprobs）。
5. tracer 拉取记录（`GET /sessions/{id}`），删除 session，并把记录转成 `Sample`。
6. 除非开启 `--generate-multi-samples`，否则会合并多条记录。

硬性要求：
- 必须启用 `--use-miles-router`
- 请求中必须能返回 token 级信息（如 `logprobs=true`，且需 prompt token ids）
- OpenAI 格式输入通常不依赖 `--apply-chat-template`

### B) `/generate` 多轮工具调用（`multi_turn.generate`）
文件：
- `miles/rollout/generate_hub/multi_turn.py`
- `miles/rollout/generate_utils/tool_call_utils.py`

流程：
1. 先构造首轮 prompt token ids。
2. 按 `--generate-max-turns` 循环：
   - 调 `/generate`
   - 用 `--generate-tool-call-parser` 解析 tool call
   - 用 `--generate-execute-tool-function-path` 执行工具
   - 把 tool response token 追加到 sample，并把对应 loss_mask 置 0、rollout_log_probs 置 0
3. 无工具调用、长度上限、终止条件或 abort 时结束。

## 奖励计算阶段
文件：`miles/rollout/rm_hub/__init__.py`。

1. 有 `--custom-rm-path` 时优先走自定义函数（可单样本或 group 批处理）。
2. 否则按 `rm_type`（样本 metadata 优先于全局 `--rm-type`）分发：
   - `remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`, `ifbench`, `random`

## 数据源与 Buffer
文件：`miles/rollout/data_source.py`。

1. `RolloutDataSource` 负责加载数据集并产出分组样本。
2. `RolloutDataSourceWithBuffer` 会先从 buffer 取，再从数据集取。
3. partial rollout 或定制流程可通过 `add_samples` 把组样本回写 buffer。

## 训练侧消费
文件：
- `miles/backends/training_utils/data.py`
- `miles/backends/megatron_utils/actor.py`（或 FSDP actor）

流程：
1. 把 rollout refs 物化为每 rank 的训练张量（`get_rollout_data`）。
2. 计算 logprobs 或 values 与优势函数。
3. 执行 actor 或 critic 优化。
4. 更新 actor 备份权重。
5. 调 `update_weights` 把新权重推送到 rollout 引擎。

## 推荐扩展顺序
没有硬性需求时，按下面顺序改，风险最低。

1. 先改 `--custom-generate-function-path`。
2. 再接 `--custom-rm-path`。
3. 再加过滤或后处理 hooks。
   - `--dynamic-sampling-filter-path`
   - `--rollout-sample-filter-path`
   - `--rollout-all-samples-process-path`
   - `--rollout-data-postprocess-path`
4. 最后才覆盖训练数据转换。
   - `--custom-convert-samples-to-train-data-path`

## 最小可用配置
OpenAI 风格 agentic RL：
```bash
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

python train.py \
  --use-miles-router \
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate \
  --custom-agent-function-path <module.path.run_agent> \
  --rollout-batch-size <N> \
  --n-samples-per-prompt <K> \
  --prompt-data <jsonl> \
  --input-key <messages_key> \
  --label-key <label_key> \
  ...
```

`/generate` 工具调用多轮：
```bash
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

python train.py \
  --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate \
  --generate-max-turns 16 \
  --generate-tool-specs-path <module.path.tool_specs> \
  --generate-tool-call-parser <parser_name> \
  --generate-execute-tool-function-path <module.path.execute_tool> \
  ...
```

## 动态参数注入规则
文件：`miles/utils/arguments.py`。

1. 先 parse known args。
2. refactor 模式下，Miles 会加载：
   - `--rollout-function-path`
   - `--custom-generate-function-path`
3. 如果函数实现了 `add_arguments(parser)`，就会自动注入自定义 CLI 参数。
4. 用这个机制扩展 agent 或工具参数，不要直接改核心 parser。

## 排障清单
1. 先核对 rollout 路径。
   - `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR`
   - 实际生效的 `rollout_function_path` 与 `custom_generate_function_path`
2. 再核对路由模式。
   - OpenAI session 路径必须 `--use-miles-router`
3. 核对训练前样本一致性。
   - `len(tokens) >= response_length`
   - `len(loss_mask) == response_length`
   - 需要时 `len(rollout_log_probs) == response_length`
4. 核对 rollout 与训练 batch 契约。
   - 常见约束：`rollout_batch_size * n_samples_per_prompt` 与训练消耗计划匹配（`global_batch_size * num_steps_per_rollout`）
5. 用调试开关定位。
   - `--debug-rollout-only`
   - `--save-debug-rollout-data`
   - `--load-debug-rollout-data`
   - `--debug-train-only`

## 优先阅读文件
- `train.py`
- `miles/ray/placement_group.py`
- `miles/ray/rollout.py`
- `miles/rollout/inference_rollout/inference_rollout_common.py`
- `miles/rollout/inference_rollout/inference_rollout_train.py`
- `miles/rollout/generate_hub/agentic_tool_call.py`
- `miles/rollout/generate_hub/multi_turn.py`
- `miles/rollout/generate_utils/openai_endpoint_utils.py`
- `miles/rollout/data_source.py`
- `miles/rollout/rm_hub/__init__.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/megatron_utils/actor.py`
