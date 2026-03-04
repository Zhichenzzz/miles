---
name: miles-agentic-rl-pipeline
description: Analyze, explain, debug, and modify the Miles agentic RL training pipeline end to end (data source, rollout generation, tool-calling agent loop, reward computation, rollout buffer, train-data conversion, actor/critic optimization, and weight sync). Use when requests mention Miles agentic or multi-turn RL, custom generate or rollout functions, OpenAI session routing, Miles Router, or rollout-to-training data flow.
---

# Miles Agentic RL Pipeline

## Goal
Understand and operate Miles agentic RL training from prompt sampling to optimizer update.

## Start Here
1. Confirm runtime mode.
   - `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` enables the new `InferenceRolloutFn` path and `generate_hub/*`.
2. Identify rollout entry.
   - `train.py` -> `create_rollout_manager` -> `RolloutManager.generate`.
3. Identify agentic generate function.
   - `--custom-generate-function-path` (fallback in refactor mode: `single_turn.generate`).
4. Identify reward source.
   - `--custom-rm-path` or `--rm-type`.
5. Confirm router type.
   - `--use-miles-router` is required for OpenAI session flow (`/sessions/...`).

## Pipeline Map (Control Plane)
`train.py` main loop:
1. Build Ray placement groups (`miles/ray/placement_group.py`).
2. Start `RolloutManager` (`miles/ray/rollout.py`).
3. Init actor or critic train groups (`miles/ray/actor_group.py` + backend actor).
4. Sync actor weights to rollout engines (`actor_model.update_weights()`).
5. For each `rollout_id`:
   - `rollout_manager.generate(rollout_id)` -> rollout samples -> train-data refs.
   - `actor_model.async_train(...)` (and critic if PPO).
   - Optional save and eval.
   - Sync updated actor weights back to rollout engines.
6. Dispose rollout manager.

## Pipeline Map (Data Plane)
`RolloutManager.generate`:
1. Call rollout function.
   - Refactor mode: `InferenceRolloutFn.__call__(RolloutFnTrainInput)`.
   - Legacy mode: `sglang_rollout.generate_rollout`.
2. Receive grouped `Sample` objects.
3. Flatten and trim or balance sample count.
4. Convert `Sample` to train dict.
   - Keys include `tokens`, `response_lengths`, `rewards`, `loss_masks`, optional `rollout_log_probs`, `rollout_routed_experts`, and metadata fields.
5. Split by DP partition and return Ray object refs to training workers.

## Refactor Rollout Internals
Files: `miles/rollout/inference_rollout/inference_rollout_common.py` and `miles/rollout/inference_rollout/inference_rollout_train.py`.

1. Build `GenerateState`.
   - tokenizer and processor
   - rollout sampling defaults
   - concurrency semaphore
   - generate function (`load_generate_function(custom_generate_path) or single_turn.generate`)
2. Sampling loop.
   - `data_source.get_samples(over_sampling_batch_size)` returns groups of size `n_samples_per_prompt`.
   - Submit one async task per group (`generate_and_rm_group`).
   - Apply dynamic filter (`--dynamic-sampling-filter-path`) on completed groups.
   - Keep accepted groups until `rollout_batch_size` is reached.
3. Abort remaining pending requests.
   - If `--partial-rollout`, cache partial groups back with `data_source.add_samples(...)`.
4. Run optional hooks.
   - `--rollout-sample-filter-path`
   - `--rollout-all-samples-process-path`

## Agentic Generate Paths
### A) OpenAI Session Agentic (`agentic_tool_call.generate`)
Files:
- `miles/rollout/generate_hub/agentic_tool_call.py`
- `miles/rollout/generate_utils/openai_endpoint_utils.py`
- `miles/router/session/sessions.py`

Flow:
1. Create session tracer (`POST /sessions`) and get `session_id`.
2. Load and call custom agent function from `--custom-agent-function-path`.
3. Agent sends one or more requests to `base_url + /v1/chat/completions`.
4. Miles Router session route records each request and response with token logprobs.
5. Tracer fetches records (`GET /sessions/{id}`), deletes session, converts records to `Sample`.
6. Merge multi-record samples unless `--generate-multi-samples` is enabled.

Hard requirements:
- `--use-miles-router`
- `logprobs=true` and prompt token ids available (`logprob_start_len=0` in wrapper)
- OpenAI-format prompt; do not rely on `--apply-chat-template` for this flow

### B) `/generate` Multi-turn Tool Calling (`multi_turn.generate`)
Files:
- `miles/rollout/generate_hub/multi_turn.py`
- `miles/rollout/generate_utils/tool_call_utils.py`

Flow:
1. Build initial prompt tokens.
2. Loop `--generate-max-turns`.
   - Call `/generate`.
   - Parse tool calls (`--generate-tool-call-parser`).
   - Execute tools (`--generate-execute-tool-function-path`).
   - Append tool-response tokens to sample with zeroed loss mask and zeroed rollout logprobs for tool text.
3. Stop on no tool call, stop condition, length limit, or abort.

## Reward Stage
File: `miles/rollout/rm_hub/__init__.py`.

1. If `--custom-rm-path` is set, call it (single or batch depending on `--group-rm`).
2. Else dispatch by `rm_type` from sample metadata or `--rm-type`.
   - `remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`, `ifbench`, `random`

## Data Source and Buffer
File: `miles/rollout/data_source.py`.

1. `RolloutDataSource` loads prompt dataset and emits grouped samples.
2. `RolloutDataSourceWithBuffer` pops from buffer first, then from dataset.
3. Partial rollout and custom flows can push groups back with `add_samples`.

## Training Consumption
Files:
- `miles/backends/training_utils/data.py`
- `miles/backends/megatron_utils/actor.py` (or FSDP actor)

Flow:
1. Materialize per-rank rollout refs to tensors (`get_rollout_data`).
2. Compute needed logprobs or values and advantages.
3. Run actor or critic optimization steps.
4. Backup or update actor weights.
5. Push latest weights to rollout engines (`update_weights`).

## Preferred Extension Order
Apply customization in this order unless there is a hard requirement to change core logic.

1. Start with `--custom-generate-function-path`.
2. Add `--custom-rm-path` for reward logic.
3. Add filtering or postprocess hooks.
   - `--dynamic-sampling-filter-path`
   - `--rollout-sample-filter-path`
   - `--rollout-all-samples-process-path`
   - `--rollout-data-postprocess-path`
4. Override data conversion last.
   - `--custom-convert-samples-to-train-data-path`

## Minimum Config Knobs
OpenAI-style agentic RL:
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

`/generate` tool-calling:
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

## Dynamic Argument Injection Rule
File: `miles/utils/arguments.py`.

1. Parser reads known args.
2. If refactor mode is on, Miles loads:
   - `--rollout-function-path`
   - `--custom-generate-function-path`
3. If loaded function exposes `add_arguments(parser)`, Miles injects custom CLI flags automatically.
4. Use this to register agent or tool-specific args without editing core parser.

## Debug Checklist
1. Verify rollout path.
   - `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR`
   - resolved `rollout_function_path` and `custom_generate_function_path`
2. Verify router mode.
   - OpenAI session path requires `--use-miles-router`
3. Verify sample integrity before training.
   - `len(tokens) >= response_length`
   - `len(loss_mask) == response_length`
   - `len(rollout_log_probs) == response_length` when expected
4. Verify training and rollout batch contract.
   - `rollout_batch_size * n_samples_per_prompt` should match training consumption plan (`global_batch_size * num_steps_per_rollout`)
5. Use debug switches as needed.
   - `--debug-rollout-only`
   - `--save-debug-rollout-data`
   - `--load-debug-rollout-data`
   - `--debug-train-only`

## Primary Files to Read First
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
