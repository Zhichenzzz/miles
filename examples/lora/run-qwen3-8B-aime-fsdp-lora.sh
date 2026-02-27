#!/bin/bash
#
# LoRA GRPO Training for Qwen3-8B on MATH level 3-5, evaluated on AIME 2025
#
# Model: Qwen3-8B (8.2B dense, post-trained with thinking mode)
# Training data: MATH (Hendrycks) level 3-5 (~9K competition math problems)
# Eval: AIME 2025 (30 competition math problems, integer answers 0-999)
# LoRA: rank=64, alpha=32
# Hardware: 8x GPU (4 training + 4 inference, non-colocated)
# Backend: FSDP
#
# Expected: AIME 2025 baseline ~15-20% → improved with GRPO RL
#

set -ex

export PYTHONBUFFERED=16

DATA_DIR=${DATA_DIR:-"/opt/tiger/shard_train/data"}
SAVE_DIR=${SAVE_DIR:-"/opt/tiger/shard_train/data/checkpoints/qwen3-8b-aime-fsdp-lora"}
NUM_GPUS=${NUM_GPUS:-2}

CKPT_ARGS=(
   --hf-checkpoint "${DATA_DIR}/Qwen3-8B"
   --load "${DATA_DIR}/Qwen3-8B"
   --ref-load "${DATA_DIR}/Qwen3-8B"
   --save "${SAVE_DIR}"
   --save-interval 20
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   --lora-dropout 0.0
   --save-lora-only
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/math_level3to5/train.jsonl"
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 6
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --rollout-max-response-len 6144
   --rollout-temperature 0.7
   --global-batch-size 96
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime2025 "${DATA_DIR}/aime2025/aime2025.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 12288
   --eval-top-k 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 3e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-lora-aime
   # --wandb-group qwen3-8b-fsdp-lora
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.60
   --sglang-decode-log-interval 1000
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_2
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --colocate
)

# launch ray (use custom port to avoid conflicts)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_PORT=${RAY_PORT:-6399}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8266}
if curl -fsS "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; then
   echo "Reusing existing Ray cluster at http://127.0.0.1:${RAY_DASHBOARD_PORT}"
else
   ray start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_PORT} --dashboard-port ${RAY_DASHBOARD_PORT} --num-gpus "${NUM_GPUS}" --disable-usage-stats
fi

MEGATRON_PATH=${MEGATRON_PATH:-"/opt/tiger/shard_train/Megatron-LM"}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

SUBMISSION_ID="fsdp-lora-$(date +%Y%m%d-%H%M%S)"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --submission-id "${SUBMISSION_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
   -- python3 train.py \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRAIN_BACKEND_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}"
