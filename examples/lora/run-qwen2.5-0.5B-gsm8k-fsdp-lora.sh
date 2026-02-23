#!/bin/bash
#
# FSDP LoRA GRPO training with Qwen2.5-0.5B-Instruct on GSM8K
#
# Model: Qwen2.5-0.5B-Instruct
# Dataset: GSM8K
# LoRA: rank=32, alpha=16
# Hardware: 1x GPU (colocated mode)
#

set -ex

export PYTHONBUFFERED=16

DATA_DIR=${DATA_DIR:-"/root/data"}
SAVE_DIR=${SAVE_DIR:-"/root/data/checkpoints/qwen2.5-0.5b-gsm8k-lora"}
NUM_GPUS=${NUM_GPUS:-1}

CKPT_ARGS=(
   --hf-checkpoint "${DATA_DIR}/Qwen2.5-0.5B-Instruct"
   --load "${DATA_DIR}/Qwen2.5-0.5B-Instruct"
   --ref-load "${DATA_DIR}/Qwen2.5-0.5B-Instruct"
   --save "${SAVE_DIR}"
   --save-interval 20
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 16
   --lora-dropout 0.0
   --save-lora-only
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/gsm8k/train.parquet"
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 110
   --rollout-batch-size 64
   --n-samples-per-prompt 5
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 320
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k "${DATA_DIR}/gsm8k/test.parquet"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
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
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project miles-lora-gsm8k
   # --wandb-group qwen2.5-0.5b-fsdp-lora
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.40
   --sglang-decode-log-interval 1000
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
ray start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_PORT} --dashboard-port ${RAY_DASHBOARD_PORT} --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
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
