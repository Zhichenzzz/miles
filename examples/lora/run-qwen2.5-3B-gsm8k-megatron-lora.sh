#!/bin/bash
#
# Reproduce verl GRPO LoRA blog results using Miles Megatron (bridge mode) backend
# Blog: https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl
#
# Model: Qwen2.5-3B-Instruct
# Dataset: GSM8K
# LoRA: rank=64, alpha=32
# Expected: GSM8K accuracy ~59% -> ~85%
# Hardware: 2x GPU (colocated mode)
#
# Requirements:
#   - megatron-bridge >= 0.2.0 (with megatron.bridge.peft support)
#   - Megatron-LM installed
#   - Pre-converted Megatron checkpoint (use tools/convert_hf_to_torch_dist.py)
#

set -ex

export PYTHONBUFFERED=16

DATA_DIR=${DATA_DIR:-"/root/data"}
SAVE_DIR=${SAVE_DIR:-"/root/data/checkpoints/qwen2.5-3b-gsm8k-megatron-lora"}
NUM_GPUS=${NUM_GPUS:-2}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${DATA_DIR}/Qwen2.5-3B-Instruct"
   # Use HF checkpoint directly for LoRA (torch_dist format causes sharded state dict mismatch
   # because LoRA adapter keys don't exist in the base checkpoint)
   --load "${DATA_DIR}/Qwen2.5-3B-Instruct"
   --ref-load "${DATA_DIR}/Qwen2.5-3B-Instruct"
   --save "${SAVE_DIR}"
   --save-interval 20
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   # Default Megatron target modules: linear_qkv,linear_proj,linear_fc1,linear_fc2
   --lora-dropout 0.0
   --lora-type lora
   --lora-a-init-method kaiming
   --lora-b-init-method zero
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
   --rollout-batch-size 128
   --n-samples-per-prompt 5
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 640
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k "${DATA_DIR}/gsm8k/test.parquet"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
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
   # --wandb-project miles-lora-gsm8k
   # --wandb-group qwen2.5-3b-megatron-lora
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.50
   --sglang-decode-log-interval 1000
)

TRAIN_BACKEND_ARGS=(
   --train-backend megatron
   --megatron-to-hf-mode bridge
   --update-weight-buffer-size 536870912
   --attention-backend flash
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --seq-length 2048
   --max-position-embeddings 32768
   --tokenizer-type NullTokenizer
   --tokenizer-model "${DATA_DIR}/Qwen2.5-3B-Instruct"
   --bf16
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
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
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${MODEL_ARGS[@]}" \
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
