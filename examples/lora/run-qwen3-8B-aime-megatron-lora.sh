#!/bin/bash
#
# LoRA GRPO Training for Qwen3-8B on MATH level 3-5, evaluated on AIME 2025
# Backend: Megatron (bridge mode)
#
# Model: Qwen3-8B (8.2B dense, post-trained with thinking mode)
# Training data: MATH (Hendrycks) level 3-5 (~9K competition math problems)
# Eval: AIME 2025 (30 competition math problems, integer answers 0-999)
# LoRA: rank=64, alpha=32
# Hardware: 8x GPU (colocated mode)
# Backend: Megatron bridge mode (required for LoRA + Megatron)
#
# Requirements:
#   - megatron-bridge >= 0.2.0 (with megatron.bridge.peft support)
#   - Megatron-LM installed
#
# Expected: AIME 2025 baseline ~15-20% -> improved with GRPO RL
#

set -ex

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

DATA_DIR=${DATA_DIR:-"/opt/tiger/shard_train/data"}
SAVE_DIR=${SAVE_DIR:-"/opt/tiger/shard_train/data/checkpoints/qwen3-8b-aime-megatron-lora"}
NUM_GPUS=${NUM_GPUS:-8}
MEGATRON_PATH=${MEGATRON_PATH:-"/opt/tiger/shard_train/Megatron-LM"}
HF_CKPT_DIR=${HF_CKPT_DIR:-"${DATA_DIR}/Qwen3-8B"}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT_DIR}"
   # verl-style design: Megatron trains, rollout uses SGLang, and base weights load from HF via bridge.
   --load "${HF_CKPT_DIR}"
   --ref-load "${HF_CKPT_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 10
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   # TP-safe LoRA targets: exclude attention projection layers for now.
   --lora-target-modules linear_fc1 linear_fc2
   --lora-dropout 0.0
   --lora-type lora
   --lora-a-init-method kaiming
   --lora-b-init-method zero
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
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   # Keep generations bounded to reduce late-token degeneration.
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --rollout-top-p 0.95
   --rollout-top-k 20
   # Qwen3 eos tokens: <|im_end|>=151645, <|endoftext|>=151643.
   --rollout-stop-token-ids 151645 151643
   --global-batch-size 12
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data aime2025 "${DATA_DIR}/aime2025/aime2025.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-k 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.0005
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
   # --wandb-project miles-lora-aime
   # --wandb-group qwen3-8b-megatron-lora
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
   --tensor-model-parallel-size 8
   --pipeline-model-parallel-size 1
   --seq-length 8192
   --max-position-embeddings 32768
   --tokenizer-type NullTokenizer
   --tokenizer-model "${HF_CKPT_DIR}"
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
   --no-offload-train
   --train-memory-margin-bytes 0
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

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

SUBMISSION_ID="megatron-lora-$(date +%Y%m%d-%H%M%S)"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --submission-id "${SUBMISSION_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
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
