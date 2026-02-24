#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/opt/tiger/shard_train/ray_live.log"
WORKDIR="/opt/tiger/miles"
STATE_DIR="${WORKDIR}/.codex"
RUN_LOG="${STATE_DIR}/auto_tune_runner.log"
LOCK_FILE="${STATE_DIR}/auto_tune_runner.lock"

MEGATRON_SCRIPT="${WORKDIR}/examples/lora/run-qwen3-8B-aime-megatron-lora.sh"
TARGET_ROLLOUT=100
POLL_SECONDS=30

mkdir -p "${STATE_DIR}"

if [[ -f "${LOCK_FILE}" ]]; then
  old_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    exit 0
  fi
fi
echo $$ > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

ts() { date +"%Y-%m-%d %H:%M:%S"; }

latest_rollout() {
  grep -oE 'rollout [0-9]+:' "${LOG_FILE}" 2>/dev/null | tail -n 1 | awk '{print $2}' | tr -d ':'
}

{
  echo "[$(ts)] auto_tune_runner started, waiting for current run to finish (target rollout=${TARGET_ROLLOUT})."
  while true; do
    r="$(latest_rollout || true)"
    if [[ -n "${r}" ]] && [[ "${r}" =~ ^[0-9]+$ ]]; then
      echo "[$(ts)] observed rollout=${r}"
      if (( r >= TARGET_ROLLOUT )); then
        echo "[$(ts)] current run reached rollout ${r}, starting Megatron LoRA run."
        break
      fi
    else
      echo "[$(ts)] rollout not found yet, keep waiting."
    fi
    sleep "${POLL_SECONDS}"
  done

  cd "${WORKDIR}"
  bash "${MEGATRON_SCRIPT}"
  echo "[$(ts)] Megatron run command completed."
} >> "${RUN_LOG}" 2>&1
