#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-graphmamba-training}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

NOTEBOOK="${REPO_ROOT}/notebooks/training.ipynb"
OUTPUT_DIR="${REPO_ROOT}/output/notebooks"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_NOTEBOOK="${OUTPUT_DIR}/training_${TIMESTAMP}.ipynb"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
KERNEL_NAME="${KERNEL_NAME:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

for command in tmux papermill; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Error: '${command}' is not installed or is not on PATH." >&2
        exit 1
    fi
done

if [[ ! -f "${NOTEBOOK}" ]]; then
    echo "Error: notebook not found: ${NOTEBOOK}" >&2
    exit 1
fi

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION_NAME}' already exists." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

printf -v run_command \
    'cd %q && export CUDA_VISIBLE_DEVICES=%q && set -o pipefail; papermill %q %q --kernel %q 2>&1 | tee -a %q; status=${PIPESTATUS[0]}; echo "Papermill exited with status ${status}" | tee -a %q; exit "${status}"' \
    "${REPO_ROOT}" "${CUDA_VISIBLE_DEVICES}" "${NOTEBOOK}" "${OUTPUT_NOTEBOOK}" "${KERNEL_NAME}" \
    "${LOG_FILE}" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" bash -lc "${run_command}"

echo "Started training in tmux session: ${SESSION_NAME}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${LOG_FILE}"
echo "Executed notebook: ${OUTPUT_NOTEBOOK}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Follow log with: tail -f ${LOG_FILE}"
