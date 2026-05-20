#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Load centralized LLM env if available (allows overriding MODEL/URL/API keys)
LLM_ENV_FILE="${LLM_ENV_FILE:-${REPO_ROOT}/configs/atomicrag/llm.env}"
if [[ -f "${LLM_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${LLM_ENV_FILE}"
  set +a
fi

DATASET_DIR="${REPO_ROOT}/dataset"
RESULTS_ROOT="${REPO_ROOT}/results/atomicrag"
LOG_FILE="${REPO_ROOT}/logs/atomicrag/atomicrag_processing.log"
ABLATION_OUTPUT_ROOT="${REPO_ROOT}/results/atomicrag_ablations"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${RUN_ID}"

mkdir -p "${RUN_OUTPUT_DIR}"

MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.openai.com/v1}"
USE_CACHE="${USE_CACHE:-false}"
CONCURRENCY="${CONCURRENCY:-3}"
SAMPLE="${SAMPLE:-}"
QA_PROMPT_TEMPLATE="${QA_PROMPT_TEMPLATE:-}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o-mini}"
EVAL_EMBED_MODEL="${EVAL_EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-20}"

SUBSETS=("medical")
ABLATIONS=(
  "baseline|"
  "no_decomp|--disable_query_decomposition"
  "no_ppr|--disable_ppr"
  "no_filter|--disable_fragment_filter"
)

save_results() {
  local subset="$1"
  local tag="$2"
  local dest="${RUN_OUTPUT_DIR}/${subset}/${tag}"

  rm -rf "${dest}"
  mkdir -p "${dest}"

  local subset_dir="${DATASET_DIR}/${subset}"
  if [[ ! -d "${subset_dir}" ]]; then
    echo "[WARN] subset directory not found: ${subset_dir}" >&2
    return
  fi

  for corpus_path in "${subset_dir}"/*; do
    [[ -d "${corpus_path}" ]] || continue
    local corpus_name
    corpus_name="$(basename "${corpus_path}")"
    local src_dir="${RESULTS_ROOT}/${corpus_name}"
    if [[ -d "${src_dir}" ]]; then
      cp -r "${src_dir}" "${dest}/${corpus_name}"
    else
      echo "[WARN] results missing for ${corpus_name} (subset=${subset}, tag=${tag})" >&2
    fi
  done

  if [[ -f "${LOG_FILE}" ]]; then
    cp "${LOG_FILE}" "${dest}/"
  fi
}

run_evaluation() {
  local subset="$1"
  local subset_dir="${DATASET_DIR}/${subset}"
  if [[ ! -d "${subset_dir}" ]]; then
    echo "[WARN] subset directory not found for evaluation: ${subset_dir}" >&2
    return
  fi

  local evaluated=0
  for corpus_path in "${subset_dir}"/*; do
    [[ -d "${corpus_path}" ]] || continue
    local corpus_name
    corpus_name="$(basename "${corpus_path}")"
    local result_dir="${RESULTS_ROOT}/${corpus_name}"
    local predictions_file="${result_dir}/predictions_${corpus_name}.json"
    local output_file="${result_dir}/eval_generation_${corpus_name}.json"
    if [[ -f "${predictions_file}" ]]; then
      echo "[Eval] ${corpus_name}"
      python -m Evaluation.generation_eval \
        --model "${EVAL_MODEL}" \
        --base_url "${LLM_BASE_URL}" \
        --embedding_model "${EVAL_EMBED_MODEL}" \
        --data_file "${predictions_file}" \
        --output_file "${output_file}" \
        --concurrency "${EVAL_CONCURRENCY}"
      evaluated=$((evaluated + 1))
    else
      echo "[WARN] predictions not found for ${corpus_name}, skipping evaluation." >&2
    fi
  done
  echo "[Eval] Completed ${evaluated} evaluations for subset ${subset}"
}

for subset in "${SUBSETS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    IFS="|" read -r tag flag_string <<< "${ablation}"
    echo "=============================="
    echo "Subset: ${subset} | Setting: ${tag}"
    echo "=============================="

    cmd=(python scripts/run_atomicrag.py
         --subset "${subset}"
         --model_name "${MODEL_NAME}"
         --embed_model_path "${EMBED_MODEL}"
         --llm_base_url "${LLM_BASE_URL}"
         --use_cache "${USE_CACHE}"
         --concurrency "${CONCURRENCY}"
         )

    if [[ -n "${SAMPLE}" ]]; then
      cmd+=(--sample "${SAMPLE}")
    fi

    # cmd+=(--qa_prompt_template precise)

    if [[ -n "${flag_string}" ]]; then
      # shellcheck disable=SC2206
      extra_flags=(${flag_string})
      cmd+=("${extra_flags[@]}")
    fi

    echo "[CMD] ${cmd[*]}"
    "${cmd[@]}"

    run_evaluation "${subset}"
    save_results "${subset}" "${tag}"

    printf '%s\n' "${cmd[*]}" > "${RUN_OUTPUT_DIR}/${subset}/${tag}/command.txt"
  done
done

echo "Ablation outputs saved under: ${RUN_OUTPUT_DIR}"
