#!/usr/bin/env bash

set -euo pipefail

# AtomicRAG All Datasets Pipeline
# Runs all datasets sequentially: MuSiQue, HotpotQA, 2WikiMultiHop, Novel, Medical

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CONFIG_DIR="${REPO_ROOT}/configs/atomicrag"
WORKSPACE_DIR="${REPO_ROOT}/workspaces/atomicrag"
RESULTS_DIR="${REPO_ROOT}/results/atomicrag"

# Python interpreter. Override with PYTHON=/path/to/python if needed.
PYTHON="${PYTHON:-python}"

# Load environment variables
if [ -f "${CONFIG_DIR}/llm.env" ]; then
    export $(grep -v '^#' "${CONFIG_DIR}/llm.env" | xargs)
fi

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        AtomicRAG All Datasets Sequential Pipeline          ║${NC}"
echo -e "${BLUE}║   MuSiQue → HotpotQA → 2WikiMultiHop → Novel → Medical     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Interactive cache option
echo -e "${YELLOW}🔧 Cache Configuration (applies to all datasets):${NC}"
echo ""
echo "  1) 🔨 Rebuild from scratch (default)"
echo "  2) 🔄 Use cached index/extraction (faster, skip completed)"
echo ""
read -p "Select option [1-2] (default: 1): " cache_option
cache_option=${cache_option:-1}

case $cache_option in
    1)
        USE_CACHE="false"
        echo -e "${GREEN}✓ Mode: Rebuilding from scratch${NC}"
        ;;
    2)
        USE_CACHE="true"
        echo -e "${GREEN}✓ Mode: Using cached data${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid option. Using default (rebuild).${NC}"
        USE_CACHE="false"
        ;;
esac

echo ""

# Track overall status
DATASETS_COMPLETED=0
DATASETS_FAILED=0
START_TIME=$(date +%s)

# Dataset configurations: subset|display_name|qa_prompt|result_pattern|needs_averaging
DATASETS=(
    "medical|Medical|default|Medical*|false"
    "musique|MuSiQue|precise|[Mm]usique*|false"
    "hotpotqa|HotpotQA|precise|[Hh]otpotqa*|false"
    "2wikimultihop|2WikiMultiHop|precise|2[Ww]iki[Mm]ulti[Hh]op*|false"
    "novel|Novel|default|Novel-*|true"
)

# Function to run evaluation for a result directory
run_evaluation() {
    local result_dir=$1
    local dataset_name=$(basename "$result_dir")
    local predictions_file="${result_dir}/predictions_${dataset_name}.json"
    local output_file="${result_dir}/eval_generation_${dataset_name}.json"

    if [ ! -f "$predictions_file" ]; then
        return 1
    fi

    # Skip if evaluation already exists and using cache mode
    if [ "$USE_CACHE" = "true" ] && [ -f "$output_file" ]; then
        echo -e "${CYAN}⏭️  Skipping ${dataset_name}: evaluation already exists${NC}"
        return 0
    fi

    echo -e "${CYAN}Evaluating: ${dataset_name}${NC}"

    $PYTHON -m Evaluation.generation_eval \
        --model gpt-4o-mini \
        --embedding_model BAAI/bge-large-en-v1.5  \
        --data_file "$predictions_file" \
        --output_file "$output_file"

    return $?
}

# Function to run a single dataset
run_dataset() {
    local subset=$1
    local display_name=$2
    local qa_prompt=$3
    local result_pattern=$4
    local needs_averaging=$5

    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Dataset: ${display_name}${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Build command
    local cmd="$PYTHON ${SCRIPT_DIR}/run_atomicrag.py \
        --subset $subset \
        --base_dir ${WORKSPACE_DIR} \
        --model_name gpt-4o-mini \
        --embed_model_path BAAI/bge-large-en-v1.5 \
        --use_cache $USE_CACHE"

    # Add qa_prompt_template only if not default
    if [ "$qa_prompt" != "default" ]; then
        cmd="$cmd --qa_prompt_template $qa_prompt"
    fi

    echo -e "${YELLOW}[Step 1] Running AtomicRAG processing for ${display_name}...${NC}"
    echo ""

    eval $cmd

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Processing failed for ${display_name}${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Processing completed for ${display_name}${NC}"
    echo ""

    echo -e "${YELLOW}[Step 2] Running evaluation for ${display_name}...${NC}"
    echo ""

    cd "${REPO_ROOT}"
    local eval_count=0

    # Find result directories
    shopt -s nullglob
    local patterns=($result_pattern)
    local result_dirs=()
    for pattern in "${patterns[@]}"; do
        for dir in ${RESULTS_DIR}/${pattern}; do
            if [ -d "$dir" ]; then
                result_dirs+=("$dir")
            fi
        done
    done
    shopt -u nullglob

    if [ ${#result_dirs[@]} -eq 0 ]; then
        echo -e "${RED}❌ No results found for ${display_name}${NC}"
        return 1
    fi

    # Evaluate each result directory
    for result_dir in "${result_dirs[@]}"; do
        if run_evaluation "$result_dir"; then
            eval_count=$((eval_count + 1))
            echo -e "${GREEN}✅ $(basename $result_dir) evaluated${NC}"
        else
            echo -e "${RED}⚠️  $(basename $result_dir) evaluation failed${NC}"
        fi
    done

    # Run averaging for Novel dataset
    if [ "$needs_averaging" = "true" ]; then
        echo ""
        echo -e "${YELLOW}[Step 3] Averaging evaluation results...${NC}"
        $PYTHON "${SCRIPT_DIR}/average_novel_results.py" \
            --results_dir "$RESULTS_DIR" \
            --output_file "${RESULTS_DIR}/eval_generation_novel_averaged.json"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Results averaged successfully${NC}"
        else
            echo -e "${RED}⚠️  Failed to average results${NC}"
        fi
    fi

    echo ""
    echo -e "${GREEN}✅ ${display_name} completed (evaluated $eval_count sub-datasets)${NC}"
    echo ""

    return 0
}

# Run all datasets sequentially
echo -e "${CYAN}📋 Running ${#DATASETS[@]} datasets sequentially${NC}"
echo ""

for dataset_config in "${DATASETS[@]}"; do
    IFS='|' read -r subset display_name qa_prompt result_pattern needs_averaging <<< "$dataset_config"

    if run_dataset "$subset" "$display_name" "$qa_prompt" "$result_pattern" "$needs_averaging"; then
        DATASETS_COMPLETED=$((DATASETS_COMPLETED + 1))
    else
        DATASETS_FAILED=$((DATASETS_FAILED + 1))
    fi

    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              All Datasets Pipeline Complete               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Completed: ${DATASETS_COMPLETED}/${#DATASETS[@]} datasets${NC}"
if [ $DATASETS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Failed: ${DATASETS_FAILED}/${#DATASETS[@]} datasets${NC}"
fi
echo -e "${CYAN}⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${RESULTS_DIR}/${NC}"
echo ""
