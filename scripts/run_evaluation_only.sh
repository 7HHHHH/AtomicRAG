#!/usr/bin/env bash

set -euo pipefail

# AtomicRAG Evaluation Only Script
# Runs evaluation on all existing prediction files

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CONFIG_DIR="${REPO_ROOT}/configs/atomicrag"
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
echo -e "${BLUE}║           AtomicRAG Evaluation Only Script                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SKIP_EXISTING=false
FORCE_REEVAL=false
SPECIFIC_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --force)
            FORCE_REEVAL=true
            shift
            ;;
        --dataset)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-existing    Skip datasets that already have evaluation results"
            echo "  --force            Force re-evaluation even if results exist"
            echo "  --dataset NAME     Only evaluate specific dataset (e.g., Medical, Musique)"
            echo "  --help, -h         Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}❌ Results directory not found: ${RESULTS_DIR}${NC}"
    echo -e "${YELLOW}Please run the main pipeline first to generate predictions.${NC}"
    exit 1
fi

# Interactive mode selection if no flags provided
if [ "$SKIP_EXISTING" = false ] && [ "$FORCE_REEVAL" = false ]; then
    echo -e "${YELLOW}🔧 Evaluation Mode:${NC}"
    echo ""
    echo "  1) 🔄 Evaluate missing only (skip existing)"
    echo "  2) 🔨 Re-evaluate all (overwrite existing)"
    echo ""
    read -p "Select option [1-2] (default: 1): " eval_option
    eval_option=${eval_option:-1}

    case $eval_option in
        1)
            SKIP_EXISTING=true
            echo -e "${GREEN}✓ Mode: Evaluate missing only${NC}"
            ;;
        2)
            FORCE_REEVAL=true
            echo -e "${GREEN}✓ Mode: Re-evaluate all${NC}"
            ;;
        *)
            SKIP_EXISTING=true
            echo -e "${YELLOW}⚠️  Invalid option. Using default (skip existing).${NC}"
            ;;
    esac
    echo ""
fi

# Track statistics
EVAL_SUCCESS=0
EVAL_FAILED=0
EVAL_SKIPPED=0
START_TIME=$(date +%s)

# Function to run evaluation for a single result directory
run_evaluation() {
    local result_dir=$1
    local dataset_name=$(basename "$result_dir")
    local predictions_file="${result_dir}/predictions_${dataset_name}.json"
    local output_file="${result_dir}/eval_generation_${dataset_name}.json"

    # Check if predictions file exists
    if [ ! -f "$predictions_file" ]; then
        echo -e "${YELLOW}⚠️  No predictions file found for ${dataset_name}${NC}"
        return 1
    fi

    # Check if we should skip existing
    if [ "$SKIP_EXISTING" = true ] && [ -f "$output_file" ] && [ "$FORCE_REEVAL" = false ]; then
        echo -e "${CYAN}⏭️  Skipping ${dataset_name}: evaluation already exists${NC}"
        EVAL_SKIPPED=$((EVAL_SKIPPED + 1))
        return 0
    fi

    echo -e "${BLUE}📊 Evaluating: ${dataset_name}${NC}"

    if $PYTHON -m Evaluation.generation_eval \
        --model gpt-4o-mini \
        --embedding_model BAAI/bge-large-en-v1.5 \
        --data_file "$predictions_file" \
        --output_file "$output_file" 2>&1; then

        echo -e "${GREEN}✅ ${dataset_name} evaluation completed${NC}"
        EVAL_SUCCESS=$((EVAL_SUCCESS + 1))
        return 0
    else
        echo -e "${RED}❌ ${dataset_name} evaluation failed${NC}"
        EVAL_FAILED=$((EVAL_FAILED + 1))
        return 1
    fi
}

# Find all result directories
echo -e "${CYAN}🔍 Scanning results directory: ${RESULTS_DIR}${NC}"
echo ""

cd "${REPO_ROOT}"

# Get list of directories to process
if [ -n "$SPECIFIC_DATASET" ]; then
    # Specific dataset requested
    if [ -d "${RESULTS_DIR}/${SPECIFIC_DATASET}" ]; then
        RESULT_DIRS=("${RESULTS_DIR}/${SPECIFIC_DATASET}")
    else
        echo -e "${RED}❌ Dataset not found: ${SPECIFIC_DATASET}${NC}"
        exit 1
    fi
else
    # All datasets
    RESULT_DIRS=()
    for dir in "${RESULTS_DIR}"/*/; do
        if [ -d "$dir" ]; then
            RESULT_DIRS+=("${dir%/}")
        fi
    done
fi

TOTAL_DATASETS=${#RESULT_DIRS[@]}

if [ $TOTAL_DATASETS -eq 0 ]; then
    echo -e "${RED}❌ No result directories found${NC}"
    exit 1
fi

echo -e "${CYAN}📋 Found ${TOTAL_DATASETS} dataset(s) to process${NC}"
echo ""

# Process each directory
for result_dir in "${RESULT_DIRS[@]}"; do
    run_evaluation "$result_dir" || true
    echo ""
done

# Run Novel averaging if we processed Novel datasets
NOVEL_COUNT=$(find "${RESULTS_DIR}" -maxdepth 1 -type d -name "Novel-*" | wc -l)
if [ $NOVEL_COUNT -gt 0 ] && [ -z "$SPECIFIC_DATASET" ]; then
    echo -e "${YELLOW}📊 Averaging Novel evaluation results...${NC}"

    if [ -f "${SCRIPT_DIR}/average_novel_results.py" ]; then
        $PYTHON "${SCRIPT_DIR}/average_novel_results.py" \
            --results_dir "$RESULTS_DIR" \
            --output_file "${RESULTS_DIR}/eval_generation_novel_averaged.json" 2>&1 || true

        if [ -f "${RESULTS_DIR}/eval_generation_novel_averaged.json" ]; then
            echo -e "${GREEN}✅ Novel results averaged successfully${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  average_novel_results.py not found, skipping averaging${NC}"
    fi
    echo ""
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Evaluation Summary                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Success: ${EVAL_SUCCESS}${NC}"
echo -e "${CYAN}⏭️  Skipped: ${EVAL_SKIPPED}${NC}"
if [ $EVAL_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Failed:  ${EVAL_FAILED}${NC}"
fi
echo -e "${CYAN}⏱️  Time: ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${RESULTS_DIR}/${NC}"
echo ""

# List generated evaluation files
echo -e "${YELLOW}📄 Generated evaluation files:${NC}"
find "${RESULTS_DIR}" -name "eval_generation_*.json" -type f | sort | while read -r f; do
    echo "   - $(basename $(dirname $f))/$(basename $f)"
done
echo ""
