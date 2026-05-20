#!/usr/bin/env bash

set -euo pipefail

# AtomicRAG Novel Dataset - Complete Pipeline
# Runs processing + evaluation + averaging for all novel sub-datasets

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CONFIG_DIR="${REPO_ROOT}/configs/atomicrag"
WORKSPACE_DIR="${REPO_ROOT}/workspaces/atomicrag"
RESULTS_DIR="${REPO_ROOT}/results/atomicrag"

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
echo -e "${BLUE}║           AtomicRAG Novel Dataset Pipeline                 ║${NC}"
echo -e "${BLUE}║         (20 sub-datasets with averaging)                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Interactive cache option
echo -e "${YELLOW}🔧 Cache Configuration:${NC}"
echo ""
echo "  1) 🔨 Rebuild from scratch (default)"
echo "  2) 🔄 Use cached index/extraction (faster, skip completed sub-datasets)"
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
        echo -e "${GREEN}✓ Mode: Using cached data (will skip sub-datasets with existing predictions)${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid option. Using default (rebuild).${NC}"
        USE_CACHE="false"
        ;;
esac

echo ""
echo -e "${YELLOW}[Step 1/3] Running AtomicRAG processing...${NC}"
echo ""

python "${SCRIPT_DIR}/run_atomicrag.py" \
    --subset novel \
    --base_dir "${WORKSPACE_DIR}" \
    --model_name gpt-4o-mini \
    --embed_model_path BAAI/bge-large-en-v1.5 \
    --use_cache "$USE_CACHE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Processing completed successfully${NC}"
else
    echo -e "${RED}❌ Processing failed${NC}"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}[Step 2/3] Running evaluation for each sub-dataset...${NC}"
echo ""

EVAL_COUNT=0
cd "${REPO_ROOT}"

# Evaluate each Novel-XXXXX subdirectory
for novel_dir in "${RESULTS_DIR}"/Novel-*; do
    if [ -d "$novel_dir" ]; then
        dataset_name=$(basename "$novel_dir")
        predictions_file="${novel_dir}/predictions_${dataset_name}.json"
        output_file="${novel_dir}/eval_generation_${dataset_name}.json"

        if [ -f "$predictions_file" ]; then
            # Skip if evaluation already exists and using cache mode
            if [ "$USE_CACHE" = "true" ] && [ -f "$output_file" ]; then
                echo -e "${CYAN}⏭️  Skipping ${dataset_name}: evaluation already exists${NC}"
                EVAL_COUNT=$((EVAL_COUNT + 1))
                continue
            fi

            echo -e "${CYAN}Evaluating: ${dataset_name}${NC}"

            python -m Evaluation.generation_eval \
                --model gpt-4o-mini \
                --embedding_model BAAI/bge-large-en-v1.5  \
                --data_file "$predictions_file" \
                --output_file "$output_file"

            if [ $? -eq 0 ]; then
                EVAL_COUNT=$((EVAL_COUNT + 1))
                echo -e "${GREEN}✅ ${dataset_name} evaluated${NC}"
            else
                echo -e "${RED}⚠️  ${dataset_name} evaluation failed${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Predictions not found for ${dataset_name}${NC}"
        fi
    fi
done

echo ""
echo -e "${GREEN}✅ Evaluated $EVAL_COUNT Novel sub-datasets${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}[Step 3/3] Averaging evaluation results...${NC}"
echo ""

python "${SCRIPT_DIR}/average_novel_results.py" \
    --results_dir "$RESULTS_DIR" \
    --output_file "${RESULTS_DIR}/eval_generation_novel_averaged.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Results averaged successfully${NC}"
else
    echo -e "${RED}❌ Failed to average results${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}✅ Novel dataset pipeline completed!${NC}"
echo -e "${BLUE}║ Individual results: ${RESULTS_DIR}/Novel-*/${NC}"
echo -e "${BLUE}║ Averaged results:   ${RESULTS_DIR}/eval_generation_novel_averaged.json${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
