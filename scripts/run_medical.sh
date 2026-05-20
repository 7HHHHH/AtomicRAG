#!/usr/bin/env bash

set -euo pipefail

# AtomicRAG Medical Dataset - Complete Pipeline
# Runs processing + evaluation for the medical subset

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
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           AtomicRAG Medical Dataset Pipeline               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Interactive cache option
echo -e "${YELLOW}🔧 Cache Configuration:${NC}"
echo ""
echo "  1) 🔨 Rebuild from scratch (default)"
echo "  2) 🔄 Use cached index/extraction (faster, skip if predictions exist)"
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
        echo -e "${GREEN}✓ Mode: Using cached data (will skip if predictions exist)${NC}"
        ;;
    *)
        echo -e "${RED}❌ Invalid option. Using default (rebuild).${NC}"
        USE_CACHE="false"
        ;;
esac

echo ""
echo -e "${YELLOW}[Step 1/2] Running AtomicRAG processing...${NC}"
echo ""

python "${SCRIPT_DIR}/run_atomicrag.py" \
    --subset medical \
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
echo -e "${YELLOW}[Step 2/2] Running evaluation...${NC}"
echo ""

# Medical has only one sub-dataset, so we evaluate it directly
PREDICTIONS_FILE="${RESULTS_DIR}/Medical/predictions_Medical.json"
EVAL_FILE="${RESULTS_DIR}/Medical/eval_generation_Medical.json"

if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo -e "${RED}❌ Predictions file not found: $PREDICTIONS_FILE${NC}"
    exit 1
fi

# Skip if evaluation already exists and using cache mode
if [ "$USE_CACHE" = "true" ] && [ -f "$EVAL_FILE" ]; then
    echo -e "${CYAN}⏭️  Skipping Medical: evaluation already exists${NC}"
    echo -e "${GREEN}✅ Evaluation skipped (already exists)${NC}"
else
    cd "${REPO_ROOT}"
    python -m Evaluation.generation_eval \
        --model gpt-4o-mini \
        --embedding_model BAAI/bge-large-en-v1.5  \
        --data_file "$PREDICTIONS_FILE" \
        --output_file "$EVAL_FILE"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Evaluation completed successfully${NC}"
    else
        echo -e "${RED}❌ Evaluation failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}✅ Medical dataset pipeline completed!${NC}"
echo -e "${BLUE}║ Results: ${RESULTS_DIR}/Medical/${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
