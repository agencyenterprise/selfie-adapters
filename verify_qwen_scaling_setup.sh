#!/bin/bash
# Verification script for Qwen2.5 scaling law setup
# Run this to check that all components are in place

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Qwen2.5 Scaling Law Setup Verification               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Counter for checks
PASSED=0
FAILED=0
WARNINGS=0

check_file() {
    local file=$1
    local description=$2
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description (missing: $file)"
        ((FAILED++))
        return 1
    fi
}

check_dir() {
    local dir=$1
    local description=$2
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description (missing: $dir)"
        ((FAILED++))
        return 1
    fi
}

warn() {
    local message=$1
    echo -e "${YELLOW}⚠${NC} $message"
    ((WARNINGS++))
}

echo -e "${BLUE}[1/6] Checking Core Scripts${NC}"
echo "─────────────────────────────────────────────────────────"
check_file "data_prep/prepare_qwen_scaling_datasets.py" "Dataset preparation script"
check_file "evals/generation_scoring/run_qwen_scaling_label_gen.sh" "Experiment runner script"
check_file "evals/generation_scoring/merge_labels_with_metadata.py" "Metadata merger script"
check_file "evals/generation_scoring/run_eval.py" "Main evaluation script"
echo ""

echo -e "${BLUE}[2/6] Checking Configuration Files${NC}"
echo "─────────────────────────────────────────────────────────"
for size in 7b 14b 32b 72b; do
    check_file "evals/generation_scoring/configs/qwen_scaling/qwen25_${size}_trained_label_gen.json" "Trained config: ${size}"
    check_file "evals/generation_scoring/configs/qwen_scaling/qwen25_${size}_untrained_label_gen.json" "Untrained config: ${size}"
done
echo ""

echo -e "${BLUE}[3/6] Checking Documentation${NC}"
echo "─────────────────────────────────────────────────────────"
check_file "evals/generation_scoring/configs/qwen_scaling/README.md" "Detailed README"
check_file "QWEN_SCALING_QUICKSTART.md" "Quick start guide"
check_file "SETUP_COMPLETE.md" "Setup completion doc"
echo ""

echo -e "${BLUE}[4/6] Checking Datasets${NC}"
echo "─────────────────────────────────────────────────────────"
check_dir "outputs/qwen_scaling" "Output directory"

# Check for existing datasets (sample counts vary by model size)
# Full VAL split: 4964 topics at every layer
for size in 7b 14b 32b 72b; do
    case $size in
        7b)  n_samples=69496 ;;
        14b) n_samples=119136 ;;
        32b) n_samples=158848 ;;
        72b) n_samples=198560 ;;
    esac
    
    master_json="outputs/qwen_scaling/qwen25_${size}_combined_val_${n_samples}_master.json"
    pt_file="outputs/qwen_scaling/qwen25_${size}_instruct_fifty_thousand_things_combined_val_${n_samples}.pt"
    metadata_file="outputs/qwen_scaling/qwen25_${size}_instruct_fifty_thousand_things_combined_val_${n_samples}_metadata.json"
    
    if [ -f "$master_json" ] && [ -f "$pt_file" ] && [ -f "$metadata_file" ]; then
        echo -e "${GREEN}✓${NC} Dataset exists: ${size} (master JSON, .pt, metadata)"
        ((PASSED++))
    else
        warn "Dataset missing: ${size} (run: python data_prep/prepare_qwen_scaling_datasets.py --model-size ${size})"
    fi
done
echo ""

echo -e "${BLUE}[5/6] Checking Trained Adapter Checkpoint Paths${NC}"
echo "─────────────────────────────────────────────────────────"
for size in 7b 14b 32b 72b; do
    config_file="evals/generation_scoring/configs/qwen_scaling/qwen25_${size}_trained_label_gen.json"
    if [ -f "$config_file" ]; then
        checkpoint_path=$(python3 -c "import json; data=json.load(open('$config_file')); print(data['label_generator_config']['adapter_checkpoint_path'])")
        if [[ "$checkpoint_path" == *"PLACEHOLDER"* ]]; then
            warn "Trained config has placeholder checkpoint: ${size} (update adapter_checkpoint_path)"
        elif [ ! -f "$checkpoint_path" ]; then
            warn "Checkpoint file not found: ${checkpoint_path} (config: ${size})"
        else
            echo -e "${GREEN}✓${NC} Checkpoint configured and exists: ${size}"
            ((PASSED++))
        fi
    fi
done
echo ""

echo -e "${BLUE}[6/6] Checking Script Permissions${NC}"
echo "─────────────────────────────────────────────────────────"
if [ -x "evals/generation_scoring/run_qwen_scaling_label_gen.sh" ]; then
    echo -e "${GREEN}✓${NC} Experiment runner is executable"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Experiment runner not executable (run: chmod +x evals/generation_scoring/run_qwen_scaling_label_gen.sh)"
    ((WARNINGS++))
fi

if [ -x "evals/generation_scoring/merge_labels_with_metadata.py" ]; then
    echo -e "${GREEN}✓${NC} Metadata merger is executable"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} Metadata merger not executable (run: chmod +x evals/generation_scoring/merge_labels_with_metadata.py)"
    ((WARNINGS++))
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Verification Summary                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Passed:   ${GREEN}${PASSED}${NC}"
echo -e "Failed:   ${RED}${FAILED}${NC}"
echo -e "Warnings: ${YELLOW}${WARNINGS}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Setup verification complete!${NC}"
    echo ""
    echo "Next steps:"
    if [ $WARNINGS -gt 0 ]; then
        echo "  1. Address any warnings above (optional)"
    fi
    echo "  2. Prepare datasets: python data_prep/prepare_qwen_scaling_datasets.py --model-size all"
    echo "  3. Update checkpoint paths in trained configs (if using)"
    echo "  4. Run experiments: ./evals/generation_scoring/run_qwen_scaling_label_gen.sh all untrained"
    echo ""
    echo "For detailed instructions, see: QWEN_SCALING_QUICKSTART.md"
else
    echo -e "${RED}❌ Setup verification failed${NC}"
    echo ""
    echo "Please resolve the failed checks above before proceeding."
fi
