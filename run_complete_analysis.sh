#!/bin/bash
# Master script to run complete evaluation and visualization analysis

echo "========================================================================"
echo "FedTPG Complete Analysis Pipeline"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "fedtpg" ]]; then
    echo -e "${YELLOW}Warning: fedtpg environment not activated${NC}"
    echo "Please run: conda activate fedtpg"
    exit 1
fi

echo -e "${BLUE}Environment: $CONDA_DEFAULT_ENV${NC}"
echo ""

# Configuration
DATA_ROOT="./data"
MODEL_DIR="output/cross_cls/fedtpg/20_8/43/"
LOAD_EPOCH=500

# Step 1: Run detailed evaluation
echo "========================================================================"
echo "Step 1: Running Detailed Evaluation on 6 Datasets"
echo "========================================================================"
echo ""

python evaluate_detailed.py \
    --root $DATA_ROOT \
    --model-dir $MODEL_DIR \
    --load-epoch $LOAD_EPOCH

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Evaluation encountered issues${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Step 1 Complete${NC}"
echo ""

# Step 2: Create basic visualizations
echo "========================================================================"
echo "Step 2: Creating Basic Performance Visualizations"
echo "========================================================================"
echo ""

python create_visualizations.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 2 Complete${NC}"
else
    echo -e "${YELLOW}⚠ Basic visualizations had issues${NC}"
fi
echo ""

# Step 3: Create advanced visualizations
echo "========================================================================"
echo "Step 3: Creating Advanced Visualizations"
echo "========================================================================"
echo ""

python create_advanced_visualizations.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 3 Complete${NC}"
else
    echo -e "${YELLOW}⚠ Advanced visualizations had issues${NC}"
fi
echo ""

# Step 4: Create prompt and feature visualizations
echo "========================================================================"
echo "Step 4: Creating Prompt and Feature Visualizations"
echo "========================================================================"
echo ""

python visualize_prompts_features.py \
    --root $DATA_ROOT \
    --model-dir $MODEL_DIR \
    --load-epoch $LOAD_EPOCH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 4 Complete${NC}"
else
    echo -e "${YELLOW}⚠ Prompt visualizations had issues (may need sklearn/umap)${NC}"
fi
echo ""

# Summary
echo "========================================================================"
echo "Analysis Complete - Summary"
echo "========================================================================"
echo ""

echo "Generated outputs:"
echo ""

echo "1. Detailed Evaluation Results:"
echo "   - evaluation_results_detailed/"
if [ -d "evaluation_results_detailed" ]; then
    num_files=$(find evaluation_results_detailed -type f | wc -l)
    echo -e "     ${GREEN}✓ $num_files files generated${NC}"
fi
echo ""

echo "2. Basic Visualizations:"
echo "   - visualizations/"
if [ -d "visualizations" ]; then
    num_viz=$(find visualizations -name "*.png" | wc -l)
    echo -e "     ${GREEN}✓ $num_viz PNG files${NC}"
fi
echo ""

echo "3. Advanced Visualizations:"
echo "   - visualizations_advanced/"
if [ -d "visualizations_advanced" ]; then
    num_adv=$(find visualizations_advanced -name "*.png" | wc -l)
    echo -e "     ${GREEN}✓ $num_adv PNG files${NC}"
fi
echo ""

echo "4. Prompt/Feature Visualizations:"
echo "   - visualizations_prompts/"
if [ -d "visualizations_prompts" ]; then
    num_prompt=$(find visualizations_prompts -name "*.png" | wc -l)
    echo -e "     ${GREEN}✓ $num_prompt PNG files${NC}"
fi
echo ""

echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo ""
echo "1. Review all visualizations:"
echo "   ${YELLOW}ls -lh visualizations*/${NC}"
echo ""
echo "2. Check detailed results:"
echo "   ${YELLOW}cat evaluation_results_detailed/detailed_results.json${NC}"
echo ""
echo "3. Create presentation slides using the generated figures"
echo ""
echo "4. Optionally, run Jupyter notebook for interactive analysis:"
echo "   ${YELLOW}jupyter notebook analysis_notebook.ipynb${NC}"
echo ""

echo "========================================================================"
echo "All Done! 🎉"
echo "========================================================================"
