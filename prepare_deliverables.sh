#!/bin/bash
# Master script to prepare all FedTPG reproduction deliverables

echo "========================================================================"
echo "FedTPG Reproduction - Deliverables Preparation"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "fedtpg" ]]; then
    echo -e "${YELLOW}Warning: fedtpg conda environment not activated${NC}"
    echo "Please run: conda activate fedtpg"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Creating output directories"
echo "------------------------------------"
mkdir -p visualizations
mkdir -p evaluation_results
mkdir -p reproduction_report
mkdir -p presentation
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

echo "Step 2: Generating visualizations"
echo "------------------------------------"
if [ -f "create_visualizations.py" ]; then
    echo "Running create_visualizations.py..."
    python create_visualizations.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Visualizations generated successfully${NC}"
    else
        echo -e "${RED}✗ Error generating visualizations${NC}"
    fi
else
    echo -e "${RED}✗ create_visualizations.py not found${NC}"
fi
echo ""

echo "Step 3: Parsing results from logs"
echo "------------------------------------"
if [ -f "parse_results.py" ]; then
    echo "Running parse_results.py..."
    python parse_results.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Results parsed successfully${NC}"
    else
        echo -e "${RED}✗ Error parsing results${NC}"
    fi
else
    echo -e "${RED}✗ parse_results.py not found${NC}"
fi
echo ""

echo "Step 4: Checking generated files"
echo "------------------------------------"
echo "Visualizations:"
for file in visualizations/*.png visualizations/*.csv; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename $file)"
    fi
done

echo ""
echo "Results:"
for file in evaluation_results/*.json evaluation_results/*.csv; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename $file)"
    fi
done
echo ""

echo "Step 5: Verifying documentation"
echo "------------------------------------"
docs=(
    "RESULTS_SUMMARY.md"
    "REPRODUCTION_README.md"
    "DELIVERABLES_CHECKLIST.md"
    "reproduction_report/fedtpg_reproduction.tex"
    "presentation/PRESENTATION_OUTLINE.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "  ${GREEN}✓${NC} $doc"
    else
        echo -e "  ${RED}✗${NC} $doc (missing)"
    fi
done
echo ""

echo "Step 6: Checking model weights"
echo "------------------------------------"
if [ -f "output/cross_cls/fedtpg/20_8/43/prompt_learner/model.pth.tar-500" ]; then
    echo -e "${GREEN}✓ Pre-trained model weights found${NC}"
else
    echo -e "${RED}✗ Model weights not found${NC}"
fi
echo ""

echo "Step 7: Creating .gitignore (if needed)"
echo "------------------------------------"
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Jupyter
.ipynb_checkpoints

# Data
data/
!data/.gitkeep

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temp
tmp/
temp/
EOF
    echo -e "${GREEN}✓ .gitignore created${NC}"
else
    echo -e "${YELLOW}  .gitignore already exists${NC}"
fi
echo ""

echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""

# Count generated files
viz_count=$(find visualizations -type f 2>/dev/null | wc -l)
results_count=$(find evaluation_results -type f 2>/dev/null | wc -l)

echo "Generated files:"
echo "  - Visualizations: $viz_count files"
echo "  - Results: $results_count files"
echo ""

echo "Next steps:"
echo ""
echo "1. Review generated visualizations:"
echo "   ${YELLOW}ls -lh visualizations/${NC}"
echo ""
echo "2. Review extracted results:"
echo "   ${YELLOW}cat evaluation_results/extracted_results.json${NC}"
echo ""
echo "3. Compile LaTeX report:"
echo "   ${YELLOW}cd reproduction_report && pdflatex fedtpg_reproduction.tex${NC}"
echo ""
echo "4. Create presentation slides:"
echo "   ${YELLOW}Use presentation/PRESENTATION_OUTLINE.md as guide${NC}"
echo ""
echo "5. Setup Git repository:"
echo "   ${YELLOW}git init && git add . && git commit -m \"Initial commit\"${NC}"
echo ""
echo "6. Review full checklist:"
echo "   ${YELLOW}cat DELIVERABLES_CHECKLIST.md${NC}"
echo ""

echo "========================================================================"
echo "Preparation complete!"
echo "========================================================================"
echo ""
echo "For detailed instructions, see: DELIVERABLES_CHECKLIST.md"
echo ""
