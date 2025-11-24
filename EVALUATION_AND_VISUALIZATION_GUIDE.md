# FedTPG Evaluation & Visualization Guide

## 🎯 Overview

This guide explains how to run complete evaluation and create comprehensive visualizations for the FedTPG reproduction study on 6 datasets.

---

## 📋 What's Been Prepared

### Evaluation Scripts
1. **`evaluate_detailed.py`** - Enhanced evaluation with detailed metrics
   - Per-class accuracies
   - Confusion matrices
   - Confidence statistics
   - Error analysis

2. **`parse_results.py`** - Extract results from existing logs
   - Quick results extraction
   - No GPU needed

### Visualization Scripts
1. **`create_visualizations.py`** - Basic performance plots
   - Base vs new comparison
   - Generalization gap
   - Method comparison
   - Performance heatmaps

2. **`create_advanced_visualizations.py`** - Advanced analysis
   - Confusion matrices for all datasets
   - Per-class accuracy analysis
   - Confidence analysis
   - Error pattern visualization

3. **`visualize_prompts_features.py`** - Prompt and feature analysis
   - Text embedding visualization (t-SNE)
   - Prompt comparison across datasets
   - Image-text alignment
   - Feature space analysis

### Analysis Tools
1. **`analysis_notebook.ipynb`** - Interactive Jupyter notebook
   - Explore results interactively
   - Statistical analysis
   - Export tables for report

2. **`run_complete_analysis.sh`** - Master script
   - Runs everything in sequence
   - Progress tracking
   - Error handling

---

## 🚀 Quick Start (Recommended)

### Option 1: Run Everything at Once

```bash
# Activate environment
conda activate fedtpg

# Run complete analysis pipeline
bash run_complete_analysis.sh
```

This will:
1. Run detailed evaluation on 6 datasets (~30-60 min with GPU)
2. Generate all basic visualizations (~5 min)
3. Create advanced visualizations (~10 min)
4. Generate prompt/feature visualizations (~15 min)

**Total Time**: ~1-1.5 hours

---

## 📊 Step-by-Step (Advanced)

### Step 1: Run Detailed Evaluation

```bash
conda activate fedtpg

python evaluate_detailed.py \
    --root ./data \
    --model-dir output/cross_cls/fedtpg/20_8/43/ \
    --load-epoch 500
```

**Output**:
- `evaluation_results_detailed/detailed_results.json` - All results
- `evaluation_results_detailed/base/{dataset}/` - Base set data
- `evaluation_results_detailed/new/{dataset}/` - New set data
  - `predictions.npy` - Model predictions
  - `labels.npy` - True labels
  - `logits.npy` - Raw logits
  - `confusion_matrix.npy` - Confusion matrix

**Time**: 30-60 minutes (with GPU)

### Step 2: Generate Basic Visualizations

```bash
python create_visualizations.py
```

**Output** (`visualizations/`):
- `base_vs_new_comparison.png`
- `generalization_gap.png`
- `method_comparison.png`
- `performance_heatmap.png`
- `results_table.png`
- `architecture_overview.png`

**Time**: ~5 minutes

### Step 3: Create Advanced Visualizations

```bash
python create_advanced_visualizations.py
```

**Output** (`visualizations_advanced/`):
- `confusion_matrix_{dataset}_{split}.png` - For each dataset
- `per_class_analysis_{dataset}_{split}.png` - Per-class performance
- `confidence_analysis_{split}.png` - Confidence distributions
- `error_analysis_{split}.png` - Top confusion pairs
- `accuracy_heatmap_comprehensive.png` - Overall heatmap
- `summary_table.png` - Publication-ready table

**Time**: ~10 minutes

### Step 4: Visualize Prompts and Features

```bash
python visualize_prompts_features.py \
    --root ./data \
    --model-dir output/cross_cls/fedtpg/20_8/43/ \
    --load-epoch 500
```

**Output** (`visualizations_prompts/`):
- `text_embeddings_tsne_base.png` - Text embedding visualization (base)
- `text_embeddings_tsne_new.png` - Text embedding visualization (new)
- `prompt_comparison.png` - Prompts across datasets
- `image_text_alignment_base.png` - Alignment visualization (base)
- `image_text_alignment_new.png` - Alignment visualization (new)

**Requirements**:
- sklearn (for t-SNE): `pip install scikit-learn`
- umap-learn (optional): `pip install umap-learn`

**Time**: ~15 minutes

### Step 5: Interactive Analysis (Optional)

```bash
jupyter notebook analysis_notebook.ipynb
```

**Features**:
- Interactive dataset exploration
- Statistical tests
- Export LaTeX tables
- Custom visualizations

---

## 📂 Output Directory Structure

After running all scripts:

```
FedTPG/
├── evaluation_results_detailed/
│   ├── detailed_results.json
│   ├── base/
│   │   ├── Caltech101/
│   │   ├── OxfordFlowers/
│   │   ├── FGVCAircraft/
│   │   ├── OxfordPets/
│   │   ├── Food101/
│   │   └── DescribableTextures/
│   └── new/
│       └── [same structure]
│
├── visualizations/
│   ├── base_vs_new_comparison.png
│   ├── generalization_gap.png
│   ├── method_comparison.png
│   ├── performance_heatmap.png
│   ├── results_table.png
│   └── architecture_overview.png
│
├── visualizations_advanced/
│   ├── confusion_matrix_*.png (12 files)
│   ├── per_class_analysis_*.png (12 files)
│   ├── confidence_analysis_*.png (2 files)
│   ├── error_analysis_*.png (2 files)
│   ├── accuracy_heatmap_comprehensive.png
│   └── summary_table.png
│
└── visualizations_prompts/
    ├── text_embeddings_tsne_base.png
    ├── text_embeddings_tsne_new.png
    ├── prompt_comparison.png
    ├── image_text_alignment_base.png
    └── image_text_alignment_new.png
```

**Total Visualizations**: ~50+ PNG files

---

## 🎨 Using Visualizations

### For Presentation

**Must-have slides**:
1. `summary_table.png` - Results overview
2. `base_vs_new_comparison.png` - Main result
3. `generalization_gap.png` - Key finding
4. `confusion_matrix_Caltech101_new.png` - Example analysis
5. `confidence_analysis_new.png` - Model confidence

**Good-to-have**:
- `per_class_analysis_*.png` - Deep dives
- `error_analysis_*.png` - Error patterns
- `text_embeddings_tsne_*.png` - Prompt visualization

### For Report

**Main Results** (Section: Results):
- `summary_table.png` or raw data
- `base_vs_new_comparison.png`
- `accuracy_heatmap_comprehensive.png`

**Analysis** (Section: Analysis):
- `confusion_matrix_*.png` (2-3 examples)
- `confidence_analysis_*.png`
- `error_analysis_*.png`

**Method** (Section: Methodology):
- `architecture_overview.png`
- `prompt_comparison.png`

---

## 🔧 Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in evaluation
```bash
python evaluate_detailed.py --batch_size 50
```

### Issue: Scikit-learn not found (t-SNE)

**Solution**: Install sklearn
```bash
conda activate fedtpg
pip install scikit-learn
```

### Issue: Evaluation is slow

**Solution**: The evaluation needs to run through all test samples. Expected times:
- With GPU (CUDA): 30-60 minutes
- CPU only: 3-4 hours

You can also just use existing results from logs:
```bash
python parse_results.py  # Extract from existing logs (fast)
```

### Issue: Matplotlib errors

**Solution**: Update matplotlib
```bash
pip install --upgrade matplotlib seaborn
```

---

## 📊 Expected Results

Based on pre-trained model on 6 datasets:

| Dataset | Base | New | Δ |
|---------|------|-----|---|
| Caltech101 | 97.2% | 95.2% | -2.0% |
| Oxford Flowers | 70.8% | 78.7% | +7.9% |
| FGVC Aircraft | 31.5% | 35.7% | +4.2% |
| Oxford Pets | 94.9% | 94.5% | -0.4% |
| Food-101 | 89.9% | 91.6% | +1.7% |
| DTD | 62.5% | 61.7% | -0.8% |
| **Average** | **74.47%** | **76.23%** | **+1.76%** |

---

## ⚡ Performance Tips

1. **Use GPU**: Makes evaluation 5-6x faster
2. **Run during off-hours**: Evaluation can take time
3. **Generate visualizations separately**: After evaluation completes
4. **Cache results**: Evaluation results are saved, visualizations can be regenerated anytime

---

## 📝 Next Steps After Visualization

1. **Review all figures**: Check quality and clarity
2. **Select key figures**: Choose 6-8 for presentation
3. **Create presentation**: Use outline in `presentation/PRESENTATION_OUTLINE.md`
4. **Write analysis**: Document findings
5. **Prepare demo**: Screen record the Jupyter notebook

---

## 🆘 Need Help?

**Common Questions**:

**Q: Do I need to run evaluation if I already have log files?**
A: No! You can use `parse_results.py` to extract results from existing logs, then run visualizations.

**Q: Can I skip prompt visualization?**
A: Yes. It's optional and requires additional libraries. Focus on performance visualizations first.

**Q: How long does everything take?**
A:
- Evaluation: 30-60 min (GPU) or 3-4 hrs (CPU)
- All visualizations: 30-40 min
- Total: ~1.5-2 hours

**Q: Which visualizations are most important?**
A:
1. `summary_table.png`
2. `base_vs_new_comparison.png`
3. `generalization_gap.png`
4. 2-3 confusion matrices

---

## ✅ Checklist

Before creating presentation:

- [ ] Evaluation completed
- [ ] All basic visualizations generated
- [ ] Advanced visualizations created
- [ ] Reviewed confusion matrices
- [ ] Checked confidence analysis
- [ ] Identified key findings
- [ ] Selected figures for presentation
- [ ] Exported summary table

---

**Happy analyzing! 📊**
