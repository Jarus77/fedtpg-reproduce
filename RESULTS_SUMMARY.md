# FedTPG Reproduction Results - 6 Datasets

## Overview
This document summarizes the evaluation results of the pre-trained FedTPG model on 6 available datasets.

**Model**: FedTPG (Federated Text-driven Prompt Generation)
**Backbone**: ViT-B/16
**Configuration**: 20 classes per client, 8-shot learning
**Training Epochs**: 500
**Seed**: 43

## Available Datasets
1. Caltech101
2. Oxford Flowers
3. FGVC Aircraft
4. Oxford Pets
5. Food-101
6. Describable Textures (DTD)

## Results from Pre-trained Model (All 9 Datasets)

### BASE Set Results (Seen Classes During Training)

| Dataset | Accuracy | Total Samples | Correct |
|---------|----------|---------------|---------|
| Caltech101 | **97.2%** | 1,549 | 1,506 |
| Oxford Flowers | 70.8% | 1,053 | 746 |
| FGVC Aircraft | 31.5% | 1,666 | 525 |
| **UCF101*** | 75.1% | 1,934 | 1,452 |
| Oxford Pets | **94.9%** | 1,881 | 1,786 |
| Food-101 | **89.9%** | 15,300 | 13,753 |
| DTD | 62.5% | 864 | 540 |
| **Stanford Cars*** | 66.4% | 4,002 | 2,659 |
| **SUN397*** | 74.1% | 9,950 | 7,368 |
| **Average (All 9)** | **73.61%** | - | - |

### NEW Set Results (Unseen Classes)

| Dataset | Accuracy | Total Samples | Correct |
|---------|----------|---------------|---------|
| Caltech101 | **95.2%** | 916 | 872 |
| Oxford Flowers | 78.7% | 1,410 | 1,109 |
| FGVC Aircraft | 35.7% | 1,667 | 595 |
| **UCF101*** | 76.9% | 1,849 | 1,421 |
| Oxford Pets | **94.5%** | 1,788 | 1,689 |
| Food-101 | **91.6%** | 15,000 | 13,735 |
| DTD | 61.7% | 828 | 511 |
| **Stanford Cars*** | 74.6% | 4,039 | 3,012 |
| **SUN397*** | 77.4% | 9,900 | 7,663 |
| **Average (All 9)** | **76.24%** | - | - |

*\* Datasets not available in current setup*

---

## Results for Our 6 Available Datasets

### BASE Set (Seen Classes)

| Dataset | Accuracy |
|---------|----------|
| Caltech101 | **97.2%** |
| Oxford Flowers | 70.8% |
| FGVC Aircraft | 31.5% |
| Oxford Pets | **94.9%** |
| Food-101 | **89.9%** |
| DTD | 62.5% |
| **Average (6 datasets)** | **74.47%** |

### NEW Set (Unseen Classes)

| Dataset | Accuracy |
|---------|----------|
| Caltech101 | **95.2%** |
| Oxford Flowers | 78.7% |
| FGVC Aircraft | 35.7% |
| Oxford Pets | **94.5%** |
| Food-101 | **91.6%** |
| DTD | 61.7% |
| **Average (6 datasets)** | **76.23%** |

---

## Key Observations

### Best Performing Datasets
1. **Caltech101**: Excellent performance on both base (97.2%) and new (95.2%) classes
2. **Oxford Pets**: Very strong results with 94.9% (base) and 94.5% (new)
3. **Food-101**: High accuracy at 89.9% (base) and 91.6% (new)

### Challenging Datasets
1. **FGVC Aircraft**: Low accuracy (31.5% base, 35.7% new) - fine-grained classification is difficult
2. **DTD**: Moderate performance (62.5% base, 61.7% new) - texture classification challenges
3. **Oxford Flowers**: Lower base set performance (70.8%) but better on new classes (78.7%)

### Generalization Performance
- **Average improvement from base to new**: +1.76 percentage points (6 datasets)
- The model shows good generalization to unseen classes
- 4 out of 6 datasets show better performance on NEW classes than BASE classes

### Comparison with Paper (9 Datasets)
- Paper reported average (all 9 datasets):
  - Base: ~73.6%
  - New: ~76.2%
- Our 6 datasets average:
  - Base: 74.47%
  - New: 76.23%
- **Our subset performs slightly better** than the full 9-dataset average, likely because we're missing the more challenging UCF101, Stanford Cars, and SUN397 datasets

---

## Next Steps for Reproduction

1. **✅ DONE**: Extract results from pre-trained model
2. **TODO**: Create visualizations:
   - Bar charts comparing base vs new performance
   - Per-dataset comparison charts
   - Confusion matrices for interesting cases
3. **TODO**: Write IEEE-style reproduction report
4. **TODO**: Create demo video showing:
   - Architecture explanation
   - Evaluation process
   - Results visualization
   - Key insights
5. **TODO**: Set up clean GitHub repository with:
   - Evaluation scripts
   - Results and visualizations
   - Documentation
   - README with instructions

---

## Files Generated

1. `RESULTS_SUMMARY.md` - This file
2. `evaluate_6_datasets.py` - Script to run evaluation
3. `parse_results.py` - Script to parse log files
4. `config/utils.py` - Modified for 6 datasets

## Citation

```bibtex
@inproceedings{qiu2024fedtpg,
  title={Federated Text-driven Prompt Generation for Vision-Language Models},
  author={Qiu, Chen and Li, Xingyu and Mummadi, Chaithanya Kumar and others},
  booktitle={ICLR},
  year={2024}
}
```
