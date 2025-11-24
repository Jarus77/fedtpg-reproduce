# FedTPG Reproduction Study

**Paper**: Federated Text-driven Prompt Generation for Vision-Language Models (ICLR 2024)

**Authors**: Chen Qiu, Xingyu Li, Chaithanya Kumar Mummadi, et al.

**Original Repository**: [https://github.com/boschresearch/FedTPG](https://github.com/boschresearch/FedTPG)

---

## 📋 Overview

This repository contains a reproduction study of FedTPG, evaluating the pre-trained model on 6 image classification datasets. Our study validates the paper's key claim that text-driven prompt generation enables effective generalization to unseen classes in federated learning settings.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Base Classes Accuracy** | 74.47% |
| **New Classes Accuracy** | 76.23% |
| **Generalization Improvement** | +1.76% |

✅ **Successfully validated**: FedTPG generalizes effectively to unseen classes

## 📊 Datasets Used

We evaluated on 6 out of 9 datasets from the original paper:

1. ✅ Caltech101 (97.2% base / 95.2% new)
2. ✅ Oxford Flowers (70.8% base / 78.7% new)
3. ✅ FGVC Aircraft (31.5% base / 35.7% new)
4. ✅ Oxford Pets (94.9% base / 94.5% new)
5. ✅ Food-101 (89.9% base / 91.6% new)
6. ✅ DTD (62.5% base / 61.7% new)

Missing datasets: UCF101, Stanford Cars, SUN397

## 🚀 Quick Start

### Prerequisites

```bash
conda create -n fedtpg python=3.10
conda activate fedtpg
pip install -r requirements.txt
```

### Download Datasets

Follow the instructions from [CoOp](https://github.com/KaiyangZhou/CoOp) to download and prepare the datasets.

### Run Evaluation

Using pre-trained weights (included in `output/cross_cls/fedtpg/20_8/43/`):

```bash
# Evaluate on 6 datasets
python evaluate_6_datasets.py --data-root ./data

# Or use the modified Launch_FL.py
python Launch_FL.py \
    --root ./data \
    --exp_name cross_cls \
    --model_name fedtpg \
    --eval-only \
    --model-dir output/cross_cls/fedtpg/20_8/43/ \
    --load-epoch 500
```

### Generate Visualizations

```bash
conda activate fedtpg
python create_visualizations.py
```

This creates:
- `visualizations/base_vs_new_comparison.png`
- `visualizations/generalization_gap.png`
- `visualizations/method_comparison.png`
- `visualizations/performance_heatmap.png`
- `visualizations/results_table.png`
- `visualizations/architecture_overview.png`

### Parse Results

```bash
python parse_results.py
```

Generates:
- `evaluation_results/comparison_table_6datasets.csv`
- `evaluation_results/extracted_results.json`

## 📁 Repository Structure

```
FedTPG/
├── clip/                           # CLIP model implementation
├── config/                         # Configuration files
│   ├── defaults.py                # Default configurations
│   └── utils.py                   # Config utilities (modified for 6 datasets)
├── dataloader/                     # Data loading utilities
├── federated/                      # Federated learning implementation
│   ├── server.py                  # FL server
│   ├── client.py                  # FL client
│   └── base_trainer.py            # Training logic
├── model/                          # Model architectures
│   ├── FedTPG.py                  # FedTPG implementation
│   ├── custom_coop.py             # CoOp baseline
│   └── prompt_net.py              # Prompt generation network
├── output/                         # Pre-trained models
│   └── cross_cls/fedtpg/20_8/43/
│       ├── log.txt                # Training logs
│       └── prompt_learner/
│           └── model.pth.tar-500  # Pre-trained weights
├── data/                           # Datasets (download separately)
├── reproduction_report/            # LaTeX report
│   └── fedtpg_reproduction.tex
├── visualizations/                 # Generated figures
├── evaluation_results/             # Evaluation outputs
├── Launch_FL.py                    # Main training/evaluation script
├── evaluate_6_datasets.py          # Simplified evaluation script
├── create_visualizations.py        # Visualization generation
├── parse_results.py                # Results parsing
├── RESULTS_SUMMARY.md              # Detailed results
├── REPRODUCTION_README.md          # This file
└── requirements.txt                # Python dependencies
```

## 🔬 Methodology

### FedTPG Architecture

1. **Prompt Generation Network**: Learns to generate text prompts conditioned on class names
2. **Text Encoder**: Frozen CLIP text encoder processes generated prompts
3. **Image Encoder**: Frozen ViT-B/16 extracts visual features
4. **Federated Training**: Clients train locally and aggregate via FedAvg

### Key Hyperparameters

- **Backbone**: ViT-B/16
- **Shots**: 8 per class
- **Classes per client**: 20
- **Context tokens**: 4
- **Context depth**: 1
- **Training epochs**: 500
- **Batch size**: 200
- **Optimizer**: SGD (momentum=0.9, lr=0.003)
- **LR scheduler**: Cosine annealing

## 📈 Detailed Results

### Per-Dataset Performance

| Dataset | Base Acc (%) | New Acc (%) | Δ (%) | Samples (Base) | Samples (New) |
|---------|--------------|-------------|-------|----------------|---------------|
| Caltech101 | 97.2 | 95.2 | -2.0 | 1,549 | 916 |
| Oxford Flowers | 70.8 | 78.7 | **+7.9** | 1,053 | 1,410 |
| FGVC Aircraft | 31.5 | 35.7 | +4.2 | 1,666 | 1,667 |
| Oxford Pets | 94.9 | 94.5 | -0.4 | 1,881 | 1,788 |
| Food-101 | 89.9 | 91.6 | +1.7 | 15,300 | 15,000 |
| DTD | 62.5 | 61.7 | -0.8 | 864 | 828 |
| **Average** | **74.47** | **76.23** | **+1.76** | - | - |

### Key Observations

✅ **Strong base performance**: Average 74.47% on seen classes

✅ **Effective generalization**: +1.76% improvement on unseen classes

✅ **Best generalization**: Oxford Flowers (+7.9%) and FGVC Aircraft (+4.2%)

⚠️ **Challenging tasks**: Fine-grained recognition (Aircraft: 31-36%)

## 🎓 Reproduction Report

The full reproduction report is available as:
- **LaTeX**: `reproduction_report/fedtpg_reproduction.tex`
- **PDF**: Compile with `pdflatex` or `overleaf`

### Compiling the Report

```bash
cd reproduction_report
pdflatex fedtpg_reproduction.tex
bibtex fedtpg_reproduction
pdflatex fedtpg_reproduction.tex
pdflatex fedtpg_reproduction.tex
```

## 🎥 Demo Video

[Link to presentation/demo video will be added]

**Contents**:
1. Problem introduction and motivation
2. FedTPG architecture overview
3. Evaluation process demonstration
4. Results visualization and analysis
5. Key insights and conclusions

## 📝 Citation

Original paper:

```bibtex
@inproceedings{qiu2024fedtpg,
  title={Federated Text-driven Prompt Generation for Vision-Language Models},
  author={Qiu, Chen and Li, Xingyu and Mummadi, Chaithanya Kumar and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

This reproduction study:

```bibtex
@misc{fedtpg_reproduction2024,
  title={Reproduction Study: Federated Text-driven Prompt Generation for Vision-Language Models},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub Repository},
  url={[Your GitHub URL]}
}
```

## 🔗 Links

- 📄 **Original Paper**: [OpenReview](https://openreview.net/forum?id=NW31gAylIm)
- 💻 **Original Code**: [GitHub](https://github.com/boschresearch/FedTPG)
- 📊 **ArXiv**: [2310.06123](https://arxiv.org/abs/2310.06123)

## 🤝 Acknowledgments

- Original FedTPG authors: Chen Qiu et al. @ Bosch Research
- CoOp framework: Kaiyang Zhou et al.
- CLIP model: OpenAI

## 📧 Contact

For questions about this reproduction:
- Email: [Your Email]
- GitHub Issues: [Your Repo Issues]

## 📜 License

This reproduction study follows the same license as the original FedTPG repository (AGPL-3.0).

---

**Last Updated**: November 2024
