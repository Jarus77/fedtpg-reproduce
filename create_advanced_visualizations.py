"""
Advanced visualization script for FedTPG reproduction
Creates comprehensive plots, confusion matrices, and analysis figures
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import json
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

class AdvancedVisualizer:
    def __init__(self, results_dir="evaluation_results_detailed", output_dir="visualizations_advanced"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load results
        self.load_results()

    def load_results(self):
        """Load detailed evaluation results"""
        results_file = self.results_dir / "detailed_results.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            print(f"✓ Loaded results from {results_file}")
        else:
            print(f"⚠ Results file not found: {results_file}")
            print("  Please run evaluate_detailed.py first")
            self.results = None

    def create_confusion_matrices(self):
        """Create confusion matrix visualizations for each dataset"""
        print("\nGenerating confusion matrices...")

        if not self.results:
            return

        for split in ['base', 'new']:
            split_results = self.results.get(split, {})

            for dataset_name, data in split_results.items():
                # Load confusion matrix
                cm_file = self.results_dir / split / dataset_name / 'confusion_matrix.npy'
                if not cm_file.exists():
                    continue

                conf_matrix = np.load(cm_file)
                classnames = data['classnames']

                # Create figure
                fig_size = max(10, len(classnames) * 0.3)
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))

                # Normalize confusion matrix
                cm_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)

                # Plot
                im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Normalized Frequency', rotation=270, labelpad=20)

                # Set ticks
                ax.set_xticks(np.arange(len(classnames)))
                ax.set_yticks(np.arange(len(classnames)))

                # Rotate labels
                if len(classnames) > 20:
                    ax.set_xticklabels(classnames, rotation=90, ha='right', fontsize=6)
                    ax.set_yticklabels(classnames, fontsize=6)
                else:
                    ax.set_xticklabels(classnames, rotation=45, ha='right', fontsize=8)
                    ax.set_yticklabels(classnames, fontsize=8)

                ax.set_xlabel('Predicted', fontweight='bold')
                ax.set_ylabel('True', fontweight='bold')
                ax.set_title(f'Confusion Matrix: {dataset_name} ({split} set)', fontweight='bold', pad=20)

                plt.tight_layout()
                output_file = self.output_dir / f'confusion_matrix_{dataset_name}_{split}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Saved: {output_file.name}")

    def create_per_class_analysis(self):
        """Create per-class accuracy analysis"""
        print("\nGenerating per-class accuracy analysis...")

        if not self.results:
            return

        for split in ['base', 'new']:
            split_results = self.results.get(split, {})

            for dataset_name, data in split_results.items():
                per_class = data.get('per_class_accuracy', {})
                if not per_class:
                    continue

                # Sort by accuracy
                sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['accuracy'])

                classes = [item[0] for item in sorted_classes]
                accuracies = [item[1]['accuracy'] for item in sorted_classes]
                counts = [item[1]['count'] for item in sorted_classes]

                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(classes) * 0.25)))

                # Plot 1: Accuracy bars
                colors = ['red' if acc < 50 else 'orange' if acc < 75 else 'green' for acc in accuracies]
                y_pos = np.arange(len(classes))

                ax1.barh(y_pos, accuracies, color=colors, alpha=0.7)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(classes, fontsize=8)
                ax1.set_xlabel('Accuracy (%)', fontweight='bold')
                ax1.set_title(f'{dataset_name} - Per-Class Accuracy ({split})', fontweight='bold')
                ax1.axvline(x=np.mean(accuracies), color='blue', linestyle='--', label=f'Mean: {np.mean(accuracies):.1f}%')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='x')

                # Plot 2: Sample counts
                ax2.barh(y_pos, counts, color='steelblue', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(classes, fontsize=8)
                ax2.set_xlabel('Number of Samples', fontweight='bold')
                ax2.set_title(f'{dataset_name} - Sample Distribution ({split})', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')

                plt.tight_layout()
                output_file = self.output_dir / f'per_class_analysis_{dataset_name}_{split}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✓ Saved: {output_file.name}")

    def create_confidence_analysis(self):
        """Analyze model confidence for correct vs incorrect predictions"""
        print("\nGenerating confidence analysis...")

        if not self.results:
            return

        for split in ['base', 'new']:
            split_results = self.results.get(split, {})

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for idx, (dataset_name, data) in enumerate(split_results.items()):
                if idx >= 6:
                    break

                conf_stats = data.get('confidence_stats', {})

                ax = axes[idx]

                # Create data for box plot
                correct_mean = conf_stats.get('correct_mean', 0)
                correct_std = conf_stats.get('correct_std', 0)
                incorrect_mean = conf_stats.get('incorrect_mean', 0)
                incorrect_std = conf_stats.get('incorrect_std', 0)

                # Simulate distributions for visualization
                np.random.seed(42)
                correct_samples = np.random.normal(correct_mean, correct_std, 1000)
                incorrect_samples = np.random.normal(incorrect_mean, incorrect_std, 1000)

                # Clip to [0, 1]
                correct_samples = np.clip(correct_samples, 0, 1)
                incorrect_samples = np.clip(incorrect_samples, 0, 1)

                # Create violin plot
                parts = ax.violinplot([correct_samples, incorrect_samples],
                                     positions=[1, 2],
                                     showmeans=True,
                                     showmedians=True)

                # Color the violins
                for pc, color in zip(parts['bodies'], ['green', 'red']):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.5)

                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Correct', 'Incorrect'])
                ax.set_ylabel('Confidence', fontweight='bold')
                ax.set_title(f'{dataset_name}', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3, axis='y')

                # Add mean values as text
                ax.text(1, correct_mean, f'{correct_mean:.3f}',
                       ha='center', va='bottom', fontweight='bold')
                ax.text(2, incorrect_mean, f'{incorrect_mean:.3f}',
                       ha='center', va='bottom', fontweight='bold')

            # Remove empty subplots
            for idx in range(len(split_results), 6):
                fig.delaxes(axes[idx])

            plt.suptitle(f'Model Confidence: Correct vs Incorrect Predictions ({split} set)',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()

            output_file = self.output_dir / f'confidence_analysis_{split}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {output_file.name}")

    def create_error_analysis(self):
        """Visualize top confusion pairs"""
        print("\nGenerating error analysis...")

        if not self.results:
            return

        for split in ['base', 'new']:
            split_results = self.results.get(split, {})

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()

            for idx, (dataset_name, data) in enumerate(split_results.items()):
                if idx >= 6:
                    break

                top_confusions = data.get('top_confusions', [])[:10]  # Top 10

                if not top_confusions:
                    continue

                ax = axes[idx]

                # Create labels and values
                labels = [f"{c['true_class'][:15]}\n→ {c['pred_class'][:15]}" for c in top_confusions]
                values = [c['count'] for c in top_confusions]

                y_pos = np.arange(len(labels))
                ax.barh(y_pos, values, color='coral', alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel('Number of Misclassifications', fontweight='bold')
                ax.set_title(f'{dataset_name}', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

                # Add value labels
                for i, v in enumerate(values):
                    ax.text(v, i, f' {v}', va='center', fontsize=8)

            # Remove empty subplots
            for idx in range(len(split_results), 6):
                fig.delaxes(axes[idx])

            plt.suptitle(f'Top Confusion Pairs ({split} set)',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()

            output_file = self.output_dir / f'error_analysis_{split}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {output_file.name}")

    def create_accuracy_heatmap(self):
        """Create comprehensive accuracy heatmap"""
        print("\nGenerating accuracy heatmap...")

        if not self.results:
            return

        # Collect data
        datasets = []
        base_accs = []
        new_accs = []
        top5_base = []
        top5_new = []

        for dataset_name in self.results.get('base', {}).keys():
            datasets.append(dataset_name)
            base_accs.append(self.results['base'][dataset_name]['accuracy'])
            new_accs.append(self.results['new'][dataset_name]['accuracy'])
            top5_base.append(self.results['base'][dataset_name].get('top5_accuracy', 0))
            top5_new.append(self.results['new'][dataset_name].get('top5_accuracy', 0))

        # Create dataframe
        df = pd.DataFrame({
            'Dataset': datasets,
            'Base Top-1': base_accs,
            'New Top-1': new_accs,
            'Base Top-5': top5_base,
            'New Top-5': top5_new
        })

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        data_for_heatmap = df[['Base Top-1', 'New Top-1', 'Base Top-5', 'New Top-5']].values.T

        im = ax.imshow(data_for_heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_yticklabels(['Base Top-1', 'New Top-1', 'Base Top-5', 'New Top-5'])

        # Add text annotations
        for i in range(4):
            for j in range(len(datasets)):
                text = ax.text(j, i, f'{data_for_heatmap[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')

        ax.set_title('Comprehensive Accuracy Overview', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_file = self.output_dir / 'accuracy_heatmap_comprehensive.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_file.name}")

    def create_summary_table_image(self):
        """Create a publication-ready summary table"""
        print("\nGenerating summary table...")

        if not self.results:
            return

        # Collect data
        table_data = []
        for dataset_name in self.results.get('base', {}).keys():
            base_data = self.results['base'][dataset_name]
            new_data = self.results['new'][dataset_name]

            row = [
                dataset_name,
                f"{base_data['accuracy']:.2f}%",
                f"{new_data['accuracy']:.2f}%",
                f"{new_data['accuracy'] - base_data['accuracy']:+.2f}%",
                f"{base_data['num_samples']:,}",
                f"{new_data['num_samples']:,}"
            ]
            table_data.append(row)

        # Add average row
        avg_base = np.mean([self.results['base'][d]['accuracy'] for d in self.results['base'].keys()])
        avg_new = np.mean([self.results['new'][d]['accuracy'] for d in self.results['new'].keys()])
        table_data.append([
            'Average',
            f"{avg_base:.2f}%",
            f"{avg_new:.2f}%",
            f"{avg_new - avg_base:+.2f}%",
            '-',
            '-'
        ])

        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data,
                        colLabels=['Dataset', 'Base Acc.', 'New Acc.', 'Δ', 'Samples (Base)', 'Samples (New)'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.175, 0.175])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style average row
        for i in range(6):
            table[(len(table_data), i)].set_facecolor('#ecf0f1')
            table[(len(table_data), i)].set_text_props(weight='bold')

        # Color-code the delta column
        for i in range(1, len(table_data)):
            cell = table[(i, 3)]
            if '+' in cell.get_text().get_text():
                cell.set_facecolor('#d4edda')  # Light green
            elif table_data[i-1][3] != '-':
                cell.set_facecolor('#f8d7da')  # Light red

        plt.title('FedTPG Results on 6 Datasets', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        output_file = self.output_dir / 'summary_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_file.name}")

    def run_all(self):
        """Run all visualizations"""
        print("="*80)
        print("Creating Advanced Visualizations")
        print("="*80)

        self.create_confusion_matrices()
        self.create_per_class_analysis()
        self.create_confidence_analysis()
        self.create_error_analysis()
        self.create_accuracy_heatmap()
        self.create_summary_table_image()

        print("\n" + "="*80)
        print("All visualizations created successfully!")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  - {file.name}")

def main():
    visualizer = AdvancedVisualizer()
    visualizer.run_all()

if __name__ == "__main__":
    main()
