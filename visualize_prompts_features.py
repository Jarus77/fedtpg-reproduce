"""
Visualize prompts and feature embeddings from FedTPG
Includes t-SNE, UMAP, and prompt analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

from config.defaults import _C as cfg_default
from config.utils import reset_cfg
from federated.server import Server
from dataloader.dm_federated import TestDataManager

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("⚠ sklearn not available for t-SNE")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ umap-learn not available for UMAP")

class PromptFeatureVisualizer:
    def __init__(self, server, cfg, output_dir="visualizations_prompts"):
        self.server = server
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = server.device

    @torch.no_grad()
    def extract_text_embeddings(self, classnames, dataname, num_samples=None):
        """Extract text embeddings for given class names"""

        if num_samples is not None:
            classnames = classnames[:num_samples]

        # Get text features from the model
        text_features_list = []

        for class_name in tqdm(classnames, desc=f"Extracting text embeddings for {dataname}"):
            # Generate prompt for this class
            # This uses the prompt generation network
            text_features = self.server.model.get_text_features([class_name], dataname)
            text_features_list.append(text_features.cpu().numpy())

        text_embeddings = np.vstack(text_features_list)
        return text_embeddings, classnames

    @torch.no_grad()
    def extract_image_features(self, dataloader, max_batches=50):
        """Extract image features from a dataset"""

        image_features_list = []
        labels_list = []
        class_names_list = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting image features")):
            if batch_idx >= max_batches:
                break

            inputs, labels, cnames = self.server.parse_batch(batch)

            # Get image features
            image_features = self.server.model.image_encoder(inputs.to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(image_features.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            class_names_list.extend(cnames)

        image_embeddings = np.vstack(image_features_list)
        labels = np.array(labels_list)
        class_names = class_names_list

        return image_embeddings, labels, class_names

    def visualize_text_embeddings_tsne(self, split='base', max_classes=50):
        """Visualize text embeddings using t-SNE"""

        if not TSNE_AVAILABLE:
            print("⚠ Skipping t-SNE: sklearn not installed")
            return

        print(f"\nGenerating t-SNE visualization for text embeddings ({split} set)...")

        dm = TestDataManager(self.cfg, split)
        datasets = dm.test_datasets

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes.flatten()

        for idx, dataset in enumerate(datasets[:6]):
            dataname = dataset.data_name
            classnames = dataset.classnames[:max_classes]

            print(f"  Processing {dataname}...")

            # Extract embeddings
            embeddings, class_list = self.extract_text_embeddings(classnames, dataname)

            # Apply t-SNE
            print(f"    Running t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(classnames)-1))
            embeddings_2d = tsne.fit_transform(embeddings)

            # Plot
            ax = axes[idx]
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=range(len(class_list)), cmap='tab20',
                               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Add labels for some points
            for i in range(min(10, len(class_list))):
                ax.annotate(class_list[i][:15], (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                          fontsize=8, alpha=0.7)

            ax.set_title(f'{dataname} ({len(class_list)} classes)', fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(datasets), 6):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Text Embeddings Visualization (t-SNE) - {split} set',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f'text_embeddings_tsne_{split}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def visualize_prompt_comparison(self):
        """Compare prompts across different datasets"""

        print("\nGenerating prompt comparison visualization...")

        dm_base = TestDataManager(self.cfg, 'base')
        datasets_base = dm_base.test_datasets

        # Select a few common concepts (if they exist)
        common_concepts = ['dog', 'cat', 'car', 'flower', 'food']

        fig, axes = plt.subplots(len(common_concepts), len(datasets_base[:3]),
                                figsize=(15, len(common_concepts)*3))

        for concept_idx, concept in enumerate(common_concepts):
            for dataset_idx, dataset in enumerate(datasets_base[:3]):
                dataname = dataset.data_name
                classnames = dataset.classnames

                # Find classes containing the concept
                matching_classes = [cn for cn in classnames if concept.lower() in cn.lower()]

                if not matching_classes:
                    continue

                # Get text embeddings for matching classes
                embeddings, class_list = self.extract_text_embeddings(matching_classes[:5], dataname)

                # Visualize as heatmap
                ax = axes[concept_idx, dataset_idx] if len(datasets_base) > 1 else axes[concept_idx]

                if len(embeddings) > 0:
                    # Normalize for visualization
                    emb_norm = (embeddings - embeddings.mean()) / (embeddings.std() + 1e-8)

                    im = ax.imshow(emb_norm[:, :50], cmap='coolwarm', aspect='auto')  # Show first 50 dims
                    ax.set_yticks(range(len(class_list)))
                    ax.set_yticklabels([cn[:20] for cn in class_list], fontsize=8)
                    ax.set_xlabel('Embedding Dimension', fontsize=8)
                    ax.set_title(f'{dataname} - "{concept}"', fontsize=10, fontweight='bold')

                    if dataset_idx == len(datasets_base[:3]) - 1:
                        plt.colorbar(im, ax=ax)

        plt.suptitle('Prompt Embeddings Comparison Across Datasets',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / 'prompt_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def visualize_image_text_alignment(self, split='base'):
        """Visualize alignment between image and text features"""

        print(f"\nGenerating image-text alignment visualization ({split} set)...")

        dm = TestDataManager(self.cfg, split)
        data_loaders = dm.test_loaders
        datasets = dm.test_datasets

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (dataloader, dataset) in enumerate(zip(data_loaders[:6], datasets[:6])):
            dataname = dataset.data_name
            classnames = dataset.classnames

            print(f"  Processing {dataname}...")

            # Extract features (limited samples)
            image_features, labels, _ = self.extract_image_features(dataloader, max_batches=10)
            text_features, _ = self.extract_text_embeddings(classnames, dataname)

            # Compute similarity matrix (sample)
            num_samples = min(50, len(image_features))
            num_classes = len(classnames)

            sample_indices = np.random.choice(len(image_features), num_samples, replace=False)
            image_sample = image_features[sample_indices]
            labels_sample = labels[sample_indices]

            # Compute cosine similarity
            similarity = np.dot(image_sample, text_features.T)

            # Visualize
            ax = axes[idx]
            im = ax.imshow(similarity, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

            ax.set_xlabel('Text Classes', fontweight='bold')
            ax.set_ylabel('Image Samples', fontweight='bold')
            ax.set_title(f'{dataname}', fontweight='bold')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Cosine Similarity')

        # Remove empty subplots
        for idx in range(len(datasets), 6):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Image-Text Feature Alignment ({split} set)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f'image_text_alignment_{split}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def run_all(self):
        """Run all prompt and feature visualizations"""
        print("="*80)
        print("Prompt and Feature Visualization")
        print("="*80)

        self.visualize_text_embeddings_tsne(split='base')
        self.visualize_text_embeddings_tsne(split='new')
        self.visualize_prompt_comparison()
        self.visualize_image_text_alignment(split='base')
        self.visualize_image_text_alignment(split='new')

        print("\n" + "="*80)
        print("All prompt/feature visualizations complete!")
        print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--exp_name", type=str, default="cross_cls")
    parser.add_argument("--model_name", type=str, default="fedtpg")
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--depth_ctx", type=int, default=1)
    parser.add_argument("--model_depth", type=int, default=0)
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_cls_per_client", type=int, default=20)
    parser.add_argument("--avail_percent", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--w", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--model-dir", type=str, default="output/cross_cls/fedtpg/20_8/43/")
    parser.add_argument("--load-epoch", type=int, default=500)

    args = parser.parse_args()

    # Setup config
    cfg = cfg_default.clone()
    reset_cfg(cfg, args)
    cfg.freeze()

    print("Initializing model...")
    fl_server = Server(cfg)

    print(f"Loading model from {args.model_dir}, epoch {args.load_epoch}...")
    fl_server.load_model(args.model_dir, epoch=args.load_epoch)
    print("✓ Model loaded\n")

    # Create visualizer
    visualizer = PromptFeatureVisualizer(fl_server, cfg)
    visualizer.run_all()

if __name__ == "__main__":
    main()
