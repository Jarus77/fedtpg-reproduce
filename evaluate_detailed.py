"""
Enhanced evaluation script for FedTPG with detailed outputs for visualization
Captures predictions, prompts, embeddings, and more
"""

import argparse
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from config.defaults import _C as cfg_default
from config.utils import reset_cfg
from utils import setup_logger
from federated.server import Server
from dataloader.dm_federated import TestDataManager
from sklearn.metrics import confusion_matrix
import pickle

class DetailedEvaluator:
    """Enhanced evaluator that captures detailed information"""

    def __init__(self, server, cfg, output_dir="evaluation_results_detailed"):
        self.server = server
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            'datasets': {},
            'overall': {}
        }

    @torch.no_grad()
    def evaluate_detailed(self, split='base'):
        """Run detailed evaluation on specified split"""

        print(f"\n{'='*80}")
        print(f"Detailed Evaluation on {split.upper()} set")
        print(f"{'='*80}\n")

        self.server.set_model_mode("eval")
        dm = TestDataManager(self.cfg, split)
        data_loaders = dm.test_loaders
        datasets = dm.test_datasets

        split_results = {}

        for i, data_loader in enumerate(data_loaders):
            dataset_name = self.cfg.DATASET.TESTNAME_SPACE[i]
            classnames = datasets[i].classnames
            dataname = datasets[i].data_name

            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} ({split} set)")
            print(f"Classes: {len(classnames)}")
            print(f"{'='*60}\n")

            # Storage for this dataset
            all_predictions = []
            all_labels = []
            all_logits = []
            all_class_names = []
            sample_prompts = {}

            # Process batches
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
                inputs, labels, cnames = self.server.parse_batch(batch)

                # Get model outputs
                outputs = self.server.model_inference(inputs, classnames, dataname)
                logits = outputs  # logits for all classes

                # Get predictions
                preds = logits.argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                logits_np = logits.cpu().numpy()

                all_predictions.extend(preds)
                all_labels.extend(labels_np)
                all_logits.extend(logits_np)
                all_class_names.extend(cnames)

                # Sample some prompts (first batch only)
                if batch_idx == 0 and len(sample_prompts) == 0:
                    # Get prompts for first few classes
                    for class_idx in range(min(5, len(classnames))):
                        class_name = classnames[class_idx]
                        sample_prompts[class_name] = {
                            'class_idx': class_idx,
                            'class_name': class_name
                        }

            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_logits = np.array(all_logits)

            # Calculate metrics
            accuracy = (all_predictions == all_labels).mean() * 100

            # Per-class accuracy
            per_class_acc = {}
            for class_idx, class_name in enumerate(classnames):
                mask = all_labels == class_idx
                if mask.sum() > 0:
                    class_acc = (all_predictions[mask] == all_labels[mask]).mean() * 100
                    per_class_acc[class_name] = {
                        'accuracy': float(class_acc),
                        'count': int(mask.sum())
                    }

            # Confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            # Top-5 accuracy
            top5_preds = np.argsort(all_logits, axis=1)[:, -5:]
            top5_correct = np.array([label in top5_preds[i] for i, label in enumerate(all_labels)])
            top5_acc = top5_correct.mean() * 100

            # Confidence statistics
            confidences = np.max(torch.softmax(torch.tensor(all_logits), dim=1).numpy(), axis=1)
            correct_confidences = confidences[all_predictions == all_labels]
            incorrect_confidences = confidences[all_predictions != all_labels]

            # Error analysis - most confused pairs
            conf_pairs = []
            for true_idx in range(len(classnames)):
                for pred_idx in range(len(classnames)):
                    if true_idx != pred_idx and conf_matrix[true_idx, pred_idx] > 0:
                        conf_pairs.append({
                            'true_class': classnames[true_idx],
                            'pred_class': classnames[pred_idx],
                            'count': int(conf_matrix[true_idx, pred_idx])
                        })
            conf_pairs.sort(key=lambda x: x['count'], reverse=True)

            # Store results
            dataset_results = {
                'dataset_name': dataset_name,
                'split': split,
                'num_classes': len(classnames),
                'num_samples': len(all_labels),
                'accuracy': float(accuracy),
                'top5_accuracy': float(top5_acc),
                'per_class_accuracy': per_class_acc,
                'confusion_matrix': conf_matrix.tolist(),
                'classnames': classnames,
                'confidence_stats': {
                    'correct_mean': float(correct_confidences.mean()),
                    'correct_std': float(correct_confidences.std()),
                    'incorrect_mean': float(incorrect_confidences.mean()),
                    'incorrect_std': float(incorrect_confidences.std())
                },
                'top_confusions': conf_pairs[:10],  # Top 10 confused pairs
                'sample_prompts': sample_prompts
            }

            split_results[dataset_name] = dataset_results

            # Save predictions and logits for this dataset
            dataset_dir = os.path.join(self.output_dir, split, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            np.save(os.path.join(dataset_dir, 'predictions.npy'), all_predictions)
            np.save(os.path.join(dataset_dir, 'labels.npy'), all_labels)
            np.save(os.path.join(dataset_dir, 'logits.npy'), all_logits)
            np.save(os.path.join(dataset_dir, 'confusion_matrix.npy'), conf_matrix)

            with open(os.path.join(dataset_dir, 'class_names.txt'), 'w') as f:
                for cn in all_class_names[:100]:  # Save first 100
                    f.write(f"{cn}\n")

            # Print summary
            print(f"\n✓ {dataset_name} Results:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
            print(f"  Confidence (correct): {correct_confidences.mean():.3f} ± {correct_confidences.std():.3f}")
            print(f"  Confidence (incorrect): {incorrect_confidences.mean():.3f} ± {incorrect_confidences.std():.3f}")
            print(f"  Saved to: {dataset_dir}")

        return split_results

    def save_results(self):
        """Save all results to JSON"""
        output_file = os.path.join(self.output_dir, 'detailed_results.json')
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ All results saved to: {output_file}")

def setup_cfg_from_args(args):
    cfg = cfg_default.clone()
    reset_cfg(cfg, args)
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Detailed FedTPG Evaluation")
    parser.add_argument("--root", type=str, default="./data", help="path to dataset")
    parser.add_argument("--exp_name", type=str, default="cross_cls", help="experiment name")
    parser.add_argument("--model_name", type=str, default="fedtpg", help="model name")
    parser.add_argument("--num_shots", type=int, default=8, help="number of shots")
    parser.add_argument("--depth_ctx", type=int, default=1, help="depth of ctx")
    parser.add_argument("--model_depth", type=int, default=0, help="model depth")
    parser.add_argument("--n_ctx", type=int, default=4, help="length of ctx")
    parser.add_argument("--num_epoch", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--num_cls_per_client", type=int, default=20, help="classes per client")
    parser.add_argument("--avail_percent", type=float, default=1.0, help="availability percent")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint")
    parser.add_argument("--seed", type=int, default=43, help="random seed")
    parser.add_argument("--w", type=int, default=0, help="weight for KgCoOp")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="CNN backbone")
    parser.add_argument("--model-dir", type=str, default="output/cross_cls/fedtpg/20_8/43/",
                       help="model directory")
    parser.add_argument("--load-epoch", type=int, default=500, help="epoch to load")
    parser.add_argument("--per-class", action="store_true", help="per-class results")

    args = parser.parse_args()

    print("="*80)
    print("FedTPG Detailed Evaluation - 6 Datasets")
    print("="*80)
    print(f"\nDatasets: {['caltech101', 'oxford_flowers', 'fgvc_aircraft', 'oxford_pets', 'food101', 'dtd']}")
    print(f"Model: {args.model_dir}")
    print(f"Epoch: {args.load_epoch}\n")

    # Setup configuration
    cfg = setup_cfg_from_args(args)

    # Setup logger
    log_dir = "evaluation_results_detailed"
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)

    # Initialize server
    print("\nInitializing model...")
    fl_server = Server(cfg)

    # Load pre-trained model
    print(f"\nLoading model from {args.model_dir}, epoch {args.load_epoch}...")
    fl_server.load_model(args.model_dir, epoch=args.load_epoch)
    print("✓ Model loaded successfully\n")

    # Create detailed evaluator
    evaluator = DetailedEvaluator(fl_server, cfg)

    # Evaluate on base and new sets
    print("\n" + "="*80)
    print("PHASE 1: Evaluating BASE set (seen classes)")
    print("="*80)
    base_results = evaluator.evaluate_detailed(split='base')
    evaluator.results['base'] = base_results

    print("\n" + "="*80)
    print("PHASE 2: Evaluating NEW set (unseen classes)")
    print("="*80)
    new_results = evaluator.evaluate_detailed(split='new')
    evaluator.results['new'] = new_results

    # Save all results
    evaluator.save_results()

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*80 + "\n")

    print("BASE Set (Seen Classes):")
    print("-" * 40)
    base_accs = []
    for dataset_name, results in base_results.items():
        print(f"  {dataset_name:25s}: {results['accuracy']:6.2f}%")
        base_accs.append(results['accuracy'])
    print(f"  {'Average':25s}: {np.mean(base_accs):6.2f}%\n")

    print("NEW Set (Unseen Classes):")
    print("-" * 40)
    new_accs = []
    for dataset_name, results in new_results.items():
        print(f"  {dataset_name:25s}: {results['accuracy']:6.2f}%")
        new_accs.append(results['accuracy'])
    print(f"  {'Average':25s}: {np.mean(new_accs):6.2f}%\n")

    print(f"Generalization Gap: {np.mean(new_accs) - np.mean(base_accs):+.2f}%\n")

    print("="*80)
    print("All detailed results saved to: evaluation_results_detailed/")
    print("="*80)
    print("\nNext steps:")
    print("1. Run visualization scripts to create plots")
    print("2. Analyze confusion matrices and error patterns")
    print("3. Visualize prompts and embeddings")

if __name__ == "__main__":
    main()
