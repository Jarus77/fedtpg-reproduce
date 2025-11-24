"""
Evaluation script for FedTPG on 6 datasets using pre-trained weights
Generates results for reproduction report
"""

import argparse
import torch
import json
import os
from config.defaults import _C as cfg_default
from config.utils import reset_cfg
from utils import setup_logger
from federated.server import Server
import sys

def setup_cfg(args):
    cfg = cfg_default.clone()
    reset_cfg(cfg, args)
    cfg.freeze()
    return cfg

def evaluate_cross_cls():
    """Evaluate cross_cls experiment on 6 datasets"""

    print("="*80)
    print("FedTPG Evaluation on 6 Datasets (cross_cls experiment)")
    print("="*80)
    print("\nDatasets: caltech101, oxford_flowers, fgvc_aircraft, oxford_pets, food101, dtd")
    print("\nLoading pre-trained model from: output/cross_cls/fedtpg/20_8/43/")
    print("="*80 + "\n")

    # Setup arguments
    class Args:
        root = "./data"
        exp_name = "cross_cls"
        model_name = "fedtpg"
        num_shots = 8
        depth_ctx = 1
        model_depth = 0
        n_ctx = 4
        num_epoch = 500
        batch_size = 200
        num_cls_per_client = 20
        avail_percent = 1.0
        output_dir = "./output"
        resume = ""
        seed = 43
        w = 0
        backbone = "ViT-B/16"
        eval_only = True
        model_dir = "output/cross_cls/fedtpg/20_8/43/"
        load_epoch = 500
        no_train = True
        per_class = False

    args = Args()
    cfg = setup_cfg(args)

    # Setup logger
    log_dir = os.path.join("./evaluation_results", "cross_cls_6datasets")
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}\n")

    # Initialize server and load model
    fl_server = Server(cfg)

    print(f"\nLoading model from epoch {args.load_epoch}...")
    fl_server.load_model(args.model_dir, epoch=args.load_epoch)
    print("✓ Model loaded successfully\n")

    # Run evaluation on base classes
    print("="*80)
    print("Evaluating on BASE classes (seen during training)")
    print("="*80 + "\n")
    base_results = fl_server.test("base")

    # Run evaluation on new classes
    print("\n" + "="*80)
    print("Evaluating on NEW classes (unseen during training)")
    print("="*80 + "\n")
    new_results = fl_server.test("new")

    # Save results
    results = {
        "experiment": "cross_cls",
        "datasets": ["caltech101", "oxford_flowers", "fgvc_aircraft", "oxford_pets", "food101", "dtd"],
        "model": "FedTPG",
        "epoch": 500,
        "base_results": base_results if base_results else "See log file",
        "new_results": new_results if new_results else "See log file"
    }

    results_file = os.path.join(log_dir, "results_summary.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: {log_dir}")
    print("Check the log files for detailed per-dataset and per-class accuracies")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate FedTPG on 6 datasets")
    parser.add_argument("--data-root", type=str, default="./data", help="Path to data directory")
    args = parser.parse_args()

    # Update data root if provided
    if args.data_root:
        print(f"Using data root: {args.data_root}\n")

    results = evaluate_cross_cls()

    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print("1. Check evaluation_results/cross_cls_6datasets/ for detailed results")
    print("2. Parse log files to extract accuracy tables")
    print("3. Create visualizations comparing with paper results")
    print("4. Generate reproduction report with findings")

if __name__ == "__main__":
    main()
