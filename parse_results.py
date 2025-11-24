"""
Parse evaluation results from FedTPG log files
Extracts accuracies and generates comparison tables
"""

import re
import os
import json
import pandas as pd
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file to extract accuracy results"""
    results = {
        'base_accuracies': {},
        'new_accuracies': {},
        'base_avg': None,
        'new_avg': None
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_split = None
    for i, line in enumerate(lines):
        # Detect split type
        if 'Evaluate on the *base* set' in line or 'base set' in line.lower():
            current_split = 'base'
        elif 'Evaluate on the *new* set' in line or 'new set' in line.lower():
            current_split = 'new'

        # Extract dataset-specific accuracies
        # Pattern: "=> result" or accuracy lines
        acc_match = re.search(r'accuracy[:\s]+([0-9.]+)', line, re.IGNORECASE)
        dataset_match = re.search(r'(Caltech101|OxfordFlowers|FGVCAircraft|OxfordPets|Food101|DescribableTextures|UCF101|StanfordCars|SUN397)', line)

        if acc_match and dataset_match:
            dataset = dataset_match.group(1)
            accuracy = float(acc_match.group(1))
            if current_split == 'base':
                results['base_accuracies'][dataset] = accuracy
            elif current_split == 'new':
                results['new_accuracies'][dataset] = accuracy

        # Extract average accuracy
        if 'average' in line.lower() and 'accuracy' in line.lower():
            avg_match = re.search(r'([0-9.]+)%?', line)
            if avg_match:
                avg_val = float(avg_match.group(1))
                if current_split == 'base':
                    results['base_avg'] = avg_val
                elif current_split == 'new':
                    results['new_avg'] = avg_val

    return results

def extract_from_output_logs():
    """Extract results from existing output directory logs"""

    print("="*80)
    print("Extracting Results from Pre-trained Model Logs")
    print("="*80 + "\n")

    # cross_cls results
    cross_cls_log = "output/cross_cls/fedtpg/20_8/43/log.txt"

    if os.path.exists(cross_cls_log):
        print(f"Found cross_cls log: {cross_cls_log}")
        print("Parsing results...\n")

        # Read the entire log file
        with open(cross_cls_log, 'r') as f:
            content = f.read()

        # Find test results section (usually at the end)
        lines = content.split('\n')

        # Look for evaluation results
        datasets_6 = ['Caltech101', 'OxfordFlowers', 'FGVCAircraft', 'OxfordPets', 'Food101', 'DescribableTextures']
        datasets_9 = datasets_6 + ['UCF101', 'StanfordCars', 'SUN397']

        print("Searching for accuracy results...\n")

        # Find all accuracy mentions
        base_results = {}
        new_results = {}

        in_base_section = False
        in_new_section = False

        for line in lines:
            if 'base' in line.lower() and ('evaluate' in line.lower() or 'test' in line.lower()):
                in_base_section = True
                in_new_section = False
                print("Found BASE evaluation section")
            elif 'new' in line.lower() and ('evaluate' in line.lower() or 'test' in line.lower()):
                in_new_section = True
                in_base_section = False
                print("Found NEW evaluation section")

            # Look for accuracy patterns
            for dataset in datasets_9:
                if dataset in line:
                    # Try to find accuracy number
                    acc_patterns = [
                        r'accuracy[:\s]+([0-9.]+)',
                        r'acc[:\s]+([0-9.]+)',
                        r'=>\s*([0-9.]+)',
                        r':\s*([0-9.]+)%'
                    ]
                    for pattern in acc_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            acc = float(match.group(1))
                            if in_base_section:
                                base_results[dataset] = acc
                            elif in_new_section:
                                new_results[dataset] = acc
                            break

        print(f"\nFound {len(base_results)} base results and {len(new_results)} new results")

        return {
            'base_results': base_results,
            'new_results': new_results
        }
    else:
        print(f"Log file not found: {cross_cls_log}")
        return None

def create_comparison_table(results, paper_results=None):
    """Create comparison table with paper results"""

    # Paper results from FedTPG (reported in Table 1)
    # These are approximate values for illustration - update with actual paper values
    paper_baseline = {
        'Caltech101': {'base': 96.32, 'new': 94.03},
        'OxfordFlowers': {'base': 98.53, 'new': 97.42},
        'FGVCAircraft': {'base': 39.83, 'new': 36.63},
        'OxfordPets': {'base': 93.95, 'new': 94.99},
        'Food101': {'base': 85.79, 'new': 87.82},
        'DescribableTextures': {'base': 78.71, 'new': 76.68},
        'UCF101': {'base': 85.93, 'new': 84.83},
        'StanfordCars': {'base': 74.99, 'new': 76.13},
        'SUN397': {'base': 77.65, 'new': 76.86}
    }

    datasets_6 = ['Caltech101', 'OxfordFlowers', 'FGVCAircraft', 'OxfordPets', 'Food101', 'DescribableTextures']

    # Create table data
    table_data = []
    for dataset in datasets_6:
        row = {
            'Dataset': dataset,
            'Paper Base': paper_baseline.get(dataset, {}).get('base', 'N/A'),
            'Paper New': paper_baseline.get(dataset, {}).get('new', 'N/A'),
            'Our Base': results.get('base_results', {}).get(dataset, 'N/A'),
            'Our New': results.get('new_results', {}).get(dataset, 'N/A')
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    return df

def main():
    print("FedTPG Results Parser\n")

    # Extract results from existing logs
    results = extract_from_output_logs()

    if results:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80 + "\n")

        print("BASE Classes (Seen during training):")
        print("-" * 40)
        for dataset, acc in sorted(results['base_results'].items()):
            print(f"{dataset:25s}: {acc:6.2f}%")

        if results['base_results']:
            avg_base = sum(results['base_results'].values()) / len(results['base_results'])
            print(f"{'Average':25s}: {avg_base:6.2f}%")

        print("\n\nNEW Classes (Unseen during training):")
        print("-" * 40)
        for dataset, acc in sorted(results['new_results'].items()):
            print(f"{dataset:25s}: {acc:6.2f}%")

        if results['new_results']:
            avg_new = sum(results['new_results'].values()) / len(results['new_results'])
            print(f"{'Average':25s}: {avg_new:6.2f}%")

        # Create comparison table
        print("\n\n" + "="*80)
        print("COMPARISON WITH PAPER")
        print("="*80 + "\n")

        df = create_comparison_table(results)
        print(df.to_string(index=False))

        # Save results
        os.makedirs("evaluation_results", exist_ok=True)
        df.to_csv("evaluation_results/comparison_table_6datasets.csv", index=False)
        print("\n✓ Comparison table saved to: evaluation_results/comparison_table_6datasets.csv")

        # Save JSON
        with open("evaluation_results/extracted_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("✓ Raw results saved to: evaluation_results/extracted_results.json")

    else:
        print("\nNo results found. Run evaluation first using:")
        print("python evaluate_6_datasets.py")

if __name__ == "__main__":
    main()
