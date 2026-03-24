"""
Merge Original and Supplementary Pretuning Results

Combines results from original pretuning and supplementary boundary experiments.
Finds the overall best parameters across both experiment sets.

Usage:
    python merge_pretuning_results.py --dataset lorenz
    python merge_pretuning_results.py --all-datasets
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_results(filepath):
    """Load results from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def merge_budget_results(original_budget, supplementary_budget):
    """Merge results for a single budget level."""
    # Combine all_results from both
    all_results = []

    if original_budget and 'all_results' in original_budget:
        all_results.extend(original_budget['all_results'])

    if supplementary_budget and 'all_results' in supplementary_budget:
        all_results.extend(supplementary_budget['all_results'])

    if not all_results:
        return original_budget or supplementary_budget

    # Find best result across combined results
    best_result = min(all_results, key=lambda x: x['cv_mean'])

    # Create merged budget result
    base = original_budget or supplementary_budget
    merged = {
        'budget_name': base['budget_name'],
        'budget_value': base['budget_value'],
        'num_nodes': base['num_nodes'],
        'n_combinations': len(all_results),
        'best_params': best_result['params'],
        'best_cv_mse': best_result['cv_mean'],
        'best_cv_r2': best_result['cv_r2_mean'],
        'cv_std': best_result['cv_std'],
        'all_results': all_results,
        'sources': {
            'original': original_budget['n_combinations'] if original_budget else 0,
            'supplementary': supplementary_budget['n_combinations'] if supplementary_budget else 0,
        }
    }

    return merged


def merge_dataset_results(dataset, results_dir):
    """Merge original and supplementary results for a dataset."""
    results_dir = Path(results_dir)

    # Load original results
    original_file = results_dir / f"pretuning_{dataset}_all_budgets.json"
    original = load_results(original_file)

    # Load supplementary results
    supp_file = results_dir / f"pretuning_{dataset}_supplementary.json"
    supplementary = load_results(supp_file)

    if not original and not supplementary:
        print(f"  ⚠️  No results found for {dataset}")
        return None

    # Create budget lookup
    orig_budgets = {}
    if original:
        for b in original.get('budgets', []):
            orig_budgets[b['budget_name']] = b

    supp_budgets = {}
    if supplementary:
        for b in supplementary.get('budgets', []):
            supp_budgets[b['budget_name']] = b

    # Merge each budget
    merged_budgets = []
    for budget_name in ['small', 'medium', 'large']:
        orig_b = orig_budgets.get(budget_name)
        supp_b = supp_budgets.get(budget_name)

        if orig_b or supp_b:
            merged = merge_budget_results(orig_b, supp_b)
            merged_budgets.append(merged)

    # Create merged result
    merged_data = {
        'dataset': dataset,
        'experiment_type': 'merged',
        'seed': (original or supplementary).get('seed', 42),
        'n_splits': (original or supplementary).get('n_splits', 5),
        'timestamp': datetime.now().isoformat(),
        'original_file': str(original_file) if original else None,
        'supplementary_file': str(supp_file) if supplementary else None,
        'budgets': merged_budgets,
    }

    return merged_data


def analyze_merged_results(merged_data):
    """Analyze and print merged results."""
    dataset = merged_data['dataset']

    print(f"\n{'='*80}")
    print(f"{dataset.upper()} - MERGED RESULTS")
    print(f"{'='*80}")

    # Check parameter consistency
    params_by_budget = {}
    for budget in merged_data['budgets']:
        budget_name = budget['budget_name']
        params_by_budget[budget_name] = budget['best_params']

        sources = budget.get('sources', {})
        print(f"\n  {budget_name.upper()}:")
        print(f"    Total combinations: {budget['n_combinations']} (orig: {sources.get('original', 'N/A')}, supp: {sources.get('supplementary', 'N/A')})")
        print(f"    Best MSE: {budget['best_cv_mse']:.6f}, R²: {budget['best_cv_r2']:.4f}")
        print(f"    Best params: spec_rad={budget['best_params']['spec_rad']}, "
              f"leakage={budget['best_params']['leakage_rate']}, "
              f"density={budget['best_params']['density']}, "
              f"frac_input={budget['best_params']['fraction_input']}")

    # Check consistency
    print(f"\n  {'─'*60}")
    print(f"  PARAMETER CONSISTENCY:")

    param_names = ['spec_rad', 'leakage_rate', 'density', 'fraction_input']
    all_consistent = True

    for param in param_names:
        values = [params_by_budget[b].get(param) for b in ['small', 'medium', 'large'] if b in params_by_budget]
        unique = set(values)
        consistent = len(unique) <= 1
        status = "✓" if consistent else "✗"
        if not consistent:
            all_consistent = False
        print(f"    {param}: {values} {status}")

    if all_consistent:
        print(f"\n  ✅ All parameters CONSISTENT across budgets")
    else:
        print(f"\n  ⚠️  Some parameters DIFFER across budgets")

    return all_consistent


def main():
    parser = argparse.ArgumentParser(description='Merge Pretuning Results')
    parser.add_argument('--dataset', type=str, choices=['lorenz', 'mackeyglass', 'santafe'])
    parser.add_argument('--all-datasets', action='store_true')
    parser.add_argument('--results-dir', type=str, default='results/pretuning')

    args = parser.parse_args()

    if not args.all_datasets and not args.dataset:
        parser.error("Must specify --dataset or --all-datasets")

    datasets = ['lorenz', 'mackeyglass', 'santafe'] if args.all_datasets else [args.dataset]
    results_dir = Path(args.results_dir)

    print("\n" + "="*80)
    print("MERGING PRETUNING RESULTS")
    print("="*80)

    for dataset in datasets:
        merged = merge_dataset_results(dataset, results_dir)

        if merged:
            # Analyze
            analyze_merged_results(merged)

            # Save merged results
            output_file = results_dir / f"pretuning_{dataset}_merged.json"
            with open(output_file, 'w') as f:
                json.dump(merged, f, indent=2)

            print(f"\n  💾 Merged results saved to: {output_file}")

    print(f"\n\n{'='*80}")
    print("MERGE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
