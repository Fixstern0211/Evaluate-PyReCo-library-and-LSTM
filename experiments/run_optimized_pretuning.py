"""
Run Optimized Pre-Tuning with Cross-Validation

This script runs 5-fold CV pre-tuning for each dataset using optimized grids
that avoid problematic hyperparameter combinations (spec_rad >= 1.0).

Supports multiple budget levels to find optimal parameters for each scale.

Usage:
    # Single budget (legacy mode)
    python run_optimized_pretuning.py --dataset lorenz --budget medium

    # All budgets for one dataset
    python run_optimized_pretuning.py --dataset lorenz --all-budgets

    # All datasets and all budgets
    python run_optimized_pretuning.py --all-datasets --all-budgets
"""

import argparse
import json
import time
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.load_dataset import set_seed
from models.pyreco_wrapper import tune_pyreco_with_cv
from optimized_grids import get_optimized_grid

# Try to import pyreco datasets, fall back to local
try:
    from pyreco.datasets import load as pyreco_load
    USE_PYRECO_DATASETS = True
except ImportError:
    USE_PYRECO_DATASETS = False

from src.utils.load_dataset import load as local_load

# Datasets supported by pyreco.datasets
PYRECO_SUPPORTED_DATASETS = {'lorentz69', 'lorenz', 'lorenz63', 'mackey_glass', 'mackeyglass', 'mg'}

# Budget definitions (same as test_model_scaling.py)
BUDGETS = {
    'small': 1000,
    'medium': 10000,
    'large': 50000,
}

NODE_CANDIDATES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def get_num_nodes_for_budget(budget, density=0.1):
    """Calculate num_nodes from budget using sqrt(budget/density)."""
    num_nodes_approx = int(math.sqrt(budget / density))
    return min(NODE_CANDIDATES, key=lambda x: abs(x - num_nodes_approx))


def convert_numpy(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


def load_dataset(dataset, length, seed, train_frac, n_in):
    """Load and prepare data for CV tuning."""
    # Use pyreco for supported datasets, otherwise fall back to local loader
    use_pyreco = USE_PYRECO_DATASETS and dataset in PYRECO_SUPPORTED_DATASETS

    if use_pyreco:
        Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = pyreco_load(
            dataset,
            n_samples=length,
            seed=seed,
            val_fraction=0.15,
            train_fraction=train_frac,
            n_in=n_in,
            n_out=1,
            standardize=True
        )
    else:
        print(f"  [Note: Using local loader for '{dataset}']")
        Xtr, Ytr, Xva, Yva, Xte, Yte, scaler = local_load(
            dataset,
            n_samples=length,
            seed=seed,
            val_fraction=0.15,
            train_fraction=train_frac,
            n_in=n_in,
            n_out=1,
            standardize=True
        )

    X_full = np.concatenate([Xtr, Xva], axis=0)
    y_full = np.concatenate([Ytr, Yva], axis=0)
    Dout = Xtr.shape[-1]

    return X_full, y_full, Dout


def run_pretuning_for_budget(dataset, budget_name, budget_value, X_full, y_full,
                              Dout, n_splits=5, verbose=True):
    """Run CV pretuning for a specific budget."""
    num_nodes = get_num_nodes_for_budget(budget_value)
    fraction_output = 1.0

    grid = get_optimized_grid(dataset, num_nodes, fraction_output)
    n_combos = int(np.prod([len(v) for v in grid.values()]))

    if verbose:
        print(f"\n{'='*80}")
        print(f"PRETUNING: {dataset.upper()} | {budget_name.upper()} ({budget_value:,} params)")
        print(f"{'='*80}")
        print(f"  num_nodes: {num_nodes}")
        print(f"  Grid: {n_combos} combinations × {n_splits} folds = {n_combos * n_splits} trainings")

    start_time = time.time()

    results = tune_pyreco_with_cv(
        X_train=X_full,
        y_train=y_full,
        param_grid=grid,
        n_splits=n_splits,
        verbose=verbose
    )

    tune_time = time.time() - start_time

    if verbose:
        print(f"\n  ✓ Best params: {results['best_params']}")
        print(f"  ✓ Best CV MSE: {results['best_score']:.6f}, R²: {results.get('best_r2', 0):.4f}")
        print(f"  ✓ Time: {tune_time:.1f}s ({tune_time/60:.1f} min)")

    return {
        'budget_name': budget_name,
        'budget_value': budget_value,
        'num_nodes': num_nodes,
        'grid': grid,
        'n_combinations': n_combos,
        'best_params': results['best_params'],
        'best_cv_mse': results['best_score'],
        'best_cv_r2': results.get('best_r2'),
        'cv_std': results.get('cv_std'),
        'tune_time_seconds': tune_time,
        'all_results': results.get('all_results', []),  # 包含每个参数组合的 MSE 和 R²
    }


def analyze_and_print_results(all_results, dataset):
    """Analyze results across budgets and generate recommendations."""
    print(f"\n{'='*100}")
    print(f"ANALYSIS: {dataset.upper()} - Optimal Parameters Across Budgets")
    print(f"{'='*100}")

    params_by_budget = {r['budget_name']: r['best_params'] for r in all_results}

    print(f"\n{'Parameter':<18} {'Small (1K)':<14} {'Medium (10K)':<14} {'Large (50K)':<14} {'Consistent?':<12}")
    print("-" * 72)

    param_names = ['spec_rad', 'leakage_rate', 'density', 'fraction_input']
    inconsistent_params = []

    for param in param_names:
        values = []
        for budget in ['small', 'medium', 'large']:
            if budget in params_by_budget:
                values.append(params_by_budget[budget].get(param, 'N/A'))
            else:
                values.append('N/A')

        unique_values = set(v for v in values if v != 'N/A')
        is_consistent = len(unique_values) <= 1
        consistency = "✓ Yes" if is_consistent else "✗ No"

        if not is_consistent:
            inconsistent_params.append((param, values, unique_values))

        val_strs = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in values]
        print(f"{param:<18} {val_strs[0]:<14} {val_strs[1]:<14} {val_strs[2]:<14} {consistency:<12}")

    # Print recommendations
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS FOR test_model_scaling.py")
    print(f"{'='*100}")

    if not inconsistent_params:
        print("\n✅ All parameters are CONSISTENT across budgets.")
        print("   The current param_grid in test_model_scaling.py is appropriate.")
    else:
        print("\n⚠️  Some parameters differ across budgets:")
        for param, values, unique in inconsistent_params:
            print(f"   {param}: small={values[0]}, medium={values[1]}, large={values[2]}")

        print("\n   Consider using budget-specific param_grids centered on these values.")

    return inconsistent_params


def main():
    parser = argparse.ArgumentParser(
        description='Run Pre-Tuning with CV for Multiple Budgets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str,
                       choices=['lorenz', 'mackeyglass', 'santafe'],
                       help='Dataset name')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Run on all datasets')
    parser.add_argument('--budget', type=str,
                       choices=['small', 'medium', 'large'],
                       help='Single budget level')
    parser.add_argument('--all-budgets', action='store_true',
                       help='Run on all budgets (small, medium, large)')
    parser.add_argument('--length', type=int, default=5000,
                       help='Dataset length')
    parser.add_argument('--train-frac', type=float, default=0.6,
                       help='Training fraction')
    parser.add_argument('--n-in', type=int, default=100,
                       help='Input window size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits')
    parser.add_argument('--output-dir', type=str, default='results/pretuning',
                       help='Output directory')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_datasets and not args.dataset:
        parser.error("Must specify --dataset or --all-datasets")

    if not args.all_budgets and not args.budget:
        parser.error("Must specify --budget or --all-budgets")

    # Determine datasets and budgets to run
    datasets = ['lorenz', 'mackeyglass', 'santafe'] if args.all_datasets else [args.dataset]
    budget_list = list(BUDGETS.keys()) if args.all_budgets else [args.budget]

    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("OPTIMIZED PRE-TUNING WITH CROSS-VALIDATION")
    print("="*100)
    print(f"\nDatasets: {datasets}")
    print(f"Budgets: {budget_list} → {[BUDGETS[b] for b in budget_list]}")
    print(f"CV folds: {args.n_splits}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")

    # Run pretuning for each dataset
    for dataset in datasets:
        print(f"\n\n{'#'*100}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*100}")

        # Load data once for this dataset
        print(f"\nLoading data...")
        X_full, y_full, Dout = load_dataset(
            dataset, args.length, args.seed, args.train_frac, args.n_in
        )
        print(f"Data shape: {X_full.shape}, Output dim: {Dout}")

        # Run pretuning for each budget
        dataset_results = []

        for budget_name in budget_list:
            budget_value = BUDGETS[budget_name]

            result = run_pretuning_for_budget(
                dataset=dataset,
                budget_name=budget_name,
                budget_value=budget_value,
                X_full=X_full,
                y_full=y_full,
                Dout=Dout,
                n_splits=args.n_splits,
                verbose=True
            )
            dataset_results.append(result)

        # Analyze results if multiple budgets
        if len(budget_list) > 1:
            analyze_and_print_results(dataset_results, dataset)

        # Save results
        output_data = {
            'dataset': dataset,
            'experiment_type': 'pretuning_cv',
            'seed': args.seed,
            'train_frac': args.train_frac,
            'n_in': args.n_in,
            'length': args.length,
            'n_splits': args.n_splits,
            'timestamp': datetime.now().isoformat(),
            'budgets': convert_numpy(dataset_results),
        }

        if len(budget_list) > 1:
            output_file = output_dir / f"pretuning_{dataset}_all_budgets.json"
        else:
            output_file = output_dir / f"pretuning_{dataset}_{budget_list[0]}.json"

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")

    # Final summary for multi-budget runs
    if len(budget_list) > 1:
        print(f"\n\n{'='*100}")
        print("PRETUNING COMPLETE - Run test_model_scaling.py to validate")
        print(f"{'='*100}")


if __name__ == '__main__':
    main()
