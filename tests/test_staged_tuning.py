"""
Staged Hyperparameter Tuning Following Best Practices

Implements three tuning strategies:
1. Stage 1: Quick exploration (spec_rad + leakage_rate)
2. Stage 2: Standard tuning (add density + fraction_input)
3. Stage 3: Full tuning (add ridge_alpha if available)

This follows the recommendations from pyreco_hyperparameter_analysis.md
"""

import argparse
import json
import time
from datetime import datetime
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use new PyReCo datasets interface
from pyreco.datasets import load as pyreco_load
from src.utils.load_dataset import set_seed
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget
from models.pyreco_wrapper import tune_pyreco_hyperparameters, tune_pyreco_with_cv


def stage1_quick_exploration(num_nodes, fraction_output):
    """
    Stage 1: Quick Exploration
    - Fix density=0.1, fraction_input=0.5
    - Explore spec_rad and leakage_rate
    - Total: 3 × 3 = 9 combinations
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.8, 0.9, 1.0],  # 3 values
        "leakage_rate": [0.2, 0.3, 0.5],  # 3 values
        "density": [0.1],  # Fixed
        "fraction_input": [0.5],  # Fixed
        "fraction_output": [fraction_output],
    }
    return grid


def stage2_standard_tuning(num_nodes, fraction_output, best_spec_rad=None, best_leakage=None):
    """
    Stage 2: Standard Tuning
    - Use best values from Stage 1 (if available)
    - Expand to include density and fraction_input
    - Total: ~27-48 combinations
    """
    # If Stage 1 results available, narrow the search around best values
    if best_spec_rad is not None and best_leakage is not None:
        # Narrow search around best values
        spec_rad_values = [
            max(0.5, best_spec_rad - 0.1),
            best_spec_rad,
            min(1.5, best_spec_rad + 0.1)
        ]
        leakage_values = [
            max(0.1, best_leakage - 0.1),
            best_leakage,
            min(0.9, best_leakage + 0.1)
        ]
    else:
        # Broader search
        spec_rad_values = [0.7, 0.8, 0.9, 1.0, 1.1]
        leakage_values = [0.2, 0.3, 0.4, 0.5]

    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": spec_rad_values,
        "leakage_rate": leakage_values,
        "density": [0.05, 0.1, 0.15, 0.2],  # 4 values
        "fraction_input": [0.5, 0.75, 1.0],  # 3 values
        "fraction_output": [fraction_output],
    }
    return grid


def stage3_full_tuning(num_nodes, fraction_output, include_alpha=False):
    """
    Stage 3: Full Tuning
    - Comprehensive search across all parameters
    - Optionally include ridge_alpha
    - Total: ~480 combinations (without alpha) or ~2400 (with alpha)
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # 6 values
        "leakage_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5 values
        "density": [0.05, 0.1, 0.15, 0.2],  # 4 values
        "fraction_input": [0.3, 0.5, 0.75, 1.0],  # 4 values
        "fraction_output": [fraction_output],
    }

    if include_alpha:
        grid["alpha"] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # 5 values

    return grid


def task_specific_grid_lorenz(num_nodes, fraction_output):
    """
    Task-specific grid for Lorenz chaotic system
    Based on empirical best practices
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.8, 0.9, 1.0, 1.1, 1.2],  # Edge of chaos
        "leakage_rate": [0.2, 0.3, 0.4, 0.5],  # Medium speed
        "density": [0.05, 0.1, 0.15],  # Sparse
        "fraction_input": [0.5, 0.75],  # Medium sparsity
        "fraction_output": [fraction_output],
    }
    return grid


def task_specific_grid_mackeyglass(num_nodes, fraction_output):
    """
    Task-specific grid for Mackey-Glass time series
    Requires longer memory
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.9, 1.0, 1.1, 1.2],  # Needs long-term memory
        "leakage_rate": [0.1, 0.2, 0.3],  # Slower dynamics
        "density": [0.1, 0.15, 0.2],  # Slightly denser
        "fraction_input": [0.5, 0.75, 1.0],
        "fraction_output": [fraction_output],
    }
    return grid


def task_specific_grid_santafe(num_nodes, fraction_output):
    """
    Task-specific grid for Santa Fe laser data
    Real-world noisy data, needs good regularization
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.7, 0.8, 0.9, 1.0],  # Moderate, avoid too unstable
        "leakage_rate": [0.3, 0.4, 0.5, 0.6],  # Medium to fast (adaptive to noise)
        "density": [0.05, 0.1],  # Sparse for regularization
        "fraction_input": [0.3, 0.5, 0.75],  # Sparse to prevent overfitting noise
        "fraction_output": [fraction_output],
    }
    return grid


def run_tuning_stage(stage_name, grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
                     use_cv=False, n_splits=5, verbose=True):
    """
    Run one tuning stage

    Args:
        stage_name: Name of the stage
        grid: Parameter grid
        Xtr, Ytr: Training data
        Xva, Yva: Validation data
        Xte, Yte: Test data
        use_cv: Whether to use cross-validation
        n_splits: Number of CV splits
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    if verbose:
        total_combos = np.prod([len(v) for v in grid.values()])
        print(f"\n{'='*100}")
        print(f"{stage_name}")
        print(f"{'='*100}")
        print(f"Grid: {grid}")
        print(f"Total combinations: {total_combos}")
        print(f"Using CV: {use_cv}")
        if use_cv:
            print(f"CV folds: {n_splits}")

    start_time = time.time()

    if use_cv:
        # Use time series cross-validation
        # Combine train and val for CV
        X_full = np.concatenate([Xtr, Xva], axis=0)
        y_full = np.concatenate([Ytr, Yva], axis=0)

        results = tune_pyreco_with_cv(
            X_train=X_full,
            y_train=y_full,
            param_grid=grid,
            n_splits=n_splits,
            default_spec_rad=1.0,
            default_leakage=0.3,
            default_density=0.1,
            default_activation="tanh",
            default_fraction_input=0.5,
            verbose=verbose
        )
    else:
        # Use simple train/val split
        results = tune_pyreco_hyperparameters(
            X_train=Xtr,
            y_train=Ytr,
            X_val=Xva,
            y_val=Yva,
            param_grid=grid,
            default_spec_rad=1.0,
            default_leakage=0.3,
            default_density=0.1,
            default_activation="tanh",
            default_fraction_input=0.5,
            verbose=verbose
        )

    tuning_time = time.time() - start_time

    # Create final model with best parameters for evaluation
    from models.pyreco_wrapper import PyReCoStandardModel
    final_model = PyReCoStandardModel(**results['best_params'], verbose=False)

    # Train on combined train+val data
    X_combined = np.concatenate([Xtr, Xva], axis=0)
    y_combined = np.concatenate([Ytr, Yva], axis=0)
    final_model.fit(X_combined, y_combined)

    # Evaluate on test set
    test_results = final_model.evaluate(Xte, Yte, metrics=['mse', 'mae', 'r2'])

    if verbose:
        print(f"\n✅ {stage_name} Complete!")
        print(f"Tuning time: {tuning_time:.2f}s")
        print(f"Best validation score: {results['best_score']:.6f}")
        print(f"Best parameters:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v}")
        print(f"Test MSE: {test_results['mse']:.6f}")
        print(f"Test MAE: {test_results['mae']:.6f}")
        print(f"Test R²: {test_results['r2']:.6f}")

    return {
        'stage': stage_name,
        'grid': grid,
        'best_params': results['best_params'],
        'best_val_score': results['best_score'],
        'test_mse': test_results['mse'],
        'test_mae': test_results['mae'],
        'test_r2': test_results['r2'],
        'tuning_time': tuning_time,
        'all_results': results.get('all_results', []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lorenz',
                       choices=['lorenz', 'mackeyglass', 'santafe'],
                       help='Dataset name')
    parser.add_argument('--length', type=int, default=5000,
                       help='Dataset length')
    parser.add_argument('--train-frac', type=float, default=0.6,
                       help='Training fraction')
    parser.add_argument('--n-in', type=int, default=100,
                       help='Input window size')
    parser.add_argument('--budget', type=int, default=1000,
                       help='Parameter budget')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Tuning strategy
    parser.add_argument('--strategy', type=str, default='staged',
                       choices=['staged', 'stage1', 'stage2', 'stage3',
                               'task_specific', 'quick'],
                       help='Tuning strategy')
    parser.add_argument('--use-cv', action='store_true',
                       help='Use cross-validation')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits')

    parser.add_argument('--output', type=str, default='results_staged_tuning.json',
                       help='Output file')

    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "="*100)
    print("STAGED HYPERPARAMETER TUNING TEST")
    print("="*100)
    print(f"\nDataset: {args.dataset}")
    print(f"Length: {args.length}")
    print(f"Budget: {args.budget:,} parameters")
    print(f"Strategy: {args.strategy}")
    print(f"Use CV: {args.use_cv}")
    print(f"Seed: {args.seed}")

    # 1) Load and prepare data
    print("\n" + "="*100)
    print("STEP 1: Load and Prepare Data")
    print("="*100)

    # Load data using new PyReCo datasets interface
    print("📊 Loading data with new PyReCo datasets interface...")
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = pyreco_load(
        args.dataset,
        n_samples=args.length,
        train_fraction=args.train_frac,
        val_fraction=0.15,  # 15% validation split
        n_in=args.n_in,
        seed=args.seed
    )
    Dout = x_train.shape[-1]
    print(f"✅ Data loaded successfully!")
    print(f"Train shape: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    print(f"Output dimension: {Dout}")
    print(f"Data is already standardized and windowed")

    # For compatibility, create aliases
    Xtr, Ytr = x_train, y_train
    Xva, Yva = x_val, y_val
    Xte, Yte = x_test, y_test

    print(f"Windows - Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

    # 2) Calculate budget
    print("\n" + "="*100)
    print("STEP 2: Calculate Budget")
    print("="*100)

    Ftarget = compute_readout_F_from_budget(args.budget, Dout)
    candidates = [50, 100, 200, 300, 500, 800, 1000, 1200, 1500]
    num_nodes, fraction_output, F_actual = best_num_nodes_and_fraction_out(Ftarget, candidates)

    print(f"Target Budget: {args.budget}")
    print(f"Target F: {Ftarget}")
    print(f"Chosen num_nodes: {num_nodes}")
    print(f"Chosen fraction_output: {fraction_output:.6f}")
    print(f"Actual F: {F_actual}")

    # 3) Run tuning based on strategy
    all_results = []

    if args.strategy == 'staged':
        # Run all three stages sequentially
        print("\n" + "="*100)
        print("STRATEGY: Staged Tuning (Stage 1 → 2 → 3)")
        print("="*100)

        # Stage 1
        grid1 = stage1_quick_exploration(num_nodes, fraction_output)
        result1 = run_tuning_stage(
            "Stage 1: Quick Exploration",
            grid1, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result1)

        # Stage 2: Use best values from Stage 1
        best_spec_rad = result1['best_params']['spec_rad']
        best_leakage = result1['best_params']['leakage_rate']
        grid2 = stage2_standard_tuning(num_nodes, fraction_output, best_spec_rad, best_leakage)
        result2 = run_tuning_stage(
            "Stage 2: Standard Tuning",
            grid2, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result2)

        # Stage 3: Full tuning
        grid3 = stage3_full_tuning(num_nodes, fraction_output, include_alpha=False)
        result3 = run_tuning_stage(
            "Stage 3: Full Tuning",
            grid3, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result3)

    elif args.strategy == 'stage1':
        grid = stage1_quick_exploration(num_nodes, fraction_output)
        result = run_tuning_stage(
            "Stage 1: Quick Exploration",
            grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result)

    elif args.strategy == 'stage2':
        grid = stage2_standard_tuning(num_nodes, fraction_output)
        result = run_tuning_stage(
            "Stage 2: Standard Tuning",
            grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result)

    elif args.strategy == 'stage3':
        grid = stage3_full_tuning(num_nodes, fraction_output)
        result = run_tuning_stage(
            "Stage 3: Full Tuning",
            grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result)

    elif args.strategy == 'task_specific':
        # Use task-specific grid
        if args.dataset == 'lorenz':
            grid = task_specific_grid_lorenz(num_nodes, fraction_output)
            stage_name = "Task-Specific: Lorenz"
        elif args.dataset == 'mackeyglass':
            grid = task_specific_grid_mackeyglass(num_nodes, fraction_output)
            stage_name = "Task-Specific: Mackey-Glass"
        elif args.dataset == 'santafe':
            grid = task_specific_grid_santafe(num_nodes, fraction_output)
            stage_name = "Task-Specific: Santa Fe"
        else:
            # Fall back to standard
            grid = stage2_standard_tuning(num_nodes, fraction_output)
            stage_name = "Task-Specific: Standard (fallback)"

        result = run_tuning_stage(
            stage_name,
            grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result)

    elif args.strategy == 'quick':
        # Minimal grid for quick testing
        grid = {
            "num_nodes": [num_nodes],
            "spec_rad": [0.9, 1.0],
            "leakage_rate": [0.3, 0.5],
            "density": [0.1],
            "fraction_input": [0.5],
            "fraction_output": [fraction_output],
        }
        result = run_tuning_stage(
            "Quick Test (4 combinations)",
            grid, Xtr, Ytr, Xva, Yva, Xte, Yte,
            use_cv=args.use_cv, n_splits=args.n_splits
        )
        all_results.append(result)

    # 4) Print final summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)

    print(f"\n{'Stage':<40} {'Val Score':<15} {'Test MSE':<15} {'Test R²':<15} {'Time(s)':<10}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['stage']:<40} {r['best_val_score']:<15.6f} "
              f"{r['test_mse']:<15.6f} {r['test_r2']:<15.6f} {r['tuning_time']:<10.2f}")

    # Find best overall
    best_result = min(all_results, key=lambda x: x['test_mse'])
    print(f"\n🏆 Best Stage: {best_result['stage']}")
    print(f"   Test MSE: {best_result['test_mse']:.6f}")
    print(f"   Test R²: {best_result['test_r2']:.6f}")
    print(f"   Best parameters:")
    for k, v in best_result['best_params'].items():
        print(f"     {k}: {v}")

    # 5) Save results
    output_data = {
        'metadata': {
            'dataset': args.dataset,
            'length': args.length,
            'seed': args.seed,
            'n_in': args.n_in,
            'budget': args.budget,
            'num_nodes': num_nodes,
            'fraction_output': fraction_output,
            'strategy': args.strategy,
            'use_cv': args.use_cv,
            'n_splits': args.n_splits,
            'timestamp': datetime.now().isoformat(),
        },
        'results': all_results,
        'best_result': best_result,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {args.output}")
    print("\n" + "="*100)
    print("TUNING COMPLETE!")
    print("="*100)


if __name__ == '__main__':
    main()
