"""
Test script for PyReCo Custom model tuning functions

Verifies that:
1. tune_pyreco_custom_hyperparameters() works correctly
2. tune_pyreco_custom_with_cv() works correctly
3. Results are consistent with expected behavior
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.load_dataset import load_data, set_seed
from src.utils import process_datasets
from models.pyreco_custom_wrapper import (
    tune_pyreco_custom_hyperparameters,
    tune_pyreco_custom_with_cv
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lorenz',
                       help='Dataset: lorenz, mackeyglass, santafe')
    parser.add_argument('--length', type=int, default=2000,
                       help='Dataset length (keep small for quick test)')
    parser.add_argument('--n-in', type=int, default=50,
                       help='Input window size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use-cv', action='store_true',
                       help='Test CV function (slower)')
    parser.add_argument('--n-splits', type=int, default=3,
                       help='Number of CV splits (if --use-cv)')
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "="*80)
    print("TEST: PyReCo Custom Model Tuning Functions")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Length: {args.length}")
    print(f"Seed: {args.seed}")
    print(f"Using CV: {args.use_cv}")
    if args.use_cv:
        print(f"CV splits: {args.n_splits}")

    # 1. Load data
    print("\n" + "="*80)
    print("STEP 1: Load and Prepare Data")
    print("="*80)

    data, meta = load_data(args.dataset, length=args.length, seed=args.seed)
    Dout = meta["Dout"]
    print(f"Data shape: {data.shape}, Output dimension: {Dout}")

    # Split
    train, test, split = process_datasets.split_datasets(data, 0.6)
    n_tr = int(0.85 * len(train))
    series_train = train[:n_tr]
    series_val = train[n_tr:]

    print(f"Train: {series_train.shape}, Val: {series_val.shape}, Test: {test.shape}")

    # Standardize
    scaler = StandardScaler().fit(series_train)
    series_train = scaler.transform(series_train)
    series_val = scaler.transform(series_val)
    series_test = scaler.transform(test)

    # Create windows
    Xtr, Ytr = process_datasets.sliding_window(series_train, args.n_in, 1)
    Xva, Yva = process_datasets.sliding_window(series_val, args.n_in, 1)
    Xte, Yte = process_datasets.sliding_window(series_test, args.n_in, 1)

    print(f"Windows - Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

    # 2. Define small parameter grid for quick testing
    print("\n" + "="*80)
    print("STEP 2: Define Parameter Grid")
    print("="*80)

    param_grid = {
        'num_nodes': [100, 200],
        'spec_rad': [0.9, 0.95, 1.0],
        'leakage_rate': [0.2, 0.3, 0.4],
        'density': [0.05, 0.1],
        'fraction_output': [1.0],
    }

    num_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {num_combinations}")

    # 3. Test tuning function
    if not args.use_cv:
        print("\n" + "="*80)
        print("STEP 3: Test tune_pyreco_custom_hyperparameters()")
        print("="*80)

        results = tune_pyreco_custom_hyperparameters(
            Xtr, Ytr, Xva, Yva,
            param_grid=param_grid,
            verbose=True
        )

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Best parameters: {results['best_params']}")
        print(f"Best validation MSE: {results['best_score']:.6f}")

        # Test on test set
        test_results = results['final_model'].evaluate(Xte, Yte, metrics=['mse', 'r2'])
        print(f"Test MSE: {test_results['mse']:.6f}")
        print(f"Test R²: {test_results['r2']:.6f}")

        # Show top 5 results
        print("\nTop 5 configurations:")
        sorted_results = sorted(results['all_results'], key=lambda x: x['val_mse'])[:5]
        for i, r in enumerate(sorted_results, 1):
            print(f"{i}. MSE={r['val_mse']:.6f} | spec_rad={r['params']['spec_rad']:.2f}, "
                  f"leakage={r['params']['leakage_rate']:.2f}, "
                  f"density={r['params']['density']:.2f}, "
                  f"nodes={r['params']['num_nodes']}")

    else:
        print("\n" + "="*80)
        print("STEP 3: Test tune_pyreco_custom_with_cv()")
        print("="*80)

        # Combine train + val for CV
        X_full = np.concatenate([Xtr, Xva], axis=0)
        y_full = np.concatenate([Ytr, Yva], axis=0)
        print(f"Full training set: {X_full.shape}")

        results = tune_pyreco_custom_with_cv(
            X_full, y_full,
            param_grid=param_grid,
            n_splits=args.n_splits,
            verbose=True
        )

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Best parameters: {results['best_params']}")
        print(f"Best mean CV MSE: {results['best_score']:.6f}")

        # Test on test set
        test_results = results['final_model'].evaluate(Xte, Yte, metrics=['mse', 'r2'])
        print(f"Test MSE: {test_results['mse']:.6f}")
        print(f"Test R²: {test_results['r2']:.6f}")

        # Show top 5 results
        print("\nTop 5 configurations:")
        sorted_results = sorted(results['all_results'], key=lambda x: x['mean_mse'])[:5]
        for i, r in enumerate(sorted_results, 1):
            print(f"{i}. MSE={r['mean_mse']:.6f}±{r['std_mse']:.6f} | "
                  f"spec_rad={r['params']['spec_rad']:.2f}, "
                  f"leakage={r['params']['leakage_rate']:.2f}, "
                  f"density={r['params']['density']:.2f}, "
                  f"nodes={r['params']['num_nodes']}")

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
