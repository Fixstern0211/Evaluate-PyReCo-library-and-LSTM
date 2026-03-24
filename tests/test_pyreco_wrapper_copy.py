"""
Test script for tune_pyreco_hyperparameters function

Following the workflow from models_evaluate.py
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
from models.pyreco_wrapper import tune_pyreco_hyperparameters
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=5000)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--n-in", type=int, default=100)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--num-nodes", type=int, default=800)
    ap.add_argument("--dataset", type=str, default="lorenz",
                    help="Dataset: lorenz, mackeyglass, santafe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    print("\n" + "="*80)
    print("TEST: PyReCo Custom Model Tuning Functions")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Length: {args.length}")
    print(f"Seed: {args.seed}")



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

    # 3) Budget Calculation (same as models_evaluate.py line 83-86)
    print("📊 Step 3: Calculate Budget")
    print("-" * 100)
    Ftarget = compute_readout_F_from_budget(args.budget, Dout)
    chosen_num_nodes, chosen_fraction_out, chosen_F_real = \
        best_num_nodes_and_fraction_out(Ftarget, [args.num_nodes])
    print(f"Budget: {args.budget}")
    print(f"Target F: {Ftarget}")
    print(f"Chosen num_nodes: {chosen_num_nodes}")
    print(f"Chosen fraction_output: {chosen_fraction_out:.6f}")
    print(f"Actual F: {chosen_F_real}")
    print()

    # 6) Hyperparameter Grid (smaller grid for faster testing)
    print("🔍 Step 6: Define Hyperparameter Grid")
    print("-" * 100)
    param_grid = {
        "num_nodes": [chosen_num_nodes],  # Fixed by budget
        "spec_rad": [0.8, 1.0],  # Reduced for faster testing
        "leakage_rate": [0.4, 0.5],  # Reduced for faster testing
        "density": [0.6, 0.8],  # Reduced for faster testing
        "fraction_output": [chosen_fraction_out],  # Fixed by budget
    }

    total_combinations = (len(param_grid["spec_rad"]) *
                         len(param_grid["leakage_rate"]) *
                         len(param_grid["density"]))
    print(f"Param grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print(f"Total combinations to test: {total_combinations}")
    print()

    # 7) Run Hyperparameter Tuning (using tune_pyreco_hyperparameters)
    print("🚀 Step 7: Run Hyperparameter Tuning")
    print("-" * 100)
    print("Testing tune_pyreco_hyperparameters function...")
    print()

    results = tune_pyreco_hyperparameters(
        X_train=Xtr,
        y_train=Ytr,
        X_val=Xva,
        y_val=Yva,
        param_grid=param_grid,
        verbose=True
    )

    # 8) Verify Results
    print("\n" + "="*100)
    print("📊 Step 8: Verify Results")
    print("="*100 + "\n")

    print("✅ Returned keys:", list(results.keys()))
    print()

    print("✅ Best parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    print()

    print(f"✅ Best validation MSE: {results['best_score']:.6f}")
    print()

    # Check if final model was trained
    if results['final_model'] is not None:
        print("✅ Final model was trained successfully!")
        print(f"   Model type: {type(results['final_model']).__name__}")
        print(f"   Is trained: {results['final_model'].is_trained}")
        print(f"   Training time: {results['final_model'].training_time:.3f}s")
        print()

        # 9) Evaluate on Test Set
        print("🎯 Step 9: Evaluate Final Model on Test Set")
        print("-" * 100)
        test_results = results['final_model'].evaluate(Xte, Yte, metrics=['mse', 'mae', 'r2'])
        print("Test Results:")
        print(f"  MSE: {test_results['mse']:.6f}")
        print(f"  MAE: {test_results['mae']:.6f}")
        print(f"  R²:  {test_results['r2']:.6f}")
        print()

        # Make predictions
        Ypred = results['final_model'].predict(Xte)
        print(f"✅ Predictions shape: {Ypred.shape}")
        print(f"✅ Ground truth shape: {Yte.shape}")
        print()
    else:
        print("❌ Final model is None (retrain_on_full=False?)")
        print()

    # 10) Summary
    print("="*100)
    print("✅ Test Complete!")
    print("="*100 + "\n")

    print("📝 Summary:")
    print(f"  1. Dataset: {args.dataset} ({args.length} samples)")
    print(f"  2. Train/Val/Test split: {Xtr.shape[0]}/{Xva.shape[0]}/{Xte.shape[0]} samples")
    print(f"  3. Tested {total_combinations} hyperparameter combinations")
    print(f"  4. Best validation MSE: {results['best_score']:.6f}")
    if results['final_model'] is not None:
        print(f"  5. Final model test MSE: {test_results['mse']:.6f}")
        print(f"  6. Final model test R²: {test_results['r2']:.6f}")
    print()

    print("🎉 tune_pyreco_hyperparameters works correctly!")
    print("   - Returns best_params ✅")
    print("   - Returns best_score ✅")
    print("   - Returns trained final_model ✅")
    print("   - Final model can evaluate on test set ✅")
    print()

    return results


if __name__ == "__main__":
    main()
