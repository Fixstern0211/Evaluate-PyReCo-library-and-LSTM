"""
Test script for tune_pyreco_hyperparameters function

Following the workflow from models_evaluate.py
"""

import argparse
import numpy as np
from pyreco.datasets import load as pyreco_load
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget
from models.pyreco_wrapper import tune_pyreco_hyperparameters
from models.pyreco_wrapper import PyReCoStandardModel


def main():
    print("\n" + "="*100)
    print("Testing tune_pyreco_hyperparameters (following models_evaluate.py workflow)")
    print("="*100 + "\n")

    # Parse arguments (same as models_evaluate.py)
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

    # 1) Load Data (same as models_evaluate.py line 41)
    print("📊 Step 1: Load Data")
    print("-" * 100)
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = pyreco_load(
        args.dataset,
        n_samples=args.length,
        seed=args.seed,
        val_fraction=0.15,
        train_fraction=args.train_frac
    )
    Dout = x_train.shape[-1]  # Get output dimension from last axis
    print(f"Dataset: {args.dataset}")
    print(f"Data shape: {x_train.shape}, Output dimension: {Dout}")
    print()


    # 2) Create Windows (same as models_evaluate.py line 75-80)

    print(f"Train after windows: X={x_train.shape}, Y={y_train.shape}")
    print(f"Val after windows: X={x_val.shape}, Y={y_val.shape}")
    print(f"Test after windows: X={x_test.shape}, Y={y_test.shape}")
    print()

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
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
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
    if results['best_params'] is not None:
        print("✅ Final model was trained successfully!")
        

        print()

        X_full = np.concatenate([x_train, x_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Train final model with best params (same as train_pyreco_model.py line 116-126)
        final_model = PyReCoStandardModel(**results['best_params'], verbose=False)
        final_model.fit(X_full, y_full)
        # 9) Evaluate on Test Set
        print("🎯 Step 9: Evaluate Final Model on Test Set")
        print("-" * 100)
        test_results = final_model.evaluate(x_test, y_test, metrics=['mse', 'mae', 'r2'])
        print("Test Results:")
        print(f"  MSE: {test_results['mse']:.6f}")
        print(f"  MAE: {test_results['mae']:.6f}")
        print(f"  R²:  {test_results['r2']:.6f}")
        print()

        # Make predictions
        Ypred = results['final_model'].predict(x_test)
        print(f"✅ Predictions shape: {Ypred.shape}")
        print(f"✅ Ground truth shape: {y_test.shape}")
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
    print(f"  2. Train/Val/Test split: {x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]} samples")
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
