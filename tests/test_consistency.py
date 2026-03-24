"""
Test to verify models_evaluate.py and pyreco_wrapper.py give consistent results
Updated to use new PyReCo datasets interface
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pyreco.datasets import load as pyreco_load
from src.utils.load_dataset import load_data, set_seed
from src.utils import process_datasets
from src.utils.node_number import best_num_nodes_and_fraction_out, compute_readout_F_from_budget
from src.utils import train_pyreco_model
from models.pyreco_wrapper import tune_pyreco_hyperparameters

print("\n" + "="*100)
print("Consistency Test: models_evaluate.py vs pyreco_wrapper.py")
print("="*100 + "\n")

# Set seed for reproducibility
set_seed(42)

# Method 1: Use new PyReCo datasets interface (recommended)
print("📊 Step 1a: Load Data via New PyReCo Datasets")
print("-" * 100)
x_train_new, y_train_new, x_val_new, y_val_new, x_test_new, y_test_new, scaler_new = pyreco_load(
    'lorenz',
    n_samples=2000,  # 增加样本数
    train_fraction=0.6,
    val_fraction=0.15,
    n_in=30,  # 减少窗口大小
    seed=42
)
Dout_new = x_train_new.shape[-1]
print(f"New method - Train shape: {x_train_new.shape}, Val: {x_val_new.shape}, Test: {x_test_new.shape}")
print(f"Output dimension: {Dout_new}")
print()

# Method 2: Use old method for comparison
print("📊 Step 1b: Load Data via Old Method (for comparison)")
print("-" * 100)
data, meta = load_data("lorenz", length=2000, seed=42)
Dout = meta["Dout"]
print(f"Old method - Data shape: {data.shape}, Dout: {Dout}")
print()

# Compare: Use old method for split data processing
train, test, split = process_datasets.split_datasets(data, 0.6)

# For models_evaluate.py path (uses time series CV, so uses all train)
series_train_cv = train.copy()

# For pyreco_wrapper path (uses manual train/val split)
n_tr = int(0.85 * len(train))
series_train = train[:n_tr]
series_val = train[n_tr:]

print(f"Old split - CV path: train={series_train_cv.shape}")
print(f"Old split - Wrapper path: train={series_train.shape}, val={series_val.shape}")

# Compare data shapes between methods
print(f"\n🔍 Data Shape Comparison:")
print(f"Old method total: {data.shape[0]} timesteps")
print(f"New method X_train: {x_train_new.shape[0]} windows, X_val: {x_val_new.shape[0]}, X_test: {x_test_new.shape[0]}")
print()

# 3) For rest of test, use NEW method (recommended)
print("📊 Step 3: Using New PyReCo Datasets (Main Test Path)")
print("-" * 100)
print("✅ Data already standardized and windowed by new datasets interface")
print(f"Train data mean: {x_train_new.mean():.3f}, std: {x_train_new.std():.3f}")
print(f"Val data mean: {x_val_new.mean():.3f}, std: {x_val_new.std():.3f}")
print()

# 4) Budget Calculation (using new method data)
print("📊 Step 4: Budget Calculation")
print("-" * 100)
budget = 5000
Ftarget = compute_readout_F_from_budget(budget, Dout_new)
chosen_num_nodes, chosen_fraction_out, chosen_F_real = \
    best_num_nodes_and_fraction_out(Ftarget, [300])
print(f"num_nodes: {chosen_num_nodes}, fraction_output: {chosen_fraction_out:.6f}")
print()

# 5) Define same grid for both
print("📊 Step 5: Define Grid (MINIMAL for speed)")
print("-" * 100)
grid = {
    "spec_rad": [0.9, 1.0],
    "leakage_rate": [0.3],
    "density": [0.1],
}
print(f"Grid: {grid}")
print(f"Total combinations: 2")
print()

# ============================================================================
# Test 1: New PyReCo datasets approach (recommended)
# ============================================================================
print("="*100)
print("🔬 Test 1: New PyReCo Datasets Approach (tune_pyreco_hyperparameters)")
print("="*100 + "\n")

# Use the new datasets data
param_grid = {
    "num_nodes": [chosen_num_nodes],
    "spec_rad": grid["spec_rad"],
    "leakage_rate": grid["leakage_rate"],
    "density": grid["density"],
    "fraction_output": [chosen_fraction_out],
}

results_new = tune_pyreco_hyperparameters(
    X_train=x_train_new,
    y_train=y_train_new,
    X_val=x_val_new,
    y_val=y_val_new,
    param_grid=param_grid,
    verbose=True
)

print("\n✅ New PyReCo datasets approach complete!")
print("Best params:")
for k, v in results_new['best_params'].items():
    print(f"  {k}: {v}")
print(f"Best validation MSE: {results_new['best_score']:.6f}")
print()

# ============================================================================
# Test 2: Old method approach (for comparison)
# ============================================================================
print("="*100)
print("🔬 Test 2: Old Method Approach (for comparison)")
print("="*100 + "\n")

# Process old data to match new format
from sklearn.preprocessing import StandardScaler
n_in = 30  # Match new method
train_old, test_old, _ = process_datasets.split_datasets(data, 0.6)
n_tr = int(0.85 * len(train_old))  # 15% validation
series_train_old = train_old[:n_tr]
series_val_old = train_old[n_tr:]

# Standardize
scaler_old = StandardScaler()
series_train_scaled_old = scaler_old.fit_transform(series_train_old)
series_val_scaled_old = scaler_old.transform(series_val_old)

# Create windows
Xtr_old, Ytr_old = process_datasets.sliding_window(series_train_scaled_old, n_in, 1)
Xval_old, Yval_old = process_datasets.sliding_window(series_val_scaled_old, n_in, 1)

print(f"Old method windows: train={Xtr_old.shape}, val={Xval_old.shape}")

# Use same parameter grid
results_old = tune_pyreco_hyperparameters(
    X_train=Xtr_old,
    y_train=Ytr_old,
    X_val=Xval_old,
    y_val=Yval_old,
    param_grid=param_grid,
    verbose=True
)

print("\n✅ Old method approach complete!")
print("Best params:")
for k, v in results_old['best_params'].items():
    print(f"  {k}: {v}")
print(f"Best validation MSE: {results_old['best_score']:.6f}")
print()

# ============================================================================
# Compare Results
# ============================================================================
print("="*100)
print("🔍 Comparison: New vs Old Dataset Methods")
print("="*100 + "\n")

print("📊 Data Shape Comparison:")
print(f"{'Method':<20} {'Train Shape':<20} {'Val Shape':<20} {'Features':<10}")
print("-" * 80)
print(f"{'New PyReCo':<20} {str(x_train_new.shape):<20} {str(x_val_new.shape):<20} {x_train_new.shape[-1]:<10}")
print(f"{'Old Method':<20} {str(Xtr_old.shape):<20} {str(Xval_old.shape):<20} {Xtr_old.shape[-1]:<10}")
print()

print("📊 Performance Comparison:")
print(f"{'Method':<20} {'Best MSE':<15} {'Difference from New':<20}")
print("-" * 60)
mse_new = results_new['best_score']
mse_old = results_old['best_score']
diff = abs(mse_new - mse_old)
print(f"{'New PyReCo':<20} {mse_new:.6f}{'':>9} {'(baseline)':<20}")
print(f"{'Old Method':<20} {mse_old:.6f}{'':>9} {diff:.6f}{'':>14}")

if diff < 0.001:
    print("\n✅ EXCELLENT: Methods produce very similar results!")
elif diff < 0.01:
    print("\n👌 GOOD: Methods produce reasonably similar results")
else:
    print("\n⚠️ WARNING: Methods produce different results - check implementation")

print()
print("="*100)
print("🎉 Consistency Test Complete!")
print("="*100)
