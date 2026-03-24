"""
Quick test to verify default values are correctly applied in tune_pyreco_hyperparameters
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.load_dataset import load_data
from src.utils import process_datasets
from models.pyreco_wrapper import tune_pyreco_hyperparameters

print("\n" + "="*100)
print("Quick Test: Verify Default Values Fix")
print("="*100 + "\n")

# 1) Load small dataset
print("📊 Loading small dataset...")
data, meta = load_data("lorenz", length=800, seed=0)
Dout = meta["Dout"]

# 2) Split
train, test, split = process_datasets.split_datasets(data, 0.6)
n_tr = int(0.85 * len(train))
series_train = train[:n_tr]
series_val = train[n_tr:]

# 3) Standardize
scX = StandardScaler().fit(series_train)
series_train = scX.transform(series_train)
series_val = scX.transform(series_val)

# 4) Windows (small n_in for speed)
Xtr, Ytr = process_datasets.sliding_window(series_train, 30, 1)
Xva, Yva = process_datasets.sliding_window(series_val, 30, 1)

print(f"Train windows: {Xtr.shape}, Val windows: {Xva.shape}")
print()

# 5) Test with MINIMAL grid (only 2 combinations)
print("🔍 Testing with minimal param_grid...")
print("   - Only testing 2 combinations")
print("   - Checking if default values are used correctly")
print()

param_grid = {
    "num_nodes": [50],
    "spec_rad": [0.8, 1.0],  # Only 2 values
    "fraction_output": [0.5],
}

print("🚀 Running tune_pyreco_hyperparameters...")
print("-" * 100)

results = tune_pyreco_hyperparameters(
    X_train=Xtr,
    y_train=Ytr,
    X_val=Xva,
    y_val=Yva,
    param_grid=param_grid,
    # ===【验证】这些默认值应该被使用（不是PyReCoStandardModel的默认值）
    default_spec_rad=1.0,
    default_leakage=0.3,
    default_density=0.1,  # ← 关键！应该是0.1，不是0.8
    default_activation="tanh",
    default_fraction_input=0.5,  # ← 关键！应该是0.5，不是1.0
    verbose=True
)

print("\n" + "="*100)
print("✅ Test Results")
print("="*100 + "\n")

print("📊 Best parameters:")
for key, value in results['best_params'].items():
    print(f"  {key}: {value}")
print()

# Verify critical defaults
best_params = results['best_params']
print("🔍 Verifying Default Values:")
print(f"  density: {best_params['density']} (expected: 0.1)")
print(f"  fraction_input: {best_params['fraction_input']} (expected: 0.5)")
print(f"  leakage_rate: {best_params['leakage_rate']} (expected: 0.3)")
print()

# Check if defaults are correct
if best_params['density'] == 0.1:
    print("✅ density default is CORRECT (0.1)")
else:
    print(f"❌ density default is WRONG (got {best_params['density']}, expected 0.1)")

if best_params['fraction_input'] == 0.5:
    print("✅ fraction_input default is CORRECT (0.5)")
else:
    print(f"❌ fraction_input default is WRONG (got {best_params['fraction_input']}, expected 0.5)")

if best_params['leakage_rate'] == 0.3:
    print("✅ leakage_rate default is CORRECT (0.3)")
else:
    print(f"❌ leakage_rate default is WRONG (got {best_params['leakage_rate']}, expected 0.3)")

print()

# Test final model
if results['final_model'] is not None:
    print("✅ Final model was trained successfully!")
    print(f"   Is trained: {results['final_model'].is_trained}")
    print()
else:
    print("❌ Final model is None!")
    print()

print("="*100)
print("🎉 Default Values Fix Test Complete!")
print("="*100)
