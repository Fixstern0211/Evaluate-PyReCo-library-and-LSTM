"""
Test script to compare PyReCo datasets.py with manual data generation.

Compares:
1. PyReCo's datasets.load() function
2. Manual approach using load_dataset.py + process_datasets.py
"""

import sys
import numpy as np
from pathlib import Path

# Add PyReCo to path
pyreco_path = Path(__file__).parent.parent.parent / "pyReCo" / "src"
sys.path.insert(0, str(pyreco_path))

# Add test project to path
test_project_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(test_project_path))

from pyreco import datasets as pyreco_datasets
from utils.load_dataset import load_data
from utils.process_datasets import split_datasets, sliding_window


def compare_lorenz_datasets():
    """Compare Lorenz dataset generation."""
    print("=" * 80)
    print("COMPARING LORENZ DATASETS")
    print("=" * 80)

    # Parameters
    n_samples = 1000
    train_frac = 0.7
    n_in = 100
    n_out = 1
    seed = 42

    # Method 1: PyReCo datasets.load()
    print("\n1. PyReCo datasets.load() method:")
    x_train_p, y_train_p, x_test_p, y_test_p = pyreco_datasets.load(
        'lorenz',
        n_samples=n_samples,
        train_fraction=train_frac,
        n_in=n_in,
        n_out=n_out,
        seed=seed
    )
    print(f"   x_train shape: {x_train_p.shape}")
    print(f"   y_train shape: {y_train_p.shape}")
    print(f"   x_test shape:  {x_test_p.shape}")
    print(f"   y_test shape:  {y_test_p.shape}")

    # Method 2: Manual approach
    print("\n2. Manual approach (load_data + split + sliding_window):")

    # Load raw data
    raw_data, meta = load_data('lorenz', length=n_samples, seed=seed)
    print(f"   Raw data shape: {raw_data.shape}")

    # Split
    train_data, test_data, split_idx = split_datasets(raw_data, train_frac)
    print(f"   Train data shape: {train_data.shape}")
    print(f"   Test data shape:  {test_data.shape}")
    print(f"   Split index: {split_idx}")

    # Sliding window
    x_train_m, y_train_m = sliding_window(train_data, n_in, n_out)
    x_test_m, y_test_m = sliding_window(test_data, n_in, n_out)
    print(f"   x_train shape: {x_train_m.shape}")
    print(f"   y_train shape: {y_train_m.shape}")
    print(f"   x_test shape:  {x_test_m.shape}")
    print(f"   y_test shape:  {y_test_m.shape}")

    # Compare
    print("\n3. Comparison:")
    print(f"   Shape match (x_train): {x_train_p.shape == x_train_m.shape}")
    print(f"   Shape match (y_train): {y_train_p.shape == y_train_m.shape}")
    print(f"   Shape match (x_test):  {x_test_p.shape == x_test_m.shape}")
    print(f"   Shape match (y_test):  {y_test_p.shape == y_test_m.shape}")

    # Check if data values are close
    if x_train_p.shape == x_train_m.shape:
        x_train_close = np.allclose(x_train_p, x_train_m, rtol=1e-10, atol=1e-10)
        y_train_close = np.allclose(y_train_p, y_train_m, rtol=1e-10, atol=1e-10)
        x_test_close = np.allclose(x_test_p, x_test_m, rtol=1e-10, atol=1e-10)
        y_test_close = np.allclose(y_test_p, y_test_m, rtol=1e-10, atol=1e-10)

        print(f"\n   Value match (x_train): {x_train_close}")
        print(f"   Value match (y_train): {y_train_close}")
        print(f"   Value match (x_test):  {x_test_close}")
        print(f"   Value match (y_test):  {y_test_close}")

        if not x_train_close:
            diff = np.abs(x_train_p - x_train_m)
            print(f"\n   Max difference in x_train: {diff.max()}")
            print(f"   Mean difference in x_train: {diff.mean()}")
            print(f"   Sample x_train[0,0,:] (PyReCo): {x_train_p[0,0,:]}")
            print(f"   Sample x_train[0,0,:] (Manual):  {x_train_m[0,0,:]}")

        all_match = x_train_close and y_train_close and x_test_close and y_test_close

        if all_match:
            print("\n   ✅ ALL DATA MATCHES! The implementations are equivalent.")
        else:
            print("\n   ❌ DATA MISMATCH! The implementations produce different results.")
    else:
        print("\n   ❌ SHAPE MISMATCH! Cannot compare values.")

    return all_match if x_train_p.shape == x_train_m.shape else False


def compare_mackey_glass_datasets():
    """Compare Mackey-Glass dataset generation."""
    print("\n" + "=" * 80)
    print("COMPARING MACKEY-GLASS DATASETS")
    print("=" * 80)

    # Parameters
    n_samples = 1000
    train_frac = 0.7
    n_in = 100
    n_out = 1
    seed = 42

    # Method 1: PyReCo datasets.load()
    print("\n1. PyReCo datasets.load() method:")
    x_train_p, y_train_p, x_test_p, y_test_p = pyreco_datasets.load(
        'mackey_glass',
        n_samples=n_samples,
        train_fraction=train_frac,
        n_in=n_in,
        n_out=n_out,
        seed=seed
    )
    print(f"   x_train shape: {x_train_p.shape}")
    print(f"   y_train shape: {y_train_p.shape}")
    print(f"   x_test shape:  {x_test_p.shape}")
    print(f"   y_test shape:  {y_test_p.shape}")

    # Method 2: Manual approach
    print("\n2. Manual approach (load_data + split + sliding_window):")

    # Load raw data
    raw_data, meta = load_data('mackeyglass', length=n_samples, seed=seed)
    print(f"   Raw data shape: {raw_data.shape}")

    # Split
    train_data, test_data, split_idx = split_datasets(raw_data, train_frac)
    print(f"   Train data shape: {train_data.shape}")
    print(f"   Test data shape:  {test_data.shape}")

    # Sliding window
    x_train_m, y_train_m = sliding_window(train_data, n_in, n_out)
    x_test_m, y_test_m = sliding_window(test_data, n_in, n_out)
    print(f"   x_train shape: {x_train_m.shape}")
    print(f"   y_train shape: {y_train_m.shape}")
    print(f"   x_test shape:  {x_test_m.shape}")
    print(f"   y_test shape:  {y_test_m.shape}")

    # Compare
    print("\n3. Comparison:")
    print(f"   Shape match (x_train): {x_train_p.shape == x_train_m.shape}")
    print(f"   Shape match (y_train): {y_train_p.shape == y_train_m.shape}")
    print(f"   Shape match (x_test):  {x_test_p.shape == x_test_m.shape}")
    print(f"   Shape match (y_test):  {y_test_p.shape == y_test_m.shape}")

    # Check if data values are close
    if x_train_p.shape == x_train_m.shape:
        x_train_close = np.allclose(x_train_p, x_train_m, rtol=1e-10, atol=1e-10)
        y_train_close = np.allclose(y_train_p, y_train_m, rtol=1e-10, atol=1e-10)
        x_test_close = np.allclose(x_test_p, x_test_m, rtol=1e-10, atol=1e-10)
        y_test_close = np.allclose(y_test_p, y_test_m, rtol=1e-10, atol=1e-10)

        print(f"\n   Value match (x_train): {x_train_close}")
        print(f"   Value match (y_train): {y_train_close}")
        print(f"   Value match (x_test):  {x_test_close}")
        print(f"   Value match (y_test):  {y_test_close}")

        if not x_train_close:
            diff = np.abs(x_train_p - x_train_m)
            print(f"\n   Max difference in x_train: {diff.max()}")
            print(f"   Mean difference in x_train: {diff.mean()}")
            print(f"   Sample x_train[0,0,:] (PyReCo): {x_train_p[0,0,:]}")
            print(f"   Sample x_train[0,0,:] (Manual):  {x_train_m[0,0,:]}")

        all_match = x_train_close and y_train_close and x_test_close and y_test_close

        if all_match:
            print("\n   ✅ ALL DATA MATCHES! The implementations are equivalent.")
        else:
            print("\n   ❌ DATA MISMATCH! The implementations produce different results.")
    else:
        print("\n   ❌ SHAPE MISMATCH! Cannot compare values.")

    return all_match if x_train_p.shape == x_train_m.shape else False


def main():
    """Run all comparisons."""
    print("\n" + "=" * 80)
    print("DATASET COMPARISON TEST")
    print("Testing PyReCo datasets.py vs Manual approach")
    print("=" * 80)

    try:
        lorenz_match = compare_lorenz_datasets()
        mg_match = compare_mackey_glass_datasets()

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Lorenz dataset:      {'✅ MATCH' if lorenz_match else '❌ MISMATCH'}")
        print(f"Mackey-Glass dataset: {'✅ MATCH' if mg_match else '❌ MISMATCH'}")

        if lorenz_match and mg_match:
            print("\n🎉 SUCCESS! Both implementations produce identical results.")
            return 0
        else:
            print("\n⚠️  WARNING! Implementations produce different results.")
            print("    This may be due to:")
            print("    1. Parameter order differences")
            print("    2. Default value differences")
            print("    3. Random seed handling")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
