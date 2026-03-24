"""
Test Multi-Step Evaluation System

This script tests the new multi-step evaluation capabilities with a simple model
to ensure the evaluation framework works correctly before using it in experiments.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyreco.datasets import load as pyreco_load
from models.pyreco_wrapper import PyReCoStandardModel
from src.utils.evaluation import evaluate_multi_step, create_evaluation_protocol_document


def test_multi_step_evaluation():
    """Test the multi-step evaluation system with Lorenz data"""
    print("🧪 Testing Multi-Step Evaluation System")
    print("=" * 50)

    # 1. Load test data
    print("📊 Loading Lorenz dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = pyreco_load(
        'lorenz',
        n_samples=1000,  # Smaller dataset for quick testing
        seed=42,
        val_fraction=0.15,
        train_fraction=0.6,
        n_in=30,         # Input window
        n_out=1,         # Single-step for training (multi-step handled by evaluation)
        standardize=True
    )

    # Create multi-step targets for evaluation (we'll generate longer sequences)
    print("📊 Generating multi-step target data...")
    # Get raw data for creating longer targets
    X_train_ms, y_train_ms, X_val_ms, y_val_ms, X_test_ms, y_test_ms, _ = pyreco_load(
        'lorenz',
        n_samples=1000,
        seed=42,
        val_fraction=0.15,
        train_fraction=0.6,
        n_in=30,
        n_out=10,        # Generate 10-step targets for testing
        standardize=True
    )

    print(f"✅ Data loaded:")
    print(f"  Train: {X_train.shape} -> {y_train.shape}")
    print(f"  Val: {X_val.shape} -> {y_val.shape}")
    print(f"  Test: {X_test.shape} -> {y_test.shape}")
    print(f"  Multi-step test: {X_test_ms.shape} -> {y_test_ms.shape}")

    # 2. Create and train a simple model
    print("\n🤖 Creating and training PyReCo model...")
    model = PyReCoStandardModel(
        num_nodes=100,
        spec_rad=0.9,
        leakage_rate=0.3,
        density=0.1,
        verbose=False
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"✅ Model trained in {train_time:.2f} seconds")

    # 3. Test multi-step evaluation
    print("\n📈 Testing multi-step evaluation...")

    try:
        # Test with limited horizons for speed
        results = evaluate_multi_step(
            model=model,
            X_test=X_test_ms[:5],  # Use multi-step test data
            y_test=y_test_ms[:5],  # Use multi-step targets
            horizons=[1, 5, 10],   # Test up to 10 steps
            mode='free_run',
            include_advanced_metrics=True
        )

        print("\n✅ Multi-step evaluation completed!")
        print("\n📊 Results:")
        for horizon, metrics in results.items():
            print(f"\nHorizon {horizon} steps:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.3f}")
            print(f"  R²: {metrics['r2']:.3f}")

            # Advanced metrics
            if 'avg_divergence_time' in metrics:
                print(f"  Avg divergence time: {metrics['avg_divergence_time']:.1f} steps")

            if 'spectral_psd_correlation' in metrics:
                print(f"  Spectral correlation: {metrics['spectral_psd_correlation']:.3f}")

            if 'stats_mean_error' in metrics:
                print(f"  Mean error: {metrics['stats_mean_error']:.3f}")
                print(f"  Variance error: {metrics['stats_variance_error']:.3f}")

    except Exception as e:
        print(f"\n❌ Multi-step evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test evaluation protocol document
    print("\n📋 Testing evaluation protocol document generation...")
    try:
        protocol = create_evaluation_protocol_document()
        print(f"✅ Protocol document created with {len(protocol)} sections")
        print(f"  Standard horizons: {protocol['standard_horizons']}")
        print(f"  Core metrics: {list(protocol['core_metrics'].keys())}")
    except Exception as e:
        print(f"❌ Protocol document generation failed: {e}")
        return False

    print("\n🎉 All tests passed! Multi-step evaluation system is ready.")
    return True


def test_evaluation_robustness():
    """Test evaluation system with edge cases"""
    print("\n🔧 Testing evaluation robustness...")

    # Test with minimal data
    X_mini = np.random.randn(2, 5, 3)
    y_mini = np.random.randn(2, 10, 3)

    from models.pyreco_wrapper import PyReCoStandardModel

    model = PyReCoStandardModel(num_nodes=20, verbose=False)
    model.fit(X_mini, y_mini[:, :1, :])  # Train on single-step

    try:
        results = evaluate_multi_step(
            model=model,
            X_test=X_mini,
            y_test=y_mini,
            horizons=[1, 2],
            mode='free_run',
            include_advanced_metrics=False  # Skip advanced metrics for robustness test
        )
        print("✅ Robustness test passed")
        return True
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    success1 = test_multi_step_evaluation()
    success2 = test_evaluation_robustness()

    if success1 and success2:
        print("\n🎯 All evaluation system tests passed!")
        print("The multi-step evaluation framework is ready for use in experiments.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)