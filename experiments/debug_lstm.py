#!/usr/bin/env python3
"""
Debug script to test LSTM training and identify failure reasons
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import traceback

print("=" * 60)
print("LSTM Debug Test")
print("=" * 60)

# Step 1: Test imports
print("\n[1] Testing imports...")
try:
    import numpy as np
    print("  ✓ numpy imported")
except Exception as e:
    print(f"  ✗ numpy failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ torch imported (version: {torch.__version__})")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    print(f"    MPS available: {torch.backends.mps.is_available()}")
except Exception as e:
    print(f"  ✗ torch failed: {e}")
    sys.exit(1)

try:
    from models.lstm_model import LSTMModel, tune_lstm_hyperparameters
    print("  ✓ LSTMModel imported")
except Exception as e:
    print(f"  ✗ LSTMModel import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from pyreco.datasets import load as pyreco_load
    print("  ✓ pyreco imported")
except Exception as e:
    print(f"  ✗ pyreco failed: {e}")
    sys.exit(1)

# Step 2: Load data
print("\n[2] Loading test data...")
try:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = pyreco_load(
        'lorenz',
        n_samples=2000,
        seed=42,
        train_fraction=0.7,
        val_fraction=0.15,
        n_in=100,
        n_out=1,
        standardize=True
    )
    print(f"  ✓ Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create LSTM model
print("\n[3] Creating LSTM model...")
try:
    lstm_config = {
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 5,  # Small number for testing
        'batch_size': 32,
        'patience': 3
    }
    print(f"  Config: {lstm_config}")

    model = LSTMModel(**lstm_config, verbose=True)
    print("  ✓ LSTM model created")
except Exception as e:
    print(f"  ✗ LSTM creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Train LSTM model
print("\n[4] Training LSTM model...")
try:
    model.fit(X_train, y_train)
    print("  ✓ LSTM training completed")
except Exception as e:
    print(f"  ✗ LSTM training failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test prediction
print("\n[5] Testing prediction...")
try:
    y_pred = model.predict(X_test[:10])
    print(f"  ✓ Prediction shape: {y_pred.shape}")
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test evaluation
print("\n[6] Testing evaluation...")
try:
    results = model.evaluate(X_test, y_test)
    print(f"  ✓ Evaluation results: MSE={results['mse']:.6f}, R²={results['r2']:.4f}")
except Exception as e:
    print(f"  ✗ Evaluation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test hyperparameter tuning
print("\n[7] Testing hyperparameter tuning...")
try:
    param_grid = {
        'hidden_size': [32],
        'num_layers': [2],
        'learning_rate': [0.001, 0.002],
        'dropout': [0.1, 0.2],
    }

    tune_result = tune_lstm_hyperparameters(
        X_train[:500], y_train[:500],  # Use smaller data for speed
        X_val[:100], y_val[:100],
        param_grid=param_grid,
        verbose=True
    )
    print(f"  ✓ Best params: {tune_result['best_params']}")
    print(f"  ✓ Best MSE: {tune_result['best_score']:.6f}")
except Exception as e:
    print(f"  ✗ Hyperparameter tuning failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! LSTM should work correctly.")
print("=" * 60)
