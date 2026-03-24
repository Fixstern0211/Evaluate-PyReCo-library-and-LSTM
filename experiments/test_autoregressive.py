"""
Autoregressive Multi-step Prediction Experiment

Evaluates PyReCo vs LSTM using autoregressive (free-run) prediction mode
across multiple prediction horizons.

Key Features:
- Loads best PyReCo params from existing 90 experiment results
- Trains LSTM with hyperparameter tuning for fair comparison
- Autoregressive prediction: feed predictions back as input
- Evaluates at multiple horizons: 1, 5, 10, 20, 30, 50 steps

Usage:
    python test_autoregressive.py --dataset lorenz --seed 42 --train-ratio 0.7
    python test_autoregressive.py --dataset lorenz --seed 42 --train-ratio 0.7 --budget medium
"""

import argparse
import json
import time
import math
import glob
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyreco.datasets import load as pyreco_load
from src.utils.load_dataset import load as local_load
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel, tune_lstm_hyperparameters
from src.utils.evaluation import multi_step_predict, normalized_rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Datasets supported by PyReCo
PYRECO_SUPPORTED_DATASETS = {'lorenz', 'mackeyglass', 'mackey-glass', 'mg'}

# Prediction horizons to evaluate
HORIZONS = [1, 5, 10, 20, 30, 50]

# Budget mapping
BUDGETS = {
    'small': 1000,
    'medium': 10000,
    'large': 50000
}


def load_best_pyreco_params(results_dir, dataset, seed, train_ratio, budget_name):
    """
    Load best PyReCo parameters from existing experiment results

    Args:
        results_dir: Directory containing result JSON files
        dataset: Dataset name
        seed: Random seed
        train_ratio: Training ratio
        budget_name: 'small', 'medium', or 'large'

    Returns:
        Best config dict or None if not found
    """
    pattern = f"{results_dir}/results_{dataset}_seed{seed}_train{train_ratio}_*.json"
    files = glob.glob(pattern)

    if not files:
        print(f"  Warning: No result file found for {dataset}/seed{seed}/train{train_ratio}")
        return None

    # Use the most recent file
    result_file = sorted(files)[-1]

    with open(result_file) as f:
        data = json.load(f)

    # Find PyReCo config for the specified budget
    budget_results = data['results'].get(budget_name, [])
    for r in budget_results:
        if r.get('model_type') == 'pyreco_standard':
            return r['config']

    print(f"  Warning: No PyReCo config found in {result_file}")
    return None


def get_default_pyreco_config(budget, dataset):
    """Get default PyReCo config if no existing results"""
    density = 0.1
    num_nodes = int(math.sqrt(budget / density))
    candidates = [50, 100, 200, 300, 400, 500, 700]
    num_nodes = min(candidates, key=lambda x: abs(x - num_nodes))

    # Dataset-specific defaults based on pre-tuning
    if dataset == 'lorenz':
        return {
            'num_nodes': num_nodes, 'density': 0.05, 'spec_rad': 0.9,
            'leakage_rate': 0.5, 'fraction_input': 0.5, 'fraction_output': 1.0,
            'activation': 'tanh', 'optimizer': 'ridge'
        }
    elif dataset == 'mackeyglass':
        return {
            'num_nodes': num_nodes, 'density': 0.1, 'spec_rad': 0.8,
            'leakage_rate': 0.3, 'fraction_input': 0.5, 'fraction_output': 1.0,
            'activation': 'tanh', 'optimizer': 'ridge'
        }
    else:  # santafe
        return {
            'num_nodes': num_nodes, 'density': 0.1, 'spec_rad': 0.8,
            'leakage_rate': 0.6, 'fraction_input': 0.5, 'fraction_output': 1.0,
            'activation': 'tanh', 'optimizer': 'ridge'
        }


def get_lstm_config(budget, n_features):
    """Get LSTM config for given budget"""
    num_layers = 2
    # Solve for hidden_size from parameter budget
    h = int((-15 + math.sqrt(225 + 48 * budget)) / 24)
    h = max(8, min(h, 128))

    return {
        'hidden_size': h,
        'num_layers': num_layers,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'patience': 10
    }


def load_data(dataset, n_samples, seed, train_fraction, val_fraction=0.15, n_in=100):
    """Load dataset with consistent interface"""
    if dataset.lower() in PYRECO_SUPPORTED_DATASETS:
        load_func = pyreco_load
    else:
        load_func = local_load

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_func(
        dataset,
        n_samples=n_samples,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        n_in=n_in,
        n_out=1,
        standardize=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def create_extended_targets(dataset, n_samples, seed, train_fraction, val_fraction, n_in, max_horizon):
    """Create extended target sequences for multi-horizon evaluation"""
    if dataset.lower() in PYRECO_SUPPORTED_DATASETS:
        load_func = pyreco_load
    else:
        load_func = local_load

    _, _, _, _, X_test, y_test_ext, _ = load_func(
        dataset,
        n_samples=n_samples,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        n_in=n_in,
        n_out=max_horizon,
        standardize=True
    )

    return X_test, y_test_ext


def evaluate_autoregressive(model, X_test, y_test_sequence, horizons):
    """
    Evaluate model using autoregressive prediction at multiple horizons
    """
    max_horizon = max(horizons)
    results = {}

    # Generate predictions using autoregressive mode
    start_time = time.time()
    y_pred_all = multi_step_predict(model, X_test, max_horizon, mode='free_run')
    inference_time = time.time() - start_time

    # Evaluate at each horizon
    for horizon in horizons:
        y_pred_h = y_pred_all[:, :horizon, :]
        y_true_h = y_test_sequence[:, :horizon, :]

        mse = float(mean_squared_error(y_true_h.flatten(), y_pred_h.flatten()))
        mae = float(mean_absolute_error(y_true_h.flatten(), y_pred_h.flatten()))
        rmse = float(np.sqrt(mse))
        nrmse = float(normalized_rmse(y_true_h, y_pred_h))
        r2 = float(r2_score(y_true_h.flatten(), y_pred_h.flatten()))

        results[horizon] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'r2': r2
        }

    results['inference_time'] = inference_time
    results['inference_time_per_sample_ms'] = (inference_time / len(X_test)) * 1000

    return results


def main():
    parser = argparse.ArgumentParser(description='Autoregressive Multi-step Prediction')
    parser.add_argument('--dataset', type=str, default='lorenz',
                       choices=['lorenz', 'mackeyglass', 'santafe'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-in', type=int, default=100)
    parser.add_argument('--budget', type=str, default='medium',
                       choices=['small', 'medium', 'large'])
    parser.add_argument('--results-dir', type=str, default='../results/final',
                       help='Directory with existing experiment results')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--tune-lstm', action='store_true',
                       help='Tune LSTM hyperparameters')

    args = parser.parse_args()

    np.random.seed(args.seed)

    val_fraction = 0.15
    if args.train_ratio + val_fraction >= 1.0:
        raise ValueError(f"train_ratio ({args.train_ratio}) + val_fraction ({val_fraction}) >= 1.0")

    if args.output is None:
        args.output = f"autoregressive_{args.dataset}_seed{args.seed}_train{args.train_ratio}_{args.budget}.json"

    budget = BUDGETS[args.budget]
    max_horizon = max(HORIZONS)

    print("=" * 70)
    print("AUTOREGRESSIVE MULTI-STEP PREDICTION EXPERIMENT")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Budget: {args.budget} ({budget:,} params)")
    print(f"Horizons: {HORIZONS}")
    print("=" * 70)

    # 1. Load training data
    print("\n[1/5] Loading training data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data(
        args.dataset, args.n_samples, args.seed, args.train_ratio,
        val_fraction=val_fraction, n_in=args.n_in
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 2. Load extended targets for evaluation
    print("\n[2/5] Loading extended targets...")
    X_test_ext, y_test_ext = create_extended_targets(
        args.dataset, args.n_samples, args.seed, args.train_ratio,
        val_fraction, args.n_in, max_horizon
    )
    print(f"  Extended test: {X_test_ext.shape} -> {y_test_ext.shape}")

    n_features = X_train.shape[-1]
    results = {}

    # 3. Train PyReCo with best params from existing results
    print("\n[3/5] Training PyReCo...")
    pyreco_config = load_best_pyreco_params(
        args.results_dir, args.dataset, args.seed, args.train_ratio, args.budget
    )
    if pyreco_config is None:
        print("  Using default config (no existing results found)")
        pyreco_config = get_default_pyreco_config(budget, args.dataset)
    else:
        print(f"  Loaded best params from existing results")

    print(f"  Config: nodes={pyreco_config['num_nodes']}, spec_rad={pyreco_config['spec_rad']}, "
          f"leakage={pyreco_config['leakage_rate']}")

    pyreco_model = PyReCoStandardModel(**pyreco_config, verbose=False)
    start_time = time.time()
    pyreco_model.fit(X_train, y_train)
    pyreco_train_time = time.time() - start_time
    print(f"  Training time: {pyreco_train_time:.2f}s")

    # 4. Train LSTM
    print("\n[4/5] Training LSTM...")
    lstm_base_config = get_lstm_config(budget, n_features)

    if args.tune_lstm:
        print("  Tuning hyperparameters...")
        param_grid = {
            'hidden_size': [lstm_base_config['hidden_size']],
            'num_layers': [lstm_base_config['num_layers']],
            'learning_rate': [0.0005, 0.001, 0.002],
            'dropout': [0.1, 0.2, 0.3],
        }
        tune_result = tune_lstm_hyperparameters(
            X_train, y_train, X_val, y_val,
            param_grid=param_grid,
            verbose=False
        )
        lstm_config = {
            **tune_result['best_params'],
            'epochs': 100, 'batch_size': 32, 'patience': 10
        }
    else:
        lstm_config = lstm_base_config

    print(f"  Config: hidden={lstm_config['hidden_size']}, layers={lstm_config['num_layers']}, "
          f"lr={lstm_config['learning_rate']}, dropout={lstm_config['dropout']}")

    lstm_model = LSTMModel(**lstm_config, verbose=False)
    start_time = time.time()
    lstm_model.fit(X_train, y_train)
    lstm_train_time = time.time() - start_time
    print(f"  Training time: {lstm_train_time:.2f}s")

    # 5. Evaluate autoregressive prediction
    print("\n[5/5] Evaluating autoregressive prediction...")

    print("\n  PyReCo evaluation...")
    pyreco_results = evaluate_autoregressive(pyreco_model, X_test_ext, y_test_ext, HORIZONS)
    results['pyreco'] = {
        'horizons': pyreco_results,
        'train_time': pyreco_train_time,
        'config': pyreco_config
    }

    print("  LSTM evaluation...")
    lstm_results = evaluate_autoregressive(lstm_model, X_test_ext, y_test_ext, HORIZONS)
    results['lstm'] = {
        'horizons': lstm_results,
        'train_time': lstm_train_time,
        'config': {k: v for k, v in lstm_config.items() if not callable(v)}
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Horizon':<10} {'PyReCo MSE':<15} {'LSTM MSE':<15} {'Winner':<10}")
    print("-" * 50)

    for h in HORIZONS:
        pyreco_mse = results['pyreco']['horizons'][h]['mse']
        lstm_mse = results['lstm']['horizons'][h]['mse']
        winner = 'PyReCo' if pyreco_mse < lstm_mse else 'LSTM'
        print(f"{h:<10} {pyreco_mse:<15.6f} {lstm_mse:<15.6f} {winner:<10}")

    print("-" * 50)
    print(f"\nTraining time: PyReCo={pyreco_train_time:.2f}s, LSTM={lstm_train_time:.2f}s")
    print(f"Inference time (50 steps): PyReCo={pyreco_results['inference_time']:.3f}s, "
          f"LSTM={lstm_results['inference_time']:.3f}s")

    # Save results
    output_data = {
        'metadata': {
            'dataset': args.dataset,
            'seed': args.seed,
            'train_ratio': args.train_ratio,
            'n_samples': args.n_samples,
            'n_in': args.n_in,
            'budget': args.budget,
            'budget_value': budget,
            'horizons': HORIZONS,
            'prediction_mode': 'autoregressive',
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
