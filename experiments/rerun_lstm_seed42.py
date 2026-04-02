#!/usr/bin/env python3
"""
Rerun LSTM for seed=42 only, updating existing V2 result files in-place.

Skips PyReCo entirely — only reruns LSTM tuning + final retrain using the
fixed _make_trainval_windows (no boundary window gap).

Usage:
    python experiments/rerun_lstm_seed42.py
    python experiments/rerun_lstm_seed42.py --dataset lorenz --budget small
    python experiments/rerun_lstm_seed42.py --dry-run
"""

import argparse
import json
import time
import sys
import os
import gc
import numpy as np
from datetime import datetime
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.load_dataset import load as local_load, set_seed, load_data, _sliding_window
from src.utils.budget_matching import (
    lstm_solve_hidden_size, lstm_total_params, lstm_layer_hidden_map,
)
from models.lstm_model import LSTMModel, tune_lstm_hyperparameters
from pyreco.datasets import load as pyreco_load

PYRECO_DATASETS = {'lorenz', 'mackeyglass'}
BUDGETS = {'small': 1000, 'medium': 10000, 'large': 50000}
DATASETS = ['lorenz', 'mackeyglass', 'santafe']
TRAIN_FRACS = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
SEED = 42

LSTM_PARAM_GRID = {
    'learning_rate': [0.0005, 0.001, 0.005, 0.01],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2],
}


def _make_trainval_windows(dataset, seed, train_frac, scaler, n_in=100, n_out=1):
    """Create sliding windows on the continuous train+val sequence."""
    set_seed(seed)
    if dataset in PYRECO_DATASETS:
        result = pyreco_load(
            dataset, 5000, seed=seed, train_fraction=train_frac,
            n_in=n_in, n_out=n_out, standardize=False,
        )
        X_raw, Y_raw = result[0], result[1]
    else:
        data, _ = load_data(dataset, length=5000, seed=seed)
        n_trainval = int(len(data) * train_frac)
        trainval_data = data[:n_trainval]
        X_raw, Y_raw = _sliding_window(trainval_data, n_in, n_out)

    N, win, D = X_raw.shape
    X_tv = scaler.transform(X_raw.reshape(-1, D)).reshape(N, win, D)
    _, out, D2 = Y_raw.shape
    Y_tv = scaler.transform(Y_raw.reshape(-1, D2)).reshape(Y_raw.shape)
    return X_tv, Y_tv


def rerun_lstm(dataset, budget_name, train_frac, seed, lstm_device='cpu', verbose=True):
    """Rerun LSTM tuning + final retrain for one condition."""
    budget = BUDGETS[budget_name]
    d_in = 3 if dataset == 'lorenz' else 1
    d_out = d_in

    # Load data
    set_seed(seed)
    load_func = pyreco_load if dataset in PYRECO_DATASETS else local_load
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_func(
        dataset, n_samples=5000, seed=seed,
        train_fraction=train_frac, val_fraction=0.15,
        n_in=100, n_out=1, standardize=True,
    )

    layer_map = lstm_layer_hidden_map(budget, d_in, d_out, [1, 2])

    # Tuning
    set_seed(seed)
    t_tune_start = time.time()
    tune_results = tune_lstm_hyperparameters(
        X_train, y_train, X_val, y_val,
        param_grid=LSTM_PARAM_GRID,
        lstm_device=lstm_device,
        verbose=verbose,
        layer_hidden_map=layer_map,
    )
    tune_time = time.time() - t_tune_start

    best_lstm_params = tune_results['best_params']
    best_lstm_val = tune_results['best_score']

    best_epoch = None
    for r in tune_results.get('all_results', []):
        if r['params'] == best_lstm_params:
            best_epoch = r.get('best_epoch', None)
            break

    del tune_results
    gc.collect()

    # Final retrain on train+val (with fixed boundary windows)
    X_full, y_full = _make_trainval_windows(dataset, seed, train_frac, scaler)

    set_seed(seed)
    final_lstm = LSTMModel(**best_lstm_params, device=lstm_device, verbose=verbose)
    if best_epoch and best_epoch > 0:
        final_lstm.epochs = best_epoch

    t0 = time.time()
    final_lstm.fit(X_full, y_full)
    lstm_train_time = time.time() - t0

    # Test evaluation
    test_pred = final_lstm.predict(X_test)
    test_pred = np.asarray(test_pred).reshape(y_test.shape)
    test_mse = float(np.mean((y_test - test_pred) ** 2))
    ss_res = np.sum((y_test - test_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test, axis=0))**2)
    test_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Inference time
    _ = final_lstm.predict(X_test[:10])
    t0 = time.time()
    _ = final_lstm.predict(X_test)
    inference_time = time.time() - t0
    per_sample_ms = inference_time / len(X_test) * 1000

    h = best_lstm_params['hidden_size']
    nl = best_lstm_params.get('num_layers', 1)
    lstm_pi = lstm_total_params(h, d_in, d_out, nl)

    result = {
        'model_type': 'lstm',
        'config': best_lstm_params,
        'param_info': {
            'trainable': lstm_pi['total'],
            'total': lstm_pi['total'],
            'hidden_size': h,
            'num_layers': nl,
        },
        'test_mse': test_mse,
        'test_r2': test_r2,
        'tune_time': tune_time,
        'best_combo_train_time': lstm_train_time,
        'final_train_time': lstm_train_time,
        'inference_time_per_sample_ms': per_sample_ms,
        'inference_time_total': inference_time,
        'n_test_samples': len(X_test),
        'val_mse': best_lstm_val,
        'best_epoch': best_epoch,
        'rerun_note': 'Rerun with fixed _make_trainval_windows (no boundary gap)',
        'rerun_timestamp': datetime.now().isoformat(),
    }

    del final_lstm
    gc.collect()

    return result


def update_result_file(filepath, budget_name, new_lstm_result):
    """Replace the LSTM entry in an existing result file."""
    with open(filepath) as f:
        data = json.load(f)

    results_list = data['results'].get(budget_name, [])

    # Remove old LSTM entry
    results_list = [r for r in results_list if r.get('model_type') != 'lstm']
    # Add new one
    results_list.append(new_lstm_result)

    data['results'][budget_name] = results_list
    data['metadata']['lstm_rerun_timestamp'] = datetime.now().isoformat()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Rerun LSTM seed=42')
    parser.add_argument('--dataset', type=str, default=None, choices=DATASETS)
    parser.add_argument('--budget', type=str, default=None, choices=list(BUDGETS.keys()))
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--results-dir', type=str, default='results/final_v2')
    parser.add_argument('--dry-run', action='store_true', help='List files to update without running')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    budgets = [args.budget] if args.budget else list(BUDGETS.keys())
    results_dir = Path(args.results_dir)

    # Collect all files to process
    tasks = []
    for dataset in datasets:
        for budget_name in budgets:
            for train_frac in TRAIN_FRACS:
                filename = f"results_{dataset}_{budget_name}_seed{SEED}_train{train_frac}.json"
                filepath = results_dir / filename
                if filepath.exists():
                    tasks.append((dataset, budget_name, train_frac, filepath))
                else:
                    print(f"WARNING: {filepath} not found, skipping")

    print(f"Will rerun LSTM for {len(tasks)} conditions (seed={SEED})")
    print(f"Device: {args.device}")

    if args.dry_run:
        for dataset, budget_name, train_frac, filepath in tasks:
            print(f"  {dataset}/{budget_name}/tf={train_frac} → {filepath.name}")
        return

    total_time = 0
    for i, (dataset, budget_name, train_frac, filepath) in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(tasks)}] {dataset} | {budget_name} | tf={train_frac} | seed={SEED}")
        print(f"{'='*70}")

        t0 = time.time()
        new_lstm = rerun_lstm(
            dataset, budget_name, train_frac, SEED,
            lstm_device=args.device, verbose=True,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        print(f"\n  R²={new_lstm['test_r2']:.6f}  MSE={new_lstm['test_mse']:.6f}  "
              f"tune={new_lstm['tune_time']:.0f}s  retrain={new_lstm['final_train_time']:.1f}s")

        # Update the file
        update_result_file(filepath, budget_name, new_lstm)
        print(f"  Updated: {filepath.name}")

        remaining = (len(tasks) - i - 1) * (total_time / (i + 1))
        print(f"  Elapsed: {elapsed:.0f}s | Total: {total_time/3600:.1f}h | "
              f"ETA: {remaining/3600:.1f}h")

    print(f"\n{'='*70}")
    print(f"DONE: {len(tasks)} LSTM runs updated in {total_time/3600:.1f}h")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
