#!/usr/bin/env python3
"""
Multi-Step Prediction Experiments V2: Budget-Constrained

Both PyReCo and LSTM use per-seed best configs from results/final_v2/.
Both models are retrained and evaluated at horizons [1, 5, 10, 20, 50].

Design:
- Load best config per (dataset, budget, train_frac, seed) for BOTH models
- Train each model on train+val data
- Evaluate multi-step free-run prediction at multiple horizons

Usage:
    python experiments/run_multi_step_v2.py --budget small --seed 42
    python experiments/run_multi_step_v2.py --quick
    python experiments/run_multi_step_v2.py  # all 270 experiments
"""

import argparse
import json
import glob
import os
import re
import sys
import time
import gc
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.load_dataset import load as local_load, set_seed, load_data, _sliding_window
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel
from src.utils.evaluation import evaluate_multi_step
from pyreco.datasets import load as pyreco_load

PYRECO_DATASETS = {'lorenz', 'mackeyglass'}


def _make_trainval_windows(dataset, seed, train_frac, scaler,
                           n_in=100, n_out=1, length=5000):
    """Create sliding windows on the continuous train+val sequence."""
    set_seed(seed)
    if dataset.lower() in PYRECO_DATASETS:
        result = pyreco_load(
            dataset, length, seed=seed, train_fraction=train_frac,
            n_in=n_in, n_out=n_out, standardize=False,
        )
        X_raw, Y_raw = result[0], result[1]
    else:
        data, _ = load_data(dataset, length=length, seed=seed)
        n_trainval = int(len(data) * train_frac)
        X_raw, Y_raw = _sliding_window(data[:n_trainval], n_in, n_out)

    N, win, D = X_raw.shape
    X_tv = scaler.transform(X_raw.reshape(-1, D)).reshape(N, win, D)
    _, out, D2 = Y_raw.shape
    Y_tv = scaler.transform(Y_raw.reshape(-1, D2)).reshape(Y_raw.shape)
    return X_tv, Y_tv
DEFAULT_HORIZONS = [1, 5, 10, 20, 50]
TEST_STRIDE = 5  # Subsample test windows to reduce redundant computation
DATASETS = ['lorenz', 'mackeyglass', 'santafe']
TRAIN_FRACS = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
SEEDS = [42, 43, 44, 45, 46]
BUDGETS = ['small', 'medium', 'large']


def load_best_configs(results_dir="results/final_v2"):
    """Load best config per (dataset, budget, train_frac, seed) for both models.

    Each V2 JSON file contains per-seed best configs for pyreco_standard and lstm.
    Returns dict keyed by (dataset, budget, train_frac, seed, model_type) -> config.
    """
    if not os.path.isabs(results_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, results_dir)

    files = glob.glob(f"{results_dir}/results_*.json")
    if not files:
        raise FileNotFoundError(f"No V2 result files in {results_dir}")

    configs = {}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        meta = d['metadata']
        budget_name = meta['budget_name']
        for entry in d['results'].get(budget_name, []):
            key = (meta['dataset'], budget_name, meta['train_frac'],
                   meta['seed'], entry['model_type'])
            configs[key] = entry['config']

    # Count per model type
    pyreco_count = sum(1 for k in configs if k[4] == 'pyreco_standard')
    lstm_count = sum(1 for k in configs if k[4] == 'lstm')
    print(f"Loaded configs from {results_dir}:")
    print(f"  PyReCo: {pyreco_count}, LSTM: {lstm_count}")

    return configs


def run_single_experiment(dataset, train_frac, budget, seed,
                          pyreco_config, lstm_config,
                          horizons, length=5000, n_in=100,
                          output_dir="results/multi_step_v2"):
    """Run multi-step experiment for both models with per-seed best configs."""

    max_horizon = max(horizons)

    print(f"\n{'='*70}")
    print(f"MULTI-STEP: {dataset} | {budget} | tf={train_frac} | seed={seed}")
    print(f"{'='*70}")

    # Load data
    set_seed(seed)
    load_func = pyreco_load if dataset.lower() in PYRECO_DATASETS else local_load

    # Single-step data for training
    X_train, y_train, X_val, y_val, X_test_1, y_test_1, scaler = load_func(
        dataset, n_samples=length, seed=seed,
        val_fraction=0.15, train_fraction=train_frac,
        n_in=n_in, n_out=1, standardize=True,
    )

    # Multi-step targets for testing
    _, _, _, _, X_test_ms, y_test_ms, _ = load_func(
        dataset, n_samples=length, seed=seed,
        val_fraction=0.15, train_fraction=train_frac,
        n_in=n_in, n_out=max_horizon, standardize=True,
    )

    X_full, y_full = _make_trainval_windows(
        dataset, seed, train_frac, scaler, n_in=n_in, n_out=1, length=length)

    # Subsample test windows: stride-1 sliding windows are highly redundant
    # (adjacent windows share 99/100 input steps and 49/50 target steps).
    # Stride=5 reduces computation ~5x with negligible metric difference
    # (empirically verified: R² changes < 1e-5, MSE changes < 5%).
    n_before = len(X_test_ms)
    X_test_ms = X_test_ms[::TEST_STRIDE]
    y_test_ms = y_test_ms[::TEST_STRIDE]
    n_after = len(X_test_ms)

    print(f"  Train: {X_full.shape}, Test: {n_before}->{n_after} windows (stride={TEST_STRIDE})")

    results = {
        'metadata': {
            'dataset': dataset,
            'train_frac': train_frac,
            'budget': budget,
            'seed': seed,
            'n_in': n_in,
            'length': length,
            'horizons': horizons,
            'test_stride': TEST_STRIDE,
            'n_test_windows': n_after,
            'version': 'v2',
            'timestamp': datetime.now().isoformat(),
        },
        'models': {}
    }

    # === Train & evaluate PyReCo ===
    print(f"  PyReCo: N={pyreco_config['num_nodes']} "
          f"d={pyreco_config['density']} fi={pyreco_config['fraction_input']} "
          f"sr={pyreco_config['spec_rad']} leak={pyreco_config['leakage_rate']}")

    try:
        set_seed(seed)
        model = PyReCoStandardModel(**pyreco_config, verbose=False)
        t0 = time.time()
        model.fit(X_full, y_full)
        train_time = time.time() - t0

        t0 = time.time()
        horizon_results = evaluate_multi_step(
            model=model, X_test=X_test_ms, y_test=y_test_ms,
            horizons=horizons, mode='free_run',
            include_advanced_metrics=False,
        )
        eval_time = time.time() - t0

        results['models']['pyreco_standard'] = {
            'config': pyreco_config,
            'train_time': train_time,
            'eval_time': eval_time,
            'horizon_results': {
                h: {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r.items()}
                for h, r in horizon_results.items()
            }
        }

        for h in [1, 10, max_horizon]:
            if h in horizon_results:
                r = horizon_results[h]
                print(f"    PyReCo h={h}: MSE={r['mse']:.6f}, R2={r['r2']:.4f}")

        del model
        gc.collect()

    except Exception as e:
        print(f"  PyReCo FAILED: {e}")
        results['models']['pyreco_standard'] = {'error': str(e)}

    # === Train & evaluate LSTM ===
    print(f"  LSTM: h={lstm_config['hidden_size']} "
          f"lr={lstm_config['learning_rate']} layers={lstm_config['num_layers']} "
          f"drop={lstm_config['dropout']}")

    try:
        full_lstm_config = {
            **lstm_config,
            'epochs': 100,
            'batch_size': 32,
            'patience': 20,
        }

        set_seed(seed)
        model = LSTMModel(**full_lstm_config, verbose=False)
        t0 = time.time()
        model.fit(X_full, y_full)
        train_time = time.time() - t0

        t0 = time.time()
        horizon_results = evaluate_multi_step(
            model=model, X_test=X_test_ms, y_test=y_test_ms,
            horizons=horizons, mode='free_run',
            include_advanced_metrics=False,
        )
        eval_time = time.time() - t0

        results['models']['lstm'] = {
            'config': lstm_config,
            'train_time': train_time,
            'eval_time': eval_time,
            'horizon_results': {
                h: {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r.items()}
                for h, r in horizon_results.items()
            }
        }

        for h in [1, 10, max_horizon]:
            if h in horizon_results:
                r = horizon_results[h]
                print(f"    LSTM   h={h}: MSE={r['mse']:.6f}, R2={r['r2']:.4f}")

        del model
        gc.collect()

    except Exception as e:
        print(f"  LSTM FAILED: {e}")
        results['models']['lstm'] = {'error': str(e)}

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname = f"multistep_{dataset}_seed{seed}_train{train_frac}_{budget}.json"
    outpath = os.path.join(output_dir, fname)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {outpath}")
    return results


def scan_completed(output_dir):
    """Scan for already-completed experiments."""
    completed = set()
    for fp in glob.glob(os.path.join(output_dir, "multistep_*.json")):
        try:
            d = json.load(open(fp))
            m = d['metadata']
            # Only count as complete if both models succeeded
            models = d.get('models', {})
            pyreco_ok = 'error' not in models.get('pyreco_standard', {'error': True})
            lstm_ok = 'error' not in models.get('lstm', {'error': True})
            if pyreco_ok and lstm_ok:
                completed.add((m['dataset'], m['budget'], m['train_frac'], m['seed']))
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def main():
    parser = argparse.ArgumentParser(description='Multi-step prediction V2')
    parser.add_argument('--dataset', type=str, default=None, choices=DATASETS)
    parser.add_argument('--budget', type=str, default=None, choices=BUDGETS)
    parser.add_argument('--seed', type=int, default=None, choices=SEEDS)
    parser.add_argument('--output-dir', type=str, default='results/multi_step_v2')
    parser.add_argument('--results-dir', type=str, default='results/final_v2',
                        help='V2 main results dir (for per-seed best configs)')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Run single quick test (lorenz, small, tf=0.7, seed=42)')
    parser.add_argument('--horizons', type=str, default='1,5,10,20,50')
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(',')]

    # Load per-seed best configs for both models
    configs = load_best_configs(args.results_dir)

    # Determine scope
    if args.quick:
        datasets = ['lorenz']
        budgets = ['small']
        train_fracs = [0.7]
        seeds = [42]
    else:
        datasets = [args.dataset] if args.dataset else DATASETS
        budgets = [args.budget] if args.budget else BUDGETS
        train_fracs = TRAIN_FRACS
        seeds = [args.seed] if args.seed else SEEDS

    # Resolve output dir
    if not os.path.isabs(args.output_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir

    # Resume
    completed = set()
    if not args.no_resume and not args.quick:
        completed = scan_completed(output_dir)
        if completed:
            print(f"Resume: {len(completed)} already done")

    total = len(datasets) * len(budgets) * len(train_fracs) * len(seeds)
    print(f"\nTotal experiments: {total}")

    exp_num = 0
    skipped = 0
    failed = 0
    for dataset in datasets:
        for budget in budgets:
            for train_frac in train_fracs:
                for seed in seeds:
                    exp_num += 1
                    key_tuple = (dataset, budget, train_frac, seed)

                    if key_tuple in completed:
                        skipped += 1
                        continue

                    # Get per-seed configs
                    pyreco_key = (dataset, budget, train_frac, seed, 'pyreco_standard')
                    lstm_key = (dataset, budget, train_frac, seed, 'lstm')

                    if pyreco_key not in configs:
                        print(f"\n[{exp_num}/{total}] SKIP: no PyReCo config for {key_tuple}")
                        failed += 1
                        continue
                    if lstm_key not in configs:
                        print(f"\n[{exp_num}/{total}] SKIP: no LSTM config for {key_tuple}")
                        failed += 1
                        continue

                    print(f"\n[{exp_num}/{total}]", end="")
                    run_single_experiment(
                        dataset=dataset,
                        train_frac=train_frac,
                        budget=budget,
                        seed=seed,
                        pyreco_config=configs[pyreco_key],
                        lstm_config=configs[lstm_key],
                        horizons=horizons,
                        output_dir=output_dir,
                    )

    print(f"\n{'='*70}")
    print(f"MULTI-STEP V2 COMPLETE: {exp_num} total, {skipped} skipped, {failed} failed")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
