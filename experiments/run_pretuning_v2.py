#!/usr/bin/env python3
"""
Pre-tuning Phase (v2): Budget-Constrained Hyperparameter Search

For each (dataset, budget) combination, performs grid search with 5-fold
forward-chaining time-series CV. N (num_nodes) is dynamically computed
for each (density, frac_in) to satisfy the total parameter budget.

Usage:
    python experiments/run_pretuning_v2.py
    python experiments/run_pretuning_v2.py --dataset lorenz --budget medium
    python experiments/run_pretuning_v2.py --quick  # 1 dataset, 1 budget, 3-fold
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
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.load_dataset import load as local_load, set_seed, load_data
from src.utils.budget_matching import esn_budget_grid, esn_total_params
from models.pyreco_wrapper import PyReCoStandardModel
from pyreco.datasets import load as pyreco_load

# Lorenz/MG use pyreco_load (pyreco's own integrators), Santa Fe uses local_load
PYRECO_DATASETS = {'lorenz', 'mackeyglass'}


# ============================================================================
# Configuration
# ============================================================================

BUDGETS = {
    'small': 1000,
    'medium': 10000,
    'large': 50000,
}

DATASETS = ['lorenz', 'mackeyglass', 'santafe']

# Structural parameters (affect N via budget constraint)
DENSITY_VALUES = {
    'small':  [0.01, 0.03, 0.05, 0.1],
    'medium': [0.01, 0.03, 0.05, 0.1],  # 0.01→0.02 (N=937 too slow); verify monotonicity, test 0.01 in final
    'large':  [0.05, 0.1],  # delta < 0.05 excluded (N > 1000)
}

FRAC_IN_VALUES = [0.1, 0.3, 0.5, 0.7, 0.8]

# Dynamics parameters (do NOT affect N or total params)
# Extended range: with budget-constrained larger N, optimal spec_rad
# shifts lower than the 0.99 found with old fixed small N.
SPEC_RAD_VALUES = [0.5, 0.7, 0.8, 0.9, 0.99]
LEAKAGE_VALUES = [0.1, 0.3, 0.5, 0.7, 0.8, 1.0]

MAX_NODES = 1000
CV_FOLDS = 5
SEED = 42
TRAIN_FRAC = 0.7


# ============================================================================
# Time-series cross-validation (forward-chaining, on raw time series)
# ============================================================================

def timeseries_cv_split_raw(raw_data, n_splits=5, n_in=100, n_out=1):
    """
    Forward-chaining time-series CV on RAW time series.
    Splits the raw data first, then creates sliding windows independently
    per fold to avoid data leakage from overlapping windows.

    Fold k: train on raw[0 : (k+1)*chunk], validate on raw[(k+1)*chunk : (k+2)*chunk].
    Each fold independently standardizes and creates sliding windows.

    Returns list of (X_train, y_train, X_val, y_val) tuples.
    """
    from sklearn.preprocessing import StandardScaler

    n_total = len(raw_data)
    chunk_size = n_total // (n_splits + 1)
    min_len = n_in + n_out

    splits = []
    for k in range(n_splits):
        train_end = (k + 1) * chunk_size
        val_end = min(train_end + chunk_size, n_total)

        if train_end < min_len or val_end - train_end < min_len:
            continue

        # Split raw data
        raw_train = raw_data[:train_end]
        raw_val = raw_data[train_end:val_end]

        # Standardize: fit on train, transform both
        scaler = StandardScaler()
        raw_train = scaler.fit_transform(raw_train)
        raw_val = scaler.transform(raw_val)

        # Create sliding windows independently
        X_train, y_train = _sliding_window(raw_train, n_in, n_out)
        X_val, y_val = _sliding_window(raw_val, n_in, n_out)

        if len(X_train) > 0 and len(X_val) > 0:
            splits.append((X_train, y_train, X_val, y_val))

    return splits


def _sliding_window(data, n_in, n_out):
    """Create sliding windows from a time series array."""
    T = len(data)
    D = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    N = T - n_in - n_out + 1
    if N <= 0:
        return np.array([]), np.array([])

    X = np.stack([data[i:i+n_in] for i in range(N)])
    Y = np.stack([data[i+n_in:i+n_in+n_out] for i in range(N)])
    return X, Y


# ============================================================================
# Single configuration evaluation
# ============================================================================

def evaluate_config(config, X_train, y_train, X_val, y_val, d_out,
                    timeout_seconds=600):
    """Train a PyReCo model and return validation MSE. Timeout protection."""
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("Training timeout")

    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_seconds)

        # Reset seed before each config for reproducibility
        set_seed(42)

        t0 = time.time()
        model = PyReCoStandardModel(
            num_nodes=config['num_nodes'],
            density=config['density'],
            activation='tanh',
            spec_rad=config['spec_rad'],
            leakage_rate=config['leakage_rate'],
            fraction_input=config['fraction_input'],
            fraction_output=1.0,
            optimizer='ridge',
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        signal.alarm(0)  # Cancel timeout
        signal.signal(signal.SIGALRM, old_handler)

        # Compute MSE
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        y_pred = np.asarray(y_pred).reshape(y_val.shape)
        mse = float(np.mean((y_val - y_pred) ** 2))
        train_time = time.time() - t0

        del model
        gc.collect()
        return mse, train_time

    except TimeoutError:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except:
            pass
        gc.collect()
        print(f"    WARNING: TIMEOUT ({timeout_seconds}s) for N={config['num_nodes']} "
              f"d={config['density']} sr={config['spec_rad']} leak={config['leakage_rate']}",
              flush=True)
        return float('inf'), timeout_seconds
    except Exception as e:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except:
            pass
        gc.collect()
        print(f"    WARNING: ERROR for N={config['num_nodes']}: {e}", flush=True)
        return float('inf'), 0


# ============================================================================
# Pretuning for one (dataset, budget)
# ============================================================================

def run_pretuning(dataset: str, budget_name: str, n_folds: int = CV_FOLDS,
                  output_dir: str = 'results/pretuning_v2', verbose: bool = True,
                  quick: bool = False):
    """
    Run budget-constrained pretuning for a single (dataset, budget).
    """
    budget = BUDGETS[budget_name]
    d_in = 3 if dataset == 'lorenz' else 1
    d_out = d_in

    print(f"\n{'='*70}")
    print(f"PRETUNING: {dataset} | {budget_name} ({budget:,} params) | "
          f"{n_folds}-fold CV | seed={SEED}")
    print(f"{'='*70}")

    # Load RAW time series (not windowed) for proper CV splitting
    set_seed(SEED)
    if dataset in PYRECO_DATASETS:
        from pyreco.datasets import _generate_lorenz, _generate_mackey_glass
        if dataset == 'lorenz':
            raw_data = _generate_lorenz(5000)
        else:
            raw_data = _generate_mackey_glass(5000, seed=SEED)
        print(f"  Data source: pyreco integrator")
    else:
        raw_data, meta = load_data(dataset, length=5000, seed=SEED)
        print(f"  Data source: local_load")

    # Use train portion (exclude test) for CV
    # Matching pyreco_load: train_fraction=0.7 means 70% for train+val, 30% for test
    n_total = len(raw_data)
    n_trainval = int(n_total * TRAIN_FRAC)
    raw_trainval = raw_data[:n_trainval]

    print(f"  Raw data: {raw_data.shape}, using {n_trainval} for CV (test={n_total - n_trainval})")

    # Generate budget-constrained structural grid
    density_values = DENSITY_VALUES[budget_name]
    frac_in_values = FRAC_IN_VALUES
    spec_rad_values = SPEC_RAD_VALUES
    leakage_values = LEAKAGE_VALUES

    if quick:
        # Reduced grid for smoke testing (~30 combos instead of 600)
        density_values = [0.01, 0.1]
        frac_in_values = [0.1, 0.5]
        spec_rad_values = [0.8, 0.99]
        leakage_values = [0.5, 1.0]

    structural_grid = esn_budget_grid(
        budget=budget,
        density_values=density_values,
        frac_in_values=frac_in_values,
        d_in=d_in, d_out=d_out,
        max_nodes=MAX_NODES,
    )

    if not structural_grid:
        print("ERROR: No feasible structural configurations!")
        return None

    print(f"\nStructural configs (delta x frac_in): {len(structural_grid)}")
    for sg in structural_grid:
        pi = sg['param_info']
        print(f"  d={sg['density']:.2f} fi={sg['fraction_input']:.1f} "
              f"N={sg['num_nodes']:>5d} trainable={pi['trainable']:>5d} "
              f"total={pi['total']:>6d}")

    dynamics_combos = list(product(spec_rad_values, leakage_values))
    total_combos = len(structural_grid) * len(dynamics_combos)
    print(f"Dynamics combos (sr x leak): {len(dynamics_combos)}")
    print(f"Total combos: {total_combos}")
    print(f"Total training runs: {total_combos * n_folds}")

    # Create CV splits on RAW time series (no window overlap leakage)
    cv_splits = timeseries_cv_split_raw(raw_trainval, n_splits=n_folds, n_in=100, n_out=1)
    print(f"CV folds: {len(cv_splits)}")
    for i, (Xt, yt, Xv, yv) in enumerate(cv_splits):
        print(f"  Fold {i+1}: train={len(Xt)} windows, val={len(Xv)} windows")

    # Grid search
    results = []
    start_time = time.time()

    for idx, (sg, (sr, leak)) in enumerate(product(structural_grid, dynamics_combos)):
        config = {
            'num_nodes': sg['num_nodes'],
            'density': sg['density'],
            'fraction_input': sg['fraction_input'],
            'fraction_output': 1.0,
            'spec_rad': sr,
            'leakage_rate': leak,
        }

        # Evaluate across CV folds
        fold_mses = []
        fold_times = []
        for Xt, yt, Xv, yv in cv_splits:
            mse, t = evaluate_config(config, Xt, yt, Xv, yv, d_out)
            fold_mses.append(mse)
            fold_times.append(t)

        mean_mse = np.mean(fold_mses)
        std_mse = np.std(fold_mses)
        mean_time = np.mean(fold_times)

        results.append({
            'config': config,
            'param_info': sg['param_info'],
            'cv_mean_mse': float(mean_mse),
            'cv_std_mse': float(std_mse),
            'cv_mean_time': float(mean_time),
            'fold_mses': [float(m) for m in fold_mses],
        })

        if verbose and (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            pct = (idx + 1) / total_combos * 100
            eta = elapsed / (idx + 1) * (total_combos - idx - 1)
            best_so_far = min(r['cv_mean_mse'] for r in results)
            print(f"  [{idx+1}/{total_combos}] ({pct:.0f}%) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s "
                  f"best_mse={best_so_far:.6f} "
                  f"current: d={config['density']} N={config['num_nodes']} "
                  f"t={mean_time:.1f}s", flush=True)

    total_time = time.time() - start_time

    # Sort by CV mean MSE
    results.sort(key=lambda r: r['cv_mean_mse'])

    # Print top 10
    print(f"\nTop 10 configurations (total time: {total_time:.0f}s):")
    print(f"{'Rank':>4s} {'d':>5s} {'fi':>4s} {'N':>5s} {'sr':>5s} "
          f"{'leak':>5s} {'MSE_mean':>10s} {'MSE_std':>10s} {'total':>6s}")
    print("-" * 65)
    for i, r in enumerate(results[:10]):
        c = r['config']
        pi = r['param_info']
        print(f"{i+1:>4d} {c['density']:>5.2f} {c['fraction_input']:>4.1f} "
              f"{c['num_nodes']:>5d} {c['spec_rad']:>5.2f} "
              f"{c['leakage_rate']:>5.2f} {r['cv_mean_mse']:>10.6f} "
              f"{r['cv_std_mse']:>10.6f} {pi['total']:>6d}")

    # Filter out failed configs (inf/nan) before saving
    valid_results = [r for r in results if np.isfinite(r['cv_mean_mse'])]
    failed_count = len(results) - len(valid_results)
    if failed_count > 0:
        print(f"\n  Filtered {failed_count} failed configs (timeout/error)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'dataset': dataset,
            'budget_name': budget_name,
            'budget': budget,
            'seed': SEED,
            'train_frac': TRAIN_FRAC,
            'n_folds': n_folds,
            'max_nodes': MAX_NODES,
            'd_in': d_in,
            'd_out': d_out,
            'total_combos': total_combos,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
        },
        'grid_config': {
            'density_values': density_values,
            'frac_in_values': FRAC_IN_VALUES,
            'spec_rad_values': SPEC_RAD_VALUES,
            'leakage_values': LEAKAGE_VALUES,
        },
        'results': valid_results,
        'failed_count': failed_count,
        'best_config': valid_results[0]['config'] if valid_results else None,
        'best_param_info': valid_results[0]['param_info'] if valid_results else None,
        'best_cv_mse': valid_results[0]['cv_mean_mse'] if valid_results else None,
    }

    filename = f"pretuning_v2_{dataset}_{budget_name}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {output_path / filename}")
    return output


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Budget-constrained pretuning v2')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=DATASETS, help='Single dataset (default: all)')
    parser.add_argument('--budget', type=str, default=None,
                        choices=list(BUDGETS.keys()), help='Single budget (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: lorenz + small + 3-fold')
    parser.add_argument('--output-dir', type=str, default='results/pretuning_v2')
    parser.add_argument('--folds', type=int, default=CV_FOLDS)
    args = parser.parse_args()

    if args.quick:
        datasets = ['lorenz']
        budgets = ['small']
        n_folds = 3
        quick_mode = True
    else:
        datasets = [args.dataset] if args.dataset else DATASETS
        budgets = [args.budget] if args.budget else list(BUDGETS.keys())
        n_folds = args.folds
        quick_mode = False

    total_groups = len(datasets) * len(budgets)
    print(f"Pretuning v2: {len(datasets)} datasets x {len(budgets)} budgets "
          f"= {total_groups} groups")
    print(f"CV folds: {n_folds}")

    all_results = {}
    for i, dataset in enumerate(datasets):
        for j, budget_name in enumerate(budgets):
            group_num = i * len(budgets) + j + 1
            print(f"\n[{group_num}/{total_groups}]")
            result = run_pretuning(dataset, budget_name, n_folds=n_folds,
                                   output_dir=args.output_dir, quick=quick_mode)
            if result:
                all_results[(dataset, budget_name)] = result

    # Summary
    print(f"\n\n{'='*70}")
    print("PRETUNING SUMMARY")
    print(f"{'='*70}")
    for (ds, bn), r in all_results.items():
        bc = r['best_config']
        pi = r['best_param_info']
        print(f"  {ds:15s} {bn:8s}: d={bc['density']:.2f} fi={bc['fraction_input']:.1f} "
              f"N={bc['num_nodes']} sr={bc['spec_rad']:.2f} leak={bc['leakage_rate']:.2f} "
              f"total={pi['total']} MSE={r['best_cv_mse']:.6f}")


if __name__ == '__main__':
    main()
