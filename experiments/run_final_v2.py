#!/usr/bin/env python3
"""
Final Experiments v2: Budget-Constrained Model Comparison

Runs PyReCo vs LSTM across all datasets, budgets, train_fracs, and seeds.
ESN grid: all (δ, f_in, sr, leak) as arrays; N computed dynamically from budget.
LSTM grid: (lr, dropout, num_layers); hidden_size computed from budget.

Usage:
    python experiments/run_final_v2.py
    python experiments/run_final_v2.py --dataset lorenz --budget small
    python experiments/run_final_v2.py --dataset lorenz --seed 42 --device cpu
    python experiments/run_final_v2.py --quick
"""

import argparse
import json
import time
import sys
import os
import gc
import tracemalloc
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.load_dataset import load as local_load, set_seed
from src.utils.budget_matching import (
    esn_solve_num_nodes, esn_total_params,
    lstm_solve_hidden_size, lstm_total_params, lstm_layer_hidden_map,
)
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel, tune_lstm_hyperparameters
from pyreco.datasets import load as pyreco_load

# Lorenz/MG use pyreco_load (pyreco's own integrators), Santa Fe uses local_load
PYRECO_DATASETS = {'lorenz', 'mackeyglass'}

# Optional: CodeCarbon for energy/carbon tracking
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

MAX_NODES = 1000


class GreenMetricsTracker:
    """Track green computing metrics: memory usage and energy consumption."""

    def __init__(self, track_carbon=False, country_iso_code="DEU"):
        self.track_carbon = track_carbon and CODECARBON_AVAILABLE
        self.country_iso_code = country_iso_code
        self.carbon_tracker = None
        self.memory_start = 0

    def start(self):
        gc.collect()
        tracemalloc.start()
        self.memory_start = tracemalloc.get_traced_memory()[0]
        if self.track_carbon:
            self.carbon_tracker = OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code, log_level='error')
            self.carbon_tracker.start()

    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = {
            'memory_current_mb': current / 1024 / 1024,
            'memory_peak_mb': peak / 1024 / 1024,
            'memory_delta_mb': (current - self.memory_start) / 1024 / 1024,
        }
        if self.track_carbon and self.carbon_tracker:
            emissions = self.carbon_tracker.stop()
            if emissions is not None:
                metrics['emissions_kg_co2'] = float(emissions)
                metrics['energy_kwh'] = float(
                    self.carbon_tracker._total_energy.kWh
                ) if hasattr(self.carbon_tracker, '_total_energy') else None
            else:
                metrics['emissions_kg_co2'] = 0.0
                metrics['energy_kwh'] = 0.0
        return metrics


# ============================================================================
# Configuration
# ============================================================================

BUDGETS = {'small': 1000, 'medium': 10000, 'large': 50000}
DATASETS = ['lorenz', 'mackeyglass', 'santafe']
TRAIN_FRACS = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
SEEDS = [42, 43, 44, 45, 46]

# LSTM tuning grid (does not affect total params)
LSTM_PARAM_GRID = {
    'learning_rate': [0.0005, 0.001, 0.005, 0.01],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2],
}

# ESN param grid per dataset (narrowed from pretuning monotonicity analysis)
# N is computed dynamically from (density, fraction_input, budget).
# Infeasible combos (N > MAX_NODES) are auto-skipped.
#
# Lorenz (3D): NO monotonicity on any param → full search needed
# MG (1D):     ALL 4 params monotonic → minimal grid sufficient
# SF (1D):     δ↓ leak↑ monotonic, fi/sr partially → moderate grid
ESN_PARAM_GRID = {
    'lorenz': {
        # δ: NO mono (reverses at large), fi: NO mono, sr: NO mono, leak: NO mono
        'density':        [0.01, 0.05, 0.1],
        'fraction_input': [0.1, 0.3, 0.5],
        'spec_rad':       [0.7, 0.8, 0.9],
        'leakage_rate':   [0.3, 0.5, 0.7, 0.8],
    },
    'mackeyglass': {
        # δ: MONO ↓ (98-100%), fi: MONO ↓ (90-93%), sr: MONO ↑ (100%), leak: MONO ↑ (100%)
        'density':        [0.01, 0.05],
        'fraction_input': [0.1, 0.3],
        'spec_rad':       [0.9, 0.99],
        'leakage_rate':   [0.8, 1.0],
    },
    'santafe': {
        # δ: MONO ↓ (70-92%), fi: NO to mostly ↓, sr: NO to MONO ↑, leak: MONO ↑ (97-100%)
        'density':        [0.01, 0.05],
        'fraction_input': [0.1, 0.3, 0.5],
        'spec_rad':       [0.7, 0.9, 0.99],
        'leakage_rate':   [0.8, 1.0],
    },
}


# ============================================================================
# Single experiment
# ============================================================================

def run_single_experiment(dataset: str, budget_name: str, train_frac: float,
                          seed: int, esn_grid: dict,
                          output_dir: str = 'results/final_v2',
                          lstm_device: str = 'auto',
                          skip_lstm: bool = False,
                          verbose: bool = True):
    """Run one experiment: ESN grid search + LSTM tuning at one (dataset, budget, tf, seed)."""

    budget = BUDGETS[budget_name]
    d_in = 3 if dataset == 'lorenz' else 1
    d_out = d_in

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {dataset} | {budget_name} ({budget:,}) | "
          f"tf={train_frac} | seed={seed}")
    print(f"{'='*70}")

    # Load data: pyreco_load for Lorenz/MG, local_load for Santa Fe
    set_seed(seed)
    load_func = pyreco_load if dataset in PYRECO_DATASETS else local_load
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_func(
        dataset, n_samples=5000, seed=seed,
        train_fraction=train_frac, val_fraction=0.15,
        n_in=100, n_out=1, standardize=True,
    )

    results = []

    # === ESN grid search ===
    best_esn_mse = float('inf')
    best_esn_result = None
    esn_combos_tried = 0
    esn_combos_skipped = 0

    for density, frac_in, sr, leak in product(
        esn_grid['density'], esn_grid['fraction_input'],
        esn_grid['spec_rad'], esn_grid['leakage_rate'],
    ):
        # Compute N from budget constraint
        N = esn_solve_num_nodes(budget, density, frac_in, d_in, d_out,
                                max_nodes=MAX_NODES)
        if N is None:
            esn_combos_skipped += 1
            continue

        param_info = esn_total_params(N, density, frac_in, d_in, d_out)
        esn_combos_tried += 1

        try:
            set_seed(seed)
            model = PyReCoStandardModel(
                num_nodes=N,
                density=density,
                activation='tanh',
                spec_rad=sr,
                leakage_rate=leak,
                fraction_input=frac_in,
                fraction_output=1.0,
                optimizer='ridge',
            )
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_pred = np.asarray(val_pred).reshape(y_val.shape)

            if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
                if verbose:
                    print(f"    ESN NaN/Inf (d={density} N={N} sr={sr} leak={leak}), skipping")
                del model
                gc.collect()
                continue

            val_mse = float(np.mean((y_val - val_pred) ** 2))

            if val_mse < best_esn_mse:
                best_esn_mse = val_mse
                best_esn_result = {
                    'config': {
                        'num_nodes': N,
                        'density': density,
                        'fraction_input': frac_in,
                        'fraction_output': 1.0,
                        'spec_rad': sr,
                        'leakage_rate': leak,
                    },
                    'param_info': param_info,
                    'val_mse': val_mse,
                }

            del model
            gc.collect()

        except Exception as e:
            if verbose:
                print(f"    ESN error (d={density} N={N} sr={sr} leak={leak}): {e}")

    if verbose:
        print(f"\n  ESN grid: {esn_combos_tried} tried, {esn_combos_skipped} infeasible")

    # Train final ESN on train+val with best config
    if best_esn_result:
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        green = GreenMetricsTracker()
        green.start()

        set_seed(seed)
        bc = best_esn_result['config']
        final_model = PyReCoStandardModel(
            num_nodes=bc['num_nodes'], density=bc['density'],
            activation='tanh', spec_rad=bc['spec_rad'],
            leakage_rate=bc['leakage_rate'],
            fraction_input=bc['fraction_input'],
            fraction_output=1.0, optimizer='ridge',
        )
        t0 = time.time()
        final_model.fit(X_full, y_full)
        final_train_time = time.time() - t0

        green_metrics = green.stop()

        # Test evaluation
        test_pred = final_model.predict(X_test)
        test_pred = np.asarray(test_pred).reshape(y_test.shape)
        test_mse = float(np.mean((y_test - test_pred) ** 2))
        ss_res = np.sum((y_test - test_pred)**2)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0))**2)
        test_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Inference time
        _ = final_model.predict(X_test[:10])  # warmup
        t0 = time.time()
        _ = final_model.predict(X_test)
        inference_time = time.time() - t0
        per_sample_ms = inference_time / len(X_test) * 1000

        results.append({
            'model_type': 'pyreco_standard',
            'config': bc,
            'param_info': best_esn_result['param_info'],
            'test_mse': test_mse,
            'test_r2': test_r2,
            'final_train_time': final_train_time,
            'inference_time_per_sample_ms': per_sample_ms,
            'inference_time_total': inference_time,
            'n_test_samples': len(X_test),
            'val_mse': best_esn_result['val_mse'],
            'esn_combos_tried': esn_combos_tried,
            'green_metrics': green_metrics,
        })

        if verbose:
            print(f"\n  ESN best: d={bc['density']} fi={bc['fraction_input']} "
                  f"N={bc['num_nodes']} sr={bc['spec_rad']} leak={bc['leakage_rate']}")
            print(f"    total={best_esn_result['param_info']['total']} "
                  f"trainable={best_esn_result['param_info']['trainable']}")
            print(f"    test R²={test_r2:.6f} MSE={test_mse:.6f} "
                  f"train_time={final_train_time:.2f}s")

        del final_model
        gc.collect()

    # === LSTM ===
    if not skip_lstm:
        layer_map = lstm_layer_hidden_map(budget, d_in, d_out, [1, 2])

        if lstm_device == 'auto':
            lstm_device = 'cpu'
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    lstm_device = 'mps'
                elif torch.cuda.is_available():
                    lstm_device = 'cuda'
            except Exception:
                pass

        set_seed(seed)
        tune_results = tune_lstm_hyperparameters(
            X_train, y_train, X_val, y_val,
            param_grid=LSTM_PARAM_GRID,
            lstm_device=lstm_device,
            verbose=verbose,
            layer_hidden_map=layer_map,
        )

        best_lstm_params = tune_results['best_params']
        best_lstm_val = tune_results['best_score']

        # Find best_epoch from tuning (fix B1)
        best_epoch = None
        for r in tune_results.get('all_results', []):
            if r['params'] == best_lstm_params:
                best_epoch = r.get('best_epoch', None)
                break

        del tune_results
        gc.collect()

        # Train final LSTM on train+val
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        green = GreenMetricsTracker()
        green.start()

        set_seed(seed)
        final_lstm = LSTMModel(**best_lstm_params, device=lstm_device, verbose=verbose)

        # Fix B1: if we know the best epoch, use it as max_epochs
        if best_epoch and best_epoch > 0:
            final_lstm.epochs = best_epoch

        t0 = time.time()
        final_lstm.fit(X_full, y_full)
        lstm_train_time = time.time() - t0

        green_metrics = green.stop()

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

        # Param info from actual config
        h = best_lstm_params['hidden_size']
        nl = best_lstm_params.get('num_layers', 1)
        lstm_pi = lstm_total_params(h, d_in, d_out, nl)

        results.append({
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
            'final_train_time': lstm_train_time,
            'inference_time_per_sample_ms': per_sample_ms,
            'inference_time_total': inference_time,
            'n_test_samples': len(X_test),
            'val_mse': best_lstm_val,
            'best_epoch': best_epoch,
            'green_metrics': green_metrics,
        })

        if verbose:
            print(f"\n  LSTM best: h={h} L={nl} lr={best_lstm_params.get('learning_rate')} "
                  f"dropout={best_lstm_params.get('dropout')}")
            print(f"    total={lstm_pi['total']} (all trainable)")
            print(f"    test R²={test_r2:.6f} MSE={test_mse:.6f} "
                  f"train_time={lstm_train_time:.2f}s")

        del final_lstm
        gc.collect()
    else:
        if verbose:
            print("\n  LSTM: SKIPPED (--skip-lstm, reusing old results)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'dataset': dataset,
            'budget_name': budget_name,
            'budget': budget,
            'seed': seed,
            'train_frac': train_frac,
            'd_in': d_in,
            'd_out': d_out,
            'timestamp': datetime.now().isoformat(),
        },
        'results': {budget_name: results},
    }

    filename = f"results_{dataset}_{budget_name}_seed{seed}_train{train_frac}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"\n  Saved: {output_path / filename}")

    return output


# ============================================================================
# Main
# ============================================================================

def scan_completed_experiments(output_dir: str) -> set:
    """Scan output directory for already-completed experiments.
    Returns set of (dataset, budget_name, train_frac, seed) tuples."""
    completed = set()
    output_path = Path(output_dir)
    if not output_path.exists():
        return completed

    for f in output_path.glob('results_*.json'):
        try:
            with open(f) as fp:
                meta = json.load(fp)['metadata']
            key = (meta['dataset'], meta['budget_name'],
                   meta['train_frac'], meta['seed'])
            completed.add(key)
        except (json.JSONDecodeError, KeyError):
            continue

    return completed


def main():
    parser = argparse.ArgumentParser(description='Final experiments v2')
    parser.add_argument('--dataset', type=str, default=None, choices=DATASETS)
    parser.add_argument('--budget', type=str, default=None, choices=list(BUDGETS.keys()))
    parser.add_argument('--output-dir', type=str, default='results/final_v2')
    parser.add_argument('--seed', type=int, default=None, choices=SEEDS,
                        help='Run only this seed (for parallel execution)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='LSTM device (use cpu when running multiple processes)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 1 dataset, 1 budget, 1 seed, 1 tf')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not skip already-completed experiments')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM (reuse old results from results/final/)')
    args = parser.parse_args()

    # Determine what to run
    if args.quick:
        datasets = ['lorenz']
        budgets = ['small']
        train_fracs = [0.7]
        seeds = [42]
    else:
        datasets = [args.dataset] if args.dataset else DATASETS
        budgets = [args.budget] if args.budget else list(BUDGETS.keys())
        train_fracs = TRAIN_FRACS
        seeds = [args.seed] if args.seed else SEEDS

    # Print ESN grid info
    print("ESN param grids (N computed from budget, infeasible combos auto-skipped):")
    for ds in datasets:
        grid = ESN_PARAM_GRID[ds]
        n_total = (len(grid['density']) * len(grid['fraction_input'])
                   * len(grid['spec_rad']) * len(grid['leakage_rate']))
        print(f"  {ds:15s}: {len(grid['density'])}δ × {len(grid['fraction_input'])}fi "
              f"× {len(grid['spec_rad'])}sr × {len(grid['leakage_rate'])}leak "
              f"= {n_total} max combos")

        # Show feasible count per budget
        for bn in budgets:
            budget = BUDGETS[bn]
            d_in = 3 if ds == 'lorenz' else 1
            feasible = 0
            for density, frac_in in product(grid['density'], grid['fraction_input']):
                N = esn_solve_num_nodes(budget, density, frac_in, d_in, d_in,
                                        max_nodes=MAX_NODES)
                if N is not None:
                    feasible += 1
            n_dyn = len(grid['spec_rad']) * len(grid['leakage_rate'])
            print(f"    {bn:8s}: {feasible} feasible struct × {n_dyn} dyn = {feasible * n_dyn} combos")

    total_exp = len(datasets) * len(budgets) * len(train_fracs) * len(seeds)
    print(f"\nTotal experiments: {total_exp}")
    print(f"LSTM grid: {len(LSTM_PARAM_GRID['learning_rate'])}lr × "
          f"{len(LSTM_PARAM_GRID['dropout'])}drop × "
          f"{len(LSTM_PARAM_GRID['num_layers'])}layers = "
          f"{len(LSTM_PARAM_GRID['learning_rate']) * len(LSTM_PARAM_GRID['dropout']) * len(LSTM_PARAM_GRID['num_layers'])} combos")

    # Resume: scan for already-completed experiments
    completed = set()
    if not args.no_resume and not args.quick:
        completed = scan_completed_experiments(args.output_dir)
        if completed:
            print(f"Resume mode: {len(completed)} experiments already completed, skipping them")

    exp_num = 0
    skipped = 0
    for dataset in datasets:
        esn_grid = ESN_PARAM_GRID[dataset]

        for budget_name in budgets:
            for train_frac in train_fracs:
                for seed in seeds:
                    exp_num += 1

                    # Skip if already completed
                    exp_key = (dataset, budget_name, train_frac, seed)
                    if exp_key in completed:
                        skipped += 1
                        print(f"\n[{exp_num}/{total_exp}] SKIP (already done): "
                              f"{dataset}/{budget_name}/tf={train_frac}/s={seed}")
                        continue

                    print(f"\n[{exp_num}/{total_exp}]")

                    try:
                        run_single_experiment(
                            dataset=dataset,
                            budget_name=budget_name,
                            train_frac=train_frac,
                            seed=seed,
                            esn_grid=esn_grid,
                            output_dir=args.output_dir,
                            lstm_device=args.device,
                            skip_lstm=args.skip_lstm,
                        )
                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE: {exp_num}/{total_exp} "
          f"(skipped {skipped} already done)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
