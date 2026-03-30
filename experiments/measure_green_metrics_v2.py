"""
Green Computing Metrics Measurement for V2 Experiments

Measures energy consumption, CO2 emissions, memory usage, and timing
for each unique (dataset, budget, model_type) condition using V2 configs.

Uses CodeCarbon OfflineEmissionsTracker with country_code='DEU'.
Only CPU power is tracked (Apple MPS GPU not supported by CodeCarbon).

Output:
  - results/green_metrics_v2/green_metrics_v2.json
  - results/tables/v2/green_metrics_comparison.csv
"""

import json
import os
import sys
import time
import gc
import tracemalloc
import platform
import csv
from datetime import datetime

import numpy as np
import psutil

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from codecarbon import OfflineEmissionsTracker
import codecarbon

from src.utils.load_dataset import load as load_dataset, set_seed
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel

# --- Constants ---

DATASETS = ['lorenz', 'mackeyglass', 'santafe']
BUDGETS = ['small', 'medium', 'large']
SEED = 42
TRAIN_FRAC = 0.5
COUNTRY_ISO = 'DEU'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'green_metrics_v2')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables', 'v2')
V2_DIR = os.path.join(PROJECT_ROOT, 'results', 'final_v2')


def load_v2_config(dataset: str, budget_name: str) -> dict:
    """Load the best config for each model from V2 results (seed=42, tf=0.5)."""
    filename = f"results_{dataset}_{budget_name}_seed{SEED}_train{TRAIN_FRAC}.json"
    filepath = os.path.join(V2_DIR, filename)
    with open(filepath) as f:
        data = json.load(f)

    configs = {}
    for entry in data['results'][budget_name]:
        model_type = entry['model_type']
        configs[model_type] = {
            'config': entry['config'],
            'param_info': entry.get('param_info', {}),
        }
    return configs


def measure_pyreco(dataset: str, budget_name: str, config: dict,
                   X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Train and measure PyReCo with energy tracking."""
    gc.collect()

    model = PyReCoStandardModel(
        num_nodes=config['num_nodes'],
        density=config['density'],
        fraction_input=config['fraction_input'],
        fraction_output=config.get('fraction_output', 1.0),
        spec_rad=config['spec_rad'],
        leakage_rate=config['leakage_rate'],
        verbose=False,
    )

    # --- Energy + timing: train ---
    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO,
        log_level='error',
        save_to_file=False,
    )
    tracemalloc.start()
    tracker.start()
    t0 = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - t0
    emissions_kg = tracker.stop()
    energy_kwh = float(tracker._total_energy.kWh) if hasattr(tracker, '_total_energy') else 0.0
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Inference timing ---
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    inference_total = time.perf_counter() - t0
    n_test = X_test.shape[0]
    inference_per_sample_ms = (inference_total / n_test) * 1000

    # Compute power
    power_w = (energy_kwh * 1000 / (train_time / 3600)) if train_time > 0 and energy_kwh > 0 else 0.0

    return {
        'dataset': dataset,
        'budget': budget_name,
        'model_type': 'pyreco_standard',
        'config': config,
        'train_time_s': round(train_time, 4),
        'energy_kwh': float(energy_kwh),
        'emissions_kg_co2': float(emissions_kg) if emissions_kg else 0.0,
        'power_w': round(power_w, 2),
        'memory_peak_mb': round(mem_peak / 1024 / 1024, 2),
        'inference_time_per_sample_ms': round(inference_per_sample_ms, 4),
        'inference_time_total_s': round(inference_total, 4),
        'n_test_samples': n_test,
    }


def measure_lstm(dataset: str, budget_name: str, config: dict,
                 X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Train and measure LSTM with energy tracking. Forces CPU for codecarbon."""
    gc.collect()
    import torch
    torch.set_num_threads(os.cpu_count() or 1)

    model = LSTMModel(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        learning_rate=config['learning_rate'],
        dropout=config['dropout'],
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 32),
        patience=config.get('patience', 10),
        device='cpu',  # Force CPU so codecarbon can track all compute
        verbose=False,
    )

    process = psutil.Process()
    rss_before = process.memory_info().rss

    # --- Energy + timing: train ---
    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO,
        log_level='error',
        save_to_file=False,
    )
    tracker.start()
    t0 = time.perf_counter()

    model.fit(X_train, y_train, X_val, y_val)

    train_time = time.perf_counter() - t0
    emissions_kg = tracker.stop()
    energy_kwh = float(tracker._total_energy.kWh) if hasattr(tracker, '_total_energy') else 0.0

    rss_after = process.memory_info().rss
    mem_peak_mb = (rss_after - rss_before) / 1024 / 1024
    # Fallback: if negative (GC freed memory), use absolute RSS
    if mem_peak_mb < 0:
        mem_peak_mb = rss_after / 1024 / 1024

    # --- Inference timing ---
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    inference_total = time.perf_counter() - t0
    n_test = X_test.shape[0]
    inference_per_sample_ms = (inference_total / n_test) * 1000

    power_w = (energy_kwh * 1000 / (train_time / 3600)) if train_time > 0 and energy_kwh > 0 else 0.0

    return {
        'dataset': dataset,
        'budget': budget_name,
        'model_type': 'lstm',
        'config': config,
        'train_time_s': round(train_time, 4),
        'energy_kwh': float(energy_kwh),
        'emissions_kg_co2': float(emissions_kg) if emissions_kg else 0.0,
        'power_w': round(power_w, 2),
        'memory_peak_mb': round(mem_peak_mb, 2),
        'inference_time_per_sample_ms': round(inference_per_sample_ms, 4),
        'inference_time_total_s': round(inference_total, 4),
        'n_test_samples': n_test,
    }


def run_all():
    """Run green metrics measurement for all 18 conditions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Machine info
    metadata = {
        'machine': platform.processor() or platform.machine(),
        'cpu_count': os.cpu_count(),
        'country_code': COUNTRY_ISO,
        'carbon_intensity_kg_per_kwh': 0.338,
        'codecarbon_version': codecarbon.__version__,
        'measurement_date': datetime.now().isoformat(),
        'seed': SEED,
        'train_frac': TRAIN_FRAC,
        'measurement_note': (
            'CPU-only energy estimate via CodeCarbon (TDP * CPU utilization). '
            'Apple MPS GPU computation is not captured. '
            'LSTM forced to CPU for comparable measurement. '
            'PyReCo memory via tracemalloc; LSTM memory via psutil RSS delta.'
        ),
    }

    results = []
    total = len(DATASETS) * len(BUDGETS) * 2  # 18 conditions
    count = 0

    for dataset in DATASETS:
        for budget_name in BUDGETS:
            # Load configs from V2 results
            v2_configs = load_v2_config(dataset, budget_name)

            # Load data once for both models
            set_seed(SEED)
            data = load_dataset(
                dataset,
                n_samples=5000,
                seed=SEED,
                train_fraction=TRAIN_FRAC,
                val_fraction=0.15,
                n_in=100,
                n_out=1,
                standardize=True,
            )
            X_train, y_train, X_val, y_val, X_test, y_test = data[:6]

            # --- PyReCo ---
            count += 1
            pyreco_cfg = v2_configs['pyreco_standard']['config']
            print(f"[{count}/{total}] {dataset} {budget_name} pyreco "
                  f"(N={pyreco_cfg['num_nodes']})...", flush=True)
            result = measure_pyreco(
                dataset, budget_name, pyreco_cfg,
                X_train, y_train, X_val, y_val, X_test, y_test,
            )
            results.append(result)
            print(f"  -> {result['train_time_s']:.1f}s, "
                  f"{result['energy_kwh']:.6f} kWh, "
                  f"{result['memory_peak_mb']:.1f} MB")

            # --- LSTM ---
            count += 1
            lstm_cfg = v2_configs['lstm']['config']
            print(f"[{count}/{total}] {dataset} {budget_name} lstm "
                  f"(h={lstm_cfg['hidden_size']})...", flush=True)
            result = measure_lstm(
                dataset, budget_name, lstm_cfg,
                X_train, y_train, X_val, y_val, X_test, y_test,
            )
            results.append(result)
            print(f"  -> {result['train_time_s']:.1f}s, "
                  f"{result['energy_kwh']:.6f} kWh, "
                  f"{result['memory_peak_mb']:.1f} MB")

    # --- Save JSON ---
    output = {'metadata': metadata, 'results': results}
    json_path = os.path.join(RESULTS_DIR, 'green_metrics_v2.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # --- Save CSV ---
    csv_path = os.path.join(TABLES_DIR, 'green_metrics_comparison.csv')
    fieldnames = [
        'dataset', 'budget', 'model_type',
        'train_time_s', 'energy_kwh', 'emissions_kg_co2',
        'power_w', 'memory_peak_mb',
        'inference_time_per_sample_ms', 'inference_time_total_s',
        'n_test_samples',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)
    print(f"CSV saved: {csv_path}")

    # --- Summary ---
    print(f"\nTotal conditions measured: {len(results)}")
    pyreco_energy = sum(r['energy_kwh'] for r in results if r['model_type'] == 'pyreco_standard')
    lstm_energy = sum(r['energy_kwh'] for r in results if r['model_type'] == 'lstm')
    print(f"Total PyReCo energy: {pyreco_energy:.6f} kWh")
    print(f"Total LSTM energy:   {lstm_energy:.6f} kWh")


if __name__ == '__main__':
    run_all()
