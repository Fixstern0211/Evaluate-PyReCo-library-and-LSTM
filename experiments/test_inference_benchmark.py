#!/usr/bin/env python3
"""
Inference Latency Benchmark for PyReCo vs LSTM

Measures detailed inference performance metrics:
1. Single-sample latency (with warmup and multiple runs)
2. Batch inference latency (various batch sizes)
3. Throughput (samples per second)
4. Latency statistics (min, max, mean, std, percentiles)

Usage:
    python experiments/test_inference_benchmark.py --dataset lorenz
    python experiments/test_inference_benchmark.py --dataset santafe --budget medium
    python experiments/test_inference_benchmark.py --all-datasets --output results/inference_benchmark/
"""

import argparse
import json
import glob
import time
import gc
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.load_dataset import load as local_load, set_seed
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel


def load_best_configs(results_dir="results/final"):
    """Load best configurations from main experiment results."""
    if not os.path.isabs(results_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, results_dir)

    files = sorted(glob.glob(f"{results_dir}/results_*.json"))
    if not files:
        print(f"WARNING: No result files found in {results_dir}")
        return None

    records = []
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        metadata = data['metadata']
        for budget_name in data['budgets'].keys():
            for model_result in data['results'][budget_name]:
                records.append({
                    'dataset': metadata['dataset'],
                    'seed': metadata['seed'],
                    'budget': budget_name,
                    'model_type': model_result['model_type'],
                    'test_mse': model_result['test_mse'],
                    'config': model_result['config']
                })

    # Find best config per (dataset, budget, model_type) by lowest avg MSE
    best_configs = {}
    for record in records:
        key = (record['dataset'], record['budget'], record['model_type'])
        if key not in best_configs:
            best_configs[key] = {'configs': [], 'mses': []}
        best_configs[key]['configs'].append(record['config'])
        best_configs[key]['mses'].append(record['test_mse'])

    final_configs = {}
    for key, data in best_configs.items():
        dataset, budget, model_type = key
        config_mse = {}
        for config, mse in zip(data['configs'], data['mses']):
            config_key = json.dumps(config, sort_keys=True)
            if config_key not in config_mse:
                config_mse[config_key] = []
            config_mse[config_key].append(mse)
        best_config_key = min(config_mse.keys(), key=lambda k: np.mean(config_mse[k]))
        best_config = json.loads(best_config_key)

        if dataset not in final_configs:
            final_configs[dataset] = {}
        if budget not in final_configs[dataset]:
            final_configs[dataset][budget] = {}
        final_configs[dataset][budget][model_type] = best_config

    return final_configs


def get_config(best_configs, dataset, budget, model_type):
    """Get config from best_configs."""
    if best_configs and dataset in best_configs:
        if budget in best_configs[dataset]:
            if model_type in best_configs[dataset][budget]:
                return best_configs[dataset][budget][model_type]
    raise ValueError(f"No config found for {dataset}/{budget}/{model_type}")


def measure_single_sample_latency(model, X_sample, n_warmup=10, n_runs=100):
    """
    Measure single-sample inference latency with proper warmup.

    Returns:
        dict with latency statistics in milliseconds
    """
    # Ensure single sample
    if len(X_sample.shape) == 2:
        X_sample = X_sample[np.newaxis, :, :]
    X_single = X_sample[:1]

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(X_single)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        _ = model.predict(X_single)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'n_runs': n_runs,
    }


def measure_batch_latency(model, X_test, batch_sizes=[1, 8, 16, 32, 64, 128], n_runs=20):
    """
    Measure batch inference latency for various batch sizes.

    Returns:
        dict mapping batch_size to latency statistics
    """
    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue

        X_batch = X_test[:batch_size]

        # Warmup
        for _ in range(5):
            _ = model.predict(X_batch)

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            gc.collect()
            start = time.perf_counter()
            _ = model.predict(X_batch)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)

        results[batch_size] = {
            'total_ms': float(np.mean(latencies)),
            'per_sample_ms': float(np.mean(latencies) / batch_size),
            'std_ms': float(np.std(latencies)),
            'throughput_samples_per_sec': float(batch_size / (np.mean(latencies) / 1000)),
        }

    return results


def measure_throughput(model, X_test, duration_seconds=5.0):
    """
    Measure sustained throughput over a duration.

    Returns:
        dict with throughput statistics
    """
    # Warmup
    _ = model.predict(X_test[:min(100, len(X_test))])

    # Measure
    start_time = time.perf_counter()
    total_samples = 0
    iterations = 0

    while (time.perf_counter() - start_time) < duration_seconds:
        _ = model.predict(X_test)
        total_samples += len(X_test)
        iterations += 1

    elapsed = time.perf_counter() - start_time

    return {
        'total_samples': total_samples,
        'total_seconds': float(elapsed),
        'throughput_samples_per_sec': float(total_samples / elapsed),
        'iterations': iterations,
        'samples_per_iteration': len(X_test),
    }


def run_inference_benchmark(dataset, budget='medium', seed=42, best_configs=None):
    """
    Run complete inference benchmark for both models.
    """
    set_seed(seed)

    print(f"\n{'='*70}")
    print(f"INFERENCE BENCHMARK: {dataset} | Budget: {budget} | Seed: {seed}")
    print(f"{'='*70}")

    # Load data
    print("\nLoading data...")
    n_in, n_out = 100, 1
    data = local_load(dataset, n_in=n_in, n_out=n_out, train_fraction=0.7)
    X_train, y_train = data[0], data[1]
    X_test, y_test = data[4], data[5]
    n_features = X_train.shape[-1]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {
        'metadata': {
            'dataset': dataset,
            'budget': budget,
            'seed': seed,
            'n_in': n_in,
            'n_out': n_out,
            'n_features': n_features,
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat(),
        },
        'models': {}
    }

    # Benchmark both models
    for model_type in ['pyreco', 'lstm']:
        # Map to config key names used in results/final
        config_key = 'pyreco_standard' if model_type == 'pyreco' else 'lstm'

        print(f"\n{'-'*50}")
        print(f"Benchmarking {model_type.upper()}")
        print(f"{'-'*50}")

        config = get_config(best_configs, dataset, budget, config_key)
        print(f"  Config: {config}")

        # Create and train model
        print("  Training model...")
        if model_type == 'pyreco':
            model = PyReCoStandardModel(
                num_nodes=config['num_nodes'],
                density=config['density'],
                spec_rad=config['spec_rad'],
                leakage_rate=config['leakage_rate'],
                fraction_input=config['fraction_input'],
                fraction_output=config['fraction_output'],
                activation=config.get('activation', 'tanh'),
                optimizer=config.get('optimizer', 'ridge'),
            )
            model.fit(X_train, y_train)
        else:
            model = LSTMModel(
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                learning_rate=config['learning_rate'],
                epochs=100,
                batch_size=32,
                verbose=False,
            )
            model.fit(X_train, y_train)

        print(f"  Training time: {model.training_time:.3f}s")

        # Single-sample latency
        print("  Measuring single-sample latency...")
        single_latency = measure_single_sample_latency(model, X_test[0])
        print(f"    Mean: {single_latency['mean_ms']:.3f} ms")
        print(f"    Std:  {single_latency['std_ms']:.3f} ms")
        print(f"    P95:  {single_latency['p95_ms']:.3f} ms")

        # Batch latency
        print("  Measuring batch latency...")
        batch_latency = measure_batch_latency(model, X_test)
        for bs, stats in batch_latency.items():
            print(f"    Batch {bs:3d}: {stats['total_ms']:.2f} ms total, "
                  f"{stats['per_sample_ms']:.3f} ms/sample, "
                  f"{stats['throughput_samples_per_sec']:.0f} samples/s")

        # Throughput
        print("  Measuring sustained throughput...")
        throughput = measure_throughput(model, X_test, duration_seconds=3.0)
        print(f"    Throughput: {throughput['throughput_samples_per_sec']:.0f} samples/s")

        results['models'][model_type] = {
            'config': config,
            'training_time': model.training_time,
            'single_sample_latency': single_latency,
            'batch_latency': {str(k): v for k, v in batch_latency.items()},
            'throughput': throughput,
        }

        # Cleanup
        del model
        gc.collect()

    return results


def print_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("INFERENCE BENCHMARK COMPARISON")
    print(f"{'='*70}")

    pyreco = results['models'].get('pyreco', {})
    lstm = results['models'].get('lstm', {})

    if not pyreco or not lstm:
        print("Missing model results")
        return

    print(f"\nDataset: {results['metadata']['dataset']} | Budget: {results['metadata']['budget']}")
    print(f"\n{'Metric':<35} {'PyReCo':>15} {'LSTM':>15} {'Ratio':>10}")
    print("-" * 75)

    # Single sample
    p_single = pyreco['single_sample_latency']['mean_ms']
    l_single = lstm['single_sample_latency']['mean_ms']
    ratio = l_single / p_single if p_single > 0 else float('inf')
    print(f"{'Single-sample latency (ms)':<35} {p_single:>15.3f} {l_single:>15.3f} {ratio:>9.2f}x")

    # P95
    p_p95 = pyreco['single_sample_latency']['p95_ms']
    l_p95 = lstm['single_sample_latency']['p95_ms']
    ratio = l_p95 / p_p95 if p_p95 > 0 else float('inf')
    print(f"{'P95 latency (ms)':<35} {p_p95:>15.3f} {l_p95:>15.3f} {ratio:>9.2f}x")

    # Throughput
    p_tp = pyreco['throughput']['throughput_samples_per_sec']
    l_tp = lstm['throughput']['throughput_samples_per_sec']
    ratio = p_tp / l_tp if l_tp > 0 else float('inf')
    print(f"{'Throughput (samples/s)':<35} {p_tp:>15.0f} {l_tp:>15.0f} {ratio:>9.2f}x")

    # Batch comparison
    print(f"\n{'Batch Size':<15} {'PyReCo (ms/sample)':>20} {'LSTM (ms/sample)':>20} {'Ratio':>10}")
    print("-" * 65)
    for bs in ['1', '8', '32', '128']:
        if bs in pyreco['batch_latency'] and bs in lstm['batch_latency']:
            p_ms = pyreco['batch_latency'][bs]['per_sample_ms']
            l_ms = lstm['batch_latency'][bs]['per_sample_ms']
            ratio = l_ms / p_ms if p_ms > 0 else float('inf')
            print(f"{bs:<15} {p_ms:>20.4f} {l_ms:>20.4f} {ratio:>9.2f}x")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Inference Latency Benchmark')
    parser.add_argument('--dataset', type=str, default='lorenz',
                       choices=['lorenz', 'mackeyglass', 'santafe'],
                       help='Dataset to use')
    parser.add_argument('--all-datasets', action='store_true',
                       help='Run on all datasets')
    parser.add_argument('--budget', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Parameter budget')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='results/inference_benchmark',
                       help='Output directory')
    parser.add_argument('--results-dir', type=str, default='results/final',
                       help='Directory with main experiment results for best configs')
    args = parser.parse_args()

    # Load best configs from main experiments
    print("Loading best configurations from main experiments...")
    best_configs = load_best_configs(args.results_dir)
    if best_configs is None:
        print("ERROR: Cannot load configs. Run main experiments first.")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['lorenz', 'mackeyglass', 'santafe'] if args.all_datasets else [args.dataset]
    all_results = []

    for dataset in datasets:
        results = run_inference_benchmark(
            dataset=dataset,
            budget=args.budget,
            seed=args.seed,
            best_configs=best_configs,
        )

        print_comparison(results)
        all_results.append(results)

        # Save results
        output_file = output_dir / f"inference_{dataset}_{args.budget}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

    if len(all_results) > 1:
        print("\nSummary across datasets:")
        print(f"{'Dataset':<15} {'PyReCo (ms)':<15} {'LSTM (ms)':<15} {'Speedup':>10}")
        print("-" * 55)
        for r in all_results:
            ds = r['metadata']['dataset']
            p_ms = r['models']['pyreco']['single_sample_latency']['mean_ms']
            l_ms = r['models']['lstm']['single_sample_latency']['mean_ms']
            speedup = l_ms / p_ms if p_ms > 0 else 0
            print(f"{ds:<15} {p_ms:<15.3f} {l_ms:<15.3f} {speedup:>9.2f}x")


if __name__ == '__main__':
    main()
