#!/usr/bin/env python3
"""
Autoregressive Multi-step Prediction Experiment Runner

Runs test_autoregressive.py for all dataset/seed/train_ratio combinations.

Configuration:
- 3 datasets (lorenz, mackeyglass, santafe)
- 5 random seeds (42, 43, 44, 45, 46)
- 6 train ratios (0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
- 1 budget (medium = 10k params)
- Horizons: [1, 5, 10, 20, 30, 50]

Total experiments: 3 × 5 × 6 = 90 experiments

Usage:
    python run_autoregressive_experiments.py
    python run_autoregressive_experiments.py --quick  # Test mode: 1 dataset × 1 seed × 1 ratio
    python run_autoregressive_experiments.py --tune-lstm  # Enable LSTM hyperparameter tuning

    # Run in background:
    nohup python -u run_autoregressive_experiments.py > ../logs/exp_autoregressive.log 2>&1 &
"""

import subprocess
import json
import time
import sys
from datetime import datetime
from pathlib import Path
import argparse


def run_experiment(dataset, seed, train_ratio, output_dir, budget='medium', tune_lstm=False):
    """
    Run one autoregressive experiment

    Returns:
        (success, runtime, output_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"autoregressive_{dataset}_seed{seed}_train{train_ratio}_{budget}_{timestamp}.json"

    cmd = [
        sys.executable, 'test_autoregressive.py',
        '--dataset', dataset,
        '--seed', str(seed),
        '--train-ratio', str(train_ratio),
        '--budget', budget,
        '--output', str(output_file)
    ]

    if tune_lstm:
        cmd.append('--tune-lstm')

    print(f"\n{'='*80}")
    print(f"Running: {dataset} | seed={seed} | train_ratio={train_ratio} | budget={budget}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n", flush=True)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        runtime = time.time() - start_time
        success = result.returncode == 0

        if not success:
            print(f"FAILED (exit code {result.returncode})")
            print(f"STDERR: {result.stderr}", flush=True)
            return False, runtime, None

        print(f"SUCCESS ({runtime:.1f}s)", flush=True)
        return True, runtime, output_file

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"TIMEOUT after {runtime:.1f}s", flush=True)
        return False, runtime, None
    except Exception as e:
        runtime = time.time() - start_time
        print(f"ERROR: {e}", flush=True)
        return False, runtime, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                       help='Quick test: 1 dataset × 1 seed × 1 ratio')
    parser.add_argument('--output-dir', type=str, default='../results/autoregressive',
                       help='Output directory')
    parser.add_argument('--budget', type=str, default='medium',
                       choices=['small', 'medium', 'large'])
    parser.add_argument('--tune-lstm', action='store_true',
                       help='Enable LSTM hyperparameter tuning')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    if args.quick:
        datasets = ['lorenz']
        seeds = [42]
        train_ratios = [0.7]
        print("\nQUICK TEST MODE")
    else:
        datasets = ['lorenz', 'mackeyglass', 'santafe']
        seeds = [42, 43, 44, 45, 46]
        train_ratios = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
        print("\nFULL EXPERIMENT MODE")

    total_experiments = len(datasets) * len(seeds) * len(train_ratios)

    print(f"\n{'='*80}")
    print("AUTOREGRESSIVE MULTI-STEP PREDICTION EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Train ratios: {train_ratios}")
    print(f"Budget: {args.budget}")
    print(f"Tune LSTM: {args.tune_lstm}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print(f"\nPrediction horizons: [1, 5, 10, 20, 30, 50]")
    print(f"{'='*80}\n", flush=True)

    # Track results
    all_results = []
    successful = 0
    failed = 0
    total_runtime = 0
    start_time = time.time()

    exp_num = 0
    for dataset in datasets:
        for seed in seeds:
            for train_ratio in train_ratios:
                exp_num += 1

                print(f"\n{'='*80}")
                print(f"EXPERIMENT {exp_num}/{total_experiments}")
                print(f"Progress: {exp_num/total_experiments*100:.1f}%")
                print(f"Elapsed: {(time.time()-start_time)/60:.1f} min")
                print(f"{'='*80}", flush=True)

                success, runtime, output_file = run_experiment(
                    dataset, seed, train_ratio, output_dir,
                    budget=args.budget, tune_lstm=args.tune_lstm
                )

                if success:
                    successful += 1
                else:
                    failed += 1

                total_runtime += runtime

                all_results.append({
                    'experiment': exp_num,
                    'dataset': dataset,
                    'seed': seed,
                    'train_ratio': train_ratio,
                    'success': success,
                    'runtime': runtime,
                    'output_file': str(output_file) if output_file else None,
                    'timestamp': datetime.now().isoformat()
                })

                # Save progress
                progress_file = output_dir / 'experiment_progress.json'
                with open(progress_file, 'w') as f:
                    json.dump({
                        'total_experiments': total_experiments,
                        'completed': exp_num,
                        'successful': successful,
                        'failed': failed,
                        'total_runtime': total_runtime,
                        'results': all_results
                    }, f, indent=2)

    # Final summary
    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("AUTOREGRESSIVE EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average per experiment: {total_time/total_experiments:.1f}s")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n", flush=True)


if __name__ == '__main__':
    main()
