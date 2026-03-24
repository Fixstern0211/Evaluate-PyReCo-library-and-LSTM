"""
Final Comprehensive Experiment Runner

Runs test_model_scaling.py with optimized configuration:
- 3 datasets (lorenz, mackeyglass, santafe)
- 5 random seeds (42, 43, 44, 45, 46)
- 6 train ratios (0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
- 3 parameter scales (small=1k, medium=10k, large=50k)
- PyReCo: 4-6 combination grid search (spec_rad, leakage_rate, density, fraction_input)
  Based on 5-fold CV pretuning results (pretuning_{dataset}_merged.json)
- LSTM: 40-combination grid search (num_layers, learning_rate, dropout)
  hidden_size computed per num_layers to match parameter budget

Fair comparison: Both models use hyperparameter tuning

Note: train_ratio=0.9 removed because train(0.9) + val(0.15) >= 1.0 leaves no test data.
      Maximum valid train_ratio is 0.8 (train=80%, val=15%, test=5%).

Total experiments: 3 × 5 × 6 = 90 experiments
Total trainings: 90 × 3 scales × (~5 PyReCo + 40 LSTM) = ~12,150 trainings

Usage:
    python run_final_experiments.py
    python run_final_experiments.py --quick  # Test with 1 dataset × 1 seed × 1 ratio

    # Run in background:
    nohup python -u run_final_experiments.py > ../logs/exp_final.log 2>&1 &
"""

import subprocess
import json
import time
import sys
from datetime import datetime
from pathlib import Path
import argparse


def run_experiment(dataset, seed, train_ratio, output_dir):
    """
    Run one experiment configuration

    Args:
        dataset: Dataset name
        seed: Random seed
        train_ratio: Training data fraction
        output_dir: Output directory

    Returns:
        (success, runtime, output_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{dataset}_seed{seed}_train{train_ratio}_{timestamp}.json"

    cmd = [
        sys.executable, 'test_model_scaling.py',
        '--dataset', dataset,
        '--seed', str(seed),
        '--train-ratio', str(train_ratio),
        '--tune-pyreco',
        '--output', str(output_file)
    ]

    print(f"\n{'='*80}")
    print(f"Running: {dataset} | seed={seed} | train_ratio={train_ratio}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=18000  # 5 hour timeout per experiment
        )

        runtime = time.time() - start_time
        success = result.returncode == 0

        if not success:
            print(f"❌ FAILED (exit code {result.returncode})")
            print(f"STDERR: {result.stderr}")
            return False, runtime, None

        print(f"✅ SUCCESS ({runtime:.1f}s)")
        return True, runtime, output_file

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"⏱️  TIMEOUT after {runtime:.1f}s")
        return False, runtime, None
    except Exception as e:
        runtime = time.time() - start_time
        print(f"❌ ERROR: {e}")
        return False, runtime, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode: 1 dataset × 1 seed × 1 train_ratio')
    parser.add_argument('--output-dir', type=str, default='../results/final',
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Configuration
    if args.quick:
        datasets = ['lorenz']
        seeds = [42]
        train_ratios = [0.7]
        print("\n🚀 QUICK TEST MODE")
    else:
        datasets = ['lorenz', 'mackeyglass', 'santafe']
        seeds = [42, 43, 44, 45, 46]
        train_ratios = [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]  # 0.9 removed: train+val>=1.0
        print("\n🚀 FULL EXPERIMENT MODE")

    total_experiments = len(datasets) * len(seeds) * len(train_ratios)

    print(f"\n{'='*80}")
    print("FINAL COMPREHENSIVE EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Train ratios: {train_ratios}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print(f"\nConfiguration:")
    print(f"  - PyReCo: 36 combinations (spec_rad, leakage_rate, density, fraction_input)")
    print(f"  - LSTM: 9 combinations (learning_rate, dropout) - FAIR COMPARISON")
    print(f"  - Budget scales: small=1k, medium=10k, large=50k")
    print(f"\nEstimated time:")
    if args.quick:
        print(f"  Quick test: ~20-30 minutes")
    else:
        print(f"  Full run: ~50-60 hours")
    print(f"{'='*80}")

    # Track results
    all_results = []
    successful = 0
    failed = 0
    total_runtime = 0

    start_time = time.time()

    # Run experiments
    for i, dataset in enumerate(datasets, 1):
        for j, seed in enumerate(seeds, 1):
            for k, train_ratio in enumerate(train_ratios, 1):
                exp_num = (i-1) * len(seeds) * len(train_ratios) + (j-1) * len(train_ratios) + k

                print(f"\n{'='*80}")
                print(f"EXPERIMENT {exp_num}/{total_experiments}")
                print(f"Progress: {exp_num/total_experiments*100:.1f}%")
                print(f"Elapsed: {(time.time()-start_time)/3600:.2f}h")
                print(f"{'='*80}")

                success, runtime, output_file = run_experiment(
                    dataset, seed, train_ratio, output_dir
                )

                if success:
                    successful += 1
                else:
                    failed += 1

                total_runtime += runtime

                # Record result
                result = {
                    'experiment': exp_num,
                    'dataset': dataset,
                    'seed': seed,
                    'train_ratio': train_ratio,
                    'success': success,
                    'runtime': runtime,
                    'output_file': str(output_file) if output_file else None,
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(result)

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
    print("EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total: {total_experiments}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"📊 Average per experiment: {total_time/total_experiments:.1f}s")
    print(f"\nResults saved to: {output_dir}")
    print(f"Progress file: {progress_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
