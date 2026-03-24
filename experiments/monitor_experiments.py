"""
Monitor Experiment Progress

Displays real-time progress of comprehensive experiments.

Usage:
    python monitor_experiments.py
    python monitor_experiments.py --results-dir results_final
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def monitor_progress(results_dir, interval=30):
    """
    Monitor experiment progress

    Args:
        results_dir: Directory containing experiment_progress.json
        interval: Update interval in seconds
    """
    progress_file = Path(results_dir) / 'experiment_progress.json'

    print(f"\n{'='*80}")
    print("EXPERIMENT PROGRESS MONITOR")
    print(f"{'='*80}")
    print(f"Monitoring: {progress_file}")
    print(f"Update interval: {interval}s")
    print(f"Press Ctrl+C to stop monitoring\n")

    last_completed = 0

    try:
        while True:
            if not progress_file.exists():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for experiments to start...")
                time.sleep(interval)
                continue

            with open(progress_file, 'r') as f:
                data = json.load(f)

            total = data['total_experiments']
            completed = data['completed']
            successful = data['successful']
            failed = data['failed']
            total_runtime = data['total_runtime']

            # Calculate progress
            progress_pct = completed / total * 100
            avg_time_per_exp = total_runtime / completed if completed > 0 else 0
            remaining = total - completed
            est_remaining_time = remaining * avg_time_per_exp

            # Check for new completions
            if completed > last_completed:
                last_result = data['results'][-1]
                status = "✅" if last_result['success'] else "❌"
                print(f"\n{status} Experiment {completed}/{total}: {last_result['dataset']} "
                      f"(seed={last_result['seed']}, ratio={last_result['train_ratio']}) "
                      f"- {format_time(last_result['runtime'])}")
                last_completed = completed

            # Display current status
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {completed}/{total} ({progress_pct:.1f}%) | "
                  f"✅ {successful} ❌ {failed} | "
                  f"Avg: {format_time(avg_time_per_exp)}/exp | "
                  f"ETA: {format_time(est_remaining_time)}", end='', flush=True)

            if completed >= total:
                print(f"\n\n{'='*80}")
                print("ALL EXPERIMENTS COMPLETE!")
                print(f"{'='*80}")
                print(f"Total: {total}")
                print(f"✅ Successful: {successful}")
                print(f"❌ Failed: {failed}")
                print(f"⏱️  Total time: {format_time(total_runtime)}")
                print(f"📊 Average per experiment: {format_time(avg_time_per_exp)}")
                print(f"{'='*80}\n")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")


def show_summary(results_dir):
    """Show final summary of experiments"""
    progress_file = Path(results_dir) / 'experiment_progress.json'

    if not progress_file.exists():
        print(f"No progress file found at: {progress_file}")
        return

    with open(progress_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total experiments: {data['total_experiments']}")
    print(f"Completed: {data['completed']}")
    print(f"✅ Successful: {data['successful']}")
    print(f"❌ Failed: {data['failed']}")
    print(f"⏱️  Total time: {format_time(data['total_runtime'])}")

    if data['completed'] > 0:
        avg_time = data['total_runtime'] / data['completed']
        print(f"📊 Average per experiment: {format_time(avg_time)}")

    # Show recent results
    print(f"\n{'='*80}")
    print("RECENT EXPERIMENTS (last 5)")
    print(f"{'='*80}\n")

    for result in data['results'][-5:]:
        status = "✅" if result['success'] else "❌"
        print(f"{status} #{result['experiment']}: {result['dataset']} "
              f"(seed={result['seed']}, ratio={result['train_ratio']}) "
              f"- {format_time(result['runtime'])}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results_final',
                       help='Results directory')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary and exit')
    args = parser.parse_args()

    if args.summary:
        show_summary(args.results_dir)
    else:
        monitor_progress(args.results_dir, args.interval)


if __name__ == '__main__':
    main()
