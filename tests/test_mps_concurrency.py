"""
Test MPS (Apple GPU) Concurrency

Tests whether multiple PyTorch processes can use MPS simultaneously
and how they share GPU resources.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from multiprocessing import Process, Queue


def train_single_lstm(gpu_id, duration, result_queue):
    """Train a single LSTM and measure time"""

    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"[Worker {gpu_id}] Using device: {device}")

    # Create a simple LSTM
    model = nn.LSTM(input_size=10, hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate random data
    X = torch.randn(100, 50, 10).to(device)  # (batch, seq, features)
    y = torch.randn(100, 64).to(device)

    start_time = time.time()
    iterations = 0

    # Train for specified duration
    while time.time() - start_time < duration:
        # Forward pass
        output, _ = model(X)
        loss = torch.mean((output[:, -1, :] - y) ** 2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterations += 1

    elapsed = time.time() - start_time
    throughput = iterations / elapsed

    result = {
        'worker_id': gpu_id,
        'device': str(device),
        'elapsed': elapsed,
        'iterations': iterations,
        'throughput': throughput
    }

    result_queue.put(result)
    print(f"[Worker {gpu_id}] Completed: {iterations} iterations in {elapsed:.2f}s "
          f"({throughput:.2f} iter/s)")


def test_concurrent_mps(num_workers, duration=10):
    """Test MPS with multiple concurrent workers"""

    print(f"\n{'='*70}")
    print(f"Testing MPS Concurrency with {num_workers} workers")
    print(f"Training duration: {duration} seconds per worker")
    print(f"{'='*70}\n")

    result_queue = Queue()
    processes = []

    # Start all workers simultaneously
    start = time.time()
    for i in range(num_workers):
        p = Process(target=train_single_lstm, args=(i, duration, result_queue))
        p.start()
        processes.append(p)
        print(f"Started Worker {i}")

    # Wait for all to complete
    for p in processes:
        p.join()

    total_time = time.time() - start

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}\n")

    print(f"Total wall-clock time: {total_time:.2f}s")
    print(f"Expected (if perfectly parallel): {duration:.2f}s")
    print(f"Overhead: {(total_time - duration):.2f}s\n")

    print("Per-worker performance:")
    print(f"{'Worker':<10} {'Device':<10} {'Iterations':<15} {'Throughput (iter/s)':<20}")
    print("-" * 70)

    total_throughput = 0
    for r in sorted(results, key=lambda x: x['worker_id']):
        print(f"{r['worker_id']:<10} {r['device']:<10} {r['iterations']:<15} {r['throughput']:<20.2f}")
        total_throughput += r['throughput']

    print(f"\nTotal throughput: {total_throughput:.2f} iter/s")
    print(f"Average per worker: {total_throughput/num_workers:.2f} iter/s")

    return results, total_throughput


def main():
    print("\n" + "="*70)
    print("MPS GPU CONCURRENCY TEST")
    print("="*70)

    if not torch.backends.mps.is_available():
        print("\n❌ MPS not available! Will use CPU instead.")
        print("This test is most useful on Apple Silicon.\n")
    else:
        print("\n✅ MPS (Apple GPU) is available!")
        print(f"PyTorch version: {torch.__version__}\n")

    # Test 1: Single worker (baseline)
    print("\n" + "="*70)
    print("TEST 1: Single Worker (Baseline)")
    print("="*70)
    results_1, throughput_1 = test_concurrent_mps(num_workers=1, duration=10)
    baseline_throughput = throughput_1

    # Test 2: 2 workers
    print("\n" + "="*70)
    print("TEST 2: 2 Concurrent Workers")
    print("="*70)
    results_2, throughput_2 = test_concurrent_mps(num_workers=2, duration=10)

    # Test 3: 4 workers
    print("\n" + "="*70)
    print("TEST 3: 4 Concurrent Workers")
    print("="*70)
    results_4, throughput_4 = test_concurrent_mps(num_workers=4, duration=10)

    # Summary
    print("\n" + "="*70)
    print("CONCURRENCY SUMMARY")
    print("="*70)

    print(f"\n{'Workers':<15} {'Total Throughput':<25} {'Speedup vs Baseline':<25} {'Efficiency':<15}")
    print("-" * 80)
    print(f"{'1 (baseline)':<15} {baseline_throughput:<25.2f} {'1.00x':<25} {'100%':<15}")
    print(f"{'2':<15} {throughput_2:<25.2f} {throughput_2/baseline_throughput:<25.2f}x "
          f"{throughput_2/baseline_throughput/2*100:<15.1f}%")
    print(f"{'4':<15} {throughput_4:<25.2f} {throughput_4/baseline_throughput:<25.2f}x "
          f"{throughput_4/baseline_throughput/4*100:<15.1f}%")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    speedup_2 = throughput_2 / baseline_throughput
    speedup_4 = throughput_4 / baseline_throughput
    efficiency_4 = speedup_4 / 4 * 100

    print(f"\n2 workers speedup: {speedup_2:.2f}x")
    print(f"4 workers speedup: {speedup_4:.2f}x")
    print(f"4 workers efficiency: {efficiency_4:.1f}%")

    if efficiency_4 > 80:
        print("\n✅ Excellent! MPS handles concurrent workloads very well.")
        print("   → Running 4 parallel experiments is highly efficient.")
    elif efficiency_4 > 60:
        print("\n✅ Good! MPS handles concurrent workloads reasonably well.")
        print("   → Running 4 parallel experiments will provide good speedup.")
    elif efficiency_4 > 40:
        print("\n⚠️  Moderate. MPS has some overhead with concurrent workloads.")
        print("   → Running 4 parallel experiments still faster than sequential.")
    else:
        print("\n⚠️  Low efficiency. MPS struggles with heavy concurrent loads.")
        print("   → Consider reducing workers to 2.")

    print("\n" + "="*70)
    print("RECOMMENDATION FOR YOUR EXPERIMENTS")
    print("="*70)

    if efficiency_4 > 70:
        print("\n✅ Use --workers 4 for optimal performance")
        print("   GPU can handle 4 concurrent LSTM training jobs efficiently")
    elif efficiency_4 > 50:
        print("\n✅ Use --workers 4 (acceptable)")
        print("   Or use --workers 3 for slightly better per-job performance")
    else:
        print("\n⚠️  Consider --workers 2")
        print("   GPU efficiency drops significantly with 4 workers")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
