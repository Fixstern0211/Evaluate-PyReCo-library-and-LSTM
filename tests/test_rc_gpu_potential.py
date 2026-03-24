"""
Test: Should RC use GPU?

Analyzes whether RC would benefit from GPU acceleration by:
1. Measuring RC training time (currently CPU-only)
2. Comparing with LSTM training time (CPU vs GPU)
3. Analyzing RC's computational characteristics

Key insight: RC training is NON-ITERATIVE
- RC: One-shot ridge regression (solve X^T X w = X^T y)
- LSTM: Iterative gradient descent (100+ epochs)
"""

import numpy as np
import time
import torch
from models.pyreco_wrapper import PyReCoStandardModel
from models.lstm_model import LSTMModel

def test_rc_training_speed():
    """Test RC training time with different data sizes"""
    print("\n" + "="*70)
    print("RC TRAINING SPEED ANALYSIS")
    print("="*70)

    sizes = [500, 1000, 2000, 4000]

    print("\nRC Training Time (CPU-only, NumPy/SciPy):")
    print("-" * 70)

    rc_times = []

    for n_samples in sizes:
        # Generate data
        X = np.random.randn(n_samples, 50, 3).astype(np.float32)
        y = np.random.randn(n_samples, 1, 3).astype(np.float32)

        # Train RC
        model = PyReCoStandardModel(num_nodes=500, verbose=False)

        start = time.time()
        model.fit(X, y)
        train_time = time.time() - start

        rc_times.append(train_time)
        print(f"  n={n_samples:5d} samples: {train_time:.3f}s")

    return sizes, rc_times


def test_lstm_training_speed():
    """Test LSTM training time with CPU vs GPU"""
    print("\n" + "="*70)
    print("LSTM TRAINING SPEED ANALYSIS")
    print("="*70)

    sizes = [500, 1000, 2000, 4000]

    # Test with CPU
    print("\nLSTM Training Time (CPU):")
    print("-" * 70)

    lstm_cpu_times = []

    for n_samples in sizes:
        X = np.random.randn(n_samples, 50, 3).astype(np.float32)
        y = np.random.randn(n_samples, 1, 3).astype(np.float32)

        model = LSTMModel(hidden_size=64, epochs=50, device='cpu', verbose=False)

        start = time.time()
        model.fit(X, y)
        train_time = time.time() - start

        lstm_cpu_times.append(train_time)
        print(f"  n={n_samples:5d} samples: {train_time:.3f}s")

    # Test with GPU
    if torch.backends.mps.is_available():
        print("\nLSTM Training Time (GPU/MPS):")
        print("-" * 70)

        lstm_gpu_times = []

        for n_samples in sizes:
            X = np.random.randn(n_samples, 50, 3).astype(np.float32)
            y = np.random.randn(n_samples, 1, 3).astype(np.float32)

            model = LSTMModel(hidden_size=64, epochs=50, device='mps', verbose=False)

            start = time.time()
            model.fit(X, y)
            train_time = time.time() - start

            lstm_gpu_times.append(train_time)
            print(f"  n={n_samples:5d} samples: {train_time:.3f}s")
    else:
        lstm_gpu_times = None

    return sizes, lstm_cpu_times, lstm_gpu_times


def analyze_rc_algorithm():
    """Analyze RC's computational characteristics"""
    print("\n" + "="*70)
    print("RC ALGORITHM ANALYSIS")
    print("="*70)

    print("\nRC Training Process:")
    print("-" * 70)
    print("1. Generate random reservoir weights (one-time, CPU)")
    print("2. Compute reservoir states: X_reservoir = tanh(W_in @ X + W_res @ h)")
    print("   → Matrix multiplication (could use GPU)")
    print("3. Solve ridge regression: (X^T X + λI)^-1 X^T y")
    print("   → Linear solve (NumPy uses optimized BLAS/LAPACK on CPU)")
    print("4. Done! (No iteration, no gradients)")

    print("\nLSTM Training Process:")
    print("-" * 70)
    print("1. Forward pass through LSTM layers (many matrix ops)")
    print("2. Compute loss")
    print("3. Backward pass (compute gradients)")
    print("4. Update weights")
    print("5. Repeat for 50-100+ epochs")
    print("   → MUCH MORE computation, GPU helps a lot!")

    print("\nKey Differences:")
    print("-" * 70)
    print("RC:   ONE-SHOT training (analytical solution)")
    print("LSTM: ITERATIVE training (gradient descent)")
    print("\nConclusion:")
    print("  GPU helps LSTM a lot (many iterations)")
    print("  GPU helps RC less (already very fast)")


def main():
    print("\n" + "="*70)
    print("SHOULD RC USE GPU? - COMPREHENSIVE ANALYSIS")
    print("="*70)

    # Test RC speed
    sizes, rc_times = test_rc_training_speed()

    # Test LSTM speed
    sizes, lstm_cpu_times, lstm_gpu_times = test_lstm_training_speed()

    # Analyze algorithm
    analyze_rc_algorithm()

    # Comparison
    print("\n" + "="*70)
    print("SPEED COMPARISON (n=2000 samples)")
    print("="*70)

    idx = 2  # n=2000
    rc_time = rc_times[idx]
    lstm_cpu_time = lstm_cpu_times[idx]

    print(f"\nRC (CPU):        {rc_time:.2f}s")
    print(f"LSTM (CPU):      {lstm_cpu_time:.2f}s")

    if lstm_gpu_times:
        lstm_gpu_time = lstm_gpu_times[idx]
        print(f"LSTM (GPU):      {lstm_gpu_time:.2f}s")

        print(f"\nSpeedups:")
        print(f"  RC vs LSTM(CPU):  {lstm_cpu_time/rc_time:.1f}x faster")
        print(f"  RC vs LSTM(GPU):  {lstm_gpu_time/rc_time:.1f}x faster")
        print(f"  LSTM GPU speedup: {lstm_cpu_time/lstm_gpu_time:.1f}x")
    else:
        print(f"\nSpeedup:")
        print(f"  RC vs LSTM(CPU):  {lstm_cpu_time/rc_time:.1f}x faster")

    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if rc_time < 5:
        print(f"\n✅ RC is ALREADY VERY FAST ({rc_time:.2f}s)")
        print("   GPU acceleration would provide minimal benefit because:")
        print("   1. Training is already < 5 seconds")
        print("   2. Non-iterative algorithm (one-shot solution)")
        print("   3. GPU data transfer overhead would dominate")
        print("   4. NumPy/SciPy already use optimized CPU libraries (MKL/OpenBLAS)")

        print("\n✅ CURRENT COMPARISON IS FAIR:")
        print("   - Each model uses its optimal implementation")
        print("   - LSTM: Iterative → needs GPU")
        print("   - RC: Analytical → already efficient on CPU")

        print("\n✅ FOR YOUR HYPOTHESIS:")
        print("   'RC performs better with faster/equal training time'")
        print("   → Comparing LSTM(GPU) vs RC(CPU) IS fair because:")
        print("     * Both use their best practical implementation")
        print("     * RC is already so efficient that GPU wouldn't help much")
        print("     * This reflects real-world deployment scenarios")
    else:
        print(f"\n⚠️  RC is slower than expected ({rc_time:.2f}s)")
        print("   GPU acceleration MIGHT help for very large datasets")
        print("   Consider implementing PyTorch-based RC for fair comparison")

    print("\n" + "="*70)
    print("CONCLUSION FOR YOUR EXPERIMENTS")
    print("="*70)
    print("\nYou have THREE valid comparison scenarios:\n")
    print("1. LSTM(GPU) vs RC(CPU) - Real-world deployment ✅ FAIR")
    print("   → Each uses optimal implementation")
    print("   → Reflects actual practice")

    print("\n2. LSTM(CPU) vs RC(CPU) - Hardware-neutral ✅ FAIR")
    print("   → Same hardware for both")
    print("   → Pure algorithm comparison")

    print("\n3. LSTM(GPU) vs RC(GPU) - Both accelerated")
    print("   → Would require rewriting RC in PyTorch")
    print("   → Probably unnecessary (RC already fast)")

    print("\n💡 RECOMMENDATION: Run scenarios 1 AND 2")
    print("   This gives you the most comprehensive evidence!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
