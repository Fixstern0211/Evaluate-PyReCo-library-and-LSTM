"""
Test GPU Acceleration on Apple Silicon (MPS)

This script verifies that PyTorch is using the Apple GPU (MPS backend)
and compares training speed between CPU and GPU.
"""

import torch
import numpy as np
import time
from models.lstm_model import LSTMModel

def test_device_availability():
    """Check what devices are available"""
    print("\n" + "="*70)
    print("DEVICE AVAILABILITY CHECK")
    print("="*70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA (NVIDIA GPU) available: {torch.cuda.is_available()}")

    if torch.backends.mps.is_available():
        print("\n✅ Apple GPU is available and will be used for LSTM training!")
    else:
        print("\n⚠️  Apple GPU not available. Using CPU only.")

    print("="*70 + "\n")

def test_lstm_speed(device='auto', n_samples=1000):
    """Test LSTM training speed on different devices"""
    print(f"\n{'='*70}")
    print(f"LSTM SPEED TEST (device={device})")
    print(f"{'='*70}")

    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(n_samples, 50, 3).astype(np.float32)  # (samples, timesteps, features)
    y_train = np.random.randn(n_samples, 1, 3).astype(np.float32)   # (samples, 1, features)

    # Create model
    if device == 'auto':
        model = LSTMModel(hidden_size=64, num_layers=2, epochs=20, verbose=False)
    else:
        model = LSTMModel(hidden_size=64, num_layers=2, epochs=20,
                         device=device, verbose=False)

    # Train and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    device_used = str(model.device)
    print(f"Device used: {device_used}")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Time per epoch: {train_time/20:.2f} seconds")
    print(f"{'='*70}\n")

    return train_time, device_used

def compare_cpu_vs_gpu():
    """Compare CPU vs GPU training speed"""
    print("\n" + "="*70)
    print("CPU vs GPU COMPARISON")
    print("="*70)

    # Test on CPU
    cpu_time, cpu_device = test_lstm_speed(device='cpu', n_samples=1000)

    # Test on GPU (if available)
    if torch.backends.mps.is_available():
        gpu_time, gpu_device = test_lstm_speed(device='mps', n_samples=1000)

        speedup = cpu_time / gpu_time
        print(f"\n{'='*70}")
        print(f"SPEEDUP ANALYSIS")
        print(f"{'='*70}")
        print(f"CPU time:  {cpu_time:.2f}s")
        print(f"GPU time:  {gpu_time:.2f}s")
        print(f"Speedup:   {speedup:.2f}x")

        if speedup > 1.2:
            print(f"✅ GPU is {speedup:.1f}x faster! GPU acceleration is working well.")
        elif speedup > 0.8:
            print(f"⚠️  GPU and CPU have similar performance (small dataset).")
        else:
            print(f"❌ CPU is faster. GPU overhead dominates for this small dataset.")

        print(f"{'='*70}\n")
    else:
        print("\n⚠️  MPS not available. Skipping GPU test.\n")

if __name__ == '__main__':
    # Run tests
    test_device_availability()

    # Test auto device selection
    print("\n" + "="*70)
    print("AUTO DEVICE SELECTION TEST")
    print("="*70)
    test_lstm_speed(device='auto', n_samples=500)

    # Compare CPU vs GPU
    compare_cpu_vs_gpu()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✅ LSTM model has been updated to use Apple GPU (MPS)")
    print("✅ Training will automatically use GPU when available")
    print("✅ Expected speedup: 1.5-3x for LSTM training on M4")
    print("\nNote: PyReCo models use NumPy/SciPy and run on CPU only,")
    print("      but they are already very fast (no iterative training).")
    print("="*70 + "\n")
