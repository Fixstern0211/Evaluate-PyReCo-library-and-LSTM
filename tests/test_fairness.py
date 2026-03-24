"""
Quick test to verify CPU vs GPU fairness
Tests one experiment with both device settings
"""

import subprocess
import time
import json
from pathlib import Path

def run_test(lstm_device):
    """Run one quick test"""
    output_file = f"test_fairness_{lstm_device}.json"

    cmd = [
        'python', 'test_model_scaling.py',
        '--dataset', 'lorenz',
        '--length', '1000',
        '--seed', '42',
        '--train-ratio', '0.8',
        '--lstm-device', lstm_device,
        '--output', output_file
    ]

    print(f"\n{'='*70}")
    print(f"Testing with LSTM device: {lstm_device}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.time() - start

    if result.returncode == 0:
        print(f"✅ Success! Runtime: {runtime:.2f}s")

        # Load and show LSTM training time
        with open(output_file) as f:
            data = json.load(f)

        for model_result in data['results'].values():
            for model_data in model_result:
                if model_data['model_type'] == 'lstm':
                    print(f"   LSTM training time: {model_data['training_time']:.2f}s")
                    print(f"   Device used: {lstm_device}")
                elif model_data['model_type'] == 'pyreco_standard':
                    print(f"   RC training time: {model_data['training_time']:.2f}s")

        return runtime, output_file
    else:
        print(f"❌ Failed!")
        print(result.stderr)
        return None, None

def main():
    print("\n" + "="*70)
    print("FAIRNESS TEST: CPU vs GPU for LSTM")
    print("="*70)
    print("\nThis test demonstrates the fairness issue:")
    print("- Scenario 1 (GPU): LSTM uses M4 GPU, RC uses CPU")
    print("- Scenario 2 (CPU): Both LSTM and RC use CPU only")
    print("="*70)

    # Test with GPU
    print("\n" + "🚀 " + "="*68)
    print("SCENARIO 1: LSTM with GPU (auto)")
    print("="*70)
    time_gpu, file_gpu = run_test('auto')

    # Test with CPU
    print("\n" + "⚖️  " + "="*68)
    print("SCENARIO 2: LSTM with CPU (fair comparison)")
    print("="*70)
    time_cpu, file_cpu = run_test('cpu')

    # Summary
    if time_gpu and time_cpu:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)

        # Load both results
        with open(file_gpu) as f:
            data_gpu = json.load(f)
        with open(file_cpu) as f:
            data_cpu = json.load(f)

        print("\n📊 Training Times:")
        print(f"\nScenario 1 (GPU):")
        for model_result in data_gpu['results'].values():
            for model_data in model_result:
                model_type = model_data['model_type']
                train_time = model_data['training_time']
                if model_type == 'lstm':
                    lstm_gpu_time = train_time
                    print(f"  LSTM (GPU): {train_time:.2f}s")
                elif model_type == 'pyreco_standard':
                    rc_time = train_time
                    print(f"  RC (CPU):   {train_time:.2f}s")

        print(f"\nScenario 2 (CPU):")
        for model_result in data_cpu['results'].values():
            for model_data in model_result:
                model_type = model_data['model_type']
                train_time = model_data['training_time']
                if model_type == 'lstm':
                    lstm_cpu_time = train_time
                    print(f"  LSTM (CPU): {train_time:.2f}s")
                elif model_type == 'pyreco_standard':
                    print(f"  RC (CPU):   {train_time:.2f}s")

        print("\n" + "="*70)
        print("ANALYSIS:")
        print("="*70)

        speedup = lstm_cpu_time / lstm_gpu_time
        print(f"\n1. GPU Speedup for LSTM: {speedup:.2f}x")

        if lstm_gpu_time < rc_time:
            print(f"\n2. Scenario 1 (GPU): LSTM is FASTER than RC")
            print(f"   → LSTM gets unfair advantage from GPU!")
        else:
            print(f"\n2. Scenario 1 (GPU): RC is still faster than LSTM")
            print(f"   → RC is so efficient that even GPU LSTM is slower!")

        if lstm_cpu_time > rc_time:
            print(f"\n3. Scenario 2 (CPU): RC is FASTER than LSTM")
            print(f"   → Fair comparison shows RC's efficiency advantage")
        else:
            print(f"\n3. Scenario 2 (CPU): LSTM is faster than RC")
            print(f"   → Even without GPU, LSTM is competitive")

        print("\n" + "="*70)
        print("RECOMMENDATION:")
        print("="*70)
        print("\n✅ Run BOTH scenarios in your full experiments:")
        print("   python run_fair_comparison.py --quick")
        print("\nThis provides TWO perspectives:")
        print("  1. Real-world deployment (Scenario 1)")
        print("  2. Algorithm fairness (Scenario 2)")
        print("\nYour hypothesis should be tested against BOTH!")
        print("="*70 + "\n")

        # Cleanup
        Path(file_gpu).unlink()
        Path(file_cpu).unlink()

if __name__ == '__main__':
    main()
