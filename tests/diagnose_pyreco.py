"""
Diagnose PyReCo Standard vs Custom API performance difference

This script compares the two APIs with the same configuration
to identify the performance bottleneck.
"""

import time
import numpy as np

print("="*80)
print("Diagnosing PyReCo Standard vs Custom API")
print("="*80)

# Generate small test data
np.random.seed(42)
n_samples = 100
n_timesteps = 100
n_features = 3

X_train = np.random.randn(n_samples, n_timesteps, n_features)
y_train = np.random.randn(n_samples, 1, n_features)

test_configs = [
    {'num_nodes': 500, 'spec_rad': 0.85, 'name': 'Small, spec_rad=0.85'},
    {'num_nodes': 500, 'spec_rad': 0.95, 'name': 'Small, spec_rad=0.95'},
    {'num_nodes': 1000, 'spec_rad': 0.85, 'name': 'Medium, spec_rad=0.85'},
    {'num_nodes': 1000, 'spec_rad': 0.95, 'name': 'Medium, spec_rad=0.95'},
]

print(f"\nTest data: {n_samples} samples, {n_timesteps} timesteps, {n_features} features\n")

# Test Standard API
print("="*80)
print("Testing Standard API (pyreco.models)")
print("="*80)

try:
    from pyreco.models import ReservoirComputer as RC_Standard

    for config in test_configs:
        print(f"\n{config['name']}:")
        print(f"  Creating model...")

        model = RC_Standard(
            num_nodes=config['num_nodes'],
            density=0.05,
            activation='tanh',
            spec_rad=config['spec_rad'],
            leakage_rate=0.3,
            fraction_input=0.5,
            fraction_output=1.0,
            optimizer='ridge'
        )

        print(f"  Training...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        print(f"  ✅ Train time: {train_time:.3f} seconds")

except Exception as e:
    print(f"  ❌ Error: {e}")

# Test Custom API
print("\n" + "="*80)
print("Testing Custom API (pyreco.custom_models)")
print("="*80)

try:
    from pyreco.custom_models import RC as RC_Custom
    from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

    for config in test_configs:
        print(f"\n{config['name']}:")
        print(f"  Creating model...")

        m = RC_Custom()
        m.add(InputLayer(input_shape=(n_timesteps, n_features)))
        m.add(RandomReservoirLayer(
            nodes=config['num_nodes'],
            density=0.05,
            activation='tanh',
            leakage_rate=0.3,
            spec_rad=config['spec_rad'],
        ))
        m.add(ReadoutLayer(
            output_shape=(None, n_features),
            fraction_out=1.0,
        ))
        m.compile(
            optimizer='ridge',
            metrics=["mse"],
            discard_transients=20
        )

        print(f"  Training...")
        start = time.time()
        model_fit_start = time.time()

        # Set a timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Training timeout after 60 seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout

        try:
            m.fit(X_train, y_train)
            signal.alarm(0)  # Cancel alarm
            train_time = time.time() - start
            print(f"  ✅ Train time: {train_time:.3f} seconds")
        except TimeoutError:
            signal.alarm(0)
            print(f"  ⏰ TIMEOUT after 60 seconds!")

except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("Diagnosis Complete")
print("="*80)
print("\nConclusions:")
print("  - If Standard API is fast (~1s) but Custom API is slow (>10s),")
print("    then Custom API has implementation issues")
print("  - If spec_rad=0.95 is significantly slower than 0.85,")
print("    then reservoir dynamics are causing the problem")
print("="*80)
