"""
Verify the optimization effect of improved node selection logic

This script shows the parameter reduction and expected speedup
from the improved num_nodes selection strategy that follows ESN best practices.

Changes:
1. Prioritize fraction_output=1.0 (standard ESN practice)
2. Minimize num_nodes to reduce O(N²) computation complexity
3. Allow slightly lower trainable params (90%+ of target)
"""

from src.utils.node_number import compute_readout_F_from_budget, best_num_nodes_and_fraction_out


def best_num_nodes_and_fraction_out_OLD(Ftarget: int, num_nodes_candidates: list[int]) -> tuple[int, float, int]:
    """Old logic for comparison"""
    sorted_candidates = sorted([n for n in num_nodes_candidates if n > 0])
    suitable_candidates = [n for n in sorted_candidates if n >= Ftarget]

    if suitable_candidates:
        chosen_num_nodes = suitable_candidates[0]
        chosen_F_real = Ftarget
        chosen_fraction_out = Ftarget / chosen_num_nodes
    else:
        chosen_num_nodes = sorted_candidates[-1]
        chosen_F_real = chosen_num_nodes
        chosen_fraction_out = 1.0

    return chosen_num_nodes, chosen_fraction_out, chosen_F_real


def analyze_config(budget, n_output_features=3):
    """Analyze configuration for a given budget"""
    Ftarget = compute_readout_F_from_budget(budget, n_output_features)

    # Unified candidates (max 3500)
    candidates = [50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 3500]

    # Old logic
    old_nodes, old_frac, old_F = best_num_nodes_and_fraction_out_OLD(Ftarget, candidates)

    # New logic (improved)
    new_nodes, new_frac, new_F = best_num_nodes_and_fraction_out(Ftarget, candidates)

    # Calculate parameters
    density = 0.1
    fraction_input = 0.5
    n_input_features = 3

    def calc_params(num_nodes, frac_out):
        input_params = int(num_nodes * n_input_features * fraction_input)
        reservoir_params = int(num_nodes * num_nodes * density)
        readout_params = int(num_nodes * frac_out * n_output_features)
        total = input_params + reservoir_params + readout_params
        return {
            'input': input_params,
            'reservoir': reservoir_params,
            'readout': readout_params,
            'total': total
        }

    old_params = calc_params(old_nodes, old_frac)
    new_params = calc_params(new_nodes, new_frac)

    # Calculate speedup (mainly from reservoir size reduction)
    # Forward pass complexity ~ O(num_nodes^2)
    speedup = (old_nodes / new_nodes) ** 2

    return {
        'budget': budget,
        'Ftarget': Ftarget,
        'old': {
            'num_nodes': old_nodes,
            'fraction_output': old_frac,
            'trainable_params': old_params['readout'],
            'total_params': old_params['total'],
            'reservoir_params': old_params['reservoir'],
        },
        'new': {
            'num_nodes': new_nodes,
            'fraction_output': new_frac,
            'trainable_params': new_params['readout'],
            'total_params': new_params['total'],
            'reservoir_params': new_params['reservoir'],
        },
        'speedup': speedup,
        'param_reduction': (1 - new_params['total'] / old_params['total']) * 100
    }


def main():
    print("\n" + "="*100)
    print("OPTIMIZATION ANALYSIS: Improved Node Selection Logic")
    print("="*100)
    print("\nChanges:")
    print("  1. Prioritize fraction_output=1.0 (standard ESN practice)")
    print("  2. Choose num_nodes closest to target (minimize O(N²) complexity)")
    print("  3. Allow 90%+ trainable params (acceptable trade-off)")
    print("="*100)

    budgets = {
        'SMALL': 1000,
        'MEDIUM': 10000,
        'LARGE': 100000,
    }

    for scale, budget in budgets.items():
        result = analyze_config(budget)

        print(f"\n{'='*100}")
        print(f"Scale: {scale} (Budget: {budget:,} trainable parameters)")
        print(f"{'='*100}")

        print(f"\n📊 OLD Logic (minimize num_nodes >= Ftarget):")
        print(f"  num_nodes:          {result['old']['num_nodes']:,}")
        print(f"  fraction_output:    {result['old']['fraction_output']:.3f} ⚠️ {'(NOT standard ESN)' if result['old']['fraction_output'] < 1.0 else '(standard)'}")
        print(f"  Trainable params:   {result['old']['trainable_params']:,} ({result['old']['trainable_params']/budget*100:.1f}% of target)")
        print(f"  Reservoir params:   {result['old']['reservoir_params']:,}")
        print(f"  Total params:       {result['old']['total_params']:,}")

        print(f"\n✅ NEW Logic (prioritize fraction_output=1.0):")
        print(f"  num_nodes:          {result['new']['num_nodes']:,}")
        print(f"  fraction_output:    {result['new']['fraction_output']:.3f} {'✅ (standard ESN)' if result['new']['fraction_output'] == 1.0 else '(adjusted)'}")
        print(f"  Trainable params:   {result['new']['trainable_params']:,} ({result['new']['trainable_params']/budget*100:.1f}% of target)")
        print(f"  Reservoir params:   {result['new']['reservoir_params']:,}")
        print(f"  Total params:       {result['new']['total_params']:,}")

        print(f"\n🚀 Improvement:")
        print(f"  Parameter reduction: {result['param_reduction']:.1f}%")
        print(f"  Expected speedup:    {result['speedup']:.2f}x")

        # Memory savings
        old_memory = result['old']['total_params'] * 8 / (1024**2)  # MB (float64)
        new_memory = result['new']['total_params'] * 8 / (1024**2)  # MB (float64)
        print(f"  Memory savings:      {old_memory:.1f} MB → {new_memory:.1f} MB ({old_memory - new_memory:.1f} MB saved)")

        # Training time estimate (per hyperparameter combination)
        print(f"\n⏱️  Estimated Training Time (per model):")
        old_time = result['old']['reservoir_params'] / 500000  # Rough estimate: 0.5M params per second
        new_time = result['new']['reservoir_params'] / 500000
        print(f"  OLD: ~{old_time:.2f}s × 96 combinations = {old_time * 96 / 60:.1f} min")
        print(f"  NEW: ~{new_time:.2f}s × 96 combinations = {new_time * 96 / 60:.1f} min")
        print(f"  Saved: {(old_time - new_time) * 96 / 60:.1f} min per scale")

    print("\n" + "="*100)
    print("SUMMARY: Improved Node Selection Logic")
    print("="*100)
    print("\n✅ Changes Implemented:")
    print("  1. Prioritize fraction_output=1.0 (follows standard ESN practice)")
    print("  2. Choose num_nodes <= Ftarget when possible (minimize O(N²) complexity)")
    print("  3. Accept 90%+ trainable params (better trade-off than large reservoir)")
    print("  4. Avoid wasting computed reservoir states")
    print("\n📚 Theoretical Support:")
    print("  - Jaeger (2001): Standard ESN uses full readout (fraction_output=1.0)")
    print("  - Lukoševičius (2012): Reservoir size typically 100-1000 nodes")
    print("  - Computation complexity O(N²) more important than trainable params")
    print("\n🚀 Performance Impact:")
    print("  - MEDIUM scale: 3000 nodes instead of 3500 (if Ftarget=3332)")
    print("  - Expected speedup: ~1.36x from reduced reservoir size")
    print("  - Memory savings: ~30% from smaller reservoir")
    print("  - Trainable params: 90% of target (acceptable)")
    print("  - fraction_output: 1.0 (standard ESN ✅)")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
