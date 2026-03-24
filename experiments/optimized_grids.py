"""
Optimized Task-Specific Grids for Pre-Tuning

Consolidated grids including original optimized ranges and supplementary
boundary extensions from parameter sensitivity analysis.

Design principles:
1. Avoid spec_rad >= 1.0 (causes extreme slowdowns / instability)
2. Extend boundaries where pretuning best values hit grid edges
3. Dataset-specific ranges based on empirical sensitivity analysis
"""


def optimized_grid_lorenz(num_nodes, fraction_output):
    """
    Optimized grid for Lorenz chaotic system

    Extended based on parameter sensitivity analysis:
    - spec_rad: trend ↗, added 0.99 (approaching but not reaching unstable 1.0)
    - leakage_rate: trend ↗, added 0.8, 0.9
    - density: trend ↘, added 0.02, 0.01
    - fraction_input: trend ↘, added 0.2, 0.1

    Grid: 5×8×5×5 = 1000 combinations
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.7, 0.8, 0.9, 0.95, 0.99],  # Added 0.99
        "leakage_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Added 0.8, 0.9
        "density": [0.01, 0.02, 0.03, 0.05, 0.1],  # Added 0.01, 0.02
        "fraction_input": [0.1, 0.2, 0.3, 0.5, 0.75],  # Added 0.1, 0.2
        "fraction_output": [fraction_output],
    }
    return grid


def optimized_grid_mackeyglass(num_nodes, fraction_output):
    """
    Optimized grid for Mackey-Glass time series

    Extended based on pretuning + supplementary parameter sensitivity analysis:
    - spec_rad: 0.95 (0.9 was at MAX edge), 0.99 (boundary extension)
    - leakage_rate: 0.4, 0.5 (0.3 at MAX edge), 0.6, 0.7 (boundary extension)
    - density: 0.05 (0.1 at MIN edge), 0.03 (boundary extension)
    - fraction_input: 0.3 (0.5 at MIN edge), 0.1, 0.2 (boundary extension)

    Grid: 4×7×4×6 = 672 combinations
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.8, 0.9, 0.95, 0.99],  # Added 0.95, 0.99
        "leakage_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Added 0.4-0.7
        "density": [0.03, 0.05, 0.1, 0.15],  # Added 0.03
        "fraction_input": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],  # Added 0.1, 0.2
        "fraction_output": [fraction_output],
    }
    return grid


def optimized_grid_santafe(num_nodes, fraction_output):
    """
    Optimized grid for Santa Fe laser data

    Extended based on pretuning + supplementary parameter sensitivity analysis:
    - spec_rad: 0.85 (finer resolution), 0.95, 0.99 (boundary extension)
    - leakage_rate: 0.7 (0.6 at upper edge), 0.8 (boundary extension)
    - density: 0.03 (0.05 at lower edge), 0.01, 0.02 (boundary extension)
    - fraction_input: 0.2 (0.3 at lower edge), 0.1 (boundary extension)

    Grid: 6×6×5×5 = 900 combinations
    """
    grid = {
        "num_nodes": [num_nodes],
        "spec_rad": [0.7, 0.8, 0.85, 0.9, 0.95, 0.99],  # Added 0.85, 0.95, 0.99
        "leakage_rate": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Added 0.7, 0.8
        "density": [0.01, 0.02, 0.03, 0.05, 0.1],  # Added 0.01, 0.02
        "fraction_input": [0.1, 0.2, 0.3, 0.5, 0.75],  # Added 0.1
        "fraction_output": [fraction_output],
    }
    return grid


def get_optimized_grid(dataset, num_nodes, fraction_output):
    """
    Get optimized grid for a specific dataset

    Args:
        dataset: 'lorenz', 'mackeyglass', or 'santafe'
        num_nodes: Number of reservoir nodes
        fraction_output: Fraction of output connections

    Returns:
        Parameter grid dictionary
    """
    if dataset == 'lorenz':
        return optimized_grid_lorenz(num_nodes, fraction_output)
    elif dataset == 'mackeyglass':
        return optimized_grid_mackeyglass(num_nodes, fraction_output)
    elif dataset == 'santafe':
        return optimized_grid_santafe(num_nodes, fraction_output)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == '__main__':
    """Test the grids"""
    import numpy as np

    datasets = ['lorenz', 'mackeyglass', 'santafe']
    num_nodes = 300
    fraction_output = 1.0

    print("\n" + "="*80)
    print("CONSOLIDATED PRE-TUNING GRIDS")
    print("="*80)

    total_combos = 0
    for dataset in datasets:
        grid = get_optimized_grid(dataset, num_nodes, fraction_output)
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        total_combos += n_combos

        print(f"\n{dataset.upper()}: {n_combos} combinations")
        for k, v in grid.items():
            if k not in ('num_nodes', 'fraction_output'):
                print(f"  {k:.<20} {v}")
        print(f"  With 5-fold CV: {n_combos * 5} trainings")

    print(f"\n{'='*80}")
    print(f"TOTAL: {total_combos} combinations, {total_combos * 5} trainings (5-fold CV)")
    print("="*80)
