"""
Budget-constrained parameter matching for ESN and LSTM.

Ensures both models have approximately equal total parameter counts
at every configuration evaluated during hyperparameter tuning.

ESN total parameters:
    N_total = N^2 * density + N * d_in * (1 - frac_in) + N * d_out
    Note: PyReCo's fraction_input = fraction of nodes EXCLUDED from input.

LSTM total parameters (PyTorch nn.LSTM + nn.Linear):
    L=1: 4h(h + d_in + 2) + h * d_out + d_out
    L>1: 4h(h + d_in + 2) + (L-1) * 4h(2h + 2) + h * d_out + d_out

    The +2 accounts for both bias_ih and bias_hh in PyTorch's LSTM.
"""

from __future__ import annotations
import math


# ============================================================================
# ESN budget computation
# ============================================================================

def esn_solve_num_nodes(budget: int, density: float, frac_in: float,
                        d_in: int, d_out: int, max_nodes: int = 1000) -> int | None:
    """
    Solve for N (num_nodes) given the budget constraint:
        N^2 * density + N * d_in * (1 - frac_in) + N * d_out = budget

    PyReCo's fraction_input specifies the fraction of nodes EXCLUDED from
    input connections, so actual input params = N * d_in * (1 - frac_in).

    Returns None if the required N exceeds max_nodes.
    """
    a = density
    b = d_in * (1 - frac_in) + d_out
    c = -budget

    if a <= 0:
        return None

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    N = (-b + math.sqrt(discriminant)) / (2 * a)
    N = max(1, int(N))

    if N > max_nodes:
        return None

    return N


def esn_total_params(num_nodes: int, density: float, frac_in: float,
                     d_in: int, d_out: int) -> dict:
    """
    Compute actual ESN parameter counts from the model configuration.
    frac_in is PyReCo's fraction_input (fraction EXCLUDED from input).
    """
    reservoir = int(num_nodes * num_nodes * density)
    input_w = int(num_nodes * d_in * (1 - frac_in))
    readout = num_nodes * d_out

    return {
        'reservoir': reservoir,
        'input': input_w,
        'readout': readout,
        'trainable': readout,
        'total': reservoir + input_w + readout,
    }


def esn_budget_grid(budget: int, density_values: list[float],
                    frac_in_values: list[float], d_in: int, d_out: int,
                    max_nodes: int = 1000) -> list[dict]:
    """
    Generate all feasible (density, frac_in, num_nodes) combinations
    that satisfy the budget constraint with N <= max_nodes.

    Each returned dict contains the structural parameters and actual param counts.
    """
    grid = []
    for density in density_values:
        for frac_in in frac_in_values:
            N = esn_solve_num_nodes(budget, density, frac_in, d_in, d_out,
                                    max_nodes=max_nodes)
            if N is None:
                continue

            params = esn_total_params(N, density, frac_in, d_in, d_out)
            grid.append({
                'num_nodes': N,
                'density': density,
                'fraction_input': frac_in,
                'fraction_output': 1.0,
                'param_info': params,
            })

    return grid


# ============================================================================
# LSTM budget computation
# ============================================================================

def lstm_solve_hidden_size(budget: int, d_in: int, d_out: int,
                           num_layers: int = 1) -> int:
    """
    Solve for hidden_size h given the budget constraint.

    PyTorch nn.LSTM actual parameter count:
        Layer 1:    4h(h + d_in + 2)          [weight_ih, weight_hh, bias_ih, bias_hh]
        Layer k>1:  4h(2h + 2)                [input is previous hidden]
        FC output:  h * d_out + d_out

    Total for L layers:
        4h(h + d_in + 2) + (L-1)*4h(2h + 2) + h*d_out + d_out = budget

    Rearranged as quadratic in h:
        a*h^2 + b*h + c = 0
        where a = 4 + (L-1)*8 = 8L - 4
              b = 4*(d_in + 2) + (L-1)*8 + d_out = 4*d_in + 8*L + d_out
              c = d_out - budget
    """
    a = 8 * num_layers - 4
    b = 4 * d_in + 8 * num_layers + d_out
    c = d_out - budget

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 1

    h = (-b + math.sqrt(discriminant)) / (2 * a)
    return max(1, int(h))


def lstm_total_params(hidden_size: int, d_in: int, d_out: int,
                      num_layers: int = 1) -> dict:
    """
    Compute actual LSTM parameter count matching PyTorch's implementation.
    """
    # Layer 1: weight_ih (4h, d_in) + weight_hh (4h, h) + bias_ih (4h) + bias_hh (4h)
    layer1 = 4 * hidden_size * (hidden_size + d_in + 2)

    # Additional layers: weight_ih (4h, h) + weight_hh (4h, h) + bias_ih (4h) + bias_hh (4h)
    additional = (num_layers - 1) * 4 * hidden_size * (2 * hidden_size + 2)

    # FC output layer: weight (d_out, h) + bias (d_out)
    fc = hidden_size * d_out + d_out

    lstm_params = layer1 + additional
    total = lstm_params + fc

    return {
        'lstm_params': lstm_params,
        'fc_params': fc,
        'trainable': total,
        'total': total,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
    }


def lstm_layer_hidden_map(budget: int, d_in: int, d_out: int,
                          layer_options: list[int] = None) -> dict[int, int]:
    """
    Compute hidden_size for each num_layers option to match the budget.
    Returns {num_layers: hidden_size} mapping.
    """
    if layer_options is None:
        layer_options = [1, 2]

    mapping = {}
    for nl in layer_options:
        h = lstm_solve_hidden_size(budget, d_in, d_out, num_layers=nl)
        if h >= 1:
            mapping[nl] = h

    return mapping


# ============================================================================
# Verification utilities
# ============================================================================

def verify_budget_match(budget: int, d_in: int, d_out: int,
                        density: float = 0.1, frac_in: float = 0.5):
    """
    Print a comparison table showing ESN and LSTM params at the given budget.
    """
    N = esn_solve_num_nodes(budget, density, frac_in, d_in, d_out)
    if N is None:
        print(f"ESN: N exceeds max_nodes for budget={budget}, d={density}, fi={frac_in}")
        return

    esn = esn_total_params(N, density, frac_in, d_in, d_out)

    print(f"Budget={budget}, d_in={d_in}, d_out={d_out}")
    print(f"  ESN: N={N}, density={density}, frac_in={frac_in}")
    print(f"    reservoir={esn['reservoir']}, input={esn['input']}, "
          f"readout={esn['readout']}")
    print(f"    trainable={esn['trainable']}, total={esn['total']}")

    for nl in [1, 2]:
        h = lstm_solve_hidden_size(budget, d_in, d_out, num_layers=nl)
        lstm = lstm_total_params(h, d_in, d_out, num_layers=nl)
        print(f"  LSTM (L={nl}): h={h}")
        print(f"    lstm_params={lstm['lstm_params']}, fc={lstm['fc_params']}")
        print(f"    trainable={lstm['trainable']}, total={lstm['total']}")


if __name__ == '__main__':
    print("=" * 60)
    print("BUDGET MATCHING VERIFICATION")
    print("=" * 60)

    for budget in [1000, 10000, 50000]:
        print()
        for d_in, d_out, name in [(3, 3, "Lorenz"), (1, 1, "1D")]:
            print(f"\n--- {name}, budget={budget} ---")
            verify_budget_match(budget, d_in, d_out, density=0.05, frac_in=0.3)

    print("\n\n" + "=" * 60)
    print("ESN BUDGET GRID (Large, Lorenz)")
    print("=" * 60)
    grid = esn_budget_grid(
        budget=50000,
        density_values=[0.01, 0.03, 0.05, 0.1],
        frac_in_values=[0.1, 0.3, 0.5],
        d_in=3, d_out=3, max_nodes=1000,
    )
    for g in grid:
        pi = g['param_info']
        print(f"  d={g['density']:.2f} fi={g['fraction_input']:.1f} "
              f"N={g['num_nodes']:>5d} trainable={pi['trainable']:>5d} "
              f"total={pi['total']:>6d}")

    excluded = 12 - len(grid)
    print(f"\n  Feasible: {len(grid)}/12, Excluded (N>1000): {excluded}")
