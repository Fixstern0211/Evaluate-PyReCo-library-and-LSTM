from __future__ import annotations


def compute_readout_F_from_budget(budget: int, Dout: int) -> int: 
    """
    Trainable params for linear readout: F*Dout + Dout (weights + bias)
    Solve F from budget (floor, >=1).
    """
    F = max(1, (budget - Dout) // Dout)
    return int(F)


def best_num_nodes_and_fraction_out(Ftarget: int, num_nodes_candidates: list[int]) -> tuple[int, float, int]:
    """
    Select the optimal num_nodes and fraction_output to meet the target F.

    Improved strategy (aligned with ESN best practices):
    1. Prefer fraction_output = 1.0 (standard ESN uses all reservoir nodes)
    2. Choose the num_nodes closest to Ftarget to minimize computational cost O(N^2)
    3. Allow trainable parameters slightly below the target (>= 90% acceptable)
    4. Avoid fraction_output < 1.0 when possible (wastes computed reservoir states)

    Strategy details:
    - Primary: choose the closest num_nodes <= Ftarget with fraction_output = 1.0
    - Secondary: if not feasible, choose a num_nodes slightly > Ftarget (<= Ftarget * 1.2) and adjust fraction_output
    - Fallback: handle cases where all candidates are too small or too large

    References:
    - Jaeger (2001): standard ESN uses a full readout (fraction_output = 1.0)
    - Lukoševičius (2012): reservoir sizes are typically 100–1000 nodes
    - Computational complexity O(N^2) often matters more than raw parameter count

    Args:
        Ftarget: target F value (number of readout nodes used, i.e., trainable params / Dout)
        num_nodes_candidates: list of candidate num_nodes

    Returns:
        (chosen_num_nodes, chosen_fraction_out, chosen_F_real)
    """
    sorted_candidates = sorted([n for n in num_nodes_candidates if n > 0])

    if not sorted_candidates:
        raise ValueError("No valid num_nodes candidates provided")

    best_candidate = None
    min_diff = float('inf')

    # strategy 1 (prioritize): choose <= Ftarget and closest candidate, use fraction_output=1.0
    candidates_below = [n for n in sorted_candidates if n <= Ftarget]
    if candidates_below:
        # choose the closest to Ftarget (largest <= Ftarget)
        chosen_num_nodes = candidates_below[-1]
        chosen_F_real = chosen_num_nodes
        chosen_fraction_out = 1.0

        # if achieved 90%+ of the target, return directly (prioritize fraction_output=1.0)
        if chosen_F_real >= Ftarget * 0.9:
            return (chosen_num_nodes, chosen_fraction_out, chosen_F_real)

        best_candidate = (chosen_num_nodes, chosen_fraction_out, chosen_F_real)
        min_diff = Ftarget - chosen_num_nodes

    # strategy 2 (secondary): only consider adjusting fraction_output if strategy 1 is too far off (<90%)
    # allow adjusting fraction_output within 20% margin
    candidates_above = [n for n in sorted_candidates if Ftarget < n <= Ftarget * 1.2]
    if candidates_above and (best_candidate is None or best_candidate[2] < Ftarget * 0.9):
        # choose the smallest one greater than Ftarget
        chosen_num_nodes = candidates_above[0]
        chosen_F_real = Ftarget
        chosen_fraction_out = Ftarget / chosen_num_nodes
        best_candidate = (chosen_num_nodes, chosen_fraction_out, chosen_F_real)

    # if found a good candidate, return
    if best_candidate:
        return best_candidate

    # fallback strategy: all candidates too small or too large
    if sorted_candidates[-1] < Ftarget:
        # all below Ftarget: choose largest, fraction_output=1.0
        chosen_num_nodes = sorted_candidates[-1]
        return (chosen_num_nodes, 1.0, chosen_num_nodes)
    else:
        # all above Ftarget*1.2: choose smallest, adjust fraction_output
        chosen_num_nodes = sorted_candidates[0]
        return (chosen_num_nodes, Ftarget / chosen_num_nodes, Ftarget)