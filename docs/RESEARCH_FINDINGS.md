# Research Findings and Technical Documentation

> **Note (2026-03-01)**: This document contains early-stage findings from Nov 2025.
> After fixing LSTM early stopping (Feb 2026), the original hypothesis was **not confirmed**.
> LSTM outperforms PyReCo in 51/54 accuracy comparisons. See `docs/RESULTS_SUMMARY.md` for current results.

## Research Objective
Compare Reservoir Computing (RC) and LSTM performance under equal parameter budgets, testing the hypothesis:
**RC achieves better performance with faster or comparable training speed under the same parameter budget**

**Outcome**: Hypothesis partially confirmed for training speed only. LSTM achieves better accuracy in nearly all configurations.

## Key Technical Discoveries

### 1. Reinterpretation of Parameter Budget (Critical Innovation)

**Problem**: Initially using "trainable parameters" as budget caused:
- RC has only readout layer trainable (~3-5% of total)
- LSTM has all parameters trainable (100% of total)
- Unfair comparison: RC requires massive reservoir to reach target trainable params

**Solution**: Use "total parameters" as budget instead
- RC total params = Input layer + Reservoir layer (fixed) + Readout layer (trainable)
- LSTM total params = All parameters (trainable)
- Fair comparison: same computational and memory cost

**Implementation**:
```python
# Calculate num_nodes from total parameter budget
# Total params ≈ N² * density (reservoir dominates)
num_nodes_approx = int(math.sqrt(budget / density))
```

**Final Budget Configuration** (Total Parameters):
| Budget Level | num_nodes | Total Params | Trainable Params | RC Train Time | LSTM hidden | LSTM Total | LSTM Train Time |
|--------------|-----------|--------------|------------------|---------------|-------------|------------|-----------------|
| SMALL (1k)   | 100 | 1,450 | 300 (20.7%) | ~1s | 16 | 3,312 | ~22s |
| MEDIUM (10k) | 300 | 10,350 | 900 (8.7%) | ~8s | 28 | 9,828 | ~36s |
| LARGE (30k)  | 550 | 33,550 | 1,650 (4.9%) | ~30s | 48 | 29,139 | ~50s |

**Why Total Parameters (not Trainable)?**
- **Literature alignment**: Jaeger (2001), Lukoševičius (2012) compare ESNs by reservoir size (total)
- **Fair comparison**: Equal memory and computational cost
- **Practical relevance**: Deployment constraints care about total params
- **Avoids conflation**: Isolates training algorithm efficiency (ridge vs backprop)

**Why Trainable Budget Failed** (方案C analysis):
- LSTM minimum viable config: hidden=16, 2 layers → 3,312 trainable params
- If budget=3,312 trainable: RC needs Ftarget=1,104 → **nodes≈900** (too large, slow)
- If budget=1,000 trainable: LSTM formula gives hidden=8, but performance too poor
- **Root issue**: RC has ~5-10% trainable, LSTM has 100% trainable → incompatible scaling

### 2. PyReCo Library Performance Issue (Implementation Bottleneck)

**Discovery**: PyReCo library training significantly slower than theoretical expectation
- **Theory**: Ridge regression solver, complexity O(F³), F = readout feature dimension
  - For F~300-3000, should be <1 second
- **Actual Measurements**:
  - 100 nodes: ~1s (close to theory) ✓
  - 300 nodes: ~8s (expected <1s) - **8x slower**
  - 1000 nodes: **104-2940s** (expected <1s) - **up to 2940x slower!**
    - Normal combinations (spec_rad=0.8, density=0.05): ~104s
    - Problematic combinations (spec_rad=0.9, density=0.1): **up to 2940s (49 min)**

**Root Cause**: Library implementation issue, not the algorithm itself
- Source code inspection revealed: `pyreco.models.ReservoirComputer` internally uses `pyreco.custom_models.RC`
- Standard API and Custom API use the same inefficient implementation
- Potential bottlenecks: layer-by-layer construction, unnecessary computations
- **Hyperparameter sensitivity**: High spec_rad (≥0.9) + high density (≥0.1) causes extreme slowdown

**Practical Constraints Identified**:
- ✅ **Safe range**: nodes ≤ 600, training time predictable (~1-30s)
- ⚠️ **Risk range**: nodes 600-1000, some combinations very slow (up to 49 min)
- ❌ **Infeasible**: nodes > 1000, unreliable and too slow for experiments

**Mitigation Strategies**:
1. **Limit num_nodes to 600** for main experiments (safe, predictable)
2. Remove Custom model testing (identical to Standard)
3. Use total parameter budget (keeps nodes ≤ 600 for SMALL/MEDIUM)
4. **Optimize hyperparameter grids**: Avoid spec_rad ≥ 1.0 in pre-tuning

**Suggested Paper Narrative**:
- "We implement RC models using the PyReCo library, noting that actual training time is longer than theoretical O(F³) expectation"
- "This may be a library implementation issue rather than algorithmic limitation, subject to future optimization"
- "Nevertheless, RC training remains significantly faster than LSTM backpropagation"

### 3. ESN Node Selection Logic Optimization (Best Practice Alignment)

**Old Logic Issues**:
```python
# Select minimum num_nodes >= Ftarget
# Then use fraction_output < 1.0 to precisely match trainable params
chosen_num_nodes = min([n for n in candidates if n >= Ftarget])
chosen_fraction_out = Ftarget / chosen_num_nodes
```
- Violates ESN standard practice (should use fraction_output=1.0)
- Wastes computed reservoir states
- Ignores O(N²) computational complexity

**Improved Logic** (literature-based):
```python
# 1. Prioritize num_nodes <= Ftarget, use fraction_output=1.0
# 2. Accept 90%+ trainable params (reduce computational complexity)
# 3. Avoid fraction_output < 1.0
candidates_below = [n for n in sorted_candidates if n <= Ftarget]
if candidates_below:
    chosen_num_nodes = candidates_below[-1]  # closest match
    chosen_fraction_out = 1.0
    if chosen_F_real >= Ftarget * 0.9:  # 90%+ acceptable
        return (chosen_num_nodes, 1.0, chosen_F_real)
```

**Theoretical Support**:
- Jaeger (2001): Standard ESN uses fully connected readout (fraction_output=1.0)
- Lukoševičius (2012): Reservoir size typically 100-1000 nodes
- Computational complexity O(N²) more important than trainable parameter count

**Impact**:
- Uses standard ESN configuration (fraction_output=1.0)
- Reduces unnecessary large reservoirs
- Expected training speedup 1.36-2.78x

### 4. Spectral Radius Effects

**Definition**: Maximum eigenvalue of reservoir weight matrix

**Physical Interpretation**:
- spec_rad < 1.0: Stable, signals gradually decay
- spec_rad ≈ 1.0: "Edge of chaos", information persists longer
- spec_rad > 1.0: Unstable, signal explosion

**Practical Recommendations**:
- General tasks: 0.8-0.9 (balance stability and memory)
- Long-term dependencies: 0.95-0.99 (longer memory, slower convergence)
- Short-term tasks: 0.5-0.7 (fast convergence)

**Performance Impact** (measured):
- spec_rad=0.85: Normal training speed
- spec_rad=0.95: Significantly slower (convergence issues)
- spec_rad=0.99: May cause timeout

**Hyperparameter Grid** (currently used):
```python
param_grid_pyreco_standard = {
    'spec_rad': [0.8, 0.85, 0.9, 0.95],     # 4 values
    'leakage_rate': [0.1, 0.3, 0.5, 0.7],   # 4 values
    'density': [0.05, 0.1],                 # 2 values
    'activation': ['tanh', 'relu', 'identity']  # 3 values
}
# Total: 4×4×2×3 = 96 combinations
```

## Experimental Design

### Datasets
- **Lorenz**: Chaotic system, tests long-term dependencies
- **Mackey-Glass**: Delay differential equation, tests nonlinear dynamics
- **Sinusoidal**: Periodic signal, baseline test

### Parameter Budget Levels
| Level | Total Params | RC Config | LSTM Config | Purpose |
|-------|--------------|-----------|-------------|---------|
| SMALL | 1,000 | 100 nodes | 16 hidden, 2 layers | Quick validation |
| MEDIUM | 10,000 | 300 nodes | 28 hidden, 2 layers | Primary comparison |
| LARGE | 100,000 | 1,000 nodes | 90 hidden, 2 layers | Scaling test |

### Training Ratios
[0.1, 0.3, 0.5, 0.7, 0.9, 0.95] - Test small sample learning ability

### Random Seeds
5 seeds [42, 123, 456, 789, 2024] - Evaluate stability

### Evaluation Metrics
1. **MSE** (Mean Squared Error)
2. **Training Time**
3. **Memory Usage**
4. **NMSE** (Normalized MSE) = MSE / var(y_true)

### Statistical Testing
Paired t-test
- Compare RC vs LSTM for each configuration (dataset, scale, train_ratio)
- Use results from 5 seeds
- Significance level α=0.05

## Technical Implementation Highlights

### 1. Memory Leak Fixes
```python
# Don't return final_model, only best config
def tune_pyreco_hyperparameters(...):
    # ...hyperparameter search...
    # BAD: return best_params, final_model
    return best_params  # ✅ Avoid memory leak

# Explicit cleanup
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

### 2. GPU Acceleration
```python
# Automatic device detection
device = None  # PyTorch auto-selects best device
# Priority: MPS (Apple Silicon) > CUDA > CPU

# LSTM uses GPU training
model = LSTMModel(..., device=device)
```

### 3. Parallel Experiments
```python
# Use multiprocessing Pool
# Each (dataset, scale) combination runs independently
# Max 3 workers (avoid memory overflow)
with Pool(processes=min(3, cpu_count())) as pool:
    results = pool.map(run_single_experiment, configs)
```

### 4. Real-time Progress Display
```python
# run_comprehensive_experiments.py: Remove capture_output
result = subprocess.run(cmd, timeout=14400)  # Show real-time output

# run_parallel_experiments.py: Keep capture_output
result = subprocess.run(cmd, capture_output=True)  # Avoid mixed output
```

## Expected Results and Research Value

### Hypothesis Testing
**H1**: RC trains significantly faster than LSTM under equal total parameter budget
- Expected: RC training ~10-100s, LSTM ~1-5min
- Speedup ratio: 10-30x

**H2**: RC performs better in small sample learning (train_ratio=0.1-0.3)
- Reason: Ridge regression has strong generalization, less prone to overfitting
- LSTM requires more data to train large parameter count

**H3**: Medium scale (MEDIUM, 10k params) is RC's sweet spot
- Too small (SMALL): Insufficient capacity
- Too large (LARGE): Library performance issues

### Research Narrative Framework

**Introduction**:
"Reservoir Computing offers a training-efficient approach to time series modeling. We systematically compare RC and LSTM performance under controlled total parameter budgets."

**Methodological Innovations**:
1. Propose fair comparison framework based on total parameter budget
2. Follow ESN best practices for node selection optimization
3. Comprehensive evaluation across datasets, scales, and training ratios

**Expected Contributions**:
1. Quantify RC's training speed advantage (~30x speedup)
2. Identify RC-suitable scenarios (small sample, medium scale)
3. Provide practical hyperparameter selection guidance
4. Reveal PyReCo library performance bottlenecks (community contribution)

**Practical Applications**:
- Resource-constrained scenarios (edge devices, real-time systems)
- Rapid prototyping (low hyperparameter search cost)
- Small sample learning tasks (medical, financial time series)

## Code Structure

```
Evaluate-PyReCo-library-and-LSTM/
├── models/
│   ├── lstm_model.py          # LSTM implementation with GPU support
│   └── pyreco_wrapper.py      # PyReCo wrapper
├── src/utils/
│   ├── node_number.py         # Node selection logic (optimized)
│   ├── train_pyreco_model.py  # RC training and tuning
│   ├── train_lstm.py          # LSTM training and tuning
│   ├── load_dataset.py        # Data loading
│   └── process_datasets.py    # Data preprocessing
├── test_model_scaling.py      # Main experiment script (total param budget)
├── run_comprehensive_experiments.py   # Sequential execution
├── run_parallel_experiments.py        # Parallel execution
├── analyze_comprehensive_results.py   # Results analysis
├── verify_total_params_budget.py      # Budget verification
└── verify_optimization.py             # Optimization verification
```

## Literature Support for Total Parameter Budget

### Why Total Parameters (Not Trainable)?

**Core Argument**: Total parameter count reflects actual computational and memory cost, making it the fair comparison metric.

**Key Literature**:

1. **Jaeger (2001)** - "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
   - Defines reservoir size N as the primary architectural parameter
   - **Quote**: "The reservoir size N determines the memory capacity and computational power"
   - Comparisons in ESN literature use reservoir size (total), not readout size (trainable)

2. **Lukoševičius (2012)** - "A Practical Guide to Applying Echo State Networks"
   - Section 2.1: "Reservoir size typically 50 to a few thousand"
   - Performance comparisons (Section 4) use total network size
   - **Standard practice**: Compare models with similar N (total nodes)

3. **Verstraeten et al. (2007)** - "An Experimental Unification of Reservoir Computing Methods"
   - Neural Networks, 20(3), 391-403
   - Compares different RC architectures using total reservoir size
   - **Quote**: "We compare architectures with equal reservoir sizes to ensure fair evaluation"

**Why Trainable Budget Fails**:
- RC: Only readout trainable (~5-10% of total)
- LSTM: All parameters trainable (100% of total)
- **Architectural mismatch**: Equal trainable params → vastly different total sizes
- **Conflates effects**: Can't separate training algorithm advantage from parameter count

### LSTM Parameter Calculation

**Formula** (2-layer LSTM with input_size=3, output_size=3):
```python
# Layer 1: 4 gates × (input weights + recurrent weights + bias)
params_layer1 = 4 × h × (input_size + h + 1) = 4h(3 + h + 1) = 16h + 4h²

# Layer 2: input from layer 1 (size h)
params_layer2 = 4 × h × (h + h + 1) = 4h(2h + 1) = 8h² + 4h

# Output layer
params_output = h × output_size + output_size = 3h + 3

# Total (simplified, ignoring some bias terms)
Total ≈ 12h² + 15h
```

**Solving for h**:
```
12h² + 15h - budget = 0
h = (-15 + sqrt(225 + 48×budget)) / 24
```

**Literature Support**:
- **Hochreiter & Schmidhuber (1997)**: "Long Short-Term Memory" - Original LSTM paper
- **PyTorch Documentation**: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
  - Defines: params = 4 × (input_size × hidden_size + hidden_size²) per layer

### Pre-Tuning Strategy

**Why Use MEDIUM Scale (budget=10000) for Pre-Tuning?**

1. **Training speed**: ~8s per combination (fast enough for CV)
2. **Hyperparameter effects visible**: Sufficient model capacity to show performance differences
3. **Cross-scale consistency**: Optimal hyperparameter *relationships* should generalize across scales
4. **Avoid LARGE scale issues**: nodes=1000 has severe performance problems (up to 49 min per combination)

**Rationale**:
- Pre-tuning finds optimal hyperparameter **regions** (e.g., spec_rad=0.8 better than 1.0)
- These relationships transfer to other scales
- Main experiments test all scales with refined grids around pre-tuned optima

**Time Savings**:
- Without pre-tuning: 96 combinations × 3 scales × 45 experiments = high risk of timeouts
- With pre-tuning: 48 combinations (CV-optimized) × 3 scales × 45 experiments = predictable runtime

## Remaining Tasks

1. [ ] Run quick test to verify configuration correctness
2. [ ] Execute full experiments (90 runs: 3 datasets × 5 seeds × 6 ratios)
3. [ ] Generate analysis report (with statistical tests)
4. [ ] Visualize results (MSE vs train_ratio, training time comparison)
5. [ ] Write paper sections

## Summary of Key Findings (From Recent Implementation)

### 1. Budget Definition Evolution
- **Initial**: Trainable parameters only → Unfair (RC needs massive reservoirs)
- **Attempted**: Trainable budget (方案C) → Failed (LSTM min config too large)
- **Final**: Total parameter budget → ✅ Fair, literature-aligned, feasible

### 2. PyReCo Performance Bottlenecks Identified
- **Theoretical**: Ridge regression O(F³) should be <1s
- **Actual**: Up to 2940x slower than theory
- **Critical finding**: spec_rad ≥ 0.9 + density ≥ 0.1 → 10-30x slowdown
- **Safe range**: nodes ≤ 600 (training time <30s, predictable)

### 3. Experimental Design Optimizations
- **Grid reduction**: 96 → 48 combinations (remove problematic spec_rad ≥ 1.0)
- **Pre-tuning strategy**: Use MEDIUM scale (10k params) with 5-fold CV
- **Budget configuration**: SMALL/MEDIUM/LARGE = 1k/10k/30k total params
- **Seeds & ratios**: 5 seeds × 3 train_ratios = 15 trials per configuration

### 4. Training Time Tracking
- Separated three time metrics:
  - `tune_time`: Total hyperparameter search time
  - `best_combo_train_time`: Best single combination training time
  - `final_train_time`: Final model training time (for fair comparison)

### 5. LSTM Minimum Configuration Constraint
- **Empirical finding**: hidden < 8 gives very poor performance
- **Standard practice**: hidden ≥ 16 for 2-layer LSTM
- **Implication**: LSTM trainable params minimum ≈ 3,312
- **Impact on budget choice**: Must use total params to allow smaller RC configs

## Key References

1. **Jaeger (2001)**: "The 'Echo State' Approach to Analysing and Training Recurrent Neural Networks"
   - GMD Report 148, German National Research Center for Information Technology
   - Defines ESN architecture and training method
   - Standard configuration uses fully connected readout
   - Establishes reservoir size as primary comparison metric

2. **Lukoševičius (2012)**: "A Practical Guide to Applying Echo State Networks"
   - In: Neural Networks: Tricks of the Trade (pp. 659-686). Springer
   - DOI: 10.1007/978-3-642-35289-8_36
   - Reservoir size typically 100-1000 nodes
   - Hyperparameter selection guidance
   - Spectral radius effects

3. **Hochreiter & Schmidhuber (1997)**: "Long Short-Term Memory"
   - Neural Computation, 9(8), 1735-1780
   - DOI: 10.1162/neco.1997.9.8.1735
   - Original LSTM paper, defines parameter structure

4. **Verstraeten et al. (2007)**: "An Experimental Unification of Reservoir Computing Methods"
   - Neural Networks, 20(3), 391-403
   - DOI: 10.1016/j.neunet.2007.04.003
   - Establishes total reservoir size as comparison standard

5. **PyTorch LSTM Documentation**:
   - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
   - Official parameter counting formulas

6. **PyReCo Documentation**:
   - Library API usage
   - Performance issue documentation (our findings)

## Experimental Log

### Initial Test Results
- LSTM device='auto' error: Fixed (converted to None)
- PyReCo Custom 3000 nodes: 1118s training time (performance issue confirmed)
- PyReCo Custom spec_rad=0.95: Timeout (>5.5 hours)

### Optimization Effects
- Total parameter budget approach:
  - SMALL: 100 nodes, ~1.3s ✅
  - MEDIUM: 300 nodes, ~11.5s ✅
  - LARGE: 1000 nodes, ~127.7s ✅
- All configurations feasible, no timeout risk

## Key Equations

**RC Total Parameters**:
```
Total = N_input + N_reservoir + N_readout
      = (N × D_in × f_in) + (N² × density) + (N × f_out × D_out)
      ≈ N² × density  (reservoir dominates)
```

**LSTM Total Parameters** (2 layers):
```
Layer 1: 4 × (D_in × H + H² + H)
Layer 2: 4 × (H × H + H² + H)
Output:  H × D_out + D_out
Total ≈ 8H² + 4HD_in + D_out(H + 1)
```

**Node Calculation from Budget**:
```
Budget ≈ N² × density
N ≈ sqrt(Budget / density)
```

## Comparison Summary Table

| Aspect | RC (PyReCo) | LSTM |
|--------|-------------|------|
| Training Method | Ridge regression | Backpropagation + Adam |
| Trainable Params | Readout only (~3-9%) | All parameters (100%) |
| Training Complexity | O(F³) where F=readout features | O(T × P) where T=epochs, P=params |
| Expected Speed | Fast (seconds) | Slower (minutes) |
| Memory (inference) | High (stores reservoir) | Moderate (hidden states only) |
| Memory (training) | Moderate (reservoir states) | High (gradients for all params) |
| Small Sample Performance | Expected: Better (ridge regularization) | Expected: Worse (overfitting risk) |
| Hyperparameter Sensitivity | High (spec_rad, leakage, density) | Moderate (lr, hidden_size, layers) |

## Notes for Paper Writing

**Abstract Points**:
- Total parameter budget framework enables fair RC-LSTM comparison
- RC achieves X% accuracy of LSTM with Y× faster training
- Most effective in small sample regime (train_ratio < 0.5)
- Identifies library implementation bottleneck (contribution to community)

**Introduction Hook**:
"While LSTMs dominate time series forecasting benchmarks, their training cost limits applicability in resource-constrained scenarios. Reservoir Computing offers an alternative: random projections + linear readout. But how do they compare under equal computational budgets?"

**Method Contribution**:
"Previous RC-LSTM comparisons use trainable parameters (unfair) or arbitrary network sizes. We propose total parameter budget: RC and LSTM use equal memory/computation, enabling direct comparison of training efficiency vs. predictive accuracy."

**Results Framing**:
- Figure 1: MSE vs train_ratio curves (all datasets, MEDIUM scale)
- Figure 2: Training time comparison (log scale)
- Figure 3: Parameter efficiency (accuracy per parameter)
- Table 1: Paired t-test results across all configurations

**Discussion**:
- When to use RC: small samples, fast iteration, edge deployment
- When to use LSTM: large datasets, maximum accuracy, ample compute
- PyReCo bottleneck: community improvement opportunity
- Future work: custom RC implementation, reservoir design optimization

---

**Document Version**: v1.0
**Last Updated**: 2025-11-05
**Status**: Ready to run full experiments
