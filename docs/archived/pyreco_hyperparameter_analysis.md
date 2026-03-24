# PyReCo Hyperparameter Analysis and Optimization Guide

## Why Do Results Differ Before and After the Fix?

### Root Cause
Even though `density` was overridden by the grid, **`fraction_input` used the wrong default value**:

```python
# Before the fix (INCORRECT):
for combo in combinations:
    params = dict(zip(keys, combo))  # Only contains grid parameters
    # Example: params = {'density': 0.5, 'spec_rad': 0.8}

    model = PyReCoStandardModel(**params, verbose=False)
    # ❌ fraction_input uses PyReCoStandardModel.__init__ default value 1.0
    # ✅ Should use train_pyreco_model.py default value 0.5
```

### Impact Analysis
- **Before fix**: `fraction_input=1.0` (100% nodes receive input)
- **After fix**: `fraction_input=0.5` (50% nodes receive input)
- **Impact**: 2x difference! Significantly affects reservoir dynamics and generalization

---

## Complete PyReCo Hyperparameter Reference

### 1. Hyperparameter Importance Ranking

| Rank | Parameter | Importance | Impact | Tuning Priority |
|------|-----------|-----------|---------|-----------------|
| 🥇 1 | `num_nodes` | ⭐⭐⭐⭐⭐ | Model capacity, computational cost | Must tune |
| 🥈 2 | `spec_rad` | ⭐⭐⭐⭐⭐ | Dynamic stability, memory length | Must tune |
| 🥉 3 | `leakage_rate` | ⭐⭐⭐⭐ | Time scale, memory decay | Must tune |
| 4 | `fraction_input` | ⭐⭐⭐⭐ | Input information flow, sparsity | Should tune |
| 5 | `density` | ⭐⭐⭐⭐ | Connection sparsity, efficiency | Should tune |
| 6 | `ridge_alpha` | ⭐⭐⭐ | Regularization, generalization | Recommended |
| 7 | `fraction_output` | ⭐⭐⭐ | Output connections, cost | Optional |
| 8 | `activation` | ⭐⭐ | Nonlinearity | Usually fixed |
| 9 | `optimizer` | ⭐⭐ | Training method | Usually fixed |

---

## 2. Detailed Parameter Analysis

### 🥇 1. `num_nodes` (Number of Reservoir Nodes)
**Purpose**: Controls reservoir capacity and representation power

**Effects**:
- ⬆️ Increase → Greater model capacity, can learn more complex patterns
- ⬇️ Decrease → Faster computation, but may underfit

**Recommended Ranges**:
- Small datasets (< 1000 samples): `[50, 100, 200]`
- Medium datasets (1000-5000): `[200, 500, 800]`
- Large datasets (> 5000): `[800, 1200, 2000]`

**Note**: Usually constrained by budget

---

### 🥈 2. `spec_rad` (Spectral Radius)
**Purpose**: Controls reservoir dynamic stability and memory length

**Effects**:
- `spec_rad < 1.0`: Stable, short-term memory, good for fast-changing signals
- `spec_rad ≈ 1.0`: Edge of Chaos, balances short/long-term memory
- `spec_rad > 1.0`: Unstable but has long-term memory, good for long dependencies

**Recommended Ranges**:
- Conservative: `[0.5, 0.7, 0.9]` (stability-focused)
- Standard: `[0.8, 0.9, 1.0, 1.1]` (explore edge of chaos)
- Aggressive: `[0.9, 1.0, 1.1, 1.2, 1.5]` (long-term dependencies)

**Experience**:
- Lorenz chaotic system: `0.9-1.2` works well
- Stationary time series: `0.5-0.9` works well
- Long-term dependency tasks: `1.0-1.5` works well

---

### 🥉 3. `leakage_rate` (Leakage Rate)
**Purpose**: Controls time scale of reservoir state updates

**Formula**: `h(t) = (1 - α) * h(t-1) + α * tanh(W*h(t-1) + U*x(t))`
- `α = leakage_rate`

**Effects**:
- `α → 0`: Long memory, slow state changes (slow dynamics)
- `α → 1`: Short memory, fast state changes (fast dynamics)

**Recommended Ranges**:
- Fast-changing signals: `[0.5, 0.7, 0.9]`
- Standard tasks: `[0.2, 0.3, 0.4, 0.5]`
- Slow dynamic systems: `[0.1, 0.2, 0.3]`

**Experience**:
- High-frequency signals (stock minute data): `0.5-0.9`
- Medium-frequency signals (Lorenz system): `0.2-0.5`
- Low-frequency signals (monthly temperature): `0.1-0.3`

---

### 4. `fraction_input` (Input Connection Fraction)
**Purpose**: Controls what fraction of reservoir nodes receive input

**Effects**:
- `fraction_input = 1.0`: All nodes receive input (dense connections)
  - Pros: Full information flow
  - Cons: May overfit, high computation
- `fraction_input = 0.5`: 50% nodes receive input (sparse connections)
  - Pros: Regularization effect, better generalization
  - Cons: May underfit

**Recommended Ranges**:
- Exploratory: `[0.3, 0.5, 0.7, 1.0]`
- Standard: `[0.5, 0.75, 1.0]`

**Experience**:
- Small datasets: `0.3-0.5` (prevent overfitting)
- Large datasets: `0.7-1.0` (utilize data fully)

---

### 5. `density` (Reservoir Connection Density)
**Purpose**: Controls sparsity of internal reservoir connections

**Effects**:
- `density → 1.0`: Fully connected (dense)
  - Pros: Complex dynamics, strong representation
  - Cons: High computation, may overfit
- `density → 0.0`: Sparse connections
  - Pros: Fast computation, better generalization
  - Cons: May underfit

**Recommended Ranges**:
- Large reservoirs (>500 nodes): `[0.05, 0.1, 0.2]`
- Medium reservoirs (100-500): `[0.1, 0.2, 0.4]`
- Small reservoirs (<100): `[0.3, 0.5, 0.8]`

**Experience**:
- Sparse connections usually perform better (Echo State Property)
- `density = 0.1` is a good starting point

---

### 6. `ridge_alpha` (Ridge Regression Regularization)
**Purpose**: Controls regularization strength for output layer training

**Effects**:
- `alpha → 0`: No regularization, may overfit
- `alpha → ∞`: Strong regularization, may underfit

**Recommended Ranges**:
- Logarithmic search: `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]`

**Experience**:
- Small datasets: `1e-2 ~ 1.0`
- Large datasets: `1e-5 ~ 1e-3`

---

### 7. `fraction_output` (Output Connection Fraction)
**Purpose**: Controls what fraction of reservoir nodes connect to output

**Effects**:
- Mainly affects computational cost
- Usually constrained by budget

**Recommendation**:
- If budget allows: `1.0` (use all nodes)
- If budget tight: determined by budget calculation

---

### 8. `activation` (Activation Function)
**Purpose**: Controls nonlinearity of reservoir

**Options**:
- `'tanh'`: Standard choice, range [-1, 1]
- `'relu'`: ReLU, suitable for certain tasks
- `'sigmoid'`: Sigmoid, range [0, 1]

**Recommendation**:
- Usually fixed to `'tanh'` (Echo State Network standard)
- If tuning, try: `['tanh', 'relu']`

---

### 9. `optimizer` (Optimizer)
**Purpose**: Controls training method for output layer

**Options**:
- `'ridge'`: Ridge regression (PyReCo built-in)
- `RidgeSK(alpha=x)`: sklearn's Ridge (custom alpha)
- `'pinv'`: Pseudo-inverse (no regularization)

**Recommendation**:
- Usually fixed to `'ridge'` or `RidgeSK`

---

## 3. Recommended Complete Grid Configurations

### 📊 Plan A: Quick Exploration (27 combinations)
```python
grid_fast = {
    "num_nodes": [800],  # Fixed (determined by budget)
    "spec_rad": [0.8, 0.9, 1.0],  # 3 values
    "leakage_rate": [0.2, 0.3, 0.5],  # 3 values
    "density": [0.05, 0.1, 0.2],  # 3 values
    "fraction_input": [0.5],  # Fixed
    "fraction_output": [chosen_fraction_out],  # Fixed (by budget)
}
# Total: 3 × 3 × 3 = 27 combinations
```

### 📊 Plan B: Standard Search (240 combinations)
```python
grid_standard = {
    "num_nodes": [800],  # Fixed (by budget)
    "spec_rad": [0.7, 0.8, 0.9, 1.0, 1.1],  # 5 values
    "leakage_rate": [0.2, 0.3, 0.4, 0.5],  # 4 values
    "density": [0.05, 0.1, 0.15, 0.2],  # 4 values
    "fraction_input": [0.5, 0.75, 1.0],  # 3 values
    "fraction_output": [chosen_fraction_out],  # Fixed (by budget)
}
# Total: 5 × 4 × 4 × 3 = 240 combinations
```

### 📊 Plan C: Comprehensive Search (with ridge_alpha, 2400 combinations)
```python
grid_comprehensive = {
    "num_nodes": [800],  # Fixed (by budget)
    "spec_rad": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # 6 values
    "leakage_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5 values
    "density": [0.05, 0.1, 0.15, 0.2],  # 4 values
    "fraction_input": [0.3, 0.5, 0.75, 1.0],  # 4 values
    "fraction_output": [chosen_fraction_out],  # Fixed (by budget)
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],  # 5 values (if using RidgeSK)
}
# Total with alpha: 6 × 5 × 4 × 4 × 5 = 2400 combinations
# Total without alpha: 6 × 5 × 4 × 4 = 480 combinations
```

---

## 4. Practical Recommendations

### 🚀 Quick Start (for beginners)
1. Start with **Plan A** for quick exploration (27 combinations)
2. Fix `fraction_input=0.5`, `density=0.1`
3. Focus on tuning `spec_rad` and `leakage_rate`

### 📈 Standard Workflow (for most tasks)
1. Use **Plan B** standard search (240 combinations)
2. Include `fraction_input` tuning
3. Use time series cross-validation

### 🎯 Ultimate Optimization (for best performance)
1. Use **Plan C** comprehensive search
2. Include `ridge_alpha` tuning
3. Consider Random Search to reduce computation
4. Use Bayesian Optimization for intelligent exploration

---

## 5. Task-Specific Hyperparameter Guidelines

### Lorenz Chaotic System
```python
grid_lorenz = {
    "spec_rad": [0.8, 0.9, 1.0, 1.1, 1.2],  # Edge of chaos
    "leakage_rate": [0.2, 0.3, 0.4, 0.5],  # Medium speed
    "density": [0.05, 0.1, 0.15],  # Sparse
    "fraction_input": [0.5, 0.75],  # Medium sparsity
}
```

### Mackey-Glass Time Series
```python
grid_mackeyglass = {
    "spec_rad": [0.9, 1.0, 1.1, 1.2],  # Needs long-term memory
    "leakage_rate": [0.1, 0.2, 0.3],  # Slower dynamics
    "density": [0.1, 0.15, 0.2],  # Slightly denser
    "fraction_input": [0.5, 0.75, 1.0],
}
```

### Financial Time Series (High-frequency)
```python
grid_finance = {
    "spec_rad": [0.5, 0.7, 0.9],  # Short-term memory
    "leakage_rate": [0.5, 0.7, 0.9],  # Fast adaptation
    "density": [0.05, 0.1],  # Sparse to prevent overfitting
    "fraction_input": [0.3, 0.5],  # Sparse for regularization
    "alpha": [1e-2, 1e-1, 1.0],  # Strong regularization
}
```

---

## 6. Tuning Strategies

### 🎯 Strategy 1: Staged Tuning
1. **Stage 1**: Fix other params, only tune `spec_rad` and `leakage_rate`
2. **Stage 2**: Use Stage 1 best values, tune `density` and `fraction_input`
3. **Stage 3**: If needed, tune `ridge_alpha`

### 🎯 Strategy 2: Random Search
- Randomly sample 50-100 combinations from complete grid
- 10-20x faster than full grid search
- Usually finds near-optimal solutions

### 🎯 Strategy 3: Bayesian Optimization
- Use `scikit-optimize` or `optuna`
- Intelligent exploration of parameter space
- Suitable for expensive large-scale searches

---

## 7. Parameter Interaction Effects

### Critical Interactions

**1. `spec_rad` × `leakage_rate`**
- High `spec_rad` + Low `leakage_rate` → Very long memory
- Low `spec_rad` + High `leakage_rate` → Very short memory
- Balance both for optimal performance

**2. `density` × `num_nodes`**
- Large reservoir → Use lower density (0.05-0.1)
- Small reservoir → Can use higher density (0.3-0.5)
- Total connections ≈ `num_nodes²` × `density`

**3. `fraction_input` × `fraction_output`**
- Both affect sparsity and regularization
- If one is low, consider increasing the other
- Total parameters ≈ `num_nodes` × `fraction_input` × `fraction_output`

---

## 8. Common Pitfalls and Solutions

### ❌ Pitfall 1: Using Wrong Default Values
**Problem**: Different functions use different defaults
**Solution**: Explicitly specify all defaults in tuning function

### ❌ Pitfall 2: Ignoring Budget Constraints
**Problem**: Large reservoirs exceed memory/time budget
**Solution**: Calculate optimal `num_nodes` and `fraction_output` from budget

### ❌ Pitfall 3: Not Accounting for Time Series Nature
**Problem**: Using random CV shuffles temporal order
**Solution**: Use time series forward chaining CV

### ❌ Pitfall 4: Over-tuning on Validation Set
**Problem**: Grid search overfits to validation data
**Solution**: Use nested CV or separate test set

---

## Summary Table

| Priority | Parameter | Recommended Range | Why Important |
|----------|-----------|------------------|---------------|
| 🔥 High | `spec_rad` | [0.7, 0.8, 0.9, 1.0, 1.1, 1.2] | Controls dynamics & memory |
| 🔥 High | `leakage_rate` | [0.1, 0.2, 0.3, 0.4, 0.5] | Controls time scale |
| 🔥 High | `num_nodes` | Determined by budget | Controls capacity |
| 🔶 Medium | `fraction_input` | [0.3, 0.5, 0.75, 1.0] | Affects information flow |
| 🔶 Medium | `density` | [0.05, 0.1, 0.15, 0.2] | Affects sparsity |
| 🔶 Medium | `ridge_alpha` | [1e-4, 1e-3, 1e-2, 1e-1, 1.0] | Controls regularization |
| 🔵 Low | `fraction_output` | Determined by budget | Affects computation |
| 🔵 Low | `activation` | 'tanh' | Usually fixed |
| 🔵 Low | `optimizer` | 'ridge' | Usually fixed |

---

## Quick Reference: Default Values

**Correct Defaults (from train_pyreco_model.py)**:
```python
default_spec_rad = 1.0
default_leakage_rate = 0.3
default_density = 0.1          # ← Important!
default_activation = "tanh"
default_fraction_input = 0.5   # ← Important!
default_alpha = 1.0 (if using RidgeSK)
```

**Incorrect Defaults (from PyReCoStandardModel.__init__)**:
```python
density = 0.8                  # ❌ Wrong! 8x difference
fraction_input = 1.0           # ❌ Wrong! 2x difference
leakage_rate = 0.5             # ❌ Wrong!
spec_rad = 0.9                 # ❌ Wrong!
```

---

## Conclusion

The key to successful PyReCo hyperparameter tuning:

1. ✅ **Use correct default values** (density=0.1, fraction_input=0.5)
2. ✅ **Prioritize important parameters** (spec_rad, leakage_rate)
3. ✅ **Use time series CV** (preserve temporal order)
4. ✅ **Start simple, expand gradually** (Plan A → B → C)
5. ✅ **Consider task characteristics** (chaotic, stationary, high-freq, etc.)

Happy tuning! 🎉
