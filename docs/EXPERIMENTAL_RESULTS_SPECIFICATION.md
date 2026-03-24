# Experimental Results Specification

**Last Updated**: November 6, 2025
**Purpose**: Document the output format, metrics, and parameter configurations of the comprehensive experiments comparing PyReCo (Reservoir Computing) and LSTM models.

---

## Table of Contents
- [1. Output File Structure](#1-output-file-structure)
- [2. Reported Metrics](#2-reported-metrics)
- [3. Parameter Budget Analysis](#3-parameter-budget-analysis)
- [4. Model Architecture Comparison](#4-model-architecture-comparison)
- [5. Missing Metrics](#5-missing-metrics)

---

## 1. Output File Structure

Each experiment produces a JSON file with the following hierarchical structure:

```json
{
  "metadata": { ... },
  "budgets": { ... },
  "results": {
    "small": [ {pyreco_model}, {lstm_model} ],
    "medium": [ {pyreco_model}, {lstm_model} ],
    "large": [ {pyreco_model}, {lstm_model} ]
  }
}
```

### 1.1 Metadata Section

Contains experiment configuration information:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `dataset` | string | Dataset name | `"lorenz"`, `"mackeyglass"`, `"santafe"` |
| `length` | integer | Time series length | `5000` |
| `seed` | integer | Random seed for reproducibility | `42`, `43`, `44`, `45`, `46` |
| `n_in` | integer | Input window size | `100` |
| `train_frac` | float | Training data fraction | `0.5`, `0.7`, `0.9` |
| `timestamp` | string | Experiment timestamp (ISO format) | `"2025-11-06T02:34:44.563016"` |

### 1.2 Budgets Section

Defines parameter budget constraints for fair comparison:

| Scale | Total Parameters | Notes |
|-------|------------------|-------|
| `small` | 1,000 | Lightweight models |
| `medium` | 10,000 | Standard scale |
| `large` | 30,000 | Large-scale models |

---

## 2. Reported Metrics

### 2.1 Model Configuration

**PyReCo Configuration**:
- `num_nodes`: Number of reservoir nodes
- `density`: Reservoir connectivity (0-1)
- `activation`: Activation function (`"tanh"`)
- `spec_rad`: Spectral radius
- `leakage_rate`: Leaky integrator parameter
- `fraction_input`: Input connectivity fraction
- `fraction_output`: Output connectivity fraction
- `optimizer`: Readout optimizer (`"ridge"`)

**LSTM Configuration**:
- `hidden_size`: Hidden layer dimension
- `num_layers`: Number of LSTM layers (2)
- `dropout`: Dropout rate (0.2)
- `learning_rate`: Learning rate (0.001, default)
- `epochs`: Maximum training epochs (100)
- `batch_size`: Mini-batch size (32)
- `patience`: Early stopping patience (10)

### 2.2 Parameter Information (param_info)

#### PyReCo Parameters

| Field | Description | Trainable? |
|-------|-------------|-----------|
| `trainable` | Readout layer parameters only | ✅ Yes |
| `total` | Input + Reservoir + Readout | Mixed |
| `input` | Input weight parameters | ❌ Fixed (random) |
| `reservoir` | Recurrent weight parameters | ❌ Fixed (random) |
| `readout` | Output weight parameters | ✅ Trainable |

**Example (Large scale)**:
```json
"param_info": {
  "trainable": 1500,     // 5.5% of total
  "total": 27250,
  "input": 750,          // Fixed random
  "reservoir": 25000,    // Fixed random
  "readout": 1500        // Trained via ridge regression
}
```

#### LSTM Parameters

| Field | Description | Trainable? |
|-------|-------------|-----------|
| `trainable` | All parameters | ✅ Yes (100%) |
| `total` | Same as trainable | ✅ Yes |
| `lstm_layers` | LSTM layer parameters | ✅ Yes |
| `output` | Output layer parameters | ✅ Yes |

**Example (Large scale)**:
```json
"param_info": {
  "trainable": 29939,    // 100% of total
  "total": 29939,
  "lstm_layers": 29792,  // LSTM cells
  "output": 147          // Final linear layer
}
```

### 2.3 Performance Metrics

| Metric | Full Name | Description | Range | Usage |
|--------|-----------|-------------|-------|-------|
| `test_mse` | Mean Squared Error | Average squared prediction error | [0, ∞) | Primary metric |
| `test_mae` | Mean Absolute Error | Average absolute prediction error | [0, ∞) | Interpretable error |
| `test_r2` | R² Score | Coefficient of determination | (-∞, 1] | Goodness of fit |
| `val_mse` | Validation MSE | MSE on validation set | [0, ∞) | Hyperparameter selection |

**Lower is better** for MSE and MAE. **Higher is better** for R² (1.0 = perfect fit).

### 2.4 Training Time Metrics

| Metric | Description | Units | Recorded For |
|--------|-------------|-------|--------------|
| `tune_time` | Total hyperparameter search time | seconds | PyReCo (36 combos), LSTM (1 combo) |
| `best_combo_train_time` | Training time of best combination | seconds | PyReCo only |
| `final_train_time` | Final model training time on train+val | seconds | Both models |

**Note**: `tune_time` includes the time for all hyperparameter combinations tested during the search phase.

---

## 3. Parameter Budget Analysis

### 3.1 Budget Matching Accuracy

| Scale | Target Budget | PyReCo Total | LSTM Total | Difference | Match Quality |
|-------|---------------|--------------|------------|------------|---------------|
| **Small** | 1,000 | 1,450 | 952 | 498 (49.8%) | ⚠️ Large deviation |
| **Medium** | 10,000 | 10,350 | 10,052 | 298 (3.0%) | ✅ Very close |
| **Large** | 30,000 | 27,250 | 29,939 | 2,689 (9.0%) | ✅ Close |

**Interpretation**:
- Medium and Large scales achieve fair parameter budget matching (<10% difference)
- Small scale shows larger deviation due to discrete architecture constraints
- Both models use similar computational resources at matched scales

### 3.2 Trainable vs. Total Parameters

A key architectural difference between PyReCo and LSTM:

| Model | Scale | Trainable | Total | Trainable Ratio |
|-------|-------|-----------|-------|-----------------|
| **PyReCo** | Small | 300 | 1,450 | 20.7% |
| | Medium | 900 | 10,350 | 8.7% |
| | Large | 1,500 | 27,250 | 5.5% |
| **LSTM** | Small | 952 | 952 | 100% |
| | Medium | 10,052 | 10,052 | 100% |
| | Large | 29,939 | 29,939 | 100% |

**Key Insight**:
- PyReCo uses **total parameter budget** matching (computational complexity)
- Only 5-20% of PyReCo parameters are trainable (readout layer)
- LSTM trains 100% of its parameters via backpropagation
- This reflects fundamental architectural differences, not unfairness

---

## 4. Model Architecture Comparison

### 4.1 PyReCo (Reservoir Computing)

**Architecture**: Random fixed dynamics + trained linear readout

```
Input (3D) → [Random Projection] → Reservoir (Fixed) → [Readout: Ridge] → Output (3D)
              ↓ 750 params          ↓ 25,000 params      ↓ 1,500 params
              Fixed Random          Fixed Random         Trainable
```

**Parameter Breakdown (Large scale)**:
1. **Input Layer** (750 params): Random projection weights (fixed after initialization)
2. **Reservoir** (25,000 params): Sparse recurrent connections (fixed random weights)
3. **Readout** (1,500 params): Linear layer trained via ridge regression

**Training**: Only readout weights trained (fast, one-shot learning)

### 4.2 LSTM (Long Short-Term Memory)

**Architecture**: Fully trainable recurrent network

```
Input (3D) → LSTM Layer 1 (49 units) → LSTM Layer 2 (49 units) → Linear → Output (3D)
              ↓ 14,896 params           ↓ 14,896 params          ↓ 147 params
              All Trainable              All Trainable            Trainable
```

**Parameter Breakdown (Large scale)**:
1. **LSTM Layer 1** (14,896 params): Input-to-hidden and hidden-to-hidden weights
2. **LSTM Layer 2** (14,896 params): Hidden-to-hidden weights for second layer
3. **Output Layer** (147 params): Final linear projection

**Training**: All parameters trained via backpropagation through time (BPTT)

### 4.3 Comparison Table

| Aspect | PyReCo | LSTM |
|--------|--------|------|
| **Trainable %** | 5-20% | 100% |
| **Training Method** | Ridge regression | Gradient descent (BPTT) |
| **Training Speed** | Fast (seconds) | Slow (minutes) |
| **Memory Usage** | Lower (sparse) | Higher (dense) |
| **Hyperparameter Sensitivity** | High (requires tuning) | Moderate (defaults work) |
| **Initialization** | Random reservoir | Random + learned |
| **Computational Bottleneck** | Reservoir state computation | Gradient computation |

---

## 5. Missing Metrics

The current experimental setup does **not** record the following metrics:

### 5.1 Not Recorded

| Metric | Why Not Included | Potential Impact |
|--------|------------------|------------------|
| `memory_usage` | Not implemented | Cannot compare memory efficiency |
| `peak_memory` | Not implemented | Cannot assess memory scalability |
| `gpu_usage` | CPU-only experiments | N/A (LSTM uses CPU for fairness) |
| `inference_time` | Not measured | Cannot compare deployment speed |
| `prediction_variance` | Not tracked | Cannot assess output uncertainty |

### 5.2 Recommendations for Future Work

1. **Add memory profiling**: Use `tracemalloc` (Python) or system monitoring
2. **Measure inference time**: Critical for real-time applications
3. **Track GPU utilization**: If enabling GPU for LSTM
4. **Record convergence metrics**: Loss curves, gradient norms
5. **Include statistical tests**: Paired t-tests for significance

---

## 6. Dataset Configuration

All three datasets use **identical length** for consistency:

| Dataset | Length | Source | Characteristics |
|---------|--------|--------|-----------------|
| **Lorenz** | 5,000 | Generated | Chaotic attractor, 3D system |
| **Mackey-Glass** | 5,000 | Generated | Delayed differential equation |
| **Santa Fe** | 5,000 | Subset of full data | Laser intensity time series |

**Data Splits** (per experiment):
- Train: 50% / 70% / 90% (2,500 / 3,500 / 4,500 samples)
- Validation: 15% of train (~375 / 525 / 675 samples)
- Test: Remaining samples

**Input/Output Format**:
- Input window: 100 timesteps × 3 features
- Output: Next 100 timesteps × 3 features (multi-step ahead prediction)

---

## 7. Experimental Design Summary

### 7.1 Factorial Design

```
Total Experiments = 3 datasets × 5 seeds × 3 train_ratios = 45 experiments
```

**Datasets**: Lorenz, Mackey-Glass, Santa Fe
**Seeds**: 42, 43, 44, 45, 46 (for reproducibility)
**Train Ratios**: 0.5, 0.7, 0.9 (data scarcity analysis)

Each experiment produces **6 models** (3 scales × 2 model types).

### 7.2 Hyperparameter Tuning Strategy

| Model | Tuning Method | Search Space | Rationale |
|-------|---------------|--------------|-----------|
| **PyReCo** | 5-fold CV grid search | 36 combinations | High sensitivity (edge-of-criticality) |
| **LSTM** | Default parameters | No search | Robust defaults (lr=0.001, dropout=0.2) |

**Justification**: Reflects real-world usage patterns where RC requires expert tuning while LSTM works well with defaults. See `LITERATURE_SUPPORT_TUNING_STRATEGY.md` for academic citations.

---

## 8. Results Aggregation

### 8.1 Per-Experiment Results

Each of the 45 experiments generates:
- 1 JSON file with 6 model results (3 scales × 2 models)
- Metadata for reproducibility
- Timing information for computational cost analysis

### 8.2 Final Analysis

After all experiments complete, the analysis will include:

1. **Performance Comparison**:
   - MSE, MAE, R² across datasets, scales, and train ratios
   - Statistical significance tests (paired t-tests)
   - Effect of data availability on performance

2. **Computational Cost**:
   - Training time comparison
   - Hyperparameter search overhead
   - Scaling behavior with parameter count

3. **Robustness Analysis**:
   - Performance variance across seeds
   - Sensitivity to train/test splits
   - Failure modes and outliers

4. **Visualizations**:
   - Performance vs. parameter budget plots
   - Training time vs. accuracy trade-offs
   - Dataset-specific comparative analysis

---

## 9. File Naming Convention

```
results_final/results_{dataset}_seed{seed}_train{train_ratio}_{timestamp}.json
```

**Example**:
```
results_lorenz_seed42_train0.7_20251106_023444.json
```

**Progress Tracking**:
```
results_final/experiment_progress.json
```

Contains real-time experiment status (completed, successful, failed, runtimes).

---

## 10. Reproducibility Information

### 10.1 Fixed Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Input window (`n_in`) | 100 | Standard for multi-step forecasting |
| Dataset length | 5,000 | Balance between data richness and compute |
| LSTM layers | 2 | Standard architecture |
| LSTM dropout | 0.2 | Prevent overfitting |
| LSTM learning rate | 0.001 | Robust default (Adam optimizer) |
| PyReCo activation | tanh | Standard for RC |
| PyReCo optimizer | ridge | Analytical solution (fast) |

### 10.2 Variable Parameters

**PyReCo** (tuned per dataset):
- `spec_rad`: 0.7, 0.8, 0.9, 1.0 (4 values)
- `leakage_rate`: 0.4, 0.5, 0.6 (3 values)
- `density`: 0.05, 0.1, 0.2 (3 values)
- **Total**: 4 × 3 × 3 = 36 combinations tested via 5-fold CV

**Both models**:
- `num_nodes` / `hidden_size`: Adjusted per scale to match parameter budget
- `seed`: 5 different seeds for variance estimation
- `train_frac`: 3 values to test data efficiency

---

**Document Version**: 1.0
**Status**: Active - Experiments Running
**Next Update**: After experiment completion with statistical analysis results
