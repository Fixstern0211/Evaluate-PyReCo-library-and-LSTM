# Time Series Prediction Evaluation Protocol

**Version**: 1.0
**Date**: November 2025
**Purpose**: Standardized evaluation framework for PyReCo vs LSTM comparison

## Overview

This document defines the standardized evaluation protocol for multi-step time series prediction, ensuring reproducible and fair comparison between Reservoir Computing (PyReCo) and LSTM models.

## Evaluation Horizons

### Standard Prediction Horizons
- **1-step**: Short-term accuracy (immediate next value)
- **5-step**: Medium-term forecasting
- **10-step**: Extended prediction capability
- **20-step**: Long-term forecasting
- **50-step**: Very long-term prediction (chaos-sensitive)

### Interpretation Guide
- **1-5 steps**: Tests model's ability to capture local dynamics
- **10-20 steps**: Tests model's understanding of medium-term patterns
- **50+ steps**: Tests model's grasp of global attractor dynamics

## Prediction Modes

### 1. Free-Run Mode (Primary)
- **Definition**: Model uses its own predictions as input for subsequent steps
- **Use case**: Autonomous prediction, realistic deployment scenario
- **Implementation**: `multi_step_predict(model, X_test, n_steps, mode='free_run')`
- **Advantage**: Tests true prediction capability without oracle information

### 2. Teacher Forcing Mode (Secondary)
- **Definition**: Model uses ground truth values as input at each step
- **Use case**: Upper bound performance, debugging
- **Implementation**: Planned for future implementation
- **Advantage**: Isolates prediction errors from input propagation errors

## Core Metrics

### Primary Metrics (Always Computed)

#### 1. Mean Squared Error (MSE)
- **Formula**: `MSE = mean((y_true - y_pred)²)`
- **Range**: [0, ∞)
- **Interpretation**: Lower is better, scale-dependent

#### 2. Mean Absolute Error (MAE)
- **Formula**: `MAE = mean(|y_true - y_pred|)`
- **Range**: [0, ∞)
- **Interpretation**: Robust to outliers, same scale as data

#### 3. Normalized Root Mean Squared Error (NRMSE)
- **Formula**: `NRMSE = sqrt(MSE) / std(y_true)`
- **Range**: [0, ∞), typically [0, 2]
- **Interpretation**:
  - **Excellent**: < 0.1
  - **Good**: 0.1 - 0.3
  - **Fair**: 0.3 - 0.5
  - **Poor**: > 0.5

#### 4. R-squared (R²)
- **Formula**: `R² = 1 - SS_res/SS_tot`
- **Range**: (-∞, 1]
- **Interpretation**:
  - **Excellent**: > 0.9
  - **Good**: 0.7 - 0.9
  - **Fair**: 0.3 - 0.7
  - **Poor**: < 0.3
  - **Note**: Can be negative for very poor predictions

## Advanced Metrics (For Detailed Analysis)

### 1. Trajectory Divergence Time
- **Purpose**: Measure prediction stability in chaotic systems
- **Method**: Time steps until prediction diverges beyond threshold
- **Threshold**: 10% of target standard deviation (adaptive)
- **Relevant for**: Lorenz, Rössler, and other chaotic systems
- **Expected behavior**: RC may have shorter divergence times due to fixed dynamics

### 2. Spectral Similarity
- **Purpose**: Assess frequency domain characteristics preservation
- **Metrics**:
  - PSD correlation: Similarity of power spectral densities
  - Spectral RMSE: Error in log-power spectrum
  - Dominant frequency error: Error in peak frequency
- **Relevant for**: Oscillatory and periodic systems
- **Expected behavior**: Important for systems with characteristic frequencies

### 3. Long-term Statistical Consistency
- **Purpose**: Evaluate attractor property preservation
- **Metrics**:
  - Mean/variance relative error
  - Skewness/kurtosis absolute error
  - Kolmogorov-Smirnov test p-value
- **Relevant for**: All chaotic systems
- **Expected behavior**: Good models preserve statistical properties even when short-term prediction fails

## Implementation Standards

### Data Requirements
- **Input format**: `(n_samples, n_timesteps, n_features)`
- **Target format**: `(n_samples, n_prediction_steps, n_features)`
- **Preprocessing**: Standardized (mean=0, std=1) using training data statistics
- **Validation**: 15% split from training data for hyperparameter tuning

### Computational Constraints
- **Advanced metrics limit**: Computed only for horizons ≤ 20 steps
- **Sample limit**: Maximum 10 samples for trajectory divergence analysis
- **Efficiency**: Use Welch's method for spectral analysis with appropriate segmentation

### Error Handling
- **Invalid metrics**: Return appropriate defaults (0.0, inf) with warnings
- **Numerical stability**: Add small epsilon (1e-12) to prevent log(0)
- **Memory management**: Process data in batches for large sequences

## Evaluation Procedure

### Step 1: Model Training
```python
model.fit(X_train, y_train)
```

### Step 2: Multi-Step Prediction
```python
results = evaluate_multi_step(
    model=model,
    X_test=X_test,
    y_test=y_test,
    horizons=[1, 5, 10, 20, 50],
    mode='free_run',
    include_advanced_metrics=True
)
```

### Step 3: Results Interpretation
```python
for horizon, metrics in results.items():
    print(f"Horizon {horizon}:")
    print(f"  NRMSE: {metrics['nrmse']:.3f}")
    print(f"  R²: {metrics['r2']:.3f}")
    if 'avg_divergence_time' in metrics:
        print(f"  Divergence time: {metrics['avg_divergence_time']:.1f} steps")
```

## Reproducibility Requirements

### Random Seeds
- **Dataset generation**: Fixed seed (42) for data loading
- **Model initialization**: Same seed for weight initialization
- **Cross-validation**: Deterministic splits

### Environment
- **Python**: 3.11.9
- **Key packages**: NumPy 2.3.3, SciPy 1.16.2, PyReCo 1.1.0
- **Hardware**: Document CPU/GPU used for timing comparisons

### Documentation
- **Parameters**: Log all model hyperparameters
- **Timing**: Record training and prediction times
- **Versions**: Include package versions in results

## Expected Outcomes

### RC vs LSTM Performance Patterns

#### Short-term (1-5 steps)
- **RC**: Fast training, competitive accuracy for simple dynamics
- **LSTM**: Potentially higher accuracy due to nonlinear capacity

#### Medium-term (10-20 steps)
- **RC**: May struggle with complex temporal dependencies
- **LSTM**: Should excel due to memory mechanisms

#### Long-term (50+ steps)
- **RC**: Fixed dynamics may provide stability
- **LSTM**: Risk of error accumulation and mode collapse

### Statistical Significance
- **Multiple runs**: 5 seeds minimum for robust statistics
- **Significance tests**: Wilcoxon signed-rank test for paired comparisons
- **Effect sizes**: Cohen's d or Cliff's delta for practical significance
- **Multiple comparisons**: Bonferroni correction when testing multiple metrics

## Training Time Measurement Methodology

### The Question: Should Hyperparameter Tuning Time Be Included?

When comparing training speed between PyReCo (RC) and LSTM, a key methodological question arises:

**Should the hyperparameter tuning process be counted as part of the model's training time?**

### Two Perspectives

| Perspective | Include Tuning Time | Exclude Tuning Time |
|-------------|--------------------|--------------------|
| **Rationale** | Real deployment requires tuning; this is the true cost | Tuning is done once; training may be repeated many times |
| **Use Case** | End-to-end workflow comparison | Pure algorithm efficiency comparison |
| **Common Practice** | Industry, AutoML research | Academic papers (more common) |

### Our Approach: Report Separately

We recommend **reporting both metrics separately** to provide a complete picture:

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Speed Comparison Report                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Hyperparameter Tuning Cost (Supplementary)                  │
│     ┌──────────────┬─────────────────┬─────────────────┐       │
│     │ Model        │ Grid Size       │ Tuning Time     │       │
│     ├──────────────┼─────────────────┼─────────────────┤       │
│     │ PyReCo       │ ~200 combos×5CV │ ~XX minutes     │       │
│     │ LSTM         │ ~9 combos×1val  │ ~XX minutes     │       │
│     └──────────────┴─────────────────┴─────────────────┘       │
│                                                                 │
│  2. Training Time with Optimal Parameters (Primary Metric)      │
│     ┌──────────────┬─────────────────┬─────────────────┐       │
│     │ Model        │ Method          │ Training Time   │       │
│     ├──────────────┼─────────────────┼─────────────────┤       │
│     │ PyReCo       │ Ridge Regression│ ~X.XX seconds   │       │
│     │ LSTM         │ Gradient Descent│ ~XX.XX seconds  │       │
│     └──────────────┴─────────────────┴─────────────────┘       │
│                                                                 │
│  3. Inference Time (Per Sample)                                 │
│     ┌──────────────┬─────────────────┐                         │
│     │ Model        │ Time/Sample     │                         │
│     ├──────────────┼─────────────────┤                         │
│     │ PyReCo       │ ~X.XX ms        │                         │
│     │ LSTM         │ ~X.XX ms        │                         │
│     └──────────────┴─────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Rationale for Separate Reporting

1. **Primary Comparison (Training Time with Optimal Parameters)**
   - This is the **fair comparison** as both models use their "best state"
   - PyReCo's advantage (fast training via linear regression) is clearly shown
   - This metric is what most academic papers report

2. **Supplementary Information (Tuning Cost)**
   - PyReCo requires tuning 4 parameters: `spec_rad`, `leakage_rate`, `density`, `fraction_input`
   - LSTM requires tuning 2 parameters: `learning_rate`, `dropout`
   - Discuss which model is easier to tune (sensitivity analysis)

3. **Practical Implications**
   - If the model needs **frequent retraining** (e.g., streaming data): Training time is more important
   - If the model is trained **once and deployed**: Tuning time should also be considered
   - If hyperparameters are **transferable** across similar tasks: Tuning cost is amortized

### What Our Pretuning Experiments Represent

The current pretuning experiments (CV-based hyperparameter search) serve as **preparation work** for fair model comparison:

- **Purpose**: Find optimal hyperparameters for each model
- **Classification**: Methodology/preprocessing step
- **Reporting**: Can be mentioned in methods section or supplementary materials
- **Not counted in**: Primary training time comparison

### Expected Findings

| Aspect | PyReCo (RC) | LSTM |
|--------|-------------|------|
| **Tuning Complexity** | 4 parameters, larger grid | 2 parameters, smaller grid |
| **Per-Config Training** | Very fast (linear regression) | Slower (multiple epochs) |
| **Total Tuning Time** | Moderate (many fast trainings) | Moderate (few slow trainings) |
| **Final Training Time** | Very fast (~seconds) | Slower (~tens of seconds) |
| **Sensitivity to Params** | Higher (reservoir dynamics) | Lower (SGD is robust) |

### Implementation Notes

When running experiments, record the following timestamps:

```python
# Timing structure for complete reporting
timing = {
    'tuning': {
        'start': timestamp,
        'end': timestamp,
        'n_configurations': int,
        'n_cv_folds': int,
    },
    'final_training': {
        'start': timestamp,
        'end': timestamp,
        'n_epochs': int,  # For LSTM
    },
    'inference': {
        'n_samples': int,
        'total_time': float,
        'time_per_sample': float,
    }
}
```

## Version History

- **v1.0** (November 2025): Initial standardized protocol
- **v1.1** (February 2026): Added training time measurement methodology
- **Planned v1.2**: Teacher forcing mode implementation
- **Planned v1.3**: Confidence intervals and uncertainty quantification

## Data Efficiency Experiments

### Research Question

**"How does model performance scale with training data size?"**

This experiment evaluates which model (PyReCo vs LSTM) is more **data-efficient** — i.e., which achieves better performance with limited training data.

### Experimental Design

#### Configuration Selection Strategy: Global Optimal Configuration

We use a **single fixed configuration** for each (dataset, budget, model_type) combination across all data lengths. This configuration is selected by:

1. Collecting results from all train_frac experiments (0.2, 0.3, 0.5, 0.6, 0.7, 0.8)
2. Averaging MSE across train_frac for each unique configuration
3. Selecting the configuration with lowest average MSE

#### Theoretical Justification

**1. Controlled Variable Principle**

The core question is: *"Given a fixed model architecture, how does performance change with data quantity?"*

- Using different configurations per data length confounds **data effect** with **configuration effect**
- Fixed configuration isolates the pure effect of data quantity on performance

```
❌ Confounded: Performance = f(data_length, config(data_length))
✅ Controlled: Performance = f(data_length) | config = constant
```

**2. Avoiding Configuration Selection Overfitting**

If we select optimal configuration for each train_frac independently:
- Small data may select configurations that happen to fit that specific split (noise)
- Averaging across multiple train_frac reduces variance in configuration selection
- This is analogous to cross-validation for hyperparameter selection

**3. Practical Relevance**

In real-world deployment scenarios:
- Users typically **fix model architecture first**, then collect data
- Re-tuning hyperparameters for each data size is impractical
- Global optimal configuration reflects "tune once, deploy many" paradigm

**4. Learning Curve Theory**

Classical learning curve studies (Hestness et al., 2017; Kaplan et al., 2020) all:
- Fix model architecture
- Vary only training data size
- Observe power-law relationship: `Performance ~ N^α`

Variable configurations would prevent fitting such scaling laws.

**5. Fair Model Comparison**

To answer "Which model is more data-efficient?":
- Both models must use their respective optimal (fixed) configurations
- Otherwise, we're comparing "model + tuning strategy" not "model alone"

### Data Lengths Tested

| Length | Train Samples (70%) | Rationale |
|--------|---------------------|-----------|
| 500    | ~280                | Minimal data regime |
| 1000   | ~560                | Low data regime |
| 2000   | ~1,120              | Moderate data |
| 3000   | ~1,680              | Transition zone |
| 5000   | ~2,800              | Standard (baseline) |
| 7000   | ~3,920              | High data regime |
| 10000  | ~5,600              | Data-rich regime |

### Expected Outcomes

1. **PyReCo**: Expected to be more data-efficient due to:
   - Fixed random reservoir (no weight training)
   - Only readout weights learned (fewer parameters)
   - Regularized linear regression (ridge) is sample-efficient

2. **LSTM**: Expected to require more data due to:
   - All weights trained via backpropagation
   - More parameters to estimate
   - Risk of overfitting with small datasets

### Metrics

Same as standard evaluation: MSE, MAE, NRMSE, R²

### Alternative Design (Not Used)

If the research question were *"How does optimal configuration change with data size?"*, we would need per-data-length configuration selection. But this answers a different question about hyperparameter sensitivity, not data efficiency.

## Ablation Studies (消融实验)

### What is Ablation Study?

**Ablation study** is an experimental methodology that systematically removes or varies individual components/parameters to evaluate their contribution to overall system performance.

**Analogy**: Like removing an organ in medicine to study its function, ablation experiments "remove" or vary a factor to study its impact.

```
Full System:    Model + n_in=100 + length=5000 + budget=medium
                            ↓
Ablation:       Vary n_in → [25, 50, 100, 150, 200]
                Fix all other parameters
                            ↓
Conclusion:     Quantify n_in's impact on performance
```

### Planned Ablation Experiments

#### 1. Input Window Size Sensitivity (n_in)

**Research Question**: How does input window size affect model performance?

**Design**:
- Fixed: dataset=lorenz, train_frac=0.7, budget=medium, length=5000
- Variable: n_in ∈ [25, 50, 100, 150, 200]
- Seeds: 5 random seeds for statistical validity

**Expected Insights**:
- Optimal n_in for each model type
- Whether RC and LSTM have different sensitivity to n_in
- Trade-off between input information and computational cost

| n_in | Expected RC Behavior | Expected LSTM Behavior |
|------|---------------------|----------------------|
| 25   | May miss long-range dependencies | Limited temporal context |
| 50   | Moderate performance | Moderate performance |
| 100  | Standard (baseline) | Standard (baseline) |
| 150  | Potential improvement | More context, slower training |
| 200  | Diminishing returns? | Risk of vanishing gradients? |

**Total Experiments**: 5 n_in × 2 models × 5 seeds = 50 experiments

#### 2. Reservoir Size Sensitivity (for RC only)

**Research Question**: How does reservoir size (num_nodes) affect RC performance independently of budget?

**Design**:
- Fixed: dataset=lorenz, train_frac=0.7, n_in=100, length=5000
- Variable: num_nodes ∈ [50, 100, 200, 500, 1000]
- Keep density constant (unlike budget experiments)

#### 3. LSTM Depth Sensitivity

**Research Question**: How does network depth affect LSTM performance?

**Design**:
- Fixed: dataset=lorenz, train_frac=0.7, n_in=100, length=5000
- Variable: num_layers ∈ [1, 2, 3, 4]
- Keep total parameters approximately constant

### Why Ablation Studies Are Important

1. **Isolate Contributing Factors**: Understand which components matter most
2. **Guide Hyperparameter Tuning**: Know where to focus optimization efforts
3. **Scientific Rigor**: Move beyond "it works" to "why it works"
4. **Reproducibility**: Help others understand key design choices

### Distinction from Main Experiments

| Aspect | Main Experiments | Ablation Studies |
|--------|-----------------|------------------|
| **Goal** | Compare RC vs LSTM | Understand individual factors |
| **Variables** | Multiple (dataset, train_frac, budget) | Single (e.g., n_in only) |
| **Scope** | Comprehensive | Focused |
| **Priority** | Primary | Secondary/Future work |

---

**References**:
- Lukoševičius, M. (2012). A practical guide to applying echo state networks.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.
- Hestness, J., et al. (2017). Deep Learning Scaling is Predictable, Empirically. arXiv:1712.00409.
- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.