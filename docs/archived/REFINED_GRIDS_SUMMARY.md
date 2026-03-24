# Refined Hyperparameter Grids Based on CV Pre-Tuning

## Summary

After completing 5-fold cross-validation pre-tuning on MEDIUM scale (budget=10,000 total parameters), we have defined **dataset-specific refined grids** that focus search around CV-validated optimal regions.

## Pre-Tuning Results (5-fold CV)

| Dataset | Best spec_rad | Best leakage | Best density | Best fraction_input | CV MSE | Tune Time |
|---------|--------------|--------------|--------------|---------------------|--------|-----------|
| Lorenz | 0.9 | 0.5 | 0.05 | 0.5 | 0.281132 | 47.2 min |
| Mackey-Glass | 0.8 | 0.3 | 0.1 | 0.5 | N/A | ~40 min |
| Santa Fe | 0.8 | 0.6 | 0.1 | 0.5 | N/A | 47.2 min |

## Refined Grids (Dataset-Specific)

### Lorenz
**Strategy**: Center search around spec_rad=0.9, leakage=0.5, density=0.05

```python
{
    'spec_rad': [0.8, 0.9, 1.0],        # Centered on CV best (0.9)
    'leakage_rate': [0.4, 0.5, 0.6],    # Centered on CV best (0.5)
    'density': [0.05, 0.1],              # Start from CV best (0.05)
    'fraction_input': [0.5, 0.75],       # Explore around CV best (0.5)
}
```
**Combinations**: 3 × 3 × 2 × 2 = **36 combinations**

### Mackey-Glass
**Strategy**: Center search around spec_rad=0.8, leakage=0.3, density=0.1

```python
{
    'spec_rad': [0.7, 0.8, 0.9],        # Centered on CV best (0.8)
    'leakage_rate': [0.2, 0.3, 0.4],    # Centered on CV best (0.3)
    'density': [0.1, 0.15],              # Start from CV best (0.1)
    'fraction_input': [0.5, 0.75],       # Explore around CV best (0.5)
}
```
**Combinations**: 3 × 3 × 2 × 2 = **36 combinations**

### Santa Fe
**Strategy**: Center search around spec_rad=0.8, leakage=0.6, density=0.1

```python
{
    'spec_rad': [0.7, 0.8, 0.9],        # Centered on CV best (0.8)
    'leakage_rate': [0.5, 0.6, 0.7],    # Centered on CV best (0.6)
    'density': [0.05, 0.1],              # Include both low and CV best
    'fraction_input': [0.5, 0.75],       # Explore around CV best (0.5)
}
```
**Combinations**: 3 × 3 × 2 × 2 = **36 combinations**

## Comparison: Original vs Refined Grids

| Metric | Original Grid | Refined Grids |
|--------|--------------|---------------|
| **Grid Type** | Unified (all datasets) | Dataset-specific |
| **Combinations per dataset** | 96 | 36 |
| **Reduction** | - | **62.5%** |
| **Design** | Broad exploration | Focused around CV optima |
| **spec_rad range** | [0.8, 0.9, 1.0, 1.1] | [0.7, 0.8, 0.9] or [0.8, 0.9, 1.0] |
| **Avoids problematic combos** | No (includes ≥1.0) | Yes (max 1.0) |

## Time Savings Estimate

### For MEDIUM Scale (budget=10,000, nodes=300)

**Original Grid**:
- 96 combinations × 8s per combo = **768s (12.8 min)** per experiment
- For 45 experiments: 96 × 8 × 45 / 3600 = **9.6 hours**

**Refined Grid**:
- 36 combinations × 8s per combo = **288s (4.8 min)** per experiment
- For 45 experiments: 36 × 8 × 45 / 3600 = **3.6 hours**

**Savings**: 9.6 - 3.6 = **6.0 hours (62.5% reduction)**

### For All Three Scales (SMALL, MEDIUM, LARGE)

**Original Grid** (96 combinations):
- PyReCo: 135 experiments × 96 combos × 8s = **28.8 hours**
- LSTM: 135 experiments × 1 × 30s = **1.1 hours**
- **Total**: ~**29.9 hours**

**Refined Grid** (36 combinations):
- PyReCo: 135 experiments × 36 combos × 8s = **10.8 hours**
- LSTM: 135 experiments × 1 × 30s = **1.1 hours**
- **Total**: ~**11.9 hours**

**Savings**: 29.9 - 11.9 = **18.0 hours (60.2% reduction)**

## Comprehensive Experiment Configuration

With refined grids, the comprehensive experiment plan:

```
Datasets:        3 (lorenz, mackeyglass, santafe)
Seeds:           5 (42, 43, 44, 45, 46)
Train ratios:    3 (0.5, 0.7, 0.9)
Budget scales:   3 (small=1k, medium=10k, large=30k)
Models:          2 (pyreco_standard, lstm)
-----------------------------------------------------------
Total experiments: 3 × 5 × 3 × 3 × 2 = 270 model trainings
Hyperparameter search: 36 combinations per PyReCo experiment
```

**Estimated Time**:
- PyReCo (with tuning): 135 experiments × 36 combos × 8s avg = **10.8 hours**
- LSTM (no tuning): 135 experiments × 1 training × 30s avg = **1.1 hours**
- **Total**: ~**11.9 hours** (vs 29.9 hours with original grid)

## Implementation Status

- Pre-tuning completed: 3/3 datasets
- Refined grids defined: 3/3 datasets
- Updated in test_model_scaling.py: lines 528-567
- Ready for comprehensive experiments

## Next Steps

1. Run comprehensive experiments with refined grids
2. Compare PyReCo vs LSTM across all configurations
3. Generate statistical analysis report
4. Create visualizations (MSE vs train_ratio, training time comparison)

---

**Literature Support**:
- Bergstra & Bengio (2012): Random search can be more efficient than grid search
- This approach uses **informed search** (CV-guided grid refinement), combining benefits of both
- Pre-tuning with CV provides statistically validated starting points for hyperparameter search
