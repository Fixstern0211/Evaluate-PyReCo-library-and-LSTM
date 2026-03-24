# Distribution Shift in Chaotic Time Series Experiments

## Date: 2025-12-16

## Key Finding

When using different data lengths for training/testing on chaotic systems (e.g., Lorenz), **standardization cannot solve distribution shift** between training and test sets.

## Experimental Evidence

### Data Configuration Comparison

| Length | Train Samples | Train y std | Test y std | Test y Range |
|--------|---------------|-------------|------------|--------------|
| 5000   | 2875          | 0.97        | 1.17       | [-2.2, 3.0]  |
| 2000   | 1090          | 0.57        | **2.17**   | [-3.2, 5.9]  |

### Impact on Model Performance

Same model (PyReCo, medium budget, train_frac=0.7, seed=42):

| Experiment Type | MSE | R² |
|-----------------|-----|-----|
| Single-step (length=5000) | 0.000415 | **0.9997** |
| Multi-step horizon=1 (length=2000) | 1.754647 | 0.6152 |

**Difference: ~4000x in MSE!**

## Root Cause Analysis

1. **Standardization uses training set parameters**
   - μ and σ are computed from training data only
   - Test data is transformed using training parameters

2. **Chaotic systems have non-stationary dynamics**
   - Lorenz system exhibits different dynamic regimes at different time periods
   - Shorter sequences may not capture full attractor

3. **Distribution shift after standardization**
   - When length=2000: test set std = 4x training set std (after standardization!)
   - Model trained on "narrow" distribution, tested on "wide" distribution
   - This is extrapolation, not interpolation

## Implications for Experimental Design

1. **Always use consistent data length** across all experiments for fair comparison
2. **Verify train/test distribution similarity** before running experiments
3. **Standardization is necessary but not sufficient** for handling scale differences
4. **Longer sequences are preferred** for chaotic systems to cover more of the attractor

## Recommendation

For multi-step prediction experiments, use **length=5000** (same as single-step experiments) to ensure:
- Consistent train/test distribution
- Fair comparison between single-step and multi-step performance
- Reproducible results

## References

- This finding aligns with standard ML practice regarding distribution shift
- Relevant to: domain adaptation, covariate shift literature
