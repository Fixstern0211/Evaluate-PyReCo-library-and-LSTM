# Cross-Validation Guide for Time Series

## When to Use Cross-Validation?

### ✅ **USE Cross-Validation When:**

1. **Small to Medium Datasets** (< 5000 samples)
   - Need to maximize data utilization
   - Cannot afford to hold out large validation set
   - Want robust hyperparameter selection

2. **Expensive Models** (training is fast enough)
   - PyReCo models: Usually FAST (seconds per model)
   - LSTM models: May be SLOW (minutes per model)
   - If CV takes too long, use simple train/val split

3. **High Variance in Performance**
   - Single validation set may give misleading results
   - CV provides mean ± std for robust comparison

4. **Final Model Evaluation**
   - Use CV to estimate generalization performance
   - More reliable than single train/val split

### ❌ **DON'T USE Cross-Validation When:**

1. **Very Large Datasets** (> 10000 samples)
   - Simple train/val/test split is sufficient
   - CV adds unnecessary computation

2. **Very Slow Models** (LSTM with large hidden size)
   - CV with 5 folds = 5x computation
   - Use simple split for speed

3. **Strong Temporal Trends** (non-stationary data)
   - Future data distribution differs from past
   - Forward chaining CV may not help much
   - Consider walk-forward validation instead

---

## Evaluation of `tune_pyreco_with_cv`

### ✅ **Advantages:**

1. **Time Series Safe** ✅
   - Uses forward chaining (not random shuffle)
   - Preserves temporal order
   - Prevents data leakage

2. **Robust Hyperparameter Selection** ✅
   - Returns mean ± std of CV scores
   - Less sensitive to validation set choice
   - Better generalization estimate

3. **Efficient for PyReCo** ✅
   - PyReCo is fast (seconds per model)
   - CV overhead is acceptable

### ⚠️ **Limitations:**

1. **Computation Cost** ⚠️
   - 5-fold CV = 5x training time
   - For large grids (>100 combinations), can be slow
   - Solution: Use smaller n_splits (3 instead of 5)

2. **Data Efficiency** ⚠️
   - Each fold uses only partial training data
   - Early folds have very small training sets
   - Solution: Minimum training size check

### 📊 **Quality Assessment:**

| Aspect | Rating | Comment |
|--------|--------|---------|
| Temporal Safety | ⭐⭐⭐⭐⭐ | Perfect - uses forward chaining |
| Robustness | ⭐⭐⭐⭐⭐ | Excellent - mean ± std |
| Speed | ⭐⭐⭐ | Good for PyReCo, slow for LSTM |
| Code Quality | ⭐⭐⭐⭐ | Good - follows train_pyreco_model.py |
| Documentation | ⭐⭐⭐⭐ | Clear docstrings |

**Overall: ⭐⭐⭐⭐ (4/5) - Good implementation, suitable for PyReCo**

---

## Should Custom Model Use Cross-Validation?

### 🎯 **Answer: YES, for consistency and fairness**

### Reasons:

1. **Fair Comparison** ✅
   - All models should use same validation strategy
   - If PyReCo-Standard uses CV, PyReCo-Custom should too
   - Ensures apples-to-apples comparison

2. **Hyperparameter Tuning** ✅
   - Custom model also has hyperparameters:
     - `num_nodes`, `density`, `spec_rad`, `leakage_rate`
     - `fraction_output`, `discard_transients`
   - CV provides robust tuning

3. **Same Implementation Pattern** ✅
   - Can reuse `tune_pyreco_with_cv` logic
   - Just swap `PyReCoStandardModel` → `PyReCoCustomModel`

### Implementation Recommendation:

```python
def tune_pyreco_custom_with_cv(
        X_train, y_train,
        param_grid,
        n_splits=5,
        default_spec_rad=0.95,
        default_leakage=0.3,
        default_density=0.05,
        default_fraction_output=1.0,
        verbose=True):
    """
    Same as tune_pyreco_with_cv but for PyReCoCustomModel
    """
    # Use same CV logic
    # Create PyReCoCustomModel instead of PyReCoStandardModel
    # Return best_params, best_score, final_model
```

---

## Comparison: Simple Split vs Cross-Validation

### Scenario Analysis

#### **Scenario 1: Small Dataset (1000 samples), PyReCo Model**
```
Simple Split (85/15):
- Train: 850 samples
- Val: 150 samples
- Time: 1 epoch
- Risk: High variance from single split

5-Fold CV:
- Fold 1: Train 200 → Val 50
- Fold 2: Train 400 → Val 50
- Fold 3: Train 600 → Val 50
- Fold 4: Train 800 → Val 50
- Fold 5: Train 1000 → Val 0 (no validation, use all)
- Time: 5 epochs
- Benefit: Robust estimate

✅ Recommendation: Use CV (5x time is acceptable for PyReCo)
```

#### **Scenario 2: Large Dataset (10000 samples), PyReCo Model**
```
Simple Split (85/15):
- Train: 8500 samples
- Val: 1500 samples (enough for reliable estimate)
- Time: 1 epoch
- Risk: Low variance

5-Fold CV:
- Time: 5 epochs
- Benefit: Marginal improvement

✅ Recommendation: Simple split (CV not worth the time)
```

#### **Scenario 3: Small Dataset (1000 samples), LSTM Model**
```
Simple Split (85/15):
- Train: 850 samples
- Val: 150 samples
- Time: 100 epochs (with early stopping)
- Risk: High variance

5-Fold CV:
- Time: 500 epochs (5x100)
- Benefit: Very robust estimate
- Problem: May take hours

⚠️ Recommendation: Use 3-Fold CV (compromise) or simple split if time-constrained
```

---

## Best Practices

### 1. **For PyReCo Models:**
- ✅ Use `tune_pyreco_with_cv` for datasets < 5000 samples
- ✅ Use 5-fold CV for robust hyperparameter selection
- ✅ Use 3-fold CV if computation time is a concern
- ✅ Use simple split for datasets > 10000 samples

### 2. **For LSTM Models:**
- ✅ Use simple split for fast iteration
- ✅ Use 3-fold CV only for final hyperparameter search
- ⚠️ Avoid 5-fold CV unless dataset is very small and you have time

### 3. **For Custom Models:**
- ✅ Use same CV strategy as Standard models
- ✅ Implement `tune_pyreco_custom_with_cv` for consistency

### 4. **For Final Evaluation:**
- ✅ Always use separate test set (never seen during training/tuning)
- ✅ Report test metrics as final performance
- ✅ Can use CV on train set for hyperparameter tuning

---

## Implementation Checklist

- [x] `tune_pyreco_hyperparameters()` - Simple train/val split
- [x] `tune_pyreco_with_cv()` - Time series CV for Standard model
- [x] `tune_pyreco_custom_hyperparameters()` - Simple train/val split for Custom model ✅
- [x] `tune_pyreco_custom_with_cv()` - Time series CV for Custom model ✅
- [ ] `tune_lstm_with_cv()` - Time series CV for LSTM (TODO, optional)

---

## Conclusion

**Summary:**
1. ✅ `tune_pyreco_with_cv` is **good** - time series safe, robust, efficient for PyReCo
2. ✅ **Use CV** for small/medium datasets (< 5000) and fast models (PyReCo)
3. ✅ **Custom model should use CV** for fair comparison
4. ⚠️ **Simple split** is better for large datasets or slow models (LSTM)

**Current Status:**
- PyReCo-Standard: Has both simple split and CV ✅
- PyReCo-Custom: Has both simple split and CV ✅
- LSTM: No CV implementation ⚠️ (acceptable, LSTM is slow)
