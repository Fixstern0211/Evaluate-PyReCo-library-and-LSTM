# Testing Guide for PyReCo and LSTM Models

## Overview

This guide covers all testing scripts created for comprehensive model evaluation and hyperparameter tuning.

---

## 📁 Created Files

### Documentation
1. **`pyreco_hyperparameter_analysis.md`** - Complete guide to PyReCo hyperparameters
2. **`cross_validation_guide.md`** - When and how to use cross-validation
3. **`TESTING_GUIDE.md`** (this file) - Usage guide for all test scripts

### Test Scripts
4. **`test_staged_tuning.py`** - Staged hyperparameter tuning (Stage 1 → 2 → 3)
5. **`test_model_scaling.py`** - Performance across parameter budgets (10³, 10⁴, 10⁵)
6. **`test_pyreco_wrapper.py`** - Unit test for tune_pyreco_hyperparameters
7. **`test_default_values.py`** - Verification of default values fix
8. **`test_consistency.py`** - Consistency check between models_evaluate.py and pyreco_wrapper.py

---

## 🚀 Quick Start

### 1. Quick Hyperparameter Test (4 combinations, ~2 minutes)
```bash
source .venv/bin/activate
python test_staged_tuning.py --strategy quick --dataset lorenz --length 2000
```

### 2. Standard Tuning (240 combinations, ~30-60 minutes)
```bash
python test_staged_tuning.py --strategy stage2 --dataset lorenz
```

### 3. Full Staged Tuning (Stage 1 → 2 → 3, ~2-3 hours)
```bash
python test_staged_tuning.py --strategy staged --dataset lorenz --use-cv
```

### 4. Parameter Scaling Test (3 scales × 3 models, ~30-60 minutes)
```bash
python test_model_scaling.py --dataset lorenz --length 5000
```

---

## 📊 Detailed Usage

### Test 1: Staged Hyperparameter Tuning

**File**: `test_staged_tuning.py`

**Purpose**: Find optimal hyperparameters using staged approach

**Strategies**:

#### A. Quick Exploration (9 combinations)
```bash
python test_staged_tuning.py --strategy stage1 --dataset lorenz
```
- Tunes: spec_rad (3 values) × leakage_rate (3 values)
- Fixed: density=0.1, fraction_input=0.5
- Time: ~5-10 minutes

#### B. Standard Tuning (240 combinations)
```bash
python test_staged_tuning.py --strategy stage2 --dataset lorenz
```
- Tunes: spec_rad (5) × leakage_rate (4) × density (4) × fraction_input (3)
- Time: ~30-60 minutes

#### C. Full Tuning (480 combinations)
```bash
python test_staged_tuning.py --strategy stage3 --dataset lorenz
```
- Comprehensive search across all parameters
- Time: ~1-2 hours

#### D. Staged Tuning (Stage 1 → 2 → 3)
```bash
python test_staged_tuning.py --strategy staged --dataset lorenz
```
- Runs all three stages sequentially
- Stage 2 uses best values from Stage 1
- Time: ~2-3 hours total

#### E. Task-Specific Tuning
```bash
# Lorenz chaotic system
python test_staged_tuning.py --strategy task_specific --dataset lorenz

# Mackey-Glass time series
python test_staged_tuning.py --strategy task_specific --dataset mackeyglass

# Santa Fe laser data
python test_staged_tuning.py --strategy task_specific --dataset santafe
```
- Uses parameter ranges optimized for specific tasks
- Time: ~20-40 minutes

**With Cross-Validation**:
```bash
python test_staged_tuning.py --strategy stage2 --dataset lorenz --use-cv --n-splits 5
```
- More robust hyperparameter selection
- 5x computation time

**All Options**:
```bash
python test_staged_tuning.py \
    --strategy staged \
    --dataset lorenz \
    --length 5000 \
    --train-frac 0.6 \
    --n-in 100 \
    --budget 10000 \
    --seed 42 \
    --use-cv \
    --n-splits 5 \
    --output results_staged.json
```

---

### Test 2: Model Scaling Test

**File**: `test_model_scaling.py`

**Purpose**: Compare PyReCo-Standard, PyReCo-Custom, and LSTM at different parameter budgets

**Parameter Budgets**:
- Small: 10³ (~1,000 parameters)
- Medium: 10⁴ (~10,000 parameters)
- Large: 10⁵ (~100,000 parameters)

**Basic Usage**:
```bash
python test_model_scaling.py --dataset lorenz --length 5000
```

**With Hyperparameter Tuning** (for PyReCo-Standard):
```bash
python test_model_scaling.py --dataset lorenz --tune-pyreco
```
- Adds hyperparameter tuning for PyReCo-Standard
- Significantly longer runtime

**All Options**:
```bash
python test_model_scaling.py \
    --dataset lorenz \
    --length 5000 \
    --train-frac 0.6 \
    --n-in 100 \
    --seed 42 \
    --tune-pyreco \
    --output results_scaling.json
```

**Output Example**:
```
SMALL Scale:
Model                     Params       Train(s)   Test MSE     Test R²
pyreco_standard           1,050        15.23      1.234567     0.678901
pyreco_custom             1,050        14.87      1.345678     0.654321
lstm                      1,008        123.45     1.123456     0.712345

MEDIUM Scale:
Model                     Params       Train(s)   Test MSE     Test R²
pyreco_standard           10,500       145.67     0.876543     0.789012
pyreco_custom             10,500       142.34     0.987654     0.745678
lstm                      9,984        1234.56    0.765432     0.823456

LARGE Scale:
Model                     Params       Train(s)   Test MSE     Test R²
pyreco_standard           105,000      1456.78    0.654321     0.845678
pyreco_custom             105,000      1423.45    0.765432     0.812345
lstm                      99,840       12345.67   0.543210     0.867890
```

---

## 📈 Recommended Testing Workflow

### Step 1: Quick Validation (5-10 minutes)
```bash
# Verify everything works with minimal testing
python test_staged_tuning.py --strategy quick --dataset lorenz --length 2000
```

### Step 2: Standard Hyperparameter Search (30-60 minutes)
```bash
# Find good hyperparameters for your dataset
python test_staged_tuning.py --strategy stage2 --dataset lorenz --length 5000
```

### Step 3: Model Scaling Comparison (30-60 minutes)
```bash
# Compare models at different parameter budgets
python test_model_scaling.py --dataset lorenz --length 5000
```

### Step 4: Full Tuning with CV (2-3 hours)
```bash
# Final hyperparameter optimization with cross-validation
python test_staged_tuning.py --strategy staged --dataset lorenz --use-cv
```

---

## 🎯 Task-Specific Recommendations

### For Lorenz Chaotic System
```bash
# Quick test
python test_staged_tuning.py --strategy task_specific --dataset lorenz --length 3000

# Full test
python test_staged_tuning.py --strategy task_specific --dataset lorenz --length 8000 --use-cv
```

**Expected Best Parameters**:
- spec_rad: 0.9-1.2 (edge of chaos)
- leakage_rate: 0.2-0.5 (medium speed)
- density: 0.05-0.15 (sparse)
- fraction_input: 0.5-0.75

### For Mackey-Glass Time Series
```bash
python test_staged_tuning.py --strategy task_specific --dataset mackeyglass --length 5000
```

**Expected Best Parameters**:
- spec_rad: 0.9-1.2 (long-term memory)
- leakage_rate: 0.1-0.3 (slower dynamics)
- density: 0.1-0.2 (slightly denser)
- fraction_input: 0.5-1.0

### For Santa Fe Laser Data
```bash
python test_staged_tuning.py --strategy task_specific --dataset santafe --length 3000
```

**Expected Best Parameters**:
- spec_rad: 0.7-1.0 (moderate stability)
- leakage_rate: 0.3-0.6 (adaptive to noise)
- density: 0.05-0.1 (sparse for regularization)
- fraction_input: 0.3-0.75 (prevent overfitting)

---

## 📝 Output Files

### Staged Tuning Output
**File**: `results_staged_tuning.json`

**Structure**:
```json
{
  "metadata": {
    "dataset": "lorenz",
    "strategy": "staged",
    "num_nodes": 800,
    "timestamp": "2025-..."
  },
  "results": [
    {
      "stage": "Stage 1: Quick Exploration",
      "best_params": {...},
      "test_mse": 1.234,
      "tuning_time": 123.45
    },
    ...
  ],
  "best_result": {...}
}
```

### Scaling Test Output
**File**: `results_scaling.json`

**Structure**:
```json
{
  "metadata": {...},
  "budgets": {
    "small": 1000,
    "medium": 10000,
    "large": 100000
  },
  "results": {
    "small": [
      {
        "model_type": "pyreco_standard",
        "param_info": {...},
        "test_mse": 1.234,
        "training_time": 15.23
      },
      ...
    ],
    ...
  }
}
```

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory
```bash
# Reduce dataset size
python test_staged_tuning.py --strategy stage1 --length 2000

# Reduce parameter budget
python test_model_scaling.py --budget 5000
```

### Issue 2: Too Slow
```bash
# Use quick strategy
python test_staged_tuning.py --strategy quick

# Disable CV
python test_staged_tuning.py --strategy stage2  # Without --use-cv

# Reduce n_splits
python test_staged_tuning.py --strategy stage2 --use-cv --n-splits 3
```

### Issue 3: Poor Results
```bash
# Try task-specific tuning
python test_staged_tuning.py --strategy task_specific --dataset lorenz

# Use longer dataset
python test_staged_tuning.py --strategy stage2 --length 8000

# Enable cross-validation
python test_staged_tuning.py --strategy stage2 --use-cv
```

---

## 📚 Additional Documentation

### 1. Hyperparameter Guide
See `pyreco_hyperparameter_analysis.md` for:
- Complete parameter descriptions
- Importance rankings
- Recommended ranges
- Parameter interactions

### 2. Cross-Validation Guide
See `cross_validation_guide.md` for:
- When to use CV
- Time series CV explanation
- trade-offs

### 3. Code Quality
All test scripts follow best practices:
- Clear argument parsing
- Comprehensive logging
- JSON output for reproducibility
- Error handling

---

## 🎓 Example Complete Workflow

```bash
# Activate environment
source .venv/bin/activate

# 1. Quick sanity check (2 minutes)
python test_staged_tuning.py --strategy quick --dataset lorenz --length 1000

# 2. Find good hyperparameters (30 minutes)
python test_staged_tuning.py --strategy stage2 --dataset lorenz --length 5000

# 3. Compare model scaling (30 minutes)
python test_model_scaling.py --dataset lorenz --length 5000

# 4. Full optimization with CV (2 hours)
python test_staged_tuning.py --strategy staged --dataset lorenz --length 8000 --use-cv

# 5. Analyze results
cat results_staged_tuning.json | python -m json.tool
```

---

## ✅ Summary

Created comprehensive testing framework with:

1. ✅ **Staged hyperparameter tuning** (Stage 1 → 2 → 3)
2. ✅ **Parameter scaling tests** (10³, 10⁴, 10⁵ parameters)
3. ✅ **Task-specific tuning** (Lorenz, Mackey-Glass, Santa Fe)
4. ✅ **Cross-validation support** (time series safe)
5. ✅ **JSON output** (reproducible results)
6. ✅ **Comprehensive documentation**

All scripts support:
- Multiple datasets (lorenz, mackeyglass, santafe)
- Flexible parameters (length, seed, budget, etc.)
- Progress logging
- Error handling
- Saved results

**Next Steps**:
1. Run quick tests to verify setup
2. Choose appropriate strategy for your task
3. Analyze results and compare models
4. Iterate with refined parameters if needed
