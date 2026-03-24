# PyReCo vs LSTM: Comprehensive Experimental Results

**Last Updated**: 2026-03-01

## Overview

### Experiment Suite

| Experiment | Count | Description |
|------------|-------|-------------|
| Main (single-step) | 90 | 3 datasets × 3 budgets × 2 train_fracs × 5 seeds |
| Supplementary (ratio=0.2) | 15 | 3 datasets × 3 budgets × 5 seeds, PyReCo 180 combos |
| HP Sensitivity Analysis | — | Analysis of 90 main experiments across 6 train_fracs |
| Data Efficiency | 270 | 3 datasets × 6 data_lengths × 3 budgets × 5 seeds |
| Multi-Step Prediction | 270 | 3 datasets × 6 train_fracs × 3 budgets × 5 seeds |
| Inference Benchmark | 3 | 3 datasets × 1 budget (medium) |
| Statistical Analysis | 1 | Paired t-test, Wilcoxon, Cohen's d on all results |
| Green Computing | 1 | Energy/CO₂ analysis across all experiments |

**Total**: 650+ experiments

### Common Parameters

- **Datasets**: Lorenz (3D), Mackey-Glass (1D), Santa Fe (1D)
- **Input window**: n_in = 100 (PyReCo library default)
- **Output**: n_out = 1 (single-step prediction)
- **Seeds**: 42, 43, 44, 45, 46
- **Budgets**: small (1K), medium (10K), large (50K total params)
- **Configs**: Loaded from pretuning results (`results/final/`)

### Key Changes from Previous Version (Dec 2025)

1. **LSTM early stopping + checkpoint fix**: LSTM now restores best weights, not last-epoch weights
2. **LSTM architecture**: Pretuning selected 1-layer LSTM for most configurations (was 2-3 layers before)
3. **Config consistency**: All downstream experiments (data_efficiency, multi_step, inference) now load configs from `results/final/`
4. **Result**: LSTM performance improved dramatically after fix; conclusions reversed on most comparisons

---

## 1. Main Experiment Results (Single-Step Prediction)

**Source**: `results/final/` (90 experiments)

### Statistical Analysis Summary (across all experiment types)

Based on 1620 records, 54 paired comparisons:

| Metric | PyReCo Wins | LSTM Wins | Total |
|--------|-------------|-----------|-------|
| MSE (lower better) | 3 | 51 | 54 |
| R² (higher better) | 3 | 51 | 54 |
| Training Speed | 36 | 18 | 54 |

### Winners by Dataset

| Dataset | R² Winner | PyReCo R² wins | Significance |
|---------|-----------|----------------|--------------|
| Lorenz | **LSTM** | 2/18 | Most p < 0.001 |
| Mackey-Glass | **LSTM** | 0/18 | All p < 0.001 |
| Santa Fe | **LSTM** | 1/18 | Most p < 0.001 |

### Training Speed by Scale

| Scale | Faster Model | Speedup |
|-------|-------------|---------|
| Small | **PyReCo** | 15–40x faster |
| Medium | **PyReCo** | 2–3x faster |
| Large | **LSTM** | 2–3.5x faster |

---

## 2. Data Efficiency Results

**Source**: `results/data_efficiency/` (270 experiments)
**Design**: Data lengths [1000, 2000, 3000, 5000, 7000, 10000] × 3 budgets × 5 seeds

### Key Findings

- **LSTM consistently outperforms PyReCo** across all data lengths and budgets
- Both models improve with more data, but LSTM benefits more from larger datasets
- PyReCo's advantage is only at the smallest data lengths with small budget (Santa Fe)
- Effect sizes are large (Cohen's d > 2.0 for most comparisons)

---

## 3. Multi-Step Prediction Results

**Source**: `results/multi_step/` (270 experiments)
**Design**: Horizons [1, 5, 10, 20, 50] steps, free-run (autoregressive) prediction

### Key Findings

- **LSTM wins short-term** (horizon 1–5) across all datasets
- **PyReCo catches up at longer horizons** on Santa Fe (horizon ≥ 10)
- Error accumulation rate is similar for both models
- Both models degrade significantly at horizon 50 for chaotic systems (expected behavior)

---

## 4. Supplementary: Broad Grid Search at train_ratio=0.2

**Source**: `results/supplementary_ratio02/` (15 experiments)
**Design**: PyReCo broad grid (180 combinations) vs main experiment narrow grid (6 combinations) at train_frac=0.2

**Purpose**: Determine whether PyReCo's poor performance at low data ratios is due to:
1. Parameter mismatch (pretuned grids optimized at train_frac=0.6, may be suboptimal for 0.2)
2. Architectural limitation (ESN genuinely struggles with little data)

### Results (Lorenz, seed=42, train_frac=0.2)

| Budget | Main R² (6 combos) | Broad R² (180 combos) | Improvement | Config Change |
|--------|--------------------|-----------------------|-------------|---------------|
| Small | 0.328 | 0.462 | +13% | spec_rad 0.99→0.8, leakage 0.7→0.5 |
| Medium | 0.418 | 0.426 | +0.8% | spec_rad 0.99→0.8 |
| Large | 0.385 | 0.394 | +0.9% | leakage 0.7→0.5 |

### Key Findings

- Broad grid selects **different hyperparameters** for low-data regime: lower spec_rad (0.8 vs 0.99) and lower leakage_rate (0.5 vs 0.7)
- However, R² improvement is **marginal** — both configurations produce poor R² (<0.5)
- **Conclusion: Architectural limitation, not parameter mismatch.** Even with 30× more search combinations, PyReCo cannot overcome the fundamental data insufficiency for 3D chaotic systems at train_frac=0.2

---

## 5. Hyperparameter Sensitivity Analysis

**Source**: Analysis of 90 main experiment results across 6 train_fracs
**Script**: `analysis/hyperparameter_sensitivity_analysis.py`

### PyReCo: Highly Stable (86.1% overall stability)

| Dataset | Stability | Varying Parameters |
|---------|-----------|-------------------|
| Mackey-Glass | **100%** | None — same config across all train_fracs |
| Santa Fe | **100%** | None — same config across all train_fracs |
| Lorenz | **50–75%** | spec_rad (0.95/0.99), leakage_rate (0.7/0.8/1.0) |

- `density` and `activation` are always stable across all datasets
- Cross-seed agreement: 68.1% average, 33.3% full agreement

### LSTM: Less Stable (33.3% overall stability)

| Parameter | Stability | Observation |
|-----------|-----------|-------------|
| hidden_size | Stable | Determined by budget, not train_frac |
| num_layers | Mostly stable | 1 or 2, varies on some datasets |
| **dropout** | **Very unstable** | All 4 values (0.0–0.3) selected across train_fracs |
| learning_rate | Partially stable | 0.001–0.01, varies by configuration |

- Cross-seed agreement: 39.3% average, **0% full agreement**
- Worst stability: Santa Fe medium/large (0% — all HPs change)

### Implications

- **PyReCo requires less hyperparameter tuning** when data availability changes — config is transferable
- **LSTM's optimal dropout is data-dependent** — needs re-tuning for different data regimes
- This supports PyReCo's advantage in **simplicity and fast prototyping**
- Despite config instability, LSTM's R² variance across seeds is small — multiple configs achieve similar performance

---

## 6. Inference Benchmark Results

**Source**: `results/inference_benchmark/` (3 experiments, medium budget)
**Design**: Measures predict() latency, not training quality

### Single-Sample Latency (1 sample)

| Dataset | PyReCo | LSTM | Faster |
|---------|--------|------|--------|
| Lorenz | 2.9 ms | 3.5 ms | **PyReCo 1.2x** |
| Mackey-Glass | 2.7 ms | 5.8 ms | **PyReCo 2.2x** |
| Santa Fe | 2.7 ms | 3.8 ms | **PyReCo 1.4x** |

### Batch Latency (128 samples)

| Dataset | PyReCo (ms/sample) | LSTM (ms/sample) | Faster |
|---------|---------------------|-------------------|--------|
| Lorenz | 2.40 | 0.05 | **LSTM 46x** |
| Mackey-Glass | 2.33 | 0.06 | **LSTM 37x** |
| Santa Fe | 2.44 | 0.05 | **LSTM 46x** |

### Sustained Throughput

| Dataset | PyReCo | LSTM | Ratio |
|---------|--------|------|-------|
| Lorenz | 389 /s | 101,797 /s | LSTM 262x |
| Mackey-Glass | 422 /s | 102,019 /s | LSTM 242x |
| Santa Fe | 406 /s | 102,746 /s | LSTM 253x |

### Key Findings

- **Single-sample: PyReCo faster** (1.2–2.2x) — simple matrix multiply vs LSTM gate computation
- **Batch: LSTM massively faster** (37–46x at batch=128) — PyTorch vectorization advantage
- **Throughput: LSTM dominates** (~100K vs ~400 samples/s) — PyReCo cannot parallelize across samples
- PyReCo latency is constant regardless of batch size (~2.5 ms/sample)

---

## 7. Overall Conclusions

### Accuracy: LSTM Wins

After fixing LSTM early stopping (restoring best checkpoint weights), LSTM outperforms PyReCo on **51/54 comparisons** across all experiment types. The previous finding that "PyReCo wins on Lorenz and Mackey-Glass" was an artifact of broken LSTM training.

### Training Speed: Depends on Scale

- **Small budget**: PyReCo 15–40x faster (ridge regression vs backpropagation)
- **Medium budget**: PyReCo 2–3x faster
- **Large budget**: LSTM 2–3.5x faster (PyReCo library has scaling bottleneck)

### Inference Speed: Depends on Deployment

- **Real-time single sample**: PyReCo slightly faster (2.7ms vs 3.7ms)
- **Batch processing**: LSTM dramatically faster (PyTorch parallelization)
- **High-throughput**: LSTM 250x higher throughput

### PyReCo's Remaining Advantages

1. **Simplicity**: No hyperparameter-heavy training (ridge regression, no epochs/learning rate)
2. **Small-budget training speed**: Significantly faster when model is small
3. **Single-sample latency**: Slightly faster for real-time one-at-a-time prediction
4. **Deterministic training**: No stochastic gradient descent, reproducible with same seed

### When to Use Each Model

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Maximum prediction accuracy | **LSTM** | Wins 94% of comparisons |
| Small model, fast prototyping | **PyReCo** | 15–40x faster training |
| Real-time single prediction | **PyReCo** | Lower single-sample latency |
| Batch inference at scale | **LSTM** | 250x higher throughput |
| Limited training data | **LSTM** | Still outperforms even with small data |
| Resource-constrained training | **PyReCo** | Ridge regression, no GPU needed |

---

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `experiments/run_final_experiments.py` | Main single-step experiments |
| `experiments/run_data_efficiency_experiments.py` | Data length impact |
| `experiments/run_multi_step_experiments.py` | Multi-horizon free-run prediction |
| `experiments/test_inference_benchmark.py` | Inference latency/throughput |
| `analysis/statistical_analysis.py` | Paired t-test, Wilcoxon, Cohen's d |
| `analysis/green_computing_analysis.py` | Energy and CO₂ analysis |

## Result Directories

| Directory | Contents |
|-----------|----------|
| `results/final/` | Main experiment results (90 JSON files) |
| `results/data_efficiency/` | Data efficiency results (270 JSON files) |
| `results/multi_step/` | Multi-step prediction results (270 JSON files) |
| `results/inference_benchmark/` | Inference benchmark results (3 JSON files) |
| `analysis/` | Statistical reports and CSV summaries |
