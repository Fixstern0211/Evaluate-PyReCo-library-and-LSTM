# Experiment Redesign v2: Budget-Constrained Parameter Matching

**Date**: 2026-03-20
**Status**: Design complete, pending implementation

---

## 1. Problem: Original Design Breaks Budget Constraint

### What went wrong

The original code computes `num_nodes` from the budget using default hyperparameters (δ=0.1, f_in=0.5), then allows δ and f_in to be tuned freely. After tuning, the actual total parameters deviate significantly from the budget target.

**Formula**: `N_total = N² × δ + N × d_in × f_in + N × d_out`

**Example** (Large budget=50K, Lorenz d=3):

| Stage | δ | f_in | N | Actual N_total | Budget utilization |
|-------|---|------|---|---------------|-------------------|
| Budget computation | 0.10 | 0.50 | 700 | 52,150 | 100% |
| After tuning | 0.01 | 0.10 | 700 (unchanged) | 7,210 | **14%** |

The tuning reduced PyReCo's actual model size to 14% of the budget, while LSTM always uses 100%. This is not "matched total parameter budgets."

### Impact on results

All existing single-step experiments (results/final/) have this bug. LSTM wins 51/54 comparisons, but PyReCo was effectively using only 11-30% of its parameter budget. Results cannot be trusted.

---

## 2. Root Cause: Circular Dependency

`N`, `δ`, and `f_in` are coupled through the budget equation:

```
N²×δ + N×d_in×f_in + N×d_out = budget
```

The original code broke this by:
1. Fixing N from budget + default (δ=0.1, f_in=0.5)
2. Tuning δ and f_in independently
3. Never recomputing N to maintain the budget

---

## 3. Solution: Dynamic N Computation

For each candidate (δ, f_in) in the tuning grid, solve the quadratic equation for N:

```
N = [-b + √(b² + 4·δ·budget)] / (2·δ)
where b = d_in × f_in + d_out
```

This ensures `N_total ≈ budget` for every configuration evaluated during tuning. N becomes a **dependent variable**, not a fixed constant.

### Analogy with LSTM

This is exactly how LSTM already works: when `num_layers` changes (1→2), `hidden_size` is recomputed to maintain the budget. The new ESN design follows the same logic.

| | ESN (new) | LSTM (existing) |
|---|---|---|
| Budget constraint | N²δ + Ndf_in + Nd_out = budget | 4h(h+d+2) + hd_out + d_out = budget |
| Tunable params | δ, f_in, spec_rad, leakage | num_layers, lr, dropout |
| Dependent variable | **N** (recomputed per δ, f_in) | **h** (recomputed per num_layers) |
| Budget maintained? | **Always** | **Always** |

---

## 4. Computational Constraint: N ≤ 1000

### Evidence: Training time scales as O(N²)

From existing experiments (seed=42, train_frac=0.7):

| N | Training time | Tuning time (36 combos × 5-fold CV) |
|---|--------------|-------------------------------------|
| 100 | 2s | 12s |
| 300 | 17s | 92s |
| 700 | 97s | 515s |
| 1000 (extrapolated) | ~200s | ~1,050s |
| 2000 (extrapolated) | ~794s (13 min) | ~4,200s (70 min) |

At N=2000, a single pretuning run (90 combos × 5 folds) would take ~70 hours. This is computationally infeasible.

### Which configurations are excluded?

At Large budget (50K), low density requires N > 1000:

| δ | N (Lorenz) | N (1D) | Feasible? |
|---|-----------|--------|-----------|
| 0.01 | 2,077 | 2,181 | **No** |
| 0.02 | 1,500 | 1,553 | **No** |
| 0.03 | 1,237 | 1,272 | **No** |
| 0.05 | 967 | 989 | **Yes** |
| 0.10 | 687 | 701 | **Yes** |

At Small and Medium budgets, all density values (0.01-0.10) are feasible.

### Can relaxing N cap to 1200 help?

No. δ=0.03 requires N=1227–1272 (still exceeds 1200). Even at N=1300, estimated training time is ~5.6 min/run, making pretuning (900 runs/group) take ~84 hours for a single group—computationally infeasible. The `frac_in` parameter has negligible effect on N at low density (N changes <3% across frac_in=0.1–1.0) because the N²×δ reservoir term dominates.

### Thesis discussion point (for Ch3 Methods + Ch5 Limitations)

**Suggested text for Ch3:**
> At the large parameter budget (50K), reservoir density values below 0.05 are excluded because the budget constraint requires reservoir sizes exceeding 1,200 nodes. At this scale, PyReCo's training cost—dominated by O(N²) ridge regression on the reservoir state matrix—becomes prohibitive (estimated >5 minutes per training run, compared to <2 seconds at N=100). This computational constraint limits the ESN's ability to exploit the sparse reservoir configurations (δ=0.01–0.03) that are consistently preferred at smaller budgets (see Section~\ref{sec:monotonic_trends}).

**Suggested text for Ch5 Limitations:**
> The N≤1000 computational constraint excludes density values below 0.05 at the large budget. Since pretuning and prior experiments consistently show that lower density yields better prediction accuracy on chaotic time series, this may systematically disadvantage PyReCo at large scale. A custom ESN implementation with optimized sparse matrix operations (e.g., using SciPy sparse solvers instead of PyReCo's dense ridge regression) could potentially lift this constraint, enabling exploration of δ=0.01–0.03 with N=1,200–2,000 nodes. This represents an important direction for future work.

---

## 5. Optimal Hyperparameters from Existing Experiments

Despite the budget-matching bug, the existing experiments reveal consistent hyperparameter preferences:

### Best density and input fraction (all 30 runs per dataset)

| Dataset | Best δ | Frequency | Best f_in | Frequency |
|---------|--------|-----------|-----------|-----------|
| Lorenz | **0.01** | 30/30 (100%) | **0.1** | 30/30 (100%) |
| Mackey-Glass | **0.03** | 30/30 (100%) | **0.1** | 28/30 (93%) |
| Santa Fe | **0.01** | 30/30 (100%) | **0.1** | 30/30 (100%) |

**Finding**: Low density and low input fraction are universally preferred for chaotic time series. This is consistent with the monotonic trends reported in the thesis (Ch4/Ch5).

### Monotonic trends (from thesis hyperparameter sensitivity analysis)

| Parameter | Direction | Optimal region | Physical basis |
|-----------|-----------|---------------|----------------|
| spec_rad | ↗ monotonically increasing | → 0.99 | Edge of chaos: longer memory |
| leakage | ↗ monotonically increasing | → 0.7-1.0 | Timescale matching: fast dynamics |
| density | ↘ monotonically decreasing | → 0.01-0.03 | Implicit regularization |
| frac_input | ↘ monotonically decreasing | → 0.1 | Richer internal representations |

These trends are expected to hold in the new design (they depend on chaotic system physics, not on specific N values).

---

## 6. The "Matched Parameter Budget" Discussion

### What "parameter budget" means

The parameter budget is the total number of non-zero scalar weights that a model is allowed to use. Both ESN and LSTM are constrained to the same budget:

```
ESN:  N²×δ + N×d_in×f_in + N×d_out = budget
      |--- fixed random ---|  |- trained -|

LSTM: 4h(h+d_in+2) + h×d_out + d_out = budget
      |-------- all trainable ---------|
```

This is an **engineering-oriented** matching criterion, not a theoretical capacity match. It answers a practical deployment question: **given a device that can store B floating-point weights and perform ~B multiply-add operations per inference, which architecture predicts better?**

### What it equalizes

| Dimension | Matched? | Explanation |
|-----------|----------|-------------|
| Storage size | ✅ | Both require ~budget × 4 bytes in memory |
| Inference FLOPs | ≈ | Both perform ~budget multiply-add operations per forward pass |
| Deployment cost | ✅ | Both fit the same hardware constraints |
| Learning capacity | ❌ | ESN trains only readout; LSTM trains all weights |
| State dimensionality | ❌ | ESN N can be much larger than LSTM h at the same budget |

### Why this is the right question for this thesis

The thesis does not ask "which architecture is theoretically more powerful?" (LSTM wins trivially with more trainable parameters). Instead it asks: **under the same resource constraint, which architecture delivers better predictions on chaotic systems?** This framing is relevant to:

- Edge deployment (embedded controllers, IoT sensors) where memory is limited
- Real-time control where inference latency is bounded by model size
- Green computing where energy cost scales with computation

### What it does NOT match
- ❌ State dimensionality (ESN: N can be much larger than LSTM: h)
- ❌ Learning capacity (ESN trains only readout; LSTM trains everything)

### Why not match N = h (state dimensionality)?

If we set ESN N = LSTM h:

| Budget | N = h | ESN trainable (N×d_out) | LSTM trainable (~4h²) | Ratio |
|--------|-------|------------------------|----------------------|-------|
| Small | ~14 | 42 | ~952 | LSTM 23x |
| Medium | ~48 | 144 | ~10K | LSTM 69x |
| Large | ~109 | 327 | ~50K | LSTM 153x |

ESN would have a tiny 14-109 node reservoir with negligible trainable parameters. This is not a meaningful comparison—it handicaps ESN by giving it an absurdly small reservoir.

### Why not match trainable parameters?

If ESN trainable = LSTM trainable:

| Budget | ESN trainable = 50K → N = 16,667 (1D) | ESN reservoir matrix | LSTM total |
|--------|---------------------------------------|---------------------|------------|
| Large | N = 16,667 | 16,667² × δ ≈ 2.8M-28M | 50K |

ESN's total model size would be 50-500x larger than LSTM. Also computationally infeasible.

### Conclusion

No single matching criterion is universally fair for comparing ESN and LSTM—they are architecturally too different. The total non-zero weight count is a practical, engineering-motivated criterion that reflects deployment cost (memory + compute). Its limitations are acknowledged and discussed.

---

## 7. LSTM Parameter Formula Correction

### Bug found

The code formula for LSTM parameter count is missing one bias term per layer:

| | Formula | h=14, d_in=3, d_out=3 |
|---|---------|----------------------|
| Code (wrong) | `4h(h + d_in + 1) + h×d_out + d_out` | 1,053 |
| PyTorch actual | `4h(h + d_in + 2) + h×d_out + d_out` | **1,109** |
| param_info reported | `4h(h + d_in)` (no bias, no FC) | 952 |

The `+1` accounts for `bias_ih` only; PyTorch also has `bias_hh` (another 4h per layer).

### Correct formulas

```
1 layer:  4h(h + d_in + 2) + h×d_out + d_out
L layers: 4h(h + d_in + 2) + (L-1)×4h(2h + 2) + h×d_out + d_out
```

### Impact

Hidden size will be slightly smaller after correction (e.g., Large Lorenz: h=109→~108). Minimal impact on results.

---

## 8. Other Bugs Fixed in New Design

| Bug | Fix |
|-----|-----|
| `set_seed()` missing `torch.manual_seed()` | Add torch seed setting |
| LSTM final model no val-based early stopping | Use best epoch from tuning, or hold out monitor set |
| `param_info` computed from default config | Compute from actual (best) config |
| `pyreco.datasets.load` doesn't exist | Use `local_load` everywhere (already the runtime behavior) |

---

## 9. Final Experiment Design

### Phase 1: Pretuning (per dataset × budget = 9 groups)

**Setup**: seed=42, train_frac=0.7, 5-fold forward-chaining CV

**Grid** (layered by budget, updated after quick test):

| Budget | δ values | f_in values | δ×f_in combos |
|--------|----------|-------------|---------------|
| Small (1K) | {0.01, 0.03, 0.05, 0.1} | {0.1, 0.3, 0.5} | 12 |
| Medium (10K) | {0.01, 0.03, 0.05, 0.1} | {0.1, 0.3, 0.5} | 12 |
| Large (50K) | {0.05, 0.1} | {0.1, 0.3, 0.5} | 6 |

Dynamics grid (extended after quick test showed optimal spec_rad shifted lower with larger N):
- spec_rad: {0.5, 0.7, 0.8, 0.9, 0.99} — 5 values (added 0.5, 0.7; removed 0.95)
- leakage: {0.1, 0.3, 0.5, 0.7, 0.8, 1.0} — 6 values (added 0.1)
- dynamics combos: 5 × 6 = 30

For each combo, N is dynamically computed from budget constraint.

**Total**: (6×360 + 3×180) × 5 folds = (2160+540) × 5 = **13,500 training runs**

**Key finding from quick test (Lorenz, Small, 3-fold)**:
Best config was δ=0.01, fi=0.3, N=176, spec_rad=0.80, leakage=0.50. The optimal spec_rad=0.80 (not 0.99) confirms that larger budget-constrained N shifts the optimal spectral radius lower, invalidating the monotonic trend observed with the old fixed-N design.

**Outputs**:
- Best 1-2 (δ, f_in) per dataset/budget
- Narrowed (spec_rad, leakage) range (leveraging verified monotonicity)
- Verified N values and actual total parameters

**Cross-validation justification and implementation**:

5-fold forward-chaining CV is the standard for time series hyperparameter selection (Kohavi, 1995; scikit-learn `TimeSeriesSplit`). k=5 provides an acceptable bias-variance tradeoff.

**Why forward-chaining, not standard k-fold?** Standard k-fold shuffles data across folds, which would use future timesteps to predict past values — a form of temporal data leakage that is invalid for time series. Forward-chaining preserves chronological order by always training on past and validating on future.

**Key property: training set size grows across folds.**

```
Standard k-fold (INVALID for time series):
  Each fold: train ≈ 80% of data, val ≈ 20%
  Problem: folds use future data in training

Forward-chaining (CORRECT for time series):
  Fold 1: train = 1/6 (17%), val = next 1/6 (17%)
  Fold 2: train = 2/6 (33%), val = next 1/6 (17%)
  Fold 3: train = 3/6 (50%), val = next 1/6 (17%)
  Fold 4: train = 4/6 (67%), val = next 1/6 (17%)
  Fold 5: train = 5/6 (83%), val = next 1/6 (17%)
  → Validation size is constant; training size expands
  → Always uses past to predict future
```

**Consequence**: early folds (small training set) produce higher MSE than later folds. For chaotic systems like Lorenz, early folds may fail entirely if the training data does not cover the full attractor structure (e.g., only one wing of the butterfly attractor). The CV mean MSE is therefore higher than the final test MSE obtained with full training data (train_frac ≥ 0.7). This is expected and correct — the purpose of pretuning CV is to **rank** configurations, not to estimate absolute test performance.

**Comparison with old pretuning CV**: The old pretuning performed CV on already-windowed data, causing adjacent windows at fold boundaries to share n_in−1 = 99 timesteps (data leakage). This produced artificially low CV MSE values (e.g., 0.0009 for Lorenz vs 0.18 in the new implementation). The ranking may also have been affected — the new CV selects different optimal (spec_rad, leakage) values for Lorenz.

**Suggested text for Ch3 Methods:**

> Forward-chaining cross-validation is used instead of standard $k$-fold CV because time series data must preserve chronological order: training data must always precede validation data to avoid temporal leakage. In each of the five folds, the training set expands from approximately 17\% to 83\% of the available data, while the validation set remains a fixed-size segment immediately following the training period. This expanding-window design means that early folds evaluate model performance with very limited data, producing higher error rates, while later folds approximate the full-data regime. The CV mean MSE is therefore a conservative estimate biased upward by the data-poor early folds; its purpose is to rank hyperparameter configurations reliably, not to predict absolute test-set performance.

**Critical implementation detail: CV must split the raw time series BEFORE creating sliding windows.** If sliding windows are created first and then split into folds, adjacent windows at the fold boundary share n_in − 1 = 99 of 100 input timesteps—a severe data leakage that inflates validation scores.

Illustration (n_in=100, n_out=1):

```
WRONG (window-level split — data leakage):
  All windows created from full series: W1, W2, ..., W_N
  Fold boundary: W_k (train) and W_{k+1} (val) share 99 timesteps
  → Validation score is optimistically biased

CORRECT (raw-series split — no leakage):
  Raw series split: train = raw[0 : T_k], val = raw[T_k : T_{k+1}]
  Windows created independently within each segment
  → Train windows end at T_k − n_in, val windows start at T_k
  → Zero shared timesteps between train and val
  → Cost: first n_in−1 timesteps of val segment cannot form complete windows
```

This implementation choice is documented in the code (`timeseries_cv_split_raw` in `run_pretuning_v2.py`). Each fold independently fits a `StandardScaler` on its training segment only, preventing distributional leakage.

**Suggested text for Ch3 Methods (Cross-Validation subsection):**

> Hyperparameter selection uses 5-fold forward-chaining cross-validation on the combined training and validation data. To avoid data leakage from overlapping sliding windows, the CV split is performed on the raw time series \emph{before} creating input--output window pairs. In each fold, a StandardScaler is fitted on the training segment only and applied to the validation segment. Sliding windows of length $n_{\text{in}} = 100$ are then constructed independently within each segment, ensuring zero overlap between training and validation windows at the fold boundary. This approach sacrifices $n_{\text{in}} - 1 = 99$ timesteps at the start of each validation segment (which cannot form complete input windows) but guarantees that no raw timestep appears in both training and validation data within the same fold. The choice of $k = 5$ follows the recommendation of Kohavi (1995) for balancing bias and variance in model selection.

**References:**
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. IJCAI, 1137–1143.

### Phase 2: Final Experiments (270 groups)

**Setup**: 3 datasets × 3 budgets × 6 train_fracs × 5 seeds

**Grid**: Narrowed from pretuning results
- ESN: ~12 combos (2 δ×f_in × 6 dynamics)
- LSTM: 32 combos (4 lr × 4 dropout × 2 layers)

**Statistical validation**: 5 seeds provide paired observations for t-tests.

### Phase 3: Multi-step + Data Efficiency (540 groups)

Use best configs from Phase 2. No additional tuning.

### Estimated Runtime

| Phase | Training runs | Est. time |
|-------|--------------|-----------|
| Pretuning | 9,000 | ~50h |
| Final (ESN) | 270 × 12 = 3,240 | ~20h |
| Final (LSTM) | 270 × 32 = 8,640 | ~70h |
| Multi-step | 540 × 2 = 1,080 | ~5h |
| Data-efficiency | 540 × 2 = 1,080 | ~5h |
| **Total** | ~23,040 | **~150h** |

---

## 10. Thesis Sections Requiring Rewrite After New Experiments

| Section | What changes |
|---------|-------------|
| Abstract | All R² values, win counts, conclusions |
| Ch3 Methods | Budget matching procedure, parameter formula, grid description |
| Ch3 Table 1 | All parameter counts (use actual values) |
| Ch4 Results | All tables, figures, numerical claims |
| Ch5 Discussion | Core findings, parameter budget analysis |
| Ch6 Conclusion | Summary, recommendations |
| All figures | Regenerate from new data |

---

## 11. Supplementary Finding: ESN Parameter Efficiency from Old Experiments

### Discovery

The old experiments (results/final/, Feb 2026) had a budget-matching bug: PyReCo's num_nodes was computed from the budget using default density=0.1, but tuning then selected density=0.01, reducing the actual total parameters to 14–43% of the budget target. LSTM always used 100% of the budget.

Despite this unintended asymmetry, the old results reveal a valuable finding about ESN parameter efficiency:

| Budget target | PyReCo actual total | LSTM actual total | PyReCo budget utilization | Prediction performance |
|--------------|--------------------|--------------------|--------------------------|----------------------|
| Small (1K) | ~430 | ~950 | 43% | Both achieve similar R² |
| Medium (10K) | ~1,890 | ~10,000 | 19% | Both R² > 0.99 on Lorenz/MG |
| Large (50K) | ~7,210 | ~50,000 | 14% | LSTM slightly better |

**ESN achieves near-equivalent prediction accuracy with only 14–43% of the parameters used by LSTM.** This demonstrates the core value proposition of reservoir computing: a fixed random nonlinear feature expansion (reservoir) combined with a small trained readout can match the performance of a fully trained recurrent network that uses 2–7× more parameters.

### How to use in the thesis

The old data and new data answer **complementary questions**:

| | Old experiments (supplementary) | New experiments (primary) |
|---|---|---|
| **Matching** | Unmatched (PyReCo used 14–43% of budget) | Strictly matched (both ≈ budget) |
| **Question** | How parameter-efficient is ESN? | Under equal budget, which is better? |
| **Role in thesis** | Supplementary analysis in Ch5 Discussion | Main results in Ch4 |

### Critical Assessment of Old Data Usability

**Supports use:**
- Data split confirmed correct (pyreco_load, n_test=1400), same as new experiments
- No train/test leakage
- 5 seeds provide paired comparisons
- Actual parameter counts recoverable from saved configs

**Against use (weaknesses):**

1. **Not a controlled experiment.** The parameter asymmetry was a bug, not a designed study. This is an *observation*, not an *experimental finding*. No systematic control of parameter count.

2. **LSTM won 51/54 overall.** "ESN parameter efficiency" only applies to Lorenz/MG at high train_frac where both achieve R² > 0.99. On Santa Fe, ESN clearly lost at all budgets. The efficiency claim is dataset-dependent.

3. **Confound: state dimensionality vs parameter count.** PyReCo has 7,210 total params but a 700-dimensional state space. LSTM has 50,000 params but only 109-dimensional state space. A reviewer could argue ESN performs well due to its larger state space, not despite having fewer parameters.

4. **B1 bug may bias results.** LSTM final model used training-loss early stopping (no validation monitoring). This may have caused LSTM to overfit, making ESN appear relatively better. The "comparable performance" observation may partly reflect this LSTM disadvantage.

**Usability score: 3/5.** Valid as a cautious observation in Discussion, not as a primary finding.

### Suggested text for Ch5 Discussion (revised — cautious framing)

> An incidental observation from preliminary experiments suggests that ESN may achieve competitive accuracy with substantially fewer total parameters than LSTM. In configurations where the ESN used only 14--43\% of the parameter budget (due to reservoir density optimization without reservoir size adjustment), prediction accuracy on the Lorenz and Mackey-Glass systems remained comparable to the fully-budgeted LSTM (R\textsuperscript{2} $>$ 0.99 at medium budget). However, this observation is subject to several caveats: (1) the parameter asymmetry was not a controlled experimental variable; (2) the ESN's state dimensionality ($N = 700$) was substantially larger than the LSTM's hidden dimensionality ($h = 109$), confounding parameter count with representational capacity; and (3) the effect was limited to synthetic chaotic systems where both models approached ceiling performance. The main matched-budget experiments (Section~\ref{sec:single_step}) provide the rigorous, controlled comparison; this observation is noted as suggestive evidence for the practical parameter efficiency of reservoir computing in well-characterized dynamical systems.
