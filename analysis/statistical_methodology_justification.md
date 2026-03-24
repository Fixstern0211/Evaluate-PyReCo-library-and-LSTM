# Statistical Methodology Justification

## Why Shapiro-Wilk and Wilcoxon Are Not Used in the Final Analysis

This document provides rigorous justification for excluding the Shapiro-Wilk
normality test and Wilcoxon signed-rank test from the statistical analysis of
PyReCo vs LSTM experiments. The paired t-test (with Holm–Bonferroni correction)
is used unconditionally as the sole hypothesis test.

### Experimental Design

- 3 datasets × 3 budgets × 3 experiment types = 27 conditions
- Each condition: 5 paired observations (seeds 42–46)
- Each pair: PyReCo and LSTM trained/tested on identical data
- Primary test: paired t-test on differences d_i = PyReCo_i − LSTM_i

---

## 1. Wilcoxon Signed-Rank Test Is Structurally Powerless at n = 5

**Exact derivation.** Under H₀, each of the n = 5 paired differences is equally
likely to be positive or negative, independently. There are 2⁵ = 32 equally
probable sign assignments. The test statistic W⁺ (sum of ranks of positive
differences) ranges from 0 to 1+2+3+4+5 = 15.

The most extreme outcomes are W⁺ = 0 and W⁺ = 15, each occurring for exactly
1 of the 32 assignments. The minimum attainable two-sided p-value is therefore:

    p_min = 2 × (1/32) = 0.0625

**Since 0.0625 > 0.05, the Wilcoxon test cannot reject H₀ at α = 0.05 with
n = 5, regardless of the data.** Even if all five differences unanimously
favour one model, the test returns "not significant." This makes it useless
as a hypothesis test in this design.

By contrast, the paired t-test exploits the magnitudes (not just ranks) of the
differences via a continuous t(4) distribution. Its two-sided critical value is
t₀.₀₂₅,₄ = 2.776, which is attainable whenever the mean difference is
sufficiently large relative to the standard error.

---

## 2. Shapiro-Wilk Has Near-Zero Power at n = 5

With only 5 observations, the Shapiro-Wilk test has statistical power of
roughly 5–15% against moderate departures from normality — barely above the
nominal α = 5%. A non-significant result (p > 0.05) therefore provides
**no meaningful evidence** that the differences are normally distributed; it
simply reflects the test's inability to detect violations.

Key references on low power of normality tests at small n:

- **Razali & Wah (2011)**, "Power comparisons of Shapiro-Wilk,
  Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests," *J. Statistical
  Modeling and Analytics*, 2(1), 21–33. Even at n = 10, power against symmetric
  non-normal distributions was often below 20%.

- **Yazici & Yolacan (2007)**, "A comparison of various tests of normality,"
  *J. Statistical Computation and Simulation*, 77(2), 175–183. For n ≤ 10, all
  normality tests have power barely above the nominal alpha.

- **D'Agostino & Stephens (1986)**, *Goodness-of-Fit Techniques*, Marcel
  Dekker. Normality tests require n ≥ 20 for reasonable power; below n = 10
  they are essentially uninformative.

---

## 3. The Paired t-Test Is Robust to Non-Normality

The paired t-test maintains its nominal Type I error rate under a wide range
of departures from normality, a property established over decades of research:

- **Boneau (1960)**, "The effects of violations of assumptions underlying the
  t test," *Psychological Bulletin*, 57(1), 49–64. Classic Monte Carlo study
  showing that the t-test is robust to non-normality even at small n. Skewness
  has modest effects; kurtosis has even less.

- **Box (1953)**, "Non-normality and tests on variances," *Biometrika*,
  40(3/4), 318–335. Early theoretical demonstration that t-test significance
  levels are insensitive to non-normality when testing means.

- **Sawilowsky & Blair (1992)**, "A more realistic look at the robustness and
  Type II error properties of the t test," *Psychological Bulletin*, 111(2),
  352–360. Confirms robustness for most practical departures, with caveats
  mainly for heavily skewed distributions at extremely small n.

In the paired design specifically, the test operates on **differences**
d_i = X_i − Y_i, not on raw observations. Differencing removes shared
variation (dataset-specific, seed-specific effects), typically yielding a more
symmetric distribution than either raw variable alone.

---

## 4. The "Pre-Test Then Decide" Workflow Is Harmful

An alternative strategy — run Shapiro-Wilk first, then choose t-test (if
"normal") or Wilcoxon (if "non-normal") — is a **conditional procedure** that
is strictly worse than using the t-test unconditionally:

- **Rochon, Gondan & Kieser (2012)**, "To test or not to test: Preliminary
  assessment of normality when comparing two independent samples," *BMC Medical
  Research Methodology*, 12, 81. Simulations show the two-stage procedure has
  **worse Type I error control** than the unconditional t-test. Overall alpha
  can substantially exceed the nominal level.

- **Zimmerman (2004)**, "A note on preliminary tests of equality of variances,"
  *British J. Mathematical and Statistical Psychology*, 57(1), 173–181.
  Preliminary testing of assumptions inflates the overall Type I error rate.

- **Rasch, Kubinger & Moder (2011)**, "The two-sample t test: pre-testing its
  assumptions does not pay off," *Statistical Papers*, 52(1), 219–231.
  Concludes that unconditional use of the t-test is preferable to any
  pre-test-based adaptive strategy.

- **Albers, Boon & Kallenberg (2000)**, "The asymptotic behavior of tests for
  normal means based on a variance pre-test," *J. Statistical Planning and
  Inference*, 88(1), 47–57. Sequential testing distorts size properties because
  the conditional distribution of the main test depends on the pre-test outcome.

At n = 5 the pre-test is doubly pointless: Shapiro-Wilk almost always "passes"
(low power), and Wilcoxon cannot reach significance even if selected.

---

## 5. Conclusion

| Test | Problem at n = 5 | Consequence |
|------|-------------------|-------------|
| Wilcoxon signed-rank | min p = 0.0625 > α = 0.05 | Cannot reject H₀; structurally powerless |
| Shapiro-Wilk | Power ≈ 5–15% | Non-rejection is uninformative |
| Pre-test workflow | Inflates Type I error | Worse than unconditional t-test |

**The paired t-test is the appropriate and sufficient test for this design.**
It is complemented by:

- **Holm–Bonferroni correction** for multiple comparisons across conditions
- **95% confidence interval of the mean difference** for practical significance
- **Cohen's d_z** (paired effect size) for standardised magnitude

These three quantities — significance, precision, and effect size — fully
characterise each pairwise comparison. No additional tests are needed.
