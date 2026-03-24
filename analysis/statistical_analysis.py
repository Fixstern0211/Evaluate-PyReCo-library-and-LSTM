#!/usr/bin/env python3
"""
Unified Statistical Analysis for PyReCo vs LSTM Experiments

Supports three data sources:
- results/final/: Single-step prediction (90 experiments)
- results/multi_step/: Multi-step prediction (270 experiments)
- results/data_efficiency/: Data efficiency (270 experiments)

Statistical methods:
1. Paired t-test with 95% CI
2. Wilcoxon signed-rank test (non-parametric)
3. Shapiro-Wilk normality test
4. Cohen's d effect size with CI
5. Holm-Bonferroni multiple comparison correction

Usage:
    python analysis/statistical_analysis.py
    python analysis/statistical_analysis.py --source final
    python analysis/statistical_analysis.py --source multi_step
    python analysis/statistical_analysis.py --source data_efficiency
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import defaultdict

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Multiple comparison correction unavailable.")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_final(results_dir: str = "results/final") -> pd.DataFrame:
    """Load single-step prediction results (final format)"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/results_*.json"))

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        metadata = data['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']

        for scale, models in data['results'].items():
            for model in models:
                records.append({
                    'source': 'final',
                    'dataset': dataset,
                    'seed': seed,
                    'train_frac': train_frac,
                    'scale': scale,
                    'model_type': model['model_type'],
                    'test_mse': model['test_mse'],
                    'test_r2': model['test_r2'],
                    'train_time': model.get('final_train_time', 0),
                    'tune_time': model.get('tune_time', 0),
                    'trainable_params': model['param_info']['trainable'],
                })

    return pd.DataFrame(records)


def load_multi_step(results_dir: str = "results/multi_step") -> pd.DataFrame:
    """Load multi-step prediction results"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/multistep_*.json"))

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        metadata = data['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']
        budget = metadata['budget']

        for model_type, model_data in data['models'].items():
            horizon_results = model_data.get('horizon_results', {})
            # Use horizon=1 as baseline metric
            h1 = horizon_results.get('1', {})

            records.append({
                'source': 'multi_step',
                'dataset': dataset,
                'seed': seed,
                'train_frac': train_frac,
                'scale': budget,
                'model_type': model_type,
                'test_mse': h1.get('mse'),
                'test_r2': h1.get('r2'),
                'train_time': model_data.get('train_time', 0),
                'tune_time': 0,
                'trainable_params': None,
            })

    return pd.DataFrame(records)


def load_data_efficiency(results_dir: str = "results/data_efficiency") -> pd.DataFrame:
    """Load data efficiency experiment results"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/dataeff_*.json"))

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        metadata = data['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']
        budget = metadata['budget']
        data_length = metadata.get('data_length', 5000)

        for model_type, model_data in data['models'].items():
            metrics = model_data.get('metrics', {})

            records.append({
                'source': 'data_efficiency',
                'dataset': dataset,
                'seed': seed,
                'train_frac': train_frac,
                'scale': budget,
                'data_length': data_length,
                'model_type': model_type,
                'test_mse': metrics.get('mse'),
                'test_r2': metrics.get('r2'),
                'train_time': model_data.get('train_time', 0),
                'tune_time': 0,
                'trainable_params': None,
            })

    return pd.DataFrame(records)


def load_all_data(source: str = "all") -> pd.DataFrame:
    """Load data from specified source(s)"""
    dfs = []

    if source in ['all', 'final']:
        df = load_final()
        if len(df) > 0:
            dfs.append(df)
            print(f"  - final: {len(df)} records")

    if source in ['all', 'multi_step']:
        df = load_multi_step()
        if len(df) > 0:
            dfs.append(df)
            print(f"  - multi_step: {len(df)} records")

    if source in ['all', 'data_efficiency']:
        df = load_data_efficiency()
        if len(df) > 0:
            dfs.append(df)
            print(f"  - data_efficiency: {len(df)} records")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Statistical Test Functions
# ============================================================================

def paired_ttest_with_ci(group1, group2, confidence=0.95):
    """Paired t-test with confidence interval"""
    group1 = np.array(group1)
    group2 = np.array(group2)
    diff = group1 - group2
    n = len(diff)

    if n < 2:
        return None

    t_stat, p_value = stats.ttest_rel(group1, group2)

    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    df = n - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        'mean1': np.mean(group1),
        'std1': np.std(group1, ddof=1),
        'mean2': np.mean(group2),
        'std2': np.std(group2, ddof=1),
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'p_value': p_value,
        'df': df,
        'n': n,
    }


def wilcoxon_test(group1, group2):
    """Wilcoxon signed-rank test (non-parametric).

    NOTE: With n<=5 pairs the minimum attainable two-sided p-value is 0.0625,
    so the test cannot reach significance at alpha=0.05.  It is retained for
    completeness but flagged accordingly in the report.
    """
    n = len(group1)
    try:
        w_stat, p_value = stats.wilcoxon(group1, group2, zero_method='wilcox')
        return {
            'statistic': w_stat,
            'p_value': p_value,
            'success': True,
            'underpowered': n <= 5,
        }
    except Exception as e:
        return {'statistic': None, 'p_value': None, 'success': False, 'error': str(e)}


def shapiro_wilk_test(data):
    """Normality test using Shapiro-Wilk.

    NOTE: With n<=5 observations the test has very low power; failure to
    reject normality does NOT confirm that the data are normal.
    """
    n = len(data)
    if n < 3:
        return {'statistic': None, 'p_value': None, 'is_normal': None, 'low_power': True}

    stat, p_value = stats.shapiro(data)
    return {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05,
        'low_power': n <= 7,
    }


def cohens_d_with_ci(group1, group2, confidence=0.95):
    """Cohen's d_z effect size for paired samples with confidence interval.

    Uses d_z = mean(diff) / SD(diff), the standard effect size for paired designs.
    SE formula: sqrt(1/n + d_z^2 / (2*n))  (Algina & Keselman, 2003).
    """
    diff = np.array(group1) - np.array(group2)
    n = len(diff)

    if n < 2:
        return 0, 0, 0, "undefined"

    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return 0, 0, 0, "undefined"

    d = np.mean(diff) / sd_diff

    # SE and CI for paired d_z
    se_d = np.sqrt(1.0 / n + d**2 / (2.0 * n))
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = d - z_crit * se_d
    ci_upper = d + z_crit * se_d

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, ci_lower, ci_upper, interpretation


# ============================================================================
# Analysis Functions
# ============================================================================

def perform_pairwise_comparison(df: pd.DataFrame) -> list:
    """Perform pairwise comparison between PyReCo and LSTM"""
    comparisons = []

    # Group by source, dataset, train_frac, scale to prevent cross-experiment mixing
    groupby_cols = ['source', 'dataset', 'train_frac', 'scale']
    if 'data_length' in df.columns and df['data_length'].notna().any():
        groupby_cols.append('data_length')

    for group_key, group_df in df.groupby(groupby_cols):
        pyreco_df = group_df[group_df['model_type'] == 'pyreco_standard']
        lstm_df = group_df[group_df['model_type'] == 'lstm']

        if len(pyreco_df) < 2 or len(lstm_df) < 2:
            continue

        # Sort by seed to ensure pairing
        pyreco_df = pyreco_df.sort_values('seed')
        lstm_df = lstm_df.sort_values('seed')

        # Only use common seeds
        common_seeds = set(pyreco_df['seed']) & set(lstm_df['seed'])
        if len(common_seeds) < 2:
            continue

        pyreco_df = pyreco_df[pyreco_df['seed'].isin(common_seeds)]
        lstm_df = lstm_df[lstm_df['seed'].isin(common_seeds)]

        pyreco_mse = pyreco_df['test_mse'].values
        lstm_mse = lstm_df['test_mse'].values
        pyreco_r2 = pyreco_df['test_r2'].values
        lstm_r2 = lstm_df['test_r2'].values
        pyreco_time = pyreco_df['train_time'].values
        lstm_time = lstm_df['train_time'].values

        comparison = {
            'group': dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key])),
            'n_pairs': len(common_seeds),
        }

        # For each metric: t-test, Wilcoxon, Cohen's d, Shapiro-Wilk on diffs
        for metric_name, vals1, vals2 in [
            ('mse', pyreco_mse, lstm_mse),
            ('r2', pyreco_r2, lstm_r2),
            ('train_time', pyreco_time, lstm_time),
        ]:
            diff = vals1 - vals2
            comparison[metric_name] = {
                'pyreco_mean': np.mean(vals1),
                'pyreco_std': np.std(vals1, ddof=1),
                'lstm_mean': np.mean(vals2),
                'lstm_std': np.std(vals2, ddof=1),
                'ttest': paired_ttest_with_ci(vals1, vals2),
                'wilcoxon': wilcoxon_test(vals1, vals2),
                'effect_size': cohens_d_with_ci(vals1, vals2),
                'shapiro': shapiro_wilk_test(diff),
            }

        comparisons.append(comparison)

    # Apply Holm-Bonferroni correction for each metric separately
    if HAS_STATSMODELS and comparisons:
        for metric_name in ['mse', 'r2', 'train_time']:
            p_values = []
            indices = []
            for i, comp in enumerate(comparisons):
                if comp[metric_name]['ttest']:
                    p_values.append(comp[metric_name]['ttest']['p_value'])
                    indices.append(i)

            if p_values:
                _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
                for j, idx in enumerate(indices):
                    comparisons[idx][metric_name]['ttest']['p_corrected'] = p_corrected[j]

    return comparisons


def generate_report(df: pd.DataFrame, comparisons: list) -> str:
    """Generate statistical analysis report"""
    report = []

    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Data summary
    report.append("## Data Summary")
    report.append("-" * 40)
    for source in df['source'].unique():
        count = len(df[df['source'] == source])
        report.append(f"  {source}: {count} records")
    report.append(f"  Total: {len(df)} records")
    report.append(f"  Datasets: {sorted(df['dataset'].unique())}")
    report.append(f"  Scales: {sorted(df['scale'].unique())}")
    report.append(f"  Comparisons performed: {len(comparisons)}")
    report.append("")

    # Statistical methods
    report.append("## Statistical Methods")
    report.append("-" * 40)
    report.append("  - Paired t-test with 95% Confidence Intervals")
    report.append("  - Cohen's d_z (paired) effect size with 95% CI")
    report.append("  - Shapiro-Wilk normality test on paired differences")
    report.append("  - Wilcoxon signed-rank test (non-parametric, for reference only)")
    if HAS_STATSMODELS:
        report.append("  - Holm-Bonferroni multiple comparison correction")
    report.append("")
    report.append("## Methodological Notes")
    report.append("-" * 40)
    report.append("  - n=5 paired observations per comparison (seeds 42-46)")
    report.append("  - Wilcoxon: min attainable p=0.0625 with n=5, cannot reach alpha=0.05")
    report.append("  - Shapiro-Wilk: low power at n=5; non-rejection does not confirm normality")
    report.append("  - Cohen's d_z = mean(diff)/SD(diff), appropriate for paired designs")
    report.append("")

    # Detailed comparisons by source, then dataset
    source_labels = {'final': 'Single-Step Prediction', 'multi_step': 'Multi-Step Prediction', 'data_efficiency': 'Data Efficiency'}
    for source in sorted(df['source'].unique()):
        report.append("")
        report.append("#" * 80)
        report.append(f"EXPERIMENT: {source_labels.get(source, source).upper()}")
        report.append("#" * 80)

        source_comps = [c for c in comparisons if c['group'].get('source') == source]

        for dataset in sorted(df[df['source'] == source]['dataset'].unique()):
            report.append("")
            report.append("=" * 80)
            report.append(f"DATASET: {dataset.upper()}")
            report.append("=" * 80)

            dataset_comps = [c for c in source_comps if c['group'].get('dataset') == dataset]

            for scale in ['small', 'medium', 'large']:
                scale_comps = [c for c in dataset_comps if c['group'].get('scale') == scale]

                if not scale_comps:
                    continue

                report.append(f"\n### {scale.upper()} Scale")
                report.append("-" * 60)

                for comp in scale_comps:
                    train_frac = comp['group'].get('train_frac', 'N/A')
                    n = comp['n_pairs']

                    report.append(f"\n► Train Fraction: {train_frac} (n={n} pairs)")

                    for metric_name, metric_label, fmt, lower_better in [
                        ('mse', 'MSE', '.6f', True),
                        ('r2', 'R²', '.6f', False),
                        ('train_time', 'Training Time', '.2f', True),
                    ]:
                        m = comp[metric_name]
                        ttest_m = m['ttest']
                        d_m, d_low_m, d_up_m, d_interp_m = m['effect_size']
                        shapiro_m = m.get('shapiro', {})
                        wilcoxon_m = m.get('wilcoxon', {})

                        unit = 's' if metric_name == 'train_time' else ''
                        better = 'lower is better' if lower_better else 'higher is better'
                        report.append(f"\n  [{metric_label}] ({better})")
                        report.append(f"    PyReCo: {m['pyreco_mean']:{fmt}}{unit} ± {m['pyreco_std']:{fmt}}{unit}")
                        report.append(f"    LSTM:   {m['lstm_mean']:{fmt}}{unit} ± {m['lstm_std']:{fmt}}{unit}")

                        if ttest_m:
                            p_val = ttest_m['p_value']
                            p_corr = ttest_m.get('p_corrected', p_val)
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            sig_corr = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"

                            report.append(f"    Mean Diff: {ttest_m['mean_diff']:{fmt}} [95% CI: {ttest_m['ci_lower']:{fmt}}, {ttest_m['ci_upper']:{fmt}}]")
                            report.append(f"    t({ttest_m['df']}) = {ttest_m['t_stat']:.3f}, p = {p_val:.4f} {sig}")
                            if p_corr != p_val:
                                report.append(f"    p (Holm-Bonferroni) = {p_corr:.4f} {sig_corr}")
                            report.append(f"    Cohen's d_z = {d_m:.3f} [{d_low_m:.3f}, {d_up_m:.3f}] ({d_interp_m})")

                        # Shapiro-Wilk normality test on differences
                        if shapiro_m and shapiro_m.get('p_value') is not None:
                            normal_str = "normal" if shapiro_m['is_normal'] else "non-normal"
                            lp_note = " [low power]" if shapiro_m.get('low_power') else ""
                            report.append(f"    Shapiro-Wilk: W={shapiro_m['statistic']:.4f}, p={shapiro_m['p_value']:.4f} ({normal_str}){lp_note}")

                        # Wilcoxon signed-rank test
                        if wilcoxon_m and wilcoxon_m.get('success'):
                            w_sig = "***" if wilcoxon_m['p_value'] < 0.001 else "**" if wilcoxon_m['p_value'] < 0.01 else "*" if wilcoxon_m['p_value'] < 0.05 else "ns"
                            up_note = " [underpowered, min p=0.0625]" if wilcoxon_m.get('underpowered') else ""
                            report.append(f"    Wilcoxon: W={wilcoxon_m['statistic']:.1f}, p={wilcoxon_m['p_value']:.4f} {w_sig}{up_note}")

                        # Winner
                        if lower_better:
                            winner = "PyReCo" if m['pyreco_mean'] < m['lstm_mean'] else "LSTM"
                            if metric_name == 'train_time':
                                speedup = m['lstm_mean'] / (m['pyreco_mean'] + 1e-6)
                                if speedup > 1:
                                    report.append(f"    -> PyReCo is {speedup:.1f}x faster")
                                else:
                                    report.append(f"    -> LSTM is {1/speedup:.1f}x faster")
                            else:
                                report.append(f"    -> Winner: {winner}")
                        else:
                            winner = "PyReCo" if m['pyreco_mean'] > m['lstm_mean'] else "LSTM"
                            report.append(f"    -> Winner: {winner}")

    # Summary
    report.append("")
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    # Count winners
    pyreco_wins = {'mse': 0, 'r2': 0, 'time': 0}
    lstm_wins = {'mse': 0, 'r2': 0, 'time': 0}

    for comp in comparisons:
        if comp['mse']['pyreco_mean'] < comp['mse']['lstm_mean']:
            pyreco_wins['mse'] += 1
        else:
            lstm_wins['mse'] += 1

        if comp['r2']['pyreco_mean'] > comp['r2']['lstm_mean']:
            pyreco_wins['r2'] += 1
        else:
            lstm_wins['r2'] += 1

        if comp['train_time']['pyreco_mean'] < comp['train_time']['lstm_mean']:
            pyreco_wins['time'] += 1
        else:
            lstm_wins['time'] += 1

    total = len(comparisons)
    report.append("")
    report.append(f"Total comparisons: {total}")
    report.append(f"MSE wins:   PyReCo {pyreco_wins['mse']}/{total}, LSTM {lstm_wins['mse']}/{total}")
    report.append(f"R² wins:    PyReCo {pyreco_wins['r2']}/{total}, LSTM {lstm_wins['r2']}/{total}")
    report.append(f"Time wins:  PyReCo {pyreco_wins['time']}/{total}, LSTM {lstm_wins['time']}/{total}")
    report.append("")

    # By dataset
    report.append("Winners by Dataset:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_comps = [c for c in comparisons if c['group'].get('dataset') == dataset]
        if not dataset_comps:
            continue

        pyreco_r2_wins = sum(1 for c in dataset_comps if c['r2']['pyreco_mean'] > c['r2']['lstm_mean'])
        total_ds = len(dataset_comps)
        winner = "PyReCo" if pyreco_r2_wins > total_ds / 2 else "LSTM"
        report.append(f"  {dataset}: {winner} (R² wins: {pyreco_r2_wins}/{total_ds})")

    report.append("")
    report.append("=" * 80)
    report.append("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Statistical Analysis for PyReCo vs LSTM')
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'final', 'multi_step', 'data_efficiency'],
                        help='Data source to analyze')
    parser.add_argument('--output-dir', type=str, default='analysis',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    print(f"\nLoading data (source: {args.source})...")
    df = load_all_data(args.source)

    if len(df) == 0:
        print("ERROR: No data found!")
        return

    # Filter valid data
    df = df[df['test_mse'].notna() & df['test_r2'].notna()]
    print(f"\nValid records: {len(df)}")

    print("\nPerforming pairwise comparisons...")
    comparisons = perform_pairwise_comparison(df)
    print(f"Comparisons completed: {len(comparisons)}")

    print("\nGenerating report...")
    report = generate_report(df, comparisons)
    print("\n" + report)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "statistical_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save detailed CSV
    csv_path = output_dir / "statistical_comparisons.csv"
    comparison_records = []
    for comp in comparisons:
        rec = {**comp['group']}
        rec['n_pairs'] = comp['n_pairs']
        rec['mse_pyreco'] = comp['mse']['pyreco_mean']
        rec['mse_lstm'] = comp['mse']['lstm_mean']
        rec['mse_p_value'] = comp['mse']['ttest']['p_value'] if comp['mse']['ttest'] else None
        rec['r2_pyreco'] = comp['r2']['pyreco_mean']
        rec['r2_lstm'] = comp['r2']['lstm_mean']
        rec['time_pyreco'] = comp['train_time']['pyreco_mean']
        rec['time_lstm'] = comp['train_time']['lstm_mean']
        comparison_records.append(rec)

    pd.DataFrame(comparison_records).to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
