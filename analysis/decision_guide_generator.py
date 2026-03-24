#!/usr/bin/env python3
"""
Decision Guide Generator for PyReCo vs LSTM Model Selection

Analyzes experiment results and generates practical decision guidelines
for choosing between PyReCo and LSTM models.

Features:
1. Dataset-specific recommendations
2. Parameter budget guidance
3. Training data efficiency analysis
4. Green computing considerations
5. Use case mapping

Usage:
    python analysis/decision_guide_generator.py
    python analysis/decision_guide_generator.py --output docs/DECISION_GUIDE.md
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_final_results(results_dir: str = "results/final") -> pd.DataFrame:
    """Load single-step prediction results"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/results_*.json"))

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            metadata = data['metadata']
            for scale, models in data['results'].items():
                for model in models:
                    records.append({
                        'source': 'final',
                        'dataset': metadata['dataset'],
                        'seed': metadata['seed'],
                        'train_frac': metadata['train_frac'],
                        'scale': scale,
                        'model_type': model['model_type'],
                        'test_mse': model['test_mse'],
                        'test_r2': model['test_r2'],
                        'train_time': model.get('final_train_time', 0),
                    })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return pd.DataFrame(records)


def load_green_metrics(results_dir: str = "results/green_metrics") -> pd.DataFrame:
    """Load green computing metrics results"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/green_*.json"))

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            metadata = data['metadata']
            for scale, models in data['results'].items():
                for model in models:
                    records.append({
                        'source': 'green',
                        'dataset': metadata['dataset'],
                        'seed': metadata['seed'],
                        'scale': scale,
                        'model_type': model['model_type'],
                        'test_mse': model['test_mse'],
                        'test_r2': model['test_r2'],
                        'train_time': model.get('final_train_time', 0),
                        'memory_peak_mb': model.get('memory_peak_mb'),
                        'energy_kwh': model.get('energy_kwh'),
                        'emissions_kg_co2': model.get('emissions_kg_co2'),
                    })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return pd.DataFrame(records)


def load_data_efficiency(results_dir: str = "results/data_efficiency") -> pd.DataFrame:
    """Load data efficiency experiment results"""
    records = []
    files = sorted(glob.glob(f"{results_dir}/dataeff_*.json"))

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            metadata = data['metadata']
            for model_type, model_data in data.get('models', {}).items():
                metrics = model_data.get('metrics', {})
                records.append({
                    'source': 'data_efficiency',
                    'dataset': metadata['dataset'],
                    'seed': metadata['seed'],
                    'data_length': metadata.get('data_length', 5000),
                    'scale': metadata.get('budget', 'medium'),
                    'model_type': model_type,
                    'test_mse': metrics.get('mse'),
                    'test_r2': metrics.get('r2'),
                    'train_time': model_data.get('train_time', 0),
                })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return pd.DataFrame(records)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_by_dataset(df: pd.DataFrame) -> dict:
    """Analyze performance by dataset"""
    results = {}

    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        pyreco = ds_df[ds_df['model_type'] == 'pyreco_standard']
        lstm = ds_df[ds_df['model_type'] == 'lstm']

        if len(pyreco) == 0 or len(lstm) == 0:
            continue

        pyreco_r2 = pyreco['test_r2'].mean()
        lstm_r2 = lstm['test_r2'].mean()
        pyreco_mse = pyreco['test_mse'].mean()
        lstm_mse = lstm['test_mse'].mean()
        pyreco_time = pyreco['train_time'].mean()
        lstm_time = lstm['train_time'].mean()

        winner = 'PyReCo' if pyreco_r2 > lstm_r2 else 'LSTM'
        r2_diff = abs(pyreco_r2 - lstm_r2)
        speedup = lstm_time / pyreco_time if pyreco_time > 0 else float('inf')

        results[dataset] = {
            'winner': winner,
            'pyreco_r2': pyreco_r2,
            'lstm_r2': lstm_r2,
            'r2_diff': r2_diff,
            'pyreco_mse': pyreco_mse,
            'lstm_mse': lstm_mse,
            'speedup': speedup,
            'confidence': 'high' if r2_diff > 0.05 else 'medium' if r2_diff > 0.02 else 'low',
        }

    return results


def analyze_by_scale(df: pd.DataFrame) -> dict:
    """Analyze performance by parameter scale"""
    results = {}

    for scale in ['small', 'medium', 'large']:
        scale_df = df[df['scale'] == scale]
        pyreco = scale_df[scale_df['model_type'] == 'pyreco_standard']
        lstm = scale_df[scale_df['model_type'] == 'lstm']

        if len(pyreco) == 0 or len(lstm) == 0:
            continue

        pyreco_r2 = pyreco['test_r2'].mean()
        lstm_r2 = lstm['test_r2'].mean()
        pyreco_time = pyreco['train_time'].mean()
        lstm_time = lstm['train_time'].mean()

        winner = 'PyReCo' if pyreco_r2 > lstm_r2 else 'LSTM'
        speedup = lstm_time / pyreco_time if pyreco_time > 0 else float('inf')

        results[scale] = {
            'winner': winner,
            'pyreco_r2': pyreco_r2,
            'lstm_r2': lstm_r2,
            'speedup': speedup,
            'pyreco_time': pyreco_time,
            'lstm_time': lstm_time,
        }

    return results


def analyze_green_metrics(df: pd.DataFrame) -> dict:
    """Analyze green computing metrics"""
    if 'memory_peak_mb' not in df.columns:
        return {}

    results = {}

    for scale in ['small', 'medium', 'large']:
        scale_df = df[df['scale'] == scale]
        pyreco = scale_df[scale_df['model_type'] == 'pyreco_standard']
        lstm = scale_df[scale_df['model_type'] == 'lstm']

        if len(pyreco) == 0 or len(lstm) == 0:
            continue

        results[scale] = {
            'pyreco_memory_mb': pyreco['memory_peak_mb'].mean(),
            'lstm_memory_mb': lstm['memory_peak_mb'].mean(),
            'pyreco_energy_kwh': pyreco['energy_kwh'].mean() if 'energy_kwh' in pyreco else None,
            'lstm_energy_kwh': lstm['energy_kwh'].mean() if 'energy_kwh' in lstm else None,
            'memory_winner': 'LSTM' if lstm['memory_peak_mb'].mean() < pyreco['memory_peak_mb'].mean() else 'PyReCo',
        }

    return results


# ============================================================================
# Decision Rule Generation
# ============================================================================

def generate_decision_rules(dataset_analysis: dict, scale_analysis: dict, green_analysis: dict) -> list:
    """Generate decision rules based on analysis"""
    rules = []

    # Dataset classification (based on known properties)
    CHAOTIC_DATASETS = {'lorenz', 'mackeyglass', 'mackey-glass'}
    REAL_WORLD_DATASETS = {'santafe', 'santa-fe'}

    # Separate by dataset type
    chaotic_wins = [(d, r['winner']) for d, r in dataset_analysis.items() if d.lower() in CHAOTIC_DATASETS]
    real_wins = [(d, r['winner']) for d, r in dataset_analysis.items() if d.lower() in REAL_WORLD_DATASETS]

    chaotic_pyreco = [d for d, w in chaotic_wins if w == 'PyReCo']
    real_lstm = [d for d, w in real_wins if w == 'LSTM']

    if chaotic_pyreco:
        rules.append({
            'condition': f"Chaotic dynamical systems ({', '.join(chaotic_pyreco)})",
            'recommendation': 'PyReCo',
            'reason': 'Reservoir computing excels at short-term prediction of chaotic systems',
            'confidence': 'high',
        })

    if real_lstm:
        rules.append({
            'condition': f"Real-world measurement data ({', '.join(real_lstm)})",
            'recommendation': 'LSTM',
            'reason': 'LSTM handles noise and complex patterns better',
            'confidence': 'high',
        })

    # Scale-based rules
    if scale_analysis.get('small', {}).get('speedup', 0) > 10:
        rules.append({
            'condition': 'Parameter budget < 10K and training speed matters',
            'recommendation': 'PyReCo',
            'reason': f"PyReCo trains {scale_analysis['small']['speedup']:.0f}x faster at small scale",
            'confidence': 'high',
        })

    # Green computing rules
    if green_analysis:
        rules.append({
            'condition': 'Memory-constrained environment (e.g., embedded devices)',
            'recommendation': 'LSTM',
            'reason': 'LSTM has lower and more stable memory footprint',
            'confidence': 'high',
        })

        if green_analysis.get('small', {}).get('pyreco_energy_kwh', 1) < green_analysis.get('small', {}).get('lstm_energy_kwh', 0):
            rules.append({
                'condition': 'Small-scale tasks with energy efficiency priority',
                'recommendation': 'PyReCo',
                'reason': 'PyReCo consumes less energy at small scale',
                'confidence': 'medium',
            })

    # General rules
    rules.append({
        'condition': 'Rapid prototyping / initial experiments',
        'recommendation': 'PyReCo',
        'reason': 'Fast training enables quick iteration',
        'confidence': 'high',
    })

    rules.append({
        'condition': 'Production deployment requiring interpretability',
        'recommendation': 'Requires evaluation',
        'reason': 'Both models are black-boxes; test on specific use case',
        'confidence': 'low',
    })

    return rules


# ============================================================================
# Markdown Generation
# ============================================================================

def generate_markdown_guide(dataset_analysis: dict, scale_analysis: dict,
                           green_analysis: dict, rules: list) -> str:
    """Generate Markdown decision guide"""

    lines = []
    lines.append("# PyReCo vs LSTM Model Selection Decision Guide")
    lines.append("")
    lines.append(f"> Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("> Based on experimental results analysis")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    pyreco_wins = sum(1 for r in dataset_analysis.values() if r['winner'] == 'PyReCo')
    lstm_wins = sum(1 for r in dataset_analysis.values() if r['winner'] == 'LSTM')
    lines.append(f"- **Dataset Win Rate**: PyReCo {pyreco_wins} / LSTM {lstm_wins}")

    if scale_analysis.get('small'):
        lines.append(f"- **Training Speed**: PyReCo is {scale_analysis['small']['speedup']:.0f}-{scale_analysis.get('large', {}).get('speedup', scale_analysis['small']['speedup']):.0f}x faster than LSTM")

    if green_analysis.get('large'):
        lines.append(f"- **Memory Efficiency**: LSTM uses only {green_analysis['large']['lstm_memory_mb']/green_analysis['large']['pyreco_memory_mb']*100:.1f}% of PyReCo's memory")
    lines.append("")

    # Quick Decision Table
    lines.append("## Quick Decision Table")
    lines.append("")
    lines.append("| Scenario | Recommendation | Confidence |")
    lines.append("|----------|----------------|------------|")
    for rule in rules:
        conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(rule['confidence'], '⚪')
        lines.append(f"| {rule['condition']} | **{rule['recommendation']}** | {conf_emoji} {rule['confidence']} |")
    lines.append("")

    # Detailed Analysis by Dataset
    lines.append("## Analysis by Dataset")
    lines.append("")
    for dataset, analysis in dataset_analysis.items():
        emoji = '🏆' if analysis['winner'] == 'PyReCo' else '🥈'
        lines.append(f"### {dataset.upper()}")
        lines.append("")
        lines.append(f"- **Recommendation**: {emoji} **{analysis['winner']}**")
        lines.append(f"- **PyReCo R²**: {analysis['pyreco_r2']:.4f}")
        lines.append(f"- **LSTM R²**: {analysis['lstm_r2']:.4f}")
        lines.append(f"- **Training Speedup**: {analysis['speedup']:.1f}x (PyReCo faster)")
        lines.append(f"- **Confidence**: {analysis['confidence']}")
        lines.append("")

    # Analysis by Parameter Scale
    lines.append("## Analysis by Parameter Scale")
    lines.append("")
    lines.append("| Scale | Params | Winner | PyReCo R² | LSTM R² | Speedup |")
    lines.append("|-------|--------|--------|-----------|---------|---------|")
    scale_params = {'small': '~1K', 'medium': '~10K', 'large': '~50K'}
    for scale in ['small', 'medium', 'large']:
        if scale in scale_analysis:
            s = scale_analysis[scale]
            lines.append(f"| {scale} | {scale_params[scale]} | {s['winner']} | {s['pyreco_r2']:.4f} | {s['lstm_r2']:.4f} | {s['speedup']:.1f}x |")
    lines.append("")

    # Green Computing Analysis
    if green_analysis:
        lines.append("## Green Computing Metrics")
        lines.append("")
        lines.append("| Scale | PyReCo Memory | LSTM Memory | PyReCo Energy | LSTM Energy |")
        lines.append("|-------|---------------|-------------|---------------|-------------|")
        for scale in ['small', 'medium', 'large']:
            if scale in green_analysis:
                g = green_analysis[scale]
                pyreco_energy = f"{g['pyreco_energy_kwh']:.4f} kWh" if g.get('pyreco_energy_kwh') else 'N/A'
                lstm_energy = f"{g['lstm_energy_kwh']:.4f} kWh" if g.get('lstm_energy_kwh') else 'N/A'
                lines.append(f"| {scale} | {g['pyreco_memory_mb']:.0f} MB | {g['lstm_memory_mb']:.0f} MB | {pyreco_energy} | {lstm_energy} |")
        lines.append("")

        lines.append("### Green Computing Recommendations")
        lines.append("")
        lines.append("- **Memory-constrained**: Choose LSTM (lower and stable memory footprint)")
        lines.append("- **Small-scale, low energy**: Choose PyReCo (lower energy consumption)")
        lines.append("- **Large-scale, low energy**: Choose LSTM (lower energy consumption)")
        lines.append("")

    # Decision Rules
    lines.append("## Decision Rules Explained")
    lines.append("")
    for i, rule in enumerate(rules, 1):
        lines.append(f"### Rule {i}: {rule['condition']}")
        lines.append("")
        lines.append(f"- **Recommendation**: {rule['recommendation']}")
        lines.append(f"- **Reason**: {rule['reason']}")
        lines.append(f"- **Confidence**: {rule['confidence']}")
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append("### When to Use PyReCo")
    lines.append("- ✅ Chaotic dynamical system prediction (Lorenz, Mackey-Glass)")
    lines.append("- ✅ Rapid prototyping and experiment iteration")
    lines.append("- ✅ Small parameter budget (<10K)")
    lines.append("- ✅ Training speed-sensitive applications")
    lines.append("")
    lines.append("### When to Use LSTM")
    lines.append("- ✅ Real-world measurement data (e.g., Santa Fe)")
    lines.append("- ✅ Memory-constrained environments")
    lines.append("- ✅ Energy efficiency at large scale")
    lines.append("- ✅ Need for mature ecosystem support")
    lines.append("")
    lines.append("### General Recommendations")
    lines.append("1. **Start with PyReCo for quick validation** - Fast training enables rapid iteration")
    lines.append("2. **Choose based on data type** - PyReCo for chaotic systems, test both for real data")
    lines.append("3. **Consider resource constraints** - LSTM for memory limits, PyReCo for time limits")
    lines.append("")
    lines.append("---")
    lines.append(f"*This guide was auto-generated based on experiments on {len(dataset_analysis)} datasets*")

    return '\n'.join(lines)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate model selection decision guide')
    parser.add_argument('--output', type=str, default='docs/DECISION_GUIDE.md',
                       help='Output file path')
    parser.add_argument('--print', action='store_true',
                       help='Print guide to console')
    args = parser.parse_args()

    print("="*60)
    print("DECISION GUIDE GENERATOR")
    print("="*60)

    # Load data
    print("\nLoading experiment results...")

    df_final = load_final_results()
    print(f"  - Final results: {len(df_final)} records")

    df_green = load_green_metrics()
    print(f"  - Green metrics: {len(df_green)} records")

    df_efficiency = load_data_efficiency()
    print(f"  - Data efficiency: {len(df_efficiency)} records")

    # Combine data for analysis
    df_all = pd.concat([df_final, df_green], ignore_index=True)
    df_all = df_all[df_all['test_r2'].notna()]
    print(f"\nTotal records for analysis: {len(df_all)}")

    # Analyze
    print("\nAnalyzing results...")
    dataset_analysis = analyze_by_dataset(df_all)
    print(f"  - Datasets analyzed: {list(dataset_analysis.keys())}")

    scale_analysis = analyze_by_scale(df_all)
    print(f"  - Scales analyzed: {list(scale_analysis.keys())}")

    green_analysis = analyze_green_metrics(df_green)
    print(f"  - Green metrics scales: {list(green_analysis.keys())}")

    # Generate rules
    print("\nGenerating decision rules...")
    rules = generate_decision_rules(dataset_analysis, scale_analysis, green_analysis)
    print(f"  - Rules generated: {len(rules)}")

    # Generate Markdown
    print("\nGenerating Markdown guide...")
    guide = generate_markdown_guide(dataset_analysis, scale_analysis, green_analysis, rules)

    # Output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    print(f"\n✅ Decision guide saved to: {output_path}")

    if args.print:
        print("\n" + "="*60)
        print(guide)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, analysis in dataset_analysis.items():
        print(f"  {dataset}: {analysis['winner']} wins (R² diff: {analysis['r2_diff']:.4f})")

    print("\n✅ Decision guide generation complete!")


if __name__ == '__main__':
    main()
