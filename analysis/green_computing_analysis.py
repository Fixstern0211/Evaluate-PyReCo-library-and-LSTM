#!/usr/bin/env python3
"""
Green Computing Analysis for PyReCo vs LSTM

基于现有实验结果进行绿色计算分析，包括：
1. 训练效率分析 (Training Efficiency)
2. 参数效率分析 (Parameter Efficiency)
3. 调优成本分析 (Tuning Cost)
4. 综合绿色评分 (Green Score)

支持三种数据源：
- results/final/: 单步预测实验 (90个)
- results/multi_step/: 多步预测实验 (270个)
- results/data_efficiency/: 数据效率实验 (270个)

可用指标：
- train_time: 训练时间
- eval_time: 评估/推理时间
- test_r2, test_mse: 性能指标
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse


def load_final_results(results_dir: str = "results/final") -> list:
    """加载单步预测实验结果 (原格式)"""
    files = sorted(glob.glob(f"{results_dir}/results_*.json"))
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results


def load_multi_step_results(results_dir: str = "results/multi_step") -> list:
    """加载多步预测实验结果"""
    files = sorted(glob.glob(f"{results_dir}/multistep_*.json"))
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results


def load_data_efficiency_results(results_dir: str = "results/data_efficiency") -> list:
    """加载数据效率实验结果"""
    files = sorted(glob.glob(f"{results_dir}/dataeff_*.json"))
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results


def load_all_results(source: str = "all") -> dict:
    """加载所有实验结果

    Args:
        source: 'final', 'multi_step', 'data_efficiency', or 'all'

    Returns:
        dict with keys for each source type
    """
    all_results = {}

    if source in ['final', 'all']:
        all_results['final'] = load_final_results()

    if source in ['multi_step', 'all']:
        all_results['multi_step'] = load_multi_step_results()

    if source in ['data_efficiency', 'all']:
        all_results['data_efficiency'] = load_data_efficiency_results()

    return all_results


def extract_green_metrics_final(results: list) -> pd.DataFrame:
    """提取单步预测实验的绿色指标 (原格式)"""
    records = []

    for exp in results:
        metadata = exp['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']

        for scale, models in exp['results'].items():
            for model in models:
                record = {
                    'source': 'final',
                    'dataset': dataset,
                    'seed': seed,
                    'train_frac': train_frac,
                    'scale': scale,
                    'model_type': model['model_type'],
                    # 时间指标
                    'tune_time': model.get('tune_time', 0),
                    'train_time': model.get('final_train_time', 0),
                    # 参数指标
                    'trainable_params': model['param_info']['trainable'],
                    'total_params': model['param_info']['total'],
                    # 性能指标
                    'test_mse': model['test_mse'],
                    'test_r2': model['test_r2'],
                    # 推理时间
                    'eval_time': model.get('inference_time_total', None),
                }
                records.append(record)

    return pd.DataFrame(records)


def extract_green_metrics_multi_step(results: list) -> pd.DataFrame:
    """提取多步预测实验的绿色指标"""
    records = []

    for exp in results:
        metadata = exp['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']
        budget = metadata['budget']

        for model_type, model_data in exp['models'].items():
            # 使用 horizon=1 的结果作为基准性能
            horizon_results = model_data.get('horizon_results', {})
            h1_metrics = horizon_results.get('1', {})

            record = {
                'source': 'multi_step',
                'dataset': dataset,
                'seed': seed,
                'train_frac': train_frac,
                'scale': budget,
                'model_type': model_type,
                # 时间指标
                'tune_time': 0,  # 多步实验不含调优
                'train_time': model_data.get('train_time', 0),
                # 参数指标 (需要从config推断)
                'trainable_params': None,
                'total_params': None,
                # 性能指标 (horizon=1)
                'test_mse': h1_metrics.get('mse', None),
                'test_r2': h1_metrics.get('r2', None),
                # 评估时间
                'eval_time': model_data.get('eval_time', None),
            }
            records.append(record)

    return pd.DataFrame(records)


def extract_green_metrics_data_efficiency(results: list) -> pd.DataFrame:
    """提取数据效率实验的绿色指标"""
    records = []

    for exp in results:
        metadata = exp['metadata']
        dataset = metadata['dataset']
        seed = metadata['seed']
        train_frac = metadata['train_frac']
        budget = metadata['budget']
        data_length = metadata.get('data_length', 5000)

        for model_type, model_data in exp['models'].items():
            metrics = model_data.get('metrics', {})

            record = {
                'source': 'data_efficiency',
                'dataset': dataset,
                'seed': seed,
                'train_frac': train_frac,
                'scale': budget,
                'data_length': data_length,
                'model_type': model_type,
                # 时间指标
                'tune_time': 0,
                'train_time': model_data.get('train_time', 0),
                # 参数指标
                'trainable_params': None,
                'total_params': None,
                # 性能指标
                'test_mse': metrics.get('mse', None),
                'test_r2': metrics.get('r2', None),
                # 评估时间
                'eval_time': model_data.get('eval_time', None),
            }
            records.append(record)

    return pd.DataFrame(records)


def extract_green_metrics(all_results: dict) -> pd.DataFrame:
    """提取所有数据源的绿色指标"""
    dfs = []

    if 'final' in all_results and all_results['final']:
        df_final = extract_green_metrics_final(all_results['final'])
        dfs.append(df_final)
        print(f"  - Final experiments: {len(df_final)} records")

    if 'multi_step' in all_results and all_results['multi_step']:
        df_multi = extract_green_metrics_multi_step(all_results['multi_step'])
        dfs.append(df_multi)
        print(f"  - Multi-step experiments: {len(df_multi)} records")

    if 'data_efficiency' in all_results and all_results['data_efficiency']:
        df_dataeff = extract_green_metrics_data_efficiency(all_results['data_efficiency'])
        dfs.append(df_dataeff)
        print(f"  - Data efficiency experiments: {len(df_dataeff)} records")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def compute_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算效率指标"""
    df = df.copy()

    # 过滤有效数据
    df = df[df['test_r2'].notna() & df['train_time'].notna()]

    # 1. 训练效率: R²/训练时间 (越高越好)
    df['training_efficiency'] = df['test_r2'] / (df['train_time'] + 1e-6)

    # 2. 参数效率: R²/可训练参数数(千) (越高越好，仅对有参数信息的数据)
    df['param_efficiency'] = df.apply(
        lambda row: row['test_r2'] / (row['trainable_params'] / 1000 + 1e-6)
        if pd.notna(row['trainable_params']) else None,
        axis=1
    )

    # 3. 调优效率: R²/调优总时间 (越高越好)
    df['tuning_efficiency'] = df.apply(
        lambda row: row['test_r2'] / (row['tune_time'] + 1e-6)
        if row['tune_time'] > 0 else None,
        axis=1
    )

    # 4. 推理效率: 1/评估时间 (越高越好)
    df['inference_efficiency'] = df.apply(
        lambda row: 1.0 / (row['eval_time'] + 1e-6)
        if pd.notna(row['eval_time']) and row['eval_time'] > 0 else None,
        axis=1
    )

    return df


def compare_models_green(df: pd.DataFrame) -> dict:
    """模型间绿色计算对比"""
    comparison = {}

    for scale in ['small', 'medium', 'large']:
        scale_df = df[df['scale'] == scale]

        pyreco = scale_df[scale_df['model_type'] == 'pyreco_standard']
        lstm = scale_df[scale_df['model_type'] == 'lstm']

        if len(pyreco) == 0 or len(lstm) == 0:
            continue

        comparison[scale] = {
            'pyreco': {
                'count': len(pyreco),
                'avg_train_time': pyreco['train_time'].mean(),
                'avg_tune_time': pyreco['tune_time'].mean() if pyreco['tune_time'].notna().any() else 0,
                'avg_r2': pyreco['test_r2'].mean(),
                'avg_eval_time': pyreco['eval_time'].mean() if pyreco['eval_time'].notna().any() else None,
                'training_efficiency': pyreco['training_efficiency'].mean(),
            },
            'lstm': {
                'count': len(lstm),
                'avg_train_time': lstm['train_time'].mean(),
                'avg_tune_time': lstm['tune_time'].mean() if lstm['tune_time'].notna().any() else 0,
                'avg_r2': lstm['test_r2'].mean(),
                'avg_eval_time': lstm['eval_time'].mean() if lstm['eval_time'].notna().any() else None,
                'training_efficiency': lstm['training_efficiency'].mean(),
            }
        }

        # 计算比率
        p = comparison[scale]['pyreco']
        l = comparison[scale]['lstm']
        comparison[scale]['ratios'] = {
            'train_time_ratio': l['avg_train_time'] / (p['avg_train_time'] + 1e-6),
            'training_efficiency_ratio': p['training_efficiency'] / (l['training_efficiency'] + 1e-6),
            'r2_diff': p['avg_r2'] - l['avg_r2'],
        }

    return comparison


def generate_report(df: pd.DataFrame, comparison: dict) -> str:
    """生成绿色计算分析报告"""
    report = []
    report.append("=" * 80)
    report.append("GREEN COMPUTING ANALYSIS REPORT")
    report.append("PyReCo vs LSTM: Efficiency Comparison")
    report.append("=" * 80)
    report.append("")

    # 数据源统计
    report.append("## Data Sources")
    report.append("-" * 40)
    for source in df['source'].unique():
        count = len(df[df['source'] == source])
        report.append(f"  {source}: {count} records")
    report.append(f"  Total: {len(df)} records")
    report.append("")

    # 1. 训练时间对比
    report.append("## 1. Training Time Comparison")
    report.append("-" * 60)
    report.append(f"{'Scale':<10} {'PyReCo(s)':<15} {'LSTM(s)':<15} {'Speedup':<10} {'N':<10}")
    report.append("-" * 60)

    for scale in ['small', 'medium', 'large']:
        if scale not in comparison:
            continue
        p_time = comparison[scale]['pyreco']['avg_train_time']
        l_time = comparison[scale]['lstm']['avg_train_time']
        speedup = l_time / (p_time + 1e-6)
        n = comparison[scale]['pyreco']['count']
        report.append(f"{scale:<10} {p_time:<15.2f} {l_time:<15.2f} {speedup:<10.1f}x {n:<10}")
    report.append("")

    # 2. 性能对比 (R²)
    report.append("## 2. Performance Comparison (R²)")
    report.append("-" * 60)
    report.append(f"{'Scale':<10} {'PyReCo R²':<15} {'LSTM R²':<15} {'Diff':<10} {'Winner':<10}")
    report.append("-" * 60)

    for scale in ['small', 'medium', 'large']:
        if scale not in comparison:
            continue
        p_r2 = comparison[scale]['pyreco']['avg_r2']
        l_r2 = comparison[scale]['lstm']['avg_r2']
        diff = p_r2 - l_r2
        winner = "PyReCo" if p_r2 > l_r2 else "LSTM"
        report.append(f"{scale:<10} {p_r2:<15.4f} {l_r2:<15.4f} {diff:+<10.4f} {winner:<10}")
    report.append("")

    # 3. 训练效率对比
    report.append("## 3. Training Efficiency (R² per second)")
    report.append("-" * 60)
    report.append(f"{'Scale':<10} {'PyReCo':<15} {'LSTM':<15} {'Ratio':<10} {'Winner':<10}")
    report.append("-" * 60)

    for scale in ['small', 'medium', 'large']:
        if scale not in comparison:
            continue
        p_eff = comparison[scale]['pyreco']['training_efficiency']
        l_eff = comparison[scale]['lstm']['training_efficiency']
        ratio = p_eff / (l_eff + 1e-6)
        winner = "PyReCo" if p_eff > l_eff else "LSTM"
        report.append(f"{scale:<10} {p_eff:<15.4f} {l_eff:<15.4f} {ratio:<10.1f}x {winner:<10}")
    report.append("")

    # 4. 按数据集分析
    report.append("## 4. Performance by Dataset")
    report.append("-" * 60)

    for dataset in sorted(df['dataset'].unique()):
        ds_df = df[df['dataset'] == dataset]
        report.append(f"\n### {dataset.upper()}")

        pyreco_df = ds_df[ds_df['model_type'] == 'pyreco_standard']
        lstm_df = ds_df[ds_df['model_type'] == 'lstm']

        if len(pyreco_df) > 0 and len(lstm_df) > 0:
            pyreco_r2 = pyreco_df['test_r2'].mean()
            lstm_r2 = lstm_df['test_r2'].mean()
            pyreco_time = pyreco_df['train_time'].mean()
            lstm_time = lstm_df['train_time'].mean()

            report.append(f"  PyReCo: R²={pyreco_r2:.4f}, Train={pyreco_time:.2f}s (n={len(pyreco_df)})")
            report.append(f"  LSTM:   R²={lstm_r2:.4f}, Train={lstm_time:.2f}s (n={len(lstm_df)})")
            report.append(f"  Winner: {'PyReCo' if pyreco_r2 > lstm_r2 else 'LSTM'}")
    report.append("")

    # 5. 总结
    report.append("=" * 80)
    report.append("## SUMMARY")
    report.append("=" * 80)
    report.append("")

    # 计算总体统计
    pyreco_all = df[df['model_type'] == 'pyreco_standard']
    lstm_all = df[df['model_type'] == 'lstm']

    if len(pyreco_all) > 0 and len(lstm_all) > 0:
        avg_speedup = lstm_all['train_time'].mean() / (pyreco_all['train_time'].mean() + 1e-6)
        avg_r2_pyreco = pyreco_all['test_r2'].mean()
        avg_r2_lstm = lstm_all['test_r2'].mean()

        report.append("Key Findings:")
        report.append(f"1. Training Speed: PyReCo is {avg_speedup:.1f}x faster than LSTM on average")
        report.append(f"2. Average R²: PyReCo={avg_r2_pyreco:.4f}, LSTM={avg_r2_lstm:.4f}")
        report.append(f"3. Overall Winner: {'PyReCo' if avg_r2_pyreco > avg_r2_lstm else 'LSTM'}")

        # 按数据集的胜者统计
        dataset_winners = {}
        for dataset in df['dataset'].unique():
            ds_df = df[df['dataset'] == dataset]
            p_r2 = ds_df[ds_df['model_type'] == 'pyreco_standard']['test_r2'].mean()
            l_r2 = ds_df[ds_df['model_type'] == 'lstm']['test_r2'].mean()
            dataset_winners[dataset] = 'PyReCo' if p_r2 > l_r2 else 'LSTM'

        report.append(f"\n4. Winners by Dataset:")
        for ds, winner in dataset_winners.items():
            report.append(f"   - {ds}: {winner}")

    report.append("")
    report.append(f"Analysis based on {len(df)} experiment records.")
    report.append("")

    return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Green Computing Analysis for PyReCo vs LSTM')
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'final', 'multi_step', 'data_efficiency'],
                        help='Data source to analyze (default: all)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                        help='Output directory for reports (default: analysis)')
    args = parser.parse_args()

    print("=" * 60)
    print("GREEN COMPUTING ANALYSIS")
    print("=" * 60)

    print(f"\nLoading experiment results (source: {args.source})...")
    all_results = load_all_results(args.source)

    total_exps = sum(len(v) for v in all_results.values())
    print(f"Loaded experiments:")
    for source, results in all_results.items():
        print(f"  - {source}: {len(results)} files")

    print("\nExtracting green metrics...")
    df = extract_green_metrics(all_results)

    if len(df) == 0:
        print("ERROR: No valid data found!")
        return

    print(f"\nTotal records: {len(df)}")

    print("\nComputing efficiency metrics...")
    df = compute_efficiency_metrics(df)

    print("Comparing models...")
    comparison = compare_models_green(df)

    print("\n" + "=" * 60)
    report = generate_report(df, comparison)
    print(report)

    # 保存报告
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "green_computing_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # 保存详细数据
    csv_path = output_dir / "green_metrics_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed metrics saved to: {csv_path}")

    # 保存JSON摘要
    pyreco_df = df[df['model_type'] == 'pyreco_standard']
    lstm_df = df[df['model_type'] == 'lstm']

    summary = {
        'sources': list(all_results.keys()),
        'total_records': len(df),
        'comparison': comparison,
        'overall': {
            'pyreco_count': len(pyreco_df),
            'lstm_count': len(lstm_df),
            'pyreco_avg_train_time': float(pyreco_df['train_time'].mean()) if len(pyreco_df) > 0 else None,
            'lstm_avg_train_time': float(lstm_df['train_time'].mean()) if len(lstm_df) > 0 else None,
            'pyreco_avg_r2': float(pyreco_df['test_r2'].mean()) if len(pyreco_df) > 0 else None,
            'lstm_avg_r2': float(lstm_df['test_r2'].mean()) if len(lstm_df) > 0 else None,
        }
    }

    json_path = output_dir / "green_computing_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
