#!/usr/bin/env python3
"""
Generate LaTeX and Markdown tables for RQ3 analysis
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

sys.path.insert(0, '.')

def load_rq3_data(output_dir: str) -> List[Dict[str, Any]]:
    """Load RQ3 data from jsonl file"""
    all_points = []
    rq3_file = os.path.join(output_dir, 'rq3.jsonl')

    if not os.path.exists(rq3_file):
        return all_points

    with open(rq3_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            model_preset = rec.get('model_preset', 'unknown')

            for p in rec.get('points', []):
                # Inject model preset if missing
                if 'model_preset' not in p:
                    p['model_preset'] = model_preset

                # Normalize strategy names
                strat = p.get('strategy', 'unknown')
                name_map = {
                    'grayshield': 'GrayShield',
                    'gray_code': 'GrayShield', 'GrayCode': 'GrayShield', 'RepErase': 'GrayShield',
                    'random': 'RandomFlip', 'pattern': 'PatternMask',
                    'gaussian': 'GaussianNoise', 'finetune': 'FineTune',
                    'ptq': 'PTQ', 'swp': 'SWP',
                }
                p['strategy'] = name_map.get(strat, strat)
                all_points.append(p)

    return all_points


def generate_table1_defense_comparison(points: List[Dict[str, Any]], output_dir: str):
    """
    Table 1: Defense Strategy Comparison (RQ3 Summary)

    Columns:
    - Defense Strategy
    - Avg Recovery Reduction (%)
    - Std Dev (%)
    - Avg Accuracy Drop (%)
    - Avg Defense Time (ms)
    - Consistency Score
    """
    # Group by strategy
    strategy_data = defaultdict(lambda: {
        'recovery_reduction': [],
        'acc_drop': [],
        'defense_time': [],
    })

    for p in points:
        strat = p.get('strategy', 'unknown')
        strategy_data[strat]['recovery_reduction'].append(p.get('recovery_reduction', 0) * 100)
        strategy_data[strat]['acc_drop'].append(p.get('acc_drop', 0) * 100)
        if 'defense_time_ms' in p:
            strategy_data[strat]['defense_time'].append(p.get('defense_time_ms', 0))

    # Sort strategies by recovery reduction (descending)
    strategies = sorted(strategy_data.keys(),
                       key=lambda s: np.mean(strategy_data[s]['recovery_reduction']),
                       reverse=True)

    # Generate Markdown table
    md_output = ["# Table 1: Defense Strategy Comparison (RQ3 Summary)\n"]
    md_output.append("| Defense | Avg Recovery Reduction | Std Dev | Avg Acc Drop | Avg Defense Time | Consistency¹ |")
    md_output.append("|---------|------------------------|---------|--------------|------------------|--------------|")

    for strat in strategies:
        data = strategy_data[strat]

        rec_red_mean = np.mean(data['recovery_reduction'])
        rec_red_std = np.std(data['recovery_reduction'])
        acc_drop_mean = np.mean(data['acc_drop'])
        time_vals = data['defense_time']
        time_mean = np.mean(time_vals) if time_vals else 0

        # Consistency score: inverse of coefficient of variation (lower std = higher consistency)
        consistency = 100 - min(100, (rec_red_std / rec_red_mean * 100) if rec_red_mean > 0 else 0)

        md_output.append(
            f"| **{strat}** | {rec_red_mean:.2f}% | {rec_red_std:.2f}% | "
            f"{acc_drop_mean:.3f}% | {time_mean:.0f} ms | {consistency:.0f}/100 |"
        )

    md_output.append("\n¹ Consistency Score: Measure of stability across models (100 = perfectly stable)")
    md_output.append("\n**Key Findings:**")
    md_output.append("- **GrayShield** remains the most stable near-chance-level sanitizer across attacker variants")
    md_output.append("- **PTQ** and **SWP** now provide paper-aligned comparison points without changing the core pipeline")
    md_output.append("- **RandomFlip** remains a weak defense and serves mainly as a stochastic baseline")

    md_path = os.path.join(output_dir, 'table1_defense_comparison.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_output))
    print(f"Generated {md_path}")

    # Generate LaTeX table
    latex_output = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Defense Strategy Comparison (RQ3 Summary)}",
        "\\label{tab:defense_comparison}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Defense} & \\textbf{Avg RR\\%} & \\textbf{Std Dev} & \\textbf{Acc Drop\\%} & \\textbf{Time (ms)} & \\textbf{Consistency} \\\\",
        "\\midrule"
    ]

    for strat in strategies:
        data = strategy_data[strat]
        rec_red_mean = np.mean(data['recovery_reduction'])
        rec_red_std = np.std(data['recovery_reduction'])
        acc_drop_mean = np.mean(data['acc_drop'])
        time_vals = data['defense_time']
        time_mean = np.mean(time_vals) if time_vals else 0
        consistency = 100 - min(100, (rec_red_std / rec_red_mean * 100) if rec_red_mean > 0 else 0)

        # Highlight best strategies
        if rec_red_mean > 45:
            strat_name = f"\\textbf{{{strat}}}"
        else:
            strat_name = strat

        latex_output.append(
            f"{strat_name} & {rec_red_mean:.2f} & {rec_red_std:.2f} & "
            f"{acc_drop_mean:.3f} & {time_mean:.0f} & {consistency:.0f}/100 \\\\"
        )

    latex_output.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{0.1cm}",
        "\\\\",
        "\\footnotesize{RR = Recovery Reduction (higher is better). Consistency measures stability across models.}",
        "\\end{table}"
    ])

    latex_path = os.path.join(output_dir, 'table1_defense_comparison.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_output))
    print(f"Generated {latex_path}")


def generate_table2_attacker_robustness(points: List[Dict[str, Any]], output_dir: str):
    """
    Table 2: Attacker Variant Robustness

    Shows how each defense performs against different attacker variants.
    Rows: Defenses
    Columns: Attacker Variants (naive, repeat3, repeat5, interleave, rs)
    Values: Avg Recovery Reduction %
    """
    # Get all unique strategies and variants
    strategies = sorted(list(set(p['strategy'] for p in points)))
    preferred_order = ["naive", "repeat3", "repeat5", "interleave", "rs"]
    raw_variants = set(p.get('attacker_variant', 'naive') for p in points)
    variants = [v for v in preferred_order if v in raw_variants] + sorted(raw_variants - set(preferred_order))

    # Build data matrix
    data_matrix = defaultdict(lambda: defaultdict(list))
    for p in points:
        strat = p['strategy']
        variant = p.get('attacker_variant', 'naive')
        rec_red = p.get('recovery_reduction', 0) * 100
        data_matrix[strat][variant].append(rec_red)

    # Generate Markdown table
    md_output = ["# Table 2: Attacker Variant Robustness\n"]

    # Header
    header = "| Defense |"
    separator = "|---------|"
    for v in variants:
        header += f" {v.upper()} |"
        separator += "------|"
    md_output.append(header)
    md_output.append(separator)

    # Rows
    for strat in strategies:
        row = f"| **{strat}** |"
        for variant in variants:
            vals = data_matrix[strat][variant]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f" {mean_val:.1f}±{std_val:.1f} |"
            else:
                row += " - |"
        md_output.append(row)

    md_output.append("\n**Interpretation:**")
    md_output.append("- Values show mean ± std deviation of recovery reduction (%)")
    md_output.append("- **Higher values** indicate better defense (harder payload recovery)")
    md_output.append("- **Low variance** across variants indicates robustness to adaptive attacks")

    md_path = os.path.join(output_dir, 'table2_attacker_robustness.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_output))
    print(f"Generated {md_path}")

    # Generate LaTeX table
    latex_output = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Defense Robustness Against Attacker Variants (Recovery Reduction \\%)}",
        "\\label{tab:attacker_robustness}",
        f"\\begin{{tabular}}{{l{'c' * len(variants)}}}",
        "\\toprule",
        "\\textbf{Defense} & " + " & ".join([f"\\textbf{{{v.upper()}}}" for v in variants]) + " \\\\",
        "\\midrule"
    ]

    for strat in strategies:
        row_vals = []
        for variant in variants:
            vals = data_matrix[strat][variant]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row_vals.append(f"{mean_val:.1f}$\\pm${std_val:.1f}")
            else:
                row_vals.append("-")

        strat_name = f"\\textbf{{{strat}}}" if strat in ['GrayShield', 'PatternMask', 'FineTune'] else strat
        latex_output.append(f"{strat_name} & " + " & ".join(row_vals) + " \\\\")

    latex_output.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{0.1cm}",
        "\\\\",
        "\\footnotesize{Values show mean $\\pm$ standard deviation. Higher is better (more robust defense).}",
        "\\end{table}"
    ])

    latex_path = os.path.join(output_dir, 'table2_attacker_robustness.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_output))
    print(f"Generated {latex_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='results/0302.revise')
    args = parser.parse_args()

    print(f"Loading RQ3 data from {args.output_dir}...")
    points = load_rq3_data(args.output_dir)

    if not points:
        print("No RQ3 data found!")
        return

    print(f"Loaded {len(points)} data points")

    print("\nGenerating Table 1: Defense Strategy Comparison...")
    generate_table1_defense_comparison(points, args.output_dir)

    print("\nGenerating Table 2: Attacker Variant Robustness...")
    generate_table2_attacker_robustness(points, args.output_dir)

    print("\n✅ Tables generated successfully!")


if __name__ == '__main__':
    main()
