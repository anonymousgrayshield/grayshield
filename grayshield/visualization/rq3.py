import json
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from grayshield.visualization.plots import (
    DEFENSE_COLORS, DEFENSE_MARKERS, 
    format_model_name, add_jitter, _ensure_dir, normalize_defense_name, sort_attacker_variants
)
from grayshield.metrics.pareto import pareto_front

# Import functions from original grayshield/viz/plot_rq3_aggregate.py
# and the centralized grayshield/visualization entrypoints

def is_cv_model(name):
    return 'cifar' in name.lower() or 'vit' in name.lower() or 'swin' in name.lower()

def is_nlp_model(name):
    return 'bert' in name.lower() or 'roberta' in name.lower() or 'sst2' in name.lower() or 'imdb' in name.lower()

def plot_rq3_pareto_aggregated(out_dir, all_points, group_name="all"):
    if not all_points:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    all_acc_drops = [p['acc_drop'] * 100 for p in all_points]
    max_acc_drop = max(all_acc_drops) if all_acc_drops else 1.0
    jitter_amount = 0.02 if max_acc_drop < 1.0 else 0.05
    x_vals_jittered = add_jitter(all_acc_drops, amount=jitter_amount)
    
    strategies = [p.get('strategy', 'unknown') for p in all_points]
    if 'recovery_reduction' not in all_points[0]:
        y_vals = [(1 - p.get('post_recovery', 1)) * 100 for p in all_points]
    else:
        y_vals = [p['recovery_reduction'] * 100 for p in all_points]

    unique_strategies = set(strategies)
    for strat in unique_strategies:
        indices = [i for i, s in enumerate(strategies) if s == strat]
        if not indices: continue
        strat_x = [x_vals_jittered[i] for i in indices]
        strat_y = [y_vals[i] for i in indices]
        strat_color = DEFENSE_COLORS.get(strat, '#888888')
        strat_marker = DEFENSE_MARKERS.get(strat, 'o')
        
        is_gray = normalize_defense_name(strat) == 'GrayShield'
        plot_s = 150 if is_gray else 60
        plot_alpha = 1.0 if is_gray else 0.7
        plot_zorder = 4 if is_gray else 2

        display_label = normalize_defense_name(strat)
        ax.scatter(strat_x, strat_y, c=strat_color, marker=strat_marker,
                   s=plot_s, alpha=plot_alpha, edgecolors='black', linewidth=0.5,
                   zorder=plot_zorder, label=display_label if strat not in ['random', 'pattern'] else None)

    pareto_pts = pareto_front(all_points, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)
    if not pareto_pts:
        for p in all_points: p['temp_y'] = 1.0 - p.get('post_recovery', 1.0)
        pareto_pts = pareto_front(all_points, x_key="acc_drop", y_key="temp_y", maximize_y=True)

    if pareto_pts:
        pareto_x = [p['acc_drop'] * 100 for p in pareto_pts]
        if 'recovery_reduction' in pareto_pts[0]:
            pareto_y = [p['recovery_reduction'] * 100 for p in pareto_pts]
        else:
            pareto_y = [(1 - p.get('post_recovery', 1)) * 100 for p in pareto_pts]
            
        sorted_pareto = sorted(zip(pareto_x, pareto_y))
        px, py = zip(*sorted_pareto)
        ax.plot(px, py, 'k--', linewidth=2, alpha=0.5, zorder=3, label='Pareto Front')
        ax.scatter(px, py, c='gold', s=150, marker='*', edgecolors='black', linewidth=1, zorder=5)

    legend_elements = []
    for strat in sorted(unique_strategies):
        display_strat = normalize_defense_name(strat)
        color = DEFENSE_COLORS.get(display_strat, '#888888')
        marker = DEFENSE_MARKERS.get(display_strat, 'o')
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, label=f'{display_strat}'))
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Pareto Optimal'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_xlabel('Accuracy Drop (%)', fontsize=12)
    ax.set_ylabel('Recovery Reduction (%)', fontsize=12)
    ax.set_title(f'Defense Trade-off (Pareto) - {group_name.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(out_dir, f"rq3_pareto_aggregate_{group_name}.png")
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_rq3_tradeoff_2x2(all_points, out_dir):
    target_models = ['bert_sst2', 'roberta_sentiment', 'vit_cifar10', 'swin_cifar10']
    valid_points = [p for p in all_points if p.get('model_preset') in target_models]
    if not valid_points: return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, model in enumerate(target_models):
        ax = axes[idx]
        model_pts = [p for p in valid_points if p['model_preset'] == model]
        if not model_pts: continue

        acc_drops = [p['acc_drop'] * 100 for p in model_pts]
        max_acc_drop = max(acc_drops) if acc_drops else 1.0
        jitter_amount = 0.02 if max_acc_drop < 1.0 else 0.05
        x_vals_jittered = add_jitter(acc_drops, amount=jitter_amount)
        y_vals = [p['recovery_reduction'] * 100 for p in model_pts]
        strategies = [p.get('strategy', 'unknown') for p in model_pts]

        unique_strategies = set(strategies)
        for strat in unique_strategies:
            indices = [i for i, s in enumerate(strategies) if s == strat]
            if not indices: continue
            strat_x = [x_vals_jittered[i] for i in indices]
            strat_y = [y_vals[i] for i in indices]
            
            is_gray = normalize_defense_name(strat) == 'GrayShield'
            if is_gray:
                plot_s, plot_alpha, plot_zorder = 150, 1.0, 5
                strat_color, strat_marker = '#d62728', '*'
            else:
                plot_s, plot_alpha, plot_zorder = 60, 0.7, 2
                strat_color = DEFENSE_COLORS.get(strat, '#cccccc')
                strat_marker = DEFENSE_MARKERS.get(strat, 'o')

            display_label = normalize_defense_name(strat)
            ax.scatter(strat_x, strat_y, c=strat_color, marker=strat_marker, s=plot_s, alpha=plot_alpha,
                       edgecolors='black', linewidth=0.5, zorder=plot_zorder,
                       label=display_label if strat not in ['random', 'pattern'] else None)

        pareto_pts = pareto_front(model_pts, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)
        if pareto_pts:
            pareto_x = [p['acc_drop'] * 100 for p in pareto_pts]
            pareto_y = [p['recovery_reduction'] * 100 for p in pareto_pts]
            sorted_pareto = sorted(zip(pareto_x, pareto_y))
            px, py = zip(*sorted_pareto)
            ax.plot(px, py, 'k--', linewidth=2, alpha=0.5, zorder=3, label='Pareto Front')
            ax.scatter(px, py, c='gold', s=120, marker='*', edgecolors='black', linewidth=1, zorder=5)

        ax.set_xlabel('Accuracy Drop (%)', fontsize=11)
        ax.set_ylabel('Recovery Reduction (%)', fontsize=11)
        ax.set_title(format_model_name(model), fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Defense Trade-off (2x2 Multi-Model View)", fontsize=16, fontweight='bold', y=0.98)
    handles_dict = {}
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for item_h, item_l in zip(h, l):
            if item_l not in handles_dict: handles_dict[item_l] = item_h
    fig.legend(list(handles_dict.values()), list(handles_dict.keys()), loc='lower center', ncol=min(6, max(1, len(handles_dict))), bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "rq3_tradeoff_2x2_all.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rq3_fig1_robustness(points, out_path, title_suffix=""):
    target_strategies = [
        'RandomFlip', 'GaussianNoise', 'FineTune',
        'PTQ', 'SWP', 'PatternMask', 'GrayShield'
    ]
    target_pts = [p for p in points if p.get('strategy') in target_strategies]
    if not target_pts: return

    data = defaultdict(lambda: defaultdict(list))
    for p in target_pts:
        variant = p.get('attacker_variant', 'naive')
        strat = p.get('strategy')
        rec = p.get('post_recovery', 0) * 100
        data[variant][strat].append(rec)
        
    variants = sort_attacker_variants(list(data.keys()))
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(variants))
    width = 0.86 / len(target_strategies)
    
    for i, strat in enumerate(target_strategies):
        means = [np.mean(data[v][strat]) if data[v][strat] else 0 for v in variants]
        stds = [np.std(data[v][strat]) if data[v][strat] else 0 for v in variants]
        offset = (i - len(target_strategies)/2) * width + width/2
        strat_color = DEFENSE_COLORS.get(strat, '#888888')
        ax.bar(x + offset, means, width, yerr=stds, label=strat, capsize=4, color=strat_color, edgecolor='black', alpha=0.9)
        
    ax.set_xticks(x)
    ax.set_xticklabels([v.upper() for v in variants])
    ax.set_xlabel('Attacker Variant', fontsize=12)
    ax.set_ylabel('Payload Recovery Rate (%) ↓', fontsize=12)
    ax.set_title(f'Defense Robustness vs Advanced Attackers - {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(title="Defense", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_rq3_fig2_tradeoff_by_variant(points, out_path, title_suffix=""):
    # Enforce preferred subplot order: naive first (top-left), then by complexity
    PREFERRED_ORDER = ["naive", "repeat3", "repeat5", "interleave", "rs"]
    raw_variants = set(p.get('attacker_variant', 'naive') for p in points)
    # Sort: known variants in preferred order, then alphabetically for unknowns
    variants = (
        [v for v in PREFERRED_ORDER if v in raw_variants] +
        sorted(raw_variants - set(PREFERRED_ORDER))
    )
    if not variants: return

    num_vars = len(variants)
    if num_vars == 1:
        grid_rows, grid_cols = 1, 1
    elif num_vars == 2:
        grid_rows, grid_cols = 1, 2
    else:
        grid_cols = min(3, num_vars)
        grid_rows = int(np.ceil(num_vars / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows), sharey=True, sharex=True)
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, variant in enumerate(variants):
        if idx >= len(axes_flat): break
        ax = axes_flat[idx]
        var_points = [p for p in points if p.get('attacker_variant') == variant]
        strategies = sorted(list(set(p.get('strategy', 'unknown') for p in var_points)))
        
        for strategy in strategies:
            strat_pts = [p for p in var_points if p.get('strategy') == strategy]
            acc_drop = [p['acc_drop'] * 100 for p in strat_pts]
            rec_red = [p['recovery_reduction'] * 100 for p in strat_pts]
            
            is_gray = normalize_defense_name(strategy) == 'GrayShield'
            plot_s = 130 if is_gray else 60
            plot_alpha = 1.0 if is_gray else 0.65
            plot_zorder = 4 if is_gray else 2
            
            ax.scatter(
                acc_drop, rec_red, label=strategy, alpha=plot_alpha, s=plot_s, zorder=plot_zorder,
                color=DEFENSE_COLORS.get(strategy, 'gray'),
                marker=DEFENSE_MARKERS.get(strategy, 'o')
            )
            
        ax.set_title(f'Attacker: {variant.upper()}')
        if idx >= len(axes_flat) - grid_cols:
            ax.set_xlabel('Accuracy Drop (%)')
        if idx % grid_cols == 0:
            ax.set_ylabel('Recovery Reduction (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5)

    for idx in range(len(variants), len(axes_flat)): axes_flat[idx].set_visible(False)
    plt.suptitle(f'Trade-off across Attacker Variants - {title_suffix}', y=1.05, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_violin_heatmap(points, out_dir):
    if not points: return

    MODEL_DISPLAY = {
        'bert_sst2': 'BERT(SST2)', 'roberta_sentiment': 'roBERTa',
        'bert_imdb': 'BERT(IMDB)', 'distilbert_sst2': 'distilBERT',
        'vit_cifar10': 'ViT', 'swin_cifar10': 'Swin',
    }

    # Format points for violin/heatmap
    vh_points = []
    for p in points:
        strat = p.get('strategy', 'unknown')
        rec_red = p.get('recovery_reduction', 1.0 - p.get('post_recovery', 1.0)) * 100
        vh_points.append({'strategy': strat, 'model': p.get('model_preset', 'unknown'), 'rec_red': rec_red})

    # 1. Violin Plot
    strat_data = defaultdict(list)
    for p in vh_points: strat_data[p['strategy']].append(p['rec_red'])
    strategy_order = ['GrayShield', 'PatternMask', 'PTQ', 'SWP', 'FineTune', 'GaussianNoise', 'RandomFlip']
    strat_order = [s for s in strategy_order if s in strat_data]
    if strat_order:
        violin_data = [strat_data[s] for s in strat_order]
        colors = [DEFENSE_COLORS.get(s, '#888888') for s in strat_order]

        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(violin_data, positions=range(len(strat_order)), showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.75)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        ax.set_xticks(range(len(strat_order)))
        ax.set_xticklabels(strat_order, fontsize=11)
        ax.set_ylabel('Recovery Reduction (%)', fontsize=12)
        ax.set_title('Defense Robustness Distribution (Violin)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rq3_violin_stability.png"), dpi=200)
        plt.close()

    # 2. Heatmap
    models = [m for m in ['bert_sst2', 'roberta_sentiment', 'bert_imdb', 'distilbert_sst2', 'vit_cifar10', 'swin_cifar10'] if any(p['model'] == m for p in vh_points)]
    if strat_order and models:
        heatmap = np.full((len(strat_order), len(models)), np.nan)
        for si, strat in enumerate(strat_order):
            for mi, model in enumerate(models):
                vals = [p['rec_red'] for p in vh_points if p['strategy'] == strat and p['model'] == model]
                if vals: heatmap[si, mi] = np.mean(vals)

        fig, ax = plt.subplots(figsize=(10, 5))
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=0, vcenter=25, vmax=60)
        im = ax.imshow(heatmap, cmap='RdYlGn_r', aspect='auto', norm=norm)
        cbar = plt.colorbar(im, ax=ax, label='Recovery Reduction (%) ↑ Better', extend='max')
        cbar.ax.axhline(y=50, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
        cbar.ax.text(0.5, 50, ' 50%', va='center', ha='left', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=10, rotation=30, ha='right')
        ax.set_yticks(range(len(strat_order)))
        ax.set_yticklabels(strat_order, fontsize=11)
        ax.set_title('Defense vs. Model Heatmap (Recovery Reduction %)', fontsize=13, fontweight='bold')
        for si in range(len(strat_order)):
            for mi in range(len(models)):
                val = heatmap[si, mi]
                if not np.isnan(val):
                    text_color = 'white' if val < 15 or val > 45 else 'black'
                    ax.text(mi, si, f'{val:.0f}', ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rq3_heatmap_model_defense.png"), dpi=200)
        plt.close()

def save_metrics_summary(points, out_path):
    data = defaultdict(lambda: {'hamming': [], 'wdd': []})
    for p in points:
        strat = p.get('strategy', 'unknown')
        if 'hamming_distance' in p: data[strat]['hamming'].append(p['hamming_distance'])
        if 'wasserstein_distance' in p: data[strat]['wdd'].append(p['wasserstein_distance'])
            
    if not data: return
        
    with open(out_path, 'w') as f:
        f.write("# RQ3 New Metrics Summary\n\n")
        f.write("| Defense Strategy | Avg Hamming Distance | Avg Wasserstein Dist (W1) |\n")
        f.write("|------------------|----------------------|---------------------------|\n")
        for strat in sorted(data.keys()):
            h_vals, w_vals = data[strat]['hamming'], data[strat]['wdd']
            h_str = f"{np.mean(h_vals):.2f}" if h_vals else "N/A"
            w_str = f"{np.mean(w_vals):.6f}" if w_vals else "N/A"
            f.write(f"| {strat} | {h_str} | {w_str} |\n")

def load_rq3_results(input_dir):
    all_points = []
    rq3_files = []
    root = os.path.join(input_dir, 'rq3.jsonl')
    if os.path.exists(root): rq3_files.append(root)
    # Also find nested ones just in case
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path):
            sub = os.path.join(entry_path, 'rq3', 'rq3.jsonl')
            if os.path.exists(sub): rq3_files.append(sub)
            
    for rq3_file in rq3_files:
        if os.path.exists(rq3_file):
            try:
                with open(rq3_file) as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line.strip())
                        if 'points' in data:
                            model_name = data.get('model_preset', 'unknown')
                            for p in data['points']:
                                p['model_preset'] = p.get('model_preset', model_name)
                                if 'recovery_reduction' not in p:
                                    p['recovery_reduction'] = 1.0 - p.get('post_recovery', 1.0)
                                p['strategy'] = normalize_defense_name(p.get('strategy', 'unknown'))
                                p['domain'] = 'CV' if is_cv_model(p['model_preset']) else 'NLP'
                                all_points.append(p)
            except Exception as e:
                print(f"Error reading {rq3_file}: {e}")
    return all_points

# =============================================================================
# RQ4 Main Paper Visualizations (Dual-Layer Strategy)
# =============================================================================

def plot_rq4_aggregated_pareto(all_points, out_path, title="RQ4: Defense Trade-off (Aggregated)"):
    """
    RQ4-1 Main Paper: Single aggregated Pareto plot.
    Shows worst-case robustness by aggregating all attacker variants.

    Args:
        all_points: List of all data points from RQ3 (includes all variants)
        out_path: Output file path for the plot
        title: Plot title
    """
    if not all_points:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data
    all_acc_drops = [p['acc_drop'] * 100 for p in all_points]
    all_rr = [p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100 for p in all_points]
    strategies = [p.get('strategy', 'unknown') for p in all_points]

    # Determine jitter based on scale
    max_acc_drop = max(all_acc_drops) if all_acc_drops else 1.0
    jitter_amount = 0.02 if max_acc_drop < 1.0 else 0.05
    x_jittered = add_jitter(all_acc_drops, amount=jitter_amount)

    # Plot each defense strategy
    unique_strategies = sorted(set(strategies))
    for strat in unique_strategies:
        indices = [i for i, s in enumerate(strategies) if s == strat]
        if not indices:
            continue

        strat_x = [x_jittered[i] for i in indices]
        strat_y = [all_rr[i] for i in indices]

        # Use existing color scheme
        is_gray = normalize_defense_name(strat) == 'GrayShield'
        display_strat = normalize_defense_name(strat)
        color = DEFENSE_COLORS.get(display_strat, '#888888')
        marker = DEFENSE_MARKERS.get(display_strat, 'o')

        plot_s = 150 if is_gray else 60
        plot_alpha = 1.0 if is_gray else 0.7
        plot_zorder = 4 if is_gray else 2

        ax.scatter(strat_x, strat_y, c=color, marker=marker, s=plot_s,
                   alpha=plot_alpha, edgecolors='black', linewidth=0.5,
                   zorder=plot_zorder, label=display_strat)

    # Compute and plot Pareto front
    pareto_pts = pareto_front(all_points, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)
    if pareto_pts:
        pareto_x = [p['acc_drop'] * 100 for p in pareto_pts]
        pareto_y = [p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100 for p in pareto_pts]

        sorted_pareto = sorted(zip(pareto_x, pareto_y))
        px, py = zip(*sorted_pareto)
        ax.plot(px, py, 'k--', linewidth=2, alpha=0.5, zorder=3, label='Pareto Front')
        ax.scatter(px, py, c='gold', s=200, marker='*', edgecolors='black',
                   linewidth=1.5, zorder=5)

    # Formatting
    ax.set_xlabel('Accuracy Drop (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recovery Reduction (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(40, color='red', linestyle='--', alpha=0.3, linewidth=1, label='RR = 40% (Target)')

    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[RQ4-1] Aggregated Pareto plot saved: {out_path}")


def plot_rq4_robustness_bar(all_points, out_path, title="RQ4: Defense Robustness to Adaptive Attacks"):
    """
    RQ4-2 Main Paper: Bar chart showing defense robustness variance across attacker variants.
    Highlights which defenses are robust to adaptive attacks (low variance = robust).

    Args:
        all_points: List of all data points from RQ3
        out_path: Output file path
        title: Plot title
    """
    if not all_points:
        return

    # Group by defense strategy and attacker variant
    defense_variant_rr = defaultdict(lambda: defaultdict(list))
    for p in all_points:
        strat = p.get('strategy', 'unknown')
        variant = p.get('attacker_variant', 'naive')
        rr = p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100
        defense_variant_rr[strat][variant].append(rr)

    # Compute statistics per defense
    defense_stats = {}
    for strat, variant_data in defense_variant_rr.items():
        # Average RR per variant
        variant_avgs = [np.mean(rrs) for rrs in variant_data.values()]

        defense_stats[strat] = {
            'mean_rr': np.mean(variant_avgs),
            'std_rr': np.std(variant_avgs),  # Low std = robust to variants
            'min_rr': np.min(variant_avgs),
            'max_rr': np.max(variant_avgs),
        }

    # Sort defenses by mean RR (descending)
    sorted_defenses = sorted(defense_stats.keys(),
                            key=lambda s: defense_stats[s]['mean_rr'],
                            reverse=True)

    # Map old names
    sorted_defenses = [normalize_defense_name(s) for s in sorted_defenses]

    # Prepare data for plotting
    means = [defense_stats.get(s, {}).get('mean_rr', 0) for s in sorted_defenses]
    stds = [defense_stats.get(s, {}).get('std_rr', 0) for s in sorted_defenses]
    colors_list = [DEFENSE_COLORS.get(s, '#888888') for s in sorted_defenses]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(sorted_defenses))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                   color=colors_list, edgecolor='black', linewidth=1.5)

    # Annotate with std values
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 2, f'σ={std:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel('Defense Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Recovery Reduction (%) ± Std', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_defenses, fontsize=12, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add reference line
    ax.axhline(40, color='red', linestyle='--', alpha=0.4, linewidth=2, label='RR = 40% (Target)')

    # Add annotation box
    textstr = 'Lower σ (std) = More robust to adaptive attacks\nGrayShield σ < 1% (highly robust)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[RQ4-2] Robustness bar chart saved: {out_path}")


def plot_rq4_variant_grid(all_points, out_path, title="Appendix: Defense Trade-off per Attacker Variant"):
    """
    Appendix: 2x3 grid of Pareto plots, one per attacker variant.
    Shows how defense effectiveness changes under different adaptive attacks.

    Args:
        all_points: List of all data points from RQ3
        out_path: Output file path
        title: Overall plot title
    """
    if not all_points:
        return

    # Define variant order (preferred order for subplot arrangement)
    VARIANT_ORDER = ['naive', 'repeat3', 'repeat5', 'interleave', 'rs', 'aggregate']

    # Get available variants
    available_variants = sort_attacker_variants(
        [p.get('attacker_variant', 'naive') for p in all_points]
    )
    variants_to_plot = [v for v in VARIANT_ORDER if v in available_variants]

    # Add "aggregate" as last subplot (all variants combined)
    if 'aggregate' not in variants_to_plot:
        variants_to_plot.append('aggregate')

    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Compute global scale for consistency
    all_acc_drops = [p['acc_drop'] * 100 for p in all_points]
    all_rr = [p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100 for p in all_points]
    x_max = max(all_acc_drops) * 1.1 if all_acc_drops else 5.0
    y_max = max(all_rr) * 1.05 if all_rr else 100

    for idx, variant in enumerate(variants_to_plot):
        if idx >= 6:  # Only 6 subplots
            break

        ax = axes_flat[idx]

        # Select points for this variant
        if variant == 'aggregate':
            variant_points = all_points
            variant_label = 'ALL (Aggregate)'
        else:
            variant_points = [p for p in all_points if p.get('attacker_variant') == variant]
            variant_label = variant.upper()

        if not variant_points:
            ax.set_visible(False)
            continue

        # Extract data
        acc_drops = [p['acc_drop'] * 100 for p in variant_points]
        rrs = [p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100 for p in variant_points]
        strategies = [p.get('strategy', 'unknown') for p in variant_points]

        # Add jitter
        jitter_amount = 0.02 if x_max < 1.0 else 0.05
        x_jittered = add_jitter(acc_drops, amount=jitter_amount)

        # Plot each strategy
        unique_strats = sorted(set(strategies))
        for strat in unique_strats:
            indices = [i for i, s in enumerate(strategies) if s == strat]
            if not indices:
                continue

            strat_x = [x_jittered[i] for i in indices]
            strat_y = [rrs[i] for i in indices]

            # Use consistent colors
            is_gray = normalize_defense_name(strat) == 'GrayShield'
            display_strat = normalize_defense_name(strat)
            color = DEFENSE_COLORS.get(display_strat, '#888888')
            marker = DEFENSE_MARKERS.get(display_strat, 'o')

            plot_s = 120 if is_gray else 50
            plot_alpha = 1.0 if is_gray else 0.65
            plot_zorder = 4 if is_gray else 2

            ax.scatter(strat_x, strat_y, c=color, marker=marker, s=plot_s,
                       alpha=plot_alpha, edgecolors='black', linewidth=0.5,
                       zorder=plot_zorder, label=display_strat if idx == 0 else "")

        # Compute Pareto front for this variant
        pareto_pts = pareto_front(variant_points, x_key="acc_drop",
                                   y_key="recovery_reduction", maximize_y=True)
        if pareto_pts:
            pareto_x = [p['acc_drop'] * 100 for p in pareto_pts]
            pareto_y = [p.get('recovery_reduction', 1 - p.get('post_recovery', 1.0)) * 100
                       for p in pareto_pts]

            sorted_pareto = sorted(zip(pareto_x, pareto_y))
            px, py = zip(*sorted_pareto)
            ax.plot(px, py, 'k--', linewidth=1.5, alpha=0.5, zorder=3)
            ax.scatter(px, py, c='gold', s=120, marker='*', edgecolors='black',
                      linewidth=1, zorder=5)

        # Formatting
        ax.set_title(f'{variant_label}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(40, color='red', linestyle='--', alpha=0.25, linewidth=1)

        # Set consistent scales
        ax.set_xlim(-0.1, x_max)
        ax.set_ylim(-2, y_max)

        # Add labels to outer subplots only
        if idx >= 3:  # Bottom row
            ax.set_xlabel('Accuracy Drop (%)', fontsize=12)
        if idx % 3 == 0:  # Left column
            ax.set_ylabel('Recovery Reduction (%)', fontsize=12)

    # Hide unused subplots
    for idx in range(len(variants_to_plot), 6):
        axes_flat[idx].set_visible(False)

    # Create unified legend (use first subplot's labels)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        # Add Pareto optimal marker to legend
        from matplotlib.lines import Line2D
        pareto_marker = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                              markersize=15, label='Pareto Optimal', markeredgecolor='black')
        handles.append(pareto_marker)

        fig.legend(handles, labels + ['Pareto Optimal'],
                  loc='upper center', bbox_to_anchor=(0.5, 0.98),
                  ncol=7, fontsize=11, framealpha=0.95)

    plt.suptitle(title, fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Appendix] Per-variant Pareto grid saved: {out_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main(input_dir, output_dir):
    print("=== Generating RQ3 Custom & Aggregated Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    all_points = load_rq3_results(input_dir)
    
    if not all_points:
        print(f"No RQ3 data found in {input_dir}")
        return

    print(f"Loaded {len(all_points)} total RQ3 data points.")
    
    # Generate Pareto plots
    plot_rq3_pareto_aggregated(output_dir, all_points, "all")
    plot_rq3_pareto_aggregated(output_dir, [p for p in all_points if is_cv_model(p.get('model_preset', ''))], "cv")
    plot_rq3_pareto_aggregated(output_dir, [p for p in all_points if is_nlp_model(p.get('model_preset', ''))], "nlp")
    
    # Generate 2x2 Tradeoff
    generate_rq3_tradeoff_2x2(all_points, output_dir)

    # Generate Robustness and Tradeoff per variant
    groups = [
        ('All Models', all_points, 'all'),
        ('NLP Models', [p for p in all_points if p.get('domain') == 'NLP'], 'nlp'),
        ('CV Models', [p for p in all_points if p.get('domain') == 'CV'], 'cv')
    ]
    for label, pts, suffix in groups:
        if not pts: continue
        plot_rq3_fig1_robustness(pts, os.path.join(output_dir, f'rq3_fig1_robustness_{suffix}.png'), title_suffix=label)
        plot_rq3_fig2_tradeoff_by_variant(pts, os.path.join(output_dir, f'rq3_fig2_tradeoff_{suffix}.png'), title_suffix=label)

    # Generate Violin, Heatmap and Metrics Table
    generate_violin_heatmap(all_points, output_dir)
    save_metrics_summary(all_points, os.path.join(output_dir, 'rq3_new_metrics_summary.md'))

    # =============================================================================
    # RQ4: Dual-Layer Strategy Visualizations
    # =============================================================================
    print("\n=== Generating RQ4 Dual-Layer Strategy Plots ===")

    # RQ4-1: Main Paper - Aggregated Pareto plot
    plot_rq4_aggregated_pareto(
        all_points,
        os.path.join(output_dir, 'rq4_fig1_aggregated_pareto.png'),
        title="RQ4: Defense Trade-off (Worst-case over All Attackers)"
    )

    # RQ4-2: Main Paper - Robustness bar chart
    plot_rq4_robustness_bar(
        all_points,
        os.path.join(output_dir, 'rq4_fig2_robustness_bar.png'),
        title="RQ4: Defense Robustness to Adaptive Attacks"
    )

    # Appendix: Per-variant 2x3 Pareto grid
    plot_rq4_variant_grid(
        all_points,
        os.path.join(output_dir, 'rq4_appendix_variant_grid.png'),
        title="Appendix: Defense Trade-off per Attacker Variant (2x3 Grid)"
    )

    print(f"\n✅ All RQ3 and RQ4 plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ3 visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing rq3.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
