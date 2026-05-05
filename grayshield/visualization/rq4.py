import json
import os
import argparse
from grayshield.visualization.plots import (
    normalize_defense_name,
    plot_rq4_pareto_scatter,
    plot_rq4_strategy_summary,
)
from grayshield.metrics.pareto import pareto_front

def load_rq2_rq3_results(input_dir):
    all_points = []
    found = {"rq2": 0, "rq3": 0}
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname in ["rq2.jsonl", "rq3.jsonl"]:
                fpath = os.path.join(root, fname)
                found["rq3" if fname == "rq3.jsonl" else "rq2"] += 1
                try:
                    with open(fpath, 'r') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                if fname == "rq3.jsonl":
                                    points = record.get('points', [])
                                else:
                                    metrics = record.get('metrics', {})
                                    defense = record.get('defense', {})
                                    point = {
                                        'strategy': normalize_defense_name(defense.get('type', 'unknown')),
                                        'acc_drop': metrics.get('acc_drop_vs_base', 0),
                                        'recovery_reduction': metrics.get(
                                            'recovery_reduction',
                                            1.0 - metrics.get('post_recovery', 1.0),
                                        ),
                                        'post_recovery': metrics.get('post_recovery', 1.0),
                                        'post_acc': metrics.get('post_defense_acc', metrics.get('post_acc', 0)),
                                        'relative_l2_distance': metrics.get('relative_l2_distance'),
                                        'attacker_variant': record.get('attacker_variant', 'naive'),
                                        'model_preset': record.get('model_preset', 'unknown'),
                                        'x': record.get('x'),
                                        'defense_x': record.get('x'),
                                        'rq': 'rq2',
                                    }
                                    for key in ['flip_prob', 'pattern', 'sigma', 'n_steps', 'fraction', 'gray_version']:
                                        if key in defense:
                                            point[key] = defense[key]
                                    points = [point]
                                if points:
                                    model = record.get('model_preset', 'unknown')
                                    for p in points:
                                        p['model'] = p.get('model_preset', model)
                                        if 'recovery_reduction' not in p:
                                            p['recovery_reduction'] = 1.0 - p.get('post_recovery', 1.0)
                                        if 'relative_l2_distance' not in p and 'l2_distance' in p:
                                            p['relative_l2_distance'] = p.get('l2_distance')
                                        p['strategy'] = normalize_defense_name(p.get('strategy', 'unknown'))
                                        p['rq'] = p.get('rq', 'rq3' if fname == "rq3.jsonl" else 'rq2')
                                    all_points.extend(points)
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
    return all_points, found

def main(input_dir, output_dir):
    print("=== Generating RQ4 Pareto Scatter ===")
    os.makedirs(output_dir, exist_ok=True)
    all_points, found = load_rq2_rq3_results(input_dir)

    if found["rq2"] == 0 or found["rq3"] == 0:
        print(
            "Skipping RQ4 visualization: both rq2.jsonl and rq3.jsonl are required "
            f"(found rq2={found['rq2']}, rq3={found['rq3']})."
        )
        return
    
    if all_points:
        pareto = pareto_front(all_points, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)
        plot_rq4_pareto_scatter(
            all_points, pareto,
            out_path=os.path.join(output_dir, "rq4_pareto_scatter.png"),
            title="Defense Trade-off (All Methods)"
        )
        plot_rq4_strategy_summary(
            all_points,
            out_path=os.path.join(output_dir, "rq4_pareto_summary.png"),
            title="Defense Trade-off Summary (Mean ± Std)"
        )
        print(f"RQ4 plots saved to {output_dir}")
    else:
        print(f"No RQ2/RQ3 data found in {input_dir} for RQ4 visualization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ4 visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing rq2.jsonl/rq3.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
