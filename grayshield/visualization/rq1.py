import json
import os
import argparse
from grayshield.visualization.plots import (
    plot_rq1_clean_vs_poisoned,
    plot_rq1_feasibility_tradeoff,
    plot_rq1_heatmap
)

def load_jsonl(path):
    results = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Keep visualization robust even if a jsonl line was partially written.
                        continue
    return results


def load_rq1_results(input_dir):
    """Prefer canonical rq1.json, fall back to tolerant rq1.jsonl loading."""
    rq1_json = os.path.join(input_dir, "rq1.json")
    if os.path.exists(rq1_json):
        with open(rq1_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data

    rq1_jsonl = os.path.join(input_dir, "rq1.jsonl")
    if os.path.exists(rq1_jsonl):
        return load_jsonl(rq1_jsonl)

    # Fallback to nested search.
    for root, _, files in os.walk(input_dir):
        if "rq1.json" in files:
            with open(os.path.join(root, "rq1.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        if "rq1.jsonl" in files:
            data = load_jsonl(os.path.join(root, "rq1.jsonl"))
            if data:
                return data
    return []

def main(input_dir, output_dir):
    print("=== Generating RQ1 Standard Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    
    rq1_results = load_rq1_results(input_dir)
    
    if rq1_results:
        print(f"Loaded {len(rq1_results)} RQ1 points.")
        plot_rq1_clean_vs_poisoned(rq1_results, os.path.join(output_dir, "rq1_clean_vs_poisoned.png"))
        plot_rq1_feasibility_tradeoff(rq1_results, os.path.join(output_dir, "rq1_feasibility_tradeoff.png"))
        plot_rq1_heatmap(rq1_results, os.path.join(output_dir, "rq1_heatmap_lsb_similarity.png"), metric="lsb_similarity", title="Attack Bit Error Rate (Worst-Case)")
        plot_rq1_heatmap(rq1_results, os.path.join(output_dir, "rq1_heatmap_stealth.png"), metric="cosine_similarity", title="Stealthiness (Worst-Case)")
        print(f"RQ1 plots saved to {output_dir}")
    else:
        print(f"No RQ1 data found in {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ1 visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing rq1.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
