import json
import os
import argparse
from grayshield.visualization.plots import (
    plot_rq2_fig1_bit_accuracy_by_method,
    plot_rq2_fig2_accuracy_drop_by_method,
    plot_rq2_fig3_strength_sweep_scatter
)

def load_jsonl(path):
    results = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results

def main(input_dir, output_dir):
    print("=== Generating RQ2 Standard Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Try different possible locations for rq2 results
    rq2_file = os.path.join(input_dir, "rq2.jsonl")
    if not os.path.exists(rq2_file):
        # Fallback to search if it's nested
        rq2_files = []
        for root, dirs, files in os.walk(input_dir):
            if "rq2.jsonl" in files:
                rq2_files.append(os.path.join(root, "rq2.jsonl"))
        if rq2_files:
            rq2_file = rq2_files[0]
            
    rq2_results = load_jsonl(rq2_file)
    
    if rq2_results:
        print(f"Loaded {len(rq2_results)} RQ2 points.")
        plot_rq2_fig1_bit_accuracy_by_method(rq2_results, os.path.join(output_dir, "rq2_fig1_disruption.png"))
        plot_rq2_fig2_accuracy_drop_by_method(rq2_results, os.path.join(output_dir, "rq2_fig2_accuracy_drop.png"))
        plot_rq2_fig3_strength_sweep_scatter(rq2_results, os.path.join(output_dir, "rq2_fig3_tradeoff.png"))
        print(f"RQ2 plots saved to {output_dir}")
    else:
        print(f"No RQ2 data found in {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RQ2 visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing rq2.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
