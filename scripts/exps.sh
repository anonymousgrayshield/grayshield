#!/usr/bin/env bash
#
# GrayShield Comprehensive Experiment & Visualization Runner
# =======================================================
#
# Usage:
#   scripts/exps.sh [OPTIONS]
#   OR: cd scripts && ./exps.sh [OPTIONS]
#
# Options:
#   --visualize-only    Skip experiments and only run visualization scripts
#   --visualize         Run experiments and then run visualization scripts
#
# Inherited Options (from scripts/experiments.sh):
#   --phase PHASE       Experiment phase: main (4 models, 2 payloads) or appendix (all) [default: main]
#   --rq RQ             Research question to run (rq1|rq2|rq3|rq4|all) [default: all]
#   --task-type TYPE    Model type to test (text|vision|all) [default: all]
#   --models MODELS     Comma-separated model list
#   --output-dir DIR    Output directory [default: results/YYYY-MM-DD]
#   --x-bits X          Number of LSBs for RQ2/RQ3/RQ4 commands [default: 19]
#   ... and all other GrayShield experiment arguments
#

set -u

# Default settings
VISUALIZE=false
VISUALIZE_ONLY=false
OUTPUT_DIR="results/$(date +%F)"

# Fast pass to extract standard arguments we need here before passing the rest
for arg in "$@"; do
    case $arg in
        --visualize-only) VISUALIZE_ONLY=true; VISUALIZE=true ;;
        --visualize)      VISUALIZE=true ;;
    esac
done

# We also need to extract output-dir to know where to visualize
args_copy=("$@")
for ((i=0; i<${#args_copy[@]}; i++)); do
    if [[ "${args_copy[$i]}" == "--output-dir" ]]; then
        OUTPUT_DIR="${args_copy[$i+1]}"
        break
    fi
done

# Filter out visualization flags before passing to the experiment runner logic
EXP_ARGS=()
for arg in "$@"; do
    case $arg in
        --visualize-only|--visualize) ;;
        *) EXP_ARGS+=("$arg") ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Make sure we have the python environment variables
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# =============================================================================
# GrayShield V3 Mode (Enhanced Defense) - Default Configuration
# =============================================================================
# Generate or load GrayShield secret key for V3 mode
GRAYSHIELD_KEY_FILE="$PROJECT_ROOT/.grayshield_key"
if [ ! -f "$GRAYSHIELD_KEY_FILE" ]; then
    log_info "Generating GrayShield secret key..."
    python -c "import os; open('$GRAYSHIELD_KEY_FILE', 'w').write(os.urandom(32).hex())"
fi
export GRAYSHIELD_KEY=$(cat "$GRAYSHIELD_KEY_FILE")

# Enable V3 mode by default (HMAC + per-run salt + multi-layer sequences)
# Set to empty string to disable: export GRAYSHIELD_V3=""
export GRAYSHIELD_V3=1

log_info "GrayShield V3 mode enabled (enhanced defense)"

# Set conda correctly if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "grayshield"; then
        conda activate grayshield
    fi
fi

# Change to project root for consistent execution
cd "$PROJECT_ROOT"

# =============================================================================
# Phase 1: Experiments
# =============================================================================
if [ "$VISUALIZE_ONLY" = false ]; then
    log_section "Running GrayShield Experiments"

    # Delegate to the core experiment runner while keeping visualization
    # orchestration centralized in this wrapper.

    log_info "Executing experiments with args: ${EXP_ARGS[*]}"
    bash "$SCRIPT_DIR/experiments.sh" "${EXP_ARGS[@]}"
else
    log_info "Skipping experiments (--visualize-only flag active)"
fi

# =============================================================================
# Phase 2: Visualizations
# =============================================================================
if [ "$VISUALIZE" = true ]; then
    log_section "Generating GrayShield Visualizations"
    log_info "Using data from: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Execute new centralized visualization scripts
    python grayshield/visualization/rq1.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR"
    python grayshield/visualization/rq2.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR"
    python grayshield/visualization/rq3.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR"
    python grayshield/visualization/rq4.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR"
    
    # Table generation (if exists)
    if [ -f "$SCRIPT_DIR/generate_tables.py" ]; then
        log_info "Generating Tables"
        python "$SCRIPT_DIR/generate_tables.py" --output_dir "$OUTPUT_DIR"
    fi
    
    log_info "All visualization targets complete. View results in $OUTPUT_DIR"
fi
