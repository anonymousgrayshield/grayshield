#!/usr/bin/env bash
#
# GrayShield Comprehensive Experiment Runner
# ==========================================
#
# Usage:
#   scripts/experiments.sh [OPTIONS]
#
# Options:
#   --phase PHASE       Experiment phase: main (4 models, 2 payloads) or appendix (all) [default: main]
#   --rq RQ             Research question to run (rq1|rq2|rq3|rq4|all) [default: all]
#   --task-type TYPE    Model type to test (text|vision|all) [default: all]
#   --models MODELS     Comma-separated model list (overrides --task-type and --phase)
#   --payloads DIR      Payload directory [default: data/]
#   --n-payloads N      Number of payloads to use [default: 3]
#   --defense DEF       Defense for single RQ2 run (random|pattern|gaussian|finetune|ptq|swp|grayshield) [default: random]
#   --defense-type TYPE Optional RQ2 extension grid (all|ablation|adaptive) [default: all]
#   --x-bits X          Number of LSBs for RQ2/RQ3/RQ4 commands [default: 19]
#   --flip-probs FPS    Comma-separated flip probabilities [default: 0,0.0001,0.001,0.01,0.05,0.1]
#   --seed SEED         Legacy seed (backward compat) [default: 42]
#   --eval-seed SEED    Fixed seed for eval set selection [default: 42]
#   --run-seed SEED     Variable seed for stochastic ops [default: 1]
#   --verbosity V       Verbosity level (0-3) [default: 1]
#   --output-dir DIR    Output directory [default: results/]
#   --n-eval N          Number of evaluation samples [default: 2048]
#   --batch-size N      Batch size for evaluation [default: 16]
#   --full-eval         Use full validation/test split (ignore --n-eval)
#   --layer-range RANGE Layer range: early, mid, late, or "start,end"
#   --attacker-variant V  Attacker encoding for RQ2 (naive|repeat3|repeat5|interleave|rs) [default: naive]
#   --attacker-variants V Comma-separated variants for RQ3 [default: naive,repeat3,repeat5,interleave,rs]
#   --use-paper-payloads  Use SHA256-matched payloads for reproducibility [default]
#   --no-paper-payloads   Use any payloads from directory (faster iteration)
#   --download          Download malware samples from MalwareBazaar
#   --dry-run           Show commands without executing
#
# Defense extension grids (--defense-type):
#   all:       Run the main RQ2/RQ3 paper grids only
#   ablation:  Add RQ2 layer-wise RandomFlip ablation
#   adaptive:  Add RQ2 RandomFlip adaptive-attacker ablation
#
# set -euo pipefail
set -u

# =============================================================================
# Default Configuration
# =============================================================================
PHASE="main"    # "main" (4 models, 2 payloads) or "appendix" (all models, all payloads)
RQ="all"
TASK_TYPE="all"  # text, vision, or all
MODELS=""  # Will be set based on TASK_TYPE/PHASE if not provided
PAYLOAD_DIR="data"
N_PAYLOADS=3
DEFENSE="random"
X_BITS=19  # Default: Transition regime with visible acc_drop
SEED=42
EVAL_SEED=42        # Fixed seed for eval set (reproducible across runs)
RUN_SEED=1          # Variable seed for stochastic ops (defense, etc.)
VERBOSITY=1
OUTPUT_DIR="results/$(date +%F)"
DOWNLOAD_MALWARE=false
DRY_RUN=false
N_EVAL=2048
BATCH_SIZE=16

FULL_EVAL=false     # If true, use full validation/test split
# Paper-grade flip probabilities (matches DEFAULT_FLIP_PROBS in config.py)
FLIP_PROBS="0.0,0.0001,0.001,0.01,0.05,0.1"
DEFENSE_TYPE="all"  # random, pattern, or all
# RQ2 defense testing: use X values that show acc_drop
LSB_BITS_RANDOM="10,17,19,21,23"  # Defense effectiveness across transition → FBLA regime
LAYER_RANGE=""      # Layer range: early, mid, late, or "start,end"
ATTACKER_VARIANT="naive"  # Attacker encoding: naive, repeat3, repeat5, interleave, rs
ATTACKER_VARIANTS="naive,repeat3,repeat5,interleave,rs"  # For RQ3: comma-separated list of variants to test

# =============================================================================
# Paper Payloads (SHA256-based selection for reproducibility)
# =============================================================================
# These two payloads correspond to the extreme entropy cases used in the paper.
# Low-entropy: FakeUpdates/SocGholish (human-readable script, easiest case)
LOW_ENTROPY_PAYLOAD_SHA256="c37c0db91ab188c2fe01642e04e0db9186bc5bf54ad8b6b72512ad5aab921a88"
# High-entropy: VMProtect packed stealer (most random-like, hardest case)
HIGH_ENTROPY_PAYLOAD_SHA256="5704fabda6a0851ea156d1731b4ed4383ce102ec3a93f5d7109cc2f47f8196d0"
# Use SHA256 matching for paper experiments (default: true for reproducibility)
# Override with --no-paper-payloads to use any payloads from directory
USE_PAPER_PAYLOADS=true
FINAL_VIZ=false  # Run final generate_visualizations() block (default: OFF, per-RQ viz is sufficient)

# Pattern defense: use the same x as the attack for fair comparison
PATTERNS_BASE="all_zero,all_one,alt01,alt10"

# Model groups by task type
# Note: Only use models with matching pretrained task for valid accuracy measurements
# - Text models: bert_imdb (imdb), bert_sst2/distilbert_sst2/roberta_sentiment (sst2)
# - Vision models: vit_cifar10/swin_cifar10 (cifar10) - ImageNet models removed (class mismatch)
TEXT_MODELS="bert_imdb,bert_sst2,distilbert_sst2,roberta_sentiment"
VISION_MODELS="vit_cifar10,swin_cifar10"
ALL_MODELS="$TEXT_MODELS,$VISION_MODELS"

# Main paper models: 4 models (2 NLP + 2 CV)
MAIN_MODELS="bert_sst2,roberta_sentiment,vit_cifar10,swin_cifar10"

# Defense methods for RQ2/RQ3
# Mainline: GrayShield + matched baselines
DEFENSES_MAINLINE="random,pattern,gaussian,finetune,ptq,swp,grayshield"
DEFENSES_APPENDIX="random,pattern,gaussian,finetune,ptq,swp,grayshield"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

usage() {
    awk '
        NR == 1 { next }
        /^# set -euo pipefail/ { exit }
        /^#/ { sub(/^# ?/, ""); print }
    ' "$0"
    exit 0
}

# Find payload by SHA256 hash (fail-fast if not found)
find_payload_by_sha256() {
    local sha256=$1
    local search_dir=$2
    local found=""

    # Search for file with matching SHA256
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            local file_sha256
            file_sha256=$(sha256sum "$file" 2>/dev/null | awk '{print $1}')
            if [ "$file_sha256" = "$sha256" ]; then
                found="$file"
                break
            fi
        fi
    done < <(find "$search_dir" -type f \( -name "*.bin" -o -name "*.malware" -o -name "*.exe" \) 2>/dev/null)

    if [ -z "$found" ]; then
        log_error "Payload with SHA256 $sha256 not found in $search_dir"
        log_error "Paper experiments require the controlled payload files under data/malware/."
        log_error "If they are missing and you are in an isolated approved environment, set GRAYSHIELD_HF_DATASET_ID and run: python data/download_from_hf.py"
        exit 1
    fi
    echo "$found"
}

# Get paper payloads (low and high entropy)
get_paper_payloads() {
    local search_dir=$1
    PAPER_PAYLOAD_LOW=$(find_payload_by_sha256 "$LOW_ENTROPY_PAYLOAD_SHA256" "$search_dir")
    PAPER_PAYLOAD_HIGH=$(find_payload_by_sha256 "$HIGH_ENTROPY_PAYLOAD_SHA256" "$search_dir")
    log_info "Paper payloads found:"
    log_info "  Low-entropy:  $PAPER_PAYLOAD_LOW"
    log_info "  High-entropy: $PAPER_PAYLOAD_HIGH"
}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --rq)
            RQ="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --task-type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --payloads)
            PAYLOAD_DIR="$2"
            shift 2
            ;;
        --n-payloads)
            N_PAYLOADS="$2"
            shift 2
            ;;
        --defense)
            DEFENSE="$2"
            shift 2
            ;;
        --x-bits)
            X_BITS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --verbosity)
            VERBOSITY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --download)
            DOWNLOAD_MALWARE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --n-eval)
            N_EVAL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --flip-probs)
            FLIP_PROBS="$2"
            shift 2
            ;;
        --defense-type)
            DEFENSE_TYPE="$2"
            shift 2
            ;;
        --eval-seed)
            EVAL_SEED="$2"
            shift 2
            ;;
        --run-seed)
            RUN_SEED="$2"
            shift 2
            ;;
        --full-eval)
            FULL_EVAL=true
            shift
            ;;
        --layer-range)
            LAYER_RANGE="$2"
            shift 2
            ;;
        --attacker-variant)
            ATTACKER_VARIANT="$2"
            shift 2
            ;;
        --attacker-variants)
            ATTACKER_VARIANTS="$2"
            shift 2
            ;;
        --use-paper-payloads)
            USE_PAPER_PAYLOADS=true
            shift
            ;;
        --no-paper-payloads)
            USE_PAPER_PAYLOADS=false
            shift
            ;;
        --final-viz)
            FINAL_VIZ=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
log_section "GrayShield Experiment Runner"

# Set models based on phase and task type if not explicitly provided
if [ -z "$MODELS" ]; then
    if [ "$PHASE" = "main" ]; then
        # Main paper: 4 models (2 NLP + 2 CV)
        MODELS="$MAIN_MODELS"
        log_info "Using main paper model set: $MODELS"
    else
        # Appendix: all models based on task type
        case $TASK_TYPE in
            text)
                MODELS="$TEXT_MODELS"
                ;;
            vision)
                MODELS="$VISION_MODELS"
                ;;
            all)
                MODELS="$ALL_MODELS"
                ;;
            *)
                log_error "Unknown task type: $TASK_TYPE (use: text, vision, all)"
                exit 1
                ;;
        esac
    fi
fi

# Set defenses based on phase
if [ "$PHASE" = "main" ]; then
    DEFENSES="$DEFENSES_MAINLINE"
else
    DEFENSES="$DEFENSES_APPENDIX"
fi

log_info "Configuration:"
log_info "  Phase: $PHASE"
log_info "  RQ: $RQ"
log_info "  Task Type: $TASK_TYPE"
log_info "  Models: $MODELS"
log_info "  Defenses: $DEFENSES"
log_info "  Payloads: $PAYLOAD_DIR"
log_info "  Defense: $DEFENSE"
log_info "  X-bits: $X_BITS"
log_info "  Seeds: seed=$SEED, eval_seed=$EVAL_SEED, run_seed=$RUN_SEED"
log_info "  Full Eval: $FULL_EVAL"
log_info "  Layer Range: ${LAYER_RANGE:-all}"
log_info "  Attacker Variant (RQ2): $ATTACKER_VARIANT"
log_info "  Attacker Variants (RQ3): $ATTACKER_VARIANTS"
log_info "  Output: $OUTPUT_DIR"
log_info "  Defense Type: $DEFENSE_TYPE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Enable logging to file (append mode)
if [ "$DRY_RUN" = false ]; then
    LOG_FILE="$OUTPUT_DIR/experiment.log"
    log_info "Logging execution trace to: $LOG_FILE"
    # Redirect stdout and stderr to tee (displays to console AND appends to file)
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# Check for Python environment
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please activate your conda environment."
    exit 1
fi

# =============================================================================
# Download Malware Samples (Optional)
# =============================================================================
if [ "$DOWNLOAD_MALWARE" = true ]; then
    log_section "Downloading Malware Samples"
    log_warn "WARNING: This will download REAL malware samples."
    log_warn "Ensure you are in an isolated environment!"

    MALWARE_DIR="$PAYLOAD_DIR/malware"
    mkdir -p "$MALWARE_DIR"

    CMD="python -c \"
from grayshield.payload.malwarebazaar import download_cli
download_cli(
    output_dir='$MALWARE_DIR',
    count=$N_PAYLOADS,
    file_type='exe',
)
\""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $CMD"
    else
        eval "$CMD"
    fi
fi

# =============================================================================
# Find Payloads
# =============================================================================
log_section "Finding Payloads"

PAYLOAD_FILES=()

if [ "$USE_PAPER_PAYLOADS" = true ]; then
    # Paper mode: find payloads by SHA256 (fail-fast if not found)
    log_info "Using paper payloads (SHA256 matching)..."
    get_paper_payloads "$PAYLOAD_DIR"
    PAYLOAD_FILES=("$PAPER_PAYLOAD_LOW" "$PAPER_PAYLOAD_HIGH")
else
    # Default mode: find all payload files, sorted by size (largest first)
    if [ -d "$PAYLOAD_DIR" ]; then
        while IFS= read -r file; do
            [ -n "$file" ] && PAYLOAD_FILES+=("$file")
        done < <(find "$PAYLOAD_DIR" -type f \( -name "*.bin" -o -name "*.malware" -o -name "*.exe" \) -printf '%s %p\n' 2>/dev/null | sort -rn | head -n "$N_PAYLOADS" | awk '{print $2}')
    fi

    if [ ${#PAYLOAD_FILES[@]} -eq 0 ]; then
        log_warn "No payload files found in $PAYLOAD_DIR"
        log_info "Creating benign test payload (deterministic seed=42)..."
        mkdir -p "$PAYLOAD_DIR"
        # Deterministic benign payload for testing (NOT for paper experiments)
        python3 -c "import random; random.seed(42); open('$PAYLOAD_DIR/test_payload.bin','wb').write(bytes(random.getrandbits(8) for _ in range(10240)))" 2>/dev/null \
            || dd if=/dev/urandom of="$PAYLOAD_DIR/test_payload.bin" bs=1024 count=10 2>/dev/null
        PAYLOAD_FILES=("$PAYLOAD_DIR/test_payload.bin")
    fi
fi

log_info "Found ${#PAYLOAD_FILES[@]} payload(s)"

# =============================================================================
# Model/Task Mapping
# =============================================================================
declare -A MODEL_TASKS
# Text models - matched to their pretrained task
MODEL_TASKS["bert_imdb"]="imdb"
MODEL_TASKS["bert_sst2"]="sst2"
MODEL_TASKS["distilbert_sst2"]="sst2"
MODEL_TASKS["roberta_sentiment"]="sst2"
# Vision models - matched to their pretrained task
MODEL_TASKS["vit_cifar10"]="cifar10"
MODEL_TASKS["swin_cifar10"]="cifar10"
# Note: vit_imagenet and swin_imagenet removed - 1000-class output incompatible with CIFAR-10

# =============================================================================
# Run Experiments
# =============================================================================
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Note: Results are now organized by payload, not timestamp
# Structure: results/{payload_name}/{rq1,rq2,rq3}/

# Helper to get payload short name
get_payload_name() {
    local path=$1
    local basename
    basename=$(basename "$path")
    # Remove extension and take first 20 chars
    basename="${basename%.*}"
    echo "${basename:0:20}"
}

# Helper to create output directory
# Structure: results/YYYY-MM-DD/
get_output_dir() {
    local payload=$1
    local rq=$2
    local out_dir="$OUTPUT_DIR"
    mkdir -p "$out_dir"
    echo "$out_dir"
}

run_command() {
    local cmd="$1"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $cmd"
    else
        log_info "Running: $cmd"
        eval "$cmd"
    fi
}

# -----------------------------------------------------------------------------
# RQ1: Injection Feasibility (Capacity Boundary)
# -----------------------------------------------------------------------------
# RQ1 tests capacity boundary: bit-depth × payload size
# Paper design: X ∈ {4, 8, 16, 19, 23} (coarse-to-fine boundary sampling)
#   - X = 4, 8 → Low capacity, high stealth
#   - X = 16, 19 → Transition regime
#   - X = 23 → Full mantissa (max capacity, visible acc_drop)
# Mode: encoder_only (component sensitivity is RQ2.a, not RQ1)
# Metrics: Capacity, LSB Similarity, L2 Distance, Accuracy Drop
LSB_BITS_RQ1="4,8,16,19,21,23"  # Paper-grade: coarse-to-fine boundary sampling

run_rq1() {
    log_section "RQ1: Injection Feasibility (X ∈ {4,8,16,19,23})"

    IFS=',' read -ra LSB_ARRAY_RQ1 <<< "$LSB_BITS_RQ1"

    for model in "${MODEL_ARRAY[@]}"; do
        task="${MODEL_TASKS[$model]}"
        log_info "Testing $model on $task"

        for payload in "${PAYLOAD_FILES[@]}"; do
            local out_dir
            out_dir=$(get_output_dir "$payload" "rq1")
            log_info "  Payload: $(basename "$payload") -> $out_dir"

            # Build common args for new seed/eval params
            COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
            [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
            [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

            # Loop over LSB bits to analyze capacity boundary
            # Mode: encoder_only only (component sensitivity moved to RQ2.a)
            for x_bits in "${LSB_ARRAY_RQ1[@]}"; do
                CMD="python -m grayshield.cli rq1 \
                    --model $model \
                    --task $task \
                    --payload_path '$payload' \
                    --x $x_bits \
                    --mode encoder_only \
                    --n_eval $N_EVAL \
                    --batch_size $BATCH_SIZE \
                    $COMMON_ARGS \
                    --device cuda \
                    --output_dir '$out_dir'"

                run_command "$CMD"
            done
        done
    done
}

# -----------------------------------------------------------------------------
# RQ2: Defense Effectiveness
# -----------------------------------------------------------------------------
run_rq2_random() {
    log_info "  Running RandomFlip defense (flip_prob sweep)..."
    IFS=',' read -ra FLIP_ARRAY <<< "$FLIP_PROBS"

    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}  # Default to x=19 for RQ2

    # Build common args for new seed/eval params
    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    # Loop over flip probabilities (strength sweep)
    for fp in "${FLIP_ARRAY[@]}"; do
        CMD="python -m grayshield.cli rq2 \
            --model $model \
            --task $task \
            --payload_path '$payload' \
            --x $x_bits \
            --mode all \
            --defense random \
            --flip_prob $fp \
            --attacker_variant $ATTACKER_VARIANT \
            --n_eval $N_EVAL \
            --batch_size $BATCH_SIZE \
            $COMMON_ARGS \
            --device cuda \
            --output_dir '$out_dir'"

        run_command "$CMD"
    done
}

run_rq2_pattern() {
    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}

    log_info "  Running Pattern Defense experiments (x=$x_bits, 4 representative patterns)..."

    local PATTERN_ARRAY=(
        "$(printf '0%.0s' $(seq 1 "$x_bits"))"
        "$(printf '1%.0s' $(seq 1 "$x_bits"))"
        "$(python - <<PY
x = $x_bits
print(('01' * ((x + 1) // 2))[:x])
PY
)"
        "$(python - <<PY
x = $x_bits
print(('10' * ((x + 1) // 2))[:x])
PY
)"
    )

    # Build common args for new seed/eval params
    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    for pattern in "${PATTERN_ARRAY[@]}"; do
        CMD="python -m grayshield.cli rq2 \
            --model $model \
            --task $task \
            --payload_path '$payload' \
            --x $x_bits \
            --mode all \
            --defense pattern \
            --pattern $pattern \
            --attacker_variant $ATTACKER_VARIANT \
            --n_eval $N_EVAL \
            --batch_size $BATCH_SIZE \
            $COMMON_ARGS \
            --device cuda \
            --output_dir '$out_dir'"

        run_command "$CMD"
    done
}

run_rq2_ptq() {
    log_info "  Running PTQ defense..."
    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}

    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    CMD="python -m grayshield.cli rq2 \
        --model $model \
        --task $task \
        --payload_path '$payload' \
        --x $x_bits \
        --mode all \
        --defense ptq \
        --attacker_variant $ATTACKER_VARIANT \
        --n_eval $N_EVAL \
        --batch_size $BATCH_SIZE \
        $COMMON_ARGS \
        --device cuda \
        --output_dir '$out_dir'"
    run_command "$CMD"
}

run_rq2_swp() {
    log_info "  Running SWP defense..."
    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}

    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    CMD="python -m grayshield.cli rq2 \
        --model $model \
        --task $task \
        --payload_path '$payload' \
        --x $x_bits \
        --mode all \
        --defense swp \
        --attacker_variant $ATTACKER_VARIANT \
        --n_eval $N_EVAL \
        --batch_size $BATCH_SIZE \
        $COMMON_ARGS \
        --device cuda \
        --output_dir '$out_dir'"
    run_command "$CMD"
}

run_rq2_gaussian() {
    log_info "  Running GaussianNoise defense (sigma sweep)..."
    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}  # Default to x=19 for RQ2

    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    # Sweep over sigma values from config (GAUSSIAN_SIGMAS)
    for sigma in 0.0 1e-6 3e-6 1e-5 3e-5 1e-4; do
        CMD="python -m grayshield.cli rq2 \
            --model $model \
            --task $task \
            --payload_path '$payload' \
            --x $x_bits \
            --mode all \
            --defense gaussian \
            --sigma $sigma \
            --attacker_variant $ATTACKER_VARIANT \
            --n_eval $N_EVAL \
            --batch_size $BATCH_SIZE \
            $COMMON_ARGS \
            --device cuda \
            --output_dir '$out_dir'"
        run_command "$CMD"
    done
}

run_rq2_finetune() {
    log_info "  Running FineTune defense (steps sweep)..."
    local model=$1
    local task=$2
    local payload=$3
    local out_dir=$4
    local x_bits=${5:-19}

    local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
    [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
    [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

    # Sweep over steps from config (FINETUNE_STEPS: 50, 100, 200, 500)
    for steps in 50 100 200 500; do
        CMD="python -m grayshield.cli rq2 \
            --model $model \
            --task $task \
            --payload_path '$payload' \
            --x $x_bits \
            --mode all \
            --defense finetune \
            --finetune_steps $steps \
            --attacker_variant $ATTACKER_VARIANT \
            --n_eval $N_EVAL \
            --batch_size $BATCH_SIZE \
            $COMMON_ARGS \
            --device cuda \
            --output_dir '$out_dir'"
        run_command "$CMD"
    done
}

# -----------------------------------------------------------------------------
# RQ2.a: Layer-wise Ablation (TargetMode × layer_range)
# -----------------------------------------------------------------------------
# Paper design: WHERE is the covert channel most permissive / sanitizable?
# Models: bert_imdb, vit_cifar10 (representative text + vision)
# Strict grid: modes × layer_range × x-bits × flip_prob
run_rq2_layerwise_ablation() {
    log_section "RQ2.a: Layer-wise Ablation"

    # Paper-grade models for ablation study
    local ABLATION_MODELS=("bert_imdb" "vit_cifar10")
    local ABLATION_X_BITS="10,17,23"
    local ABLATION_FLIP_PROBS="0.01,0.1,0.3"
    local ABLATION_MODES=("embeddings" "attention" "ffn" "encoder_only" "full_model")
    local ABLATION_LAYER_RANGES=("early" "mid" "late")

    IFS=',' read -ra X_ARRAY <<< "$ABLATION_X_BITS"
    IFS=',' read -ra FP_ARRAY <<< "$ABLATION_FLIP_PROBS"

    for model in "${ABLATION_MODELS[@]}"; do
        task="${MODEL_TASKS[$model]}"
        log_info "Layer-wise ablation on $model"

        for payload in "${PAYLOAD_FILES[@]}"; do
            local out_dir
            out_dir=$(get_output_dir "$payload" "rq2")
            log_info "  Payload: $(basename "$payload") -> $out_dir"

            local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
            [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"

            for mode in "${ABLATION_MODES[@]}"; do
                for x_bits in "${X_ARRAY[@]}"; do
                    for fp in "${FP_ARRAY[@]}"; do
                        # Layer range only applies to: attention, ffn, encoder_only
                        if [[ "$mode" == "attention" || "$mode" == "ffn" || "$mode" == "encoder_only" ]]; then
                            for lr in "${ABLATION_LAYER_RANGES[@]}"; do
                                CMD="python -m grayshield.cli rq2 \
                                    --model $model \
                                    --task $task \
                                    --payload_path '$payload' \
                                    --x $x_bits \
                                    --mode $mode \
                                    --defense random \
                                    --flip_prob $fp \
                                    --layer_range $lr \
                                    --attacker_variant naive \
                                    --n_eval $N_EVAL \
                                    --batch_size $BATCH_SIZE \
                                    $COMMON_ARGS \
                                    --device cuda \
                                    --output_dir '$out_dir'"
                                run_command "$CMD"
                            done
                        else
                            # embeddings, full_model: no layer_range
                            CMD="python -m grayshield.cli rq2 \
                                --model $model \
                                --task $task \
                                --payload_path '$payload' \
                                --x $x_bits \
                                --mode $mode \
                                --defense random \
                                --flip_prob $fp \
                                --attacker_variant naive \
                                --n_eval $N_EVAL \
                                --batch_size $BATCH_SIZE \
                                $COMMON_ARGS \
                                --device cuda \
                                --output_dir '$out_dir'"
                            run_command "$CMD"
                        fi
                    done
                done
            done
        done
    done
}

# -----------------------------------------------------------------------------
# RQ2.b: Adaptive Attacker Analysis
# -----------------------------------------------------------------------------
# Paper design: Do defenses remain effective against non-naive attackers?
# Models: bert_imdb, vit_cifar10
# Fixed x=10 (documented choice: transition regime where defense matters)
# Attacker variants: naive, repeat3, repeat5, interleave, rs
run_rq2_adaptive() {
    log_section "RQ2.b: Adaptive Attacker Analysis"

    local ADAPTIVE_MODELS=("bert_imdb" "vit_cifar10")
    local ADAPTIVE_X_BITS=10  # Fixed: transition regime
    local ADAPTIVE_FLIP_PROBS="0.01,0.1,0.3"
    local ADAPTIVE_VARIANTS=("naive" "repeat3" "repeat5" "interleave" "rs")

    IFS=',' read -ra FP_ARRAY <<< "$ADAPTIVE_FLIP_PROBS"

    for model in "${ADAPTIVE_MODELS[@]}"; do
        task="${MODEL_TASKS[$model]}"
        log_info "Adaptive attacker analysis on $model"

        for payload in "${PAYLOAD_FILES[@]}"; do
            local out_dir
            out_dir=$(get_output_dir "$payload" "rq2")
            log_info "  Payload: $(basename "$payload") -> $out_dir"

            local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
            [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"

            for variant in "${ADAPTIVE_VARIANTS[@]}"; do
                for fp in "${FP_ARRAY[@]}"; do
                    CMD="python -m grayshield.cli rq2 \
                        --model $model \
                        --task $task \
                        --payload_path '$payload' \
                        --x $ADAPTIVE_X_BITS \
                        --mode encoder_only \
                        --defense random \
                        --flip_prob $fp \
                        --attacker_variant $variant \
                        --n_eval $N_EVAL \
                        --batch_size $BATCH_SIZE \
                        $COMMON_ARGS \
                        --device cuda \
                        --output_dir '$out_dir'"
                    run_command "$CMD"
                done
            done
        done
    done
}

run_rq2() {
    log_section "RQ2: Defense Effectiveness"

    # Determine defenses based on phase (main paper vs appendix)
    local defenses_to_run
    if [ "$PHASE" = "main" ]; then
        defenses_to_run="random pattern gaussian finetune ptq swp grayshield"
    else
        defenses_to_run="random pattern gaussian finetune ptq swp grayshield"
    fi

    log_info "Phase: $PHASE"
    log_info "Defenses to run: $defenses_to_run"
    log_info "Default x: $X_BITS"

    for model in "${MODEL_ARRAY[@]}"; do
        task="${MODEL_TASKS[$model]}"
        log_info "Testing defenses on $model"

        for payload in "${PAYLOAD_FILES[@]}"; do
            local out_dir
            out_dir=$(get_output_dir "$payload" "rq2")
            log_info "  Payload: $(basename "$payload") -> $out_dir"

            local COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
            [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
            [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

            # Run all defenses for this phase
            for defense in $defenses_to_run; do
                case $defense in
                    random)
                        run_rq2_random "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    gaussian)
                        run_rq2_gaussian "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    finetune)
                        run_rq2_finetune "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    ptq)
                        run_rq2_ptq "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    swp)
                        run_rq2_swp "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    pattern)
                        run_rq2_pattern "$model" "$task" "$payload" "$out_dir" "$X_BITS"
                        ;;
                    grayshield)
                        CMD="python -m grayshield.cli rq2 \
                            --model $model \
                            --task $task \
                            --payload_path '$payload' \
                            --x $X_BITS \
                            --mode all \
                            --defense grayshield \
                            --attacker_variant $ATTACKER_VARIANT \
                            --n_eval $N_EVAL \
                            --batch_size $BATCH_SIZE \
                            $COMMON_ARGS \
                            --device cuda \
                            --output_dir '$out_dir'"
                        run_command "$CMD"
                        ;;
                    *)
                        log_warn "Unknown defense: $defense"
                        ;;
                esac
            done
        done
    done

    # Run ablation and adaptive experiments if requested via DEFENSE_TYPE
    case $DEFENSE_TYPE in
        ablation)
            run_rq2_layerwise_ablation
            ;;
        adaptive)
            run_rq2_adaptive
            ;;
        all)
            # Note: ablation and adaptive are NOT included in 'all' by default
            # Use --defense-type ablation or --defense-type adaptive explicitly
            ;;
    esac
}

# -----------------------------------------------------------------------------
# RQ3: Strategy Comparison
# -----------------------------------------------------------------------------
run_rq3() {
    log_section "RQ3: Strategy Comparison"

    for model in "${MODEL_ARRAY[@]}"; do
        task="${MODEL_TASKS[$model]}"
        log_info "Comparing strategies for $model"

        for payload in "${PAYLOAD_FILES[@]}"; do
            local out_dir
            out_dir=$(get_output_dir "$payload" "rq3")
            log_info "  Payload: $(basename "$payload") -> $out_dir"

            # Build common args for new seed/eval params
            COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
            [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"
            [ -n "$LAYER_RANGE" ] && COMMON_ARGS="$COMMON_ARGS --layer_range $LAYER_RANGE"

            CMD="python -m grayshield.cli rq3 \
                --model $model \
                --task $task \
                --payload_path '$payload' \
                --x $X_BITS \
                --mode all \
                --attacker_variants '$ATTACKER_VARIANTS' \
                --n_eval $N_EVAL \
                --batch_size $BATCH_SIZE \
                $COMMON_ARGS \
                --device cuda \
                --output_dir '$out_dir'"

            run_command "$CMD"
        done
    done

}

# =============================================================================
# Per-RQ Visualization Generation
# =============================================================================
generate_rq1_viz() {
    log_info "Generating RQ1 visualizations..."
    python grayshield/visualization/rq1.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ1 visualization may have failed"
}

generate_rq2_viz() {
    log_info "Generating RQ2 visualizations..."
    python grayshield/visualization/rq2.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ2 visualization may have failed"
}

generate_rq3_viz() {
    log_info "Generating RQ3 visualizations..."
    python grayshield/visualization/rq3.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ3 visualization may have failed"
}

# -----------------------------------------------------------------------------
# RQ4: Trade-off / Pareto Analysis
# -----------------------------------------------------------------------------
run_rq4() {
    log_section "RQ4: Trade-off / Pareto Analysis"

    for payload in "${PAYLOAD_FILES[@]}"; do
        local out_dir
        out_dir=$(get_output_dir "$payload" "rq4")
        local results_dir
        results_dir=$(get_output_dir "$payload" "rq3")
        log_info "  Generating RQ4 analysis for payload: $(basename "$payload") -> $out_dir"

        # Build common args
        COMMON_ARGS="--seed $SEED --eval_seed $EVAL_SEED --run_seed $RUN_SEED"
        [ "$FULL_EVAL" = true ] && COMMON_ARGS="$COMMON_ARGS --full_eval"

        # Use first model for RQ4 (it reads existing results)
        IFS=',' read -ra FIRST_MODEL <<< "$MODELS"
        local model="${FIRST_MODEL[0]}"
        task="${MODEL_TASKS[$model]}"

        CMD="python -m grayshield.cli rq4 \
            --model $model \
            --task $task \
            --payload_path '$payload' \
            --x $X_BITS \
            --mode encoder_only \
            --results_dir '$results_dir' \
            --generate_table \
            --n_eval $N_EVAL \
            --batch_size $BATCH_SIZE \
            $COMMON_ARGS \
            --device cuda \
            --output_dir '$out_dir'"

        run_command "$CMD"
    done
}

generate_rq4_viz() {
    log_info "Generating RQ4 visualizations..."
    python grayshield/visualization/rq4.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ4 visualization may have failed"
}

# =============================================================================
# Execute Selected RQs (with auto-visualization)
# =============================================================================
case $RQ in
    rq1)
        run_rq1
        [ "$DRY_RUN" = false ] && generate_rq1_viz
        ;;
    rq2)
        run_rq2
        [ "$DRY_RUN" = false ] && generate_rq2_viz
        ;;
    rq3)
        run_rq3
        [ "$DRY_RUN" = false ] && generate_rq3_viz
        ;;
    rq4)
        run_rq4
        [ "$DRY_RUN" = false ] && generate_rq4_viz
        ;;
    all)
        run_rq1
        [ "$DRY_RUN" = false ] && generate_rq1_viz
        run_rq2
        [ "$DRY_RUN" = false ] && generate_rq2_viz
        run_rq3
        [ "$DRY_RUN" = false ] && generate_rq3_viz
        run_rq4
        [ "$DRY_RUN" = false ] && generate_rq4_viz
        ;;
    *)
        log_error "Unknown RQ: $RQ"
        exit 1
        ;;
esac

# =============================================================================
# Generate Visualizations
# =============================================================================
generate_visualizations() {
    log_section "Generating Visualizations"

    # Generate visualizations for each payload directory
    log_info "Generating visualizations for payload-organized results..."

    # Generate visualizations using dedicated scripts
    log_info "Generating visualizations using python scripts..."
    
    python grayshield/visualization/rq1.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ1 viz failed"
    python grayshield/visualization/rq2.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ2 viz failed"
    python grayshield/visualization/rq3.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" || log_warn "RQ3 viz failed"
    
    log_info "Visualization generation complete!"
}

# Generate final visualizations only if explicitly requested (--final-viz)
# Per-RQ visualizations are generated after each RQ, so this is redundant by default
if [ "$DRY_RUN" = false ] && [ "$FINAL_VIZ" = true ]; then
    generate_visualizations
fi

# =============================================================================
# Summary
# =============================================================================
log_section "Experiment Complete"
log_info "Results saved to: $OUTPUT_DIR"

# Count result files
if [ "$DRY_RUN" = false ]; then
    N_RESULTS=$(find "$OUTPUT_DIR" -name "*.jsonl" -newer "$0" | wc -l)
    log_info "Generated $N_RESULTS result files"
fi

log_info "Done!"
