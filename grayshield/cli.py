"""
GrayShield Command Line Interface

Usage:
    python -m grayshield.cli <command> [options]

Commands:
    rq1         Run RQ1: Injection Feasibility
    rq2         Run RQ2: Defense Effectiveness
    rq3         Run RQ3: Strategy Comparison
    rq_all      Run all RQs
    download    Download malware samples from MalwareBazaar
"""
from __future__ import annotations
import argparse
import sys
from .config import (
    ExperimentArgs, DefenseArgs, LAYER_RANGE_ALIASES,
    MAIN_MODELS, ALL_MODELS, MAINLINE_DEFENSES,
    GAUSSIAN_SIGMAS, FINETUNE_STEPS,
    RQ3_ATTACKER_VARIANTS, PhaseConfig,
    MALWARE_DIR,
)
from .experiments.runner import run_rq1, run_rq2, run_rq3, run_rq4, run_all
from .utils.logging import get_logger, set_verbosity, VERBOSITY_QUIET, VERBOSITY_VERBOSE, VERBOSITY_DEBUG

logger = get_logger()


def _normalize_defense_id(name: str) -> str:
    aliases = {
        "gray_code": "grayshield",
        "GrayShield": "grayshield",
        "grayShield": "grayshield",
    }
    return aliases.get(name, name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grayshield",
        description="GrayShield: Gray-Code-Guided Bit-Level Sanitization for Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    p.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for verbose, -vv for debug)"
    )
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (errors only)"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        """Add common experiment arguments."""
        sp.add_argument(
            "--model", dest="model_preset", required=True,
            choices=[
                # Text models
                "bert_imdb", "bert_sst2", "distilbert_sst2", "roberta_sentiment",
                # Vision models
                "vit_cifar10", "swin_cifar10", "vit_imagenet", "swin_imagenet"
            ],
            help="Model preset to use"
        )
        sp.add_argument(
            "--task", required=True,
            choices=["sst2", "imdb", "cifar10"],
            help="Task/dataset to use"
        )
        sp.add_argument(
            "--payload_path", required=True,
            help="Path to payload file"
        )
        sp.add_argument(
            "--x", type=int, default=4,
            choices=list(range(1, 24)),  # 1-23 (float32 mantissa = 23 bits)
            help="Number of LSBs to use (default: 4). Paper presets: 4(HBLA), 8, 12(stealthy), 16(HMLA), 20, 21, 23(FMLA)"
        )
        sp.add_argument(
            "--mode", dest="target_mode", default="encoder_only",
            choices=["attention", "ffn", "embeddings", "encoder_only", "all", "full_model"],
            help="Target parameter selection mode (default: encoder_only). "
                 "Options: attention, ffn, embeddings, encoder_only, all (deprecated), full_model"
        )
        sp.add_argument(
            "--layer_range", type=str, default=None,
            help='Layer range: "0,3" for layers 0-3, or alias "early"/"mid"/"late"'
        )
        sp.add_argument(
            "--n_eval", type=int, default=2048,
            help="Number of evaluation samples (default: 2048, ignored if --full_eval)"
        )
        sp.add_argument(
            "--full_eval", action="store_true",
            help="Use full validation/test split instead of n_eval sampling"
        )
        sp.add_argument(
            "--batch_size", type=int, default=16,
            help="Batch size for evaluation (default: 16)"
        )
        # Seed configuration for reproducibility
        sp.add_argument(
            "--seed", type=int, default=42,
            help="Legacy seed (sets both eval_seed and run_seed, default: 42)"
        )
        sp.add_argument(
            "--eval_seed", type=int, default=None,
            help="Seed for eval data sampling (fixed across runs for paired comparison)"
        )
        sp.add_argument(
            "--run_seed", type=int, default=None,
            help="Seed for stochastic ops (defense, etc.) - vary this for mean±std"
        )
        sp.add_argument(
            "--device", type=str, default="cuda",
            choices=["cpu", "cuda"],
            help="Device to use (default: cuda)"
        )
        sp.add_argument(
            "--output_dir", type=str, default=None,
            help="Output directory for results (default: auto-generated timestamp dir)"
        )

    def add_defense(sp):
        """Add defense-specific arguments."""
        sp.add_argument(
            "--defense", default="random",
            choices=["random", "pattern", "gaussian", "finetune", "ptq", "swp", "grayshield"],
            help="Defense type (default: random). Options: random, pattern, gaussian, finetune, ptq, swp, grayshield"
        )
        # RandomFlip parameters
        sp.add_argument(
            "--flip_prob", type=float, default=0.1,
            help="Flip probability for random defense (default: 0.1)"
        )
        # Pattern parameters (appendix only)
        sp.add_argument(
            "--pattern", type=str, default=None,
            help="Binary pattern for pattern defense (e.g., '00' for x=2)"
        )
        # GaussianNoise parameters
        sp.add_argument(
            "--sigma", type=float, default=1e-5,
            help="Sigma (std dev) for Gaussian noise defense (default: 1e-5)"
        )
        # FineTune parameters
        sp.add_argument(
            "--finetune_steps", type=int, default=100,
            help="Number of fine-tuning steps (default: 100)"
        )
        sp.add_argument(
            "--finetune_lr", type=float, default=1e-5,
            help="Learning rate for fine-tuning (default: 1e-5)"
        )
        sp.add_argument(
            "--finetune_samples", type=int, default=256,
            help="Number of samples for fine-tuning (default: 256)"
        )
        sp.add_argument(
            "--ptq_calibration_samples", type=int, default=256,
            help="Number of calibration samples for PTQ (default: 256)"
        )
        sp.add_argument(
            "--ptq_calibration_batches", type=int, default=8,
            help="Maximum number of calibration batches for PTQ (default: 8)"
        )
        sp.add_argument(
            "--swp_fraction", type=float, default=0.20,
            help="Fraction of low-magnitude weights perturbed by SWP (default: 0.20)"
        )
    # RQ1 subcommand
    s1 = sub.add_parser("rq1", help="RQ1: Injection Feasibility")
    add_common(s1)

    # RQ2 subcommand
    s2 = sub.add_parser("rq2", help="RQ2: Defense Effectiveness")
    add_common(s2)
    add_defense(s2)
    s2.add_argument(
        "--attacker_variant", type=str, default="naive",
        choices=["naive", "repeat3", "repeat5", "interleave", "rs"],
        help="Attacker encoding strategy (default: naive)"
    )

    # RQ3 subcommand
    s3 = sub.add_parser("rq3", help="RQ3: Strategy Comparison")
    add_common(s3)
    s3.add_argument(
        "--attacker_variants", type=str, default="naive",
        help="Comma-separated attacker variants to test (default: naive). Options: naive,repeat3,repeat5,interleave,rs"
    )
    s3.add_argument(
        "--defenses", type=str, default="random,pattern,gaussian,finetune,ptq,swp,grayshield",
        help="Comma-separated defense methods to test"
    )

    # RQ4 subcommand (Trade-off / Pareto analysis)
    s4 = sub.add_parser("rq4", help="RQ4: Trade-off / Pareto Analysis")
    add_common(s4)
    s4.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing RQ2/RQ3 results to aggregate (default: same as output_dir)"
    )
    s4.add_argument(
        "--generate_table", action="store_true",
        help="Generate operating points table (RQ4-Table1)"
    )

    # Run all
    sa = sub.add_parser("rq_all", help="Run all RQs")
    add_common(sa)
    add_defense(sa)
    sa.add_argument(
        "--attacker_variants", type=str, default="naive",
        help="Comma-separated attacker variants for RQ3"
    )

    # Download command
    sd = sub.add_parser(
        "download",
        help="Download malware samples from MalwareBazaar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Download malware samples from MalwareBazaar for research.

Examples:
  # Download by single SHA256 hash
  python -m grayshield.cli download --malware-hash 9d5b59d0ee9914baaa0c45b0748126c3bf585fc8380b34270dc03aca9bdcb46e

  # Download multiple samples by hash list
  python -m grayshield.cli download --malware-list HASH1,HASH2,HASH3

  # Download from a file containing hashes (one per line)
  python -m grayshield.cli download --malware-list @hashes.txt

  # Query and download recent samples
  python -m grayshield.cli download --count 5 --file-type exe
        """
    )
    sd.add_argument(
        "--output-dir", dest="output_dir", default=str(MALWARE_DIR),
        help=f"Output directory for downloaded samples (default: {MALWARE_DIR})"
    )
    sd.add_argument(
        "--malware-hash", dest="malware_hash", type=str, default=None,
        help="Download a single sample by SHA256 hash"
    )
    sd.add_argument(
        "--malware-list", dest="malware_list", type=str, default=None,
        help="Download multiple samples: HASH1,HASH2,... or @file.txt"
    )
    sd.add_argument(
        "--count", type=int, default=3,
        help="Number of samples to download when querying (default: 3)"
    )
    sd.add_argument(
        "--file-type", dest="file_type", type=str, default=None,
        help="Filter by file type (e.g., 'exe', 'dll', 'elf')"
    )
    sd.add_argument(
        "--tag", type=str, default=None,
        help="Filter by malware tag (e.g., 'emotet', 'cobalt-strike')"
    )

    return p


def parse_layer_range(s: str | None):
    """Parse layer range string to tuple.

    Supports:
        - Numeric range: "0,3" -> (0, 3)
        - Aliases: "early", "mid", "late" -> mapped from LAYER_RANGE_ALIASES
    """
    if not s:
        return None

    # Check for alias first
    s_lower = s.strip().lower()
    if s_lower in LAYER_RANGE_ALIASES:
        return LAYER_RANGE_ALIASES[s_lower]

    # Parse numeric range
    parts = s.split(",")
    if len(parts) != 2:
        valid_aliases = ", ".join(LAYER_RANGE_ALIASES.keys())
        raise ValueError(
            f"Invalid layer_range: {s}. "
            f"Expected format: 'start,end' or alias ({valid_aliases})"
        )
    return (int(parts[0].strip()), int(parts[1].strip()))


def cmd_download(ns):
    """Handle download command."""
    from .payload.malwarebazaar import download_cli
    download_cli(
        output_dir=ns.output_dir,
        count=ns.count,
        file_type=ns.file_type,
        tag=ns.tag,
        malware_hash=getattr(ns, "malware_hash", None),
        malware_list=getattr(ns, "malware_list", None),
    )


def main():
    parser = build_parser()
    ns = parser.parse_args()

    # Set verbosity
    if ns.quiet:
        set_verbosity(VERBOSITY_QUIET)
    elif ns.verbose >= 2:
        set_verbosity(VERBOSITY_DEBUG)
    elif ns.verbose >= 1:
        set_verbosity(VERBOSITY_VERBOSE)

    # Handle download command separately
    if ns.cmd == "download":
        cmd_download(ns)
        return

    # Parse common arguments
    try:
        layer_range = parse_layer_range(getattr(ns, "layer_range", None))
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Handle seed separation:
    # - If eval_seed/run_seed not specified, inherit from legacy seed
    eval_seed = ns.eval_seed if ns.eval_seed is not None else ns.seed
    run_seed = ns.run_seed if ns.run_seed is not None else ns.seed

    args = ExperimentArgs(
        model_preset=ns.model_preset,
        task=ns.task,
        payload_path=ns.payload_path,
        x=ns.x,
        target_mode=ns.target_mode,
        layer_range=layer_range,
        n_eval=ns.n_eval,
        batch_size=ns.batch_size,
        seed=ns.seed,
        eval_seed=eval_seed,
        run_seed=run_seed,
        full_eval=getattr(ns, "full_eval", False),
        device=ns.device,
    )

    # Build defense args with all new parameters
    pattern = getattr(ns, "pattern", None) or ("0" * ns.x)
    defense_id = _normalize_defense_id(getattr(ns, "defense", "random"))
    dargs = DefenseArgs(
        defense=defense_id,
        flip_prob=getattr(ns, "flip_prob", 0.1),
        pattern=pattern,
        sigma=getattr(ns, "sigma", 1e-5),
        finetune_steps=getattr(ns, "finetune_steps", 100),
        finetune_lr=getattr(ns, "finetune_lr", 1e-5),
        finetune_samples=getattr(ns, "finetune_samples", 256),
        ptq_calibration_samples=getattr(ns, "ptq_calibration_samples", 256),
        ptq_calibration_batches=getattr(ns, "ptq_calibration_batches", 8),
        swp_fraction=getattr(ns, "swp_fraction", 0.20),
    )

    # Get output directory if specified
    out_dir = getattr(ns, "output_dir", None)

    # Run experiment
    try:
        if ns.cmd == "rq1":
            out = run_rq1(args, out_dir=out_dir)
        elif ns.cmd == "rq2":
            attacker_variant = getattr(ns, "attacker_variant", "naive")
            out = run_rq2(args, dargs, out_dir=out_dir, attacker_variant=attacker_variant)
        elif ns.cmd == "rq3":
            # Parse comma-separated attacker variants
            attacker_variants_str = getattr(ns, "attacker_variants", "naive")
            attacker_variants = [v.strip() for v in attacker_variants_str.split(",") if v.strip()]
            # Parse comma-separated defenses
            defenses_str = getattr(ns, "defenses", "random,pattern,gaussian,finetune,ptq,swp,grayshield")
            defenses = [_normalize_defense_id(d.strip()) for d in defenses_str.split(",") if d.strip()]
            out = run_rq3(args, dargs, out_dir=out_dir, attacker_variants=attacker_variants, defenses=defenses)
        elif ns.cmd == "rq4":
            results_dir = getattr(ns, "results_dir", out_dir)
            generate_table = getattr(ns, "generate_table", False)
            out = run_rq4(args, out_dir=out_dir, results_dir=results_dir, generate_table=generate_table)
        elif ns.cmd == "rq_all":
            attacker_variants_str = getattr(ns, "attacker_variants", "naive")
            attacker_variants = [v.strip() for v in attacker_variants_str.split(",") if v.strip()]
            out = run_all(args, dargs, attacker_variants=attacker_variants)
        else:
            logger.error(f"Unknown command: {ns.cmd}")
            sys.exit(1)

        # Report results
        if isinstance(out, dict) and "rq1" in out:
            # run_all result
            for rq in ["rq1", "rq2", "rq3"]:
                if rq in out and "out_dir" in out[rq]:
                    logger.info(f"{rq.upper()} results: {out[rq]['out_dir']}")
        elif "out_dir" in out:
            logger.info(f"Results saved to: {out['out_dir']}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if ns.verbose >= 1:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
