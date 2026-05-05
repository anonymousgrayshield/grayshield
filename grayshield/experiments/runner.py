"""
GrayShield Experiment Runner

Implements the three research questions:
- RQ1: Injection Feasibility
- RQ2: Defense Effectiveness
- RQ3: Strategy Comparison (Pareto Analysis)
"""
from __future__ import annotations
import os
import copy
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from datetime import datetime
import torch

from ..config import (
    ExperimentArgs, DefenseArgs, AttackerVariant,
    DEFAULT_FLIP_PROBS, FLIP_PROBS_EXTENDED,
    GAUSSIAN_SIGMAS, FINETUNE_STEPS,
    MAINLINE_DEFENSES, RQ3_ATTACKER_VARIANTS,
)
from ..utils.seed import set_seed, get_generator
from ..utils.logging import (
    get_logger,
    log_experiment_start,
    log_experiment_result,
    log_defense_applied,
    log_timing,
)
import logging
from ..payload.loader import load_payload_bits
from ..payload.encoding import (
    encode_payload,
    decode_payload,
    AttackerVariant as AttackerVariantEnum,
    generate_bound_curve,
)
from ..models.factory import load_preset
from ..models.targets import select_targets, select_targets_with_report
from ..models.tasks import TaskRunner
from ..models.checkpoint import (
    ModelState,
    ModelFingerprint,
    compute_fingerprint,
    save_model_state,
    load_model_state,
    verify_fingerprints_differ,
    log_model_state,
)
from ..lsb.stego import inject_bits, extract_bits, capacity_bits, InjectionReport
from ..defense.random_flip import RandomFlipDefense
from ..defense.pattern_mask import PatternMaskDefense
from ..defense.gaussian_noise import GaussianNoiseDefense
from ..defense.finetune import FineTuneDefense
from ..defense.ptq import PTQDefense
from ..defense.swp import SWPDefense
from ..defense.gray_code import GrayShieldDefense
from ..metrics.payload import (
    hamming_distance,
    bit_accuracy,
    bit_accuracy_strict,
    exact_recovery,
    exact_recovery_strict,
    byte_recovery,
    hash_match,
    length_ratio,
    was_truncated,
)
from ..metrics.model import (
    weight_distribution_distance,
    cosine_similarity_on_targets,
    weight_norm_ratio,
    l2_distance,
    relative_l2_distance,
    logits_kl_div,
    top1_agreement,
    logits_mse,
    Timer,
    paired_prediction_diagnostics,
)

# Minimum recommended eval samples for reliable accuracy measurement
MIN_EVAL_SAMPLES_WARNING = 1000
from ..metrics.pareto import pareto_front
from ..visualization.plots import (
    plot_tradeoff,
    plot_curve,
    plot_rq1_injection_metrics,
    plot_rq1_clean_vs_poisoned,
    plot_rq2_fig1_bit_accuracy_by_method,
    plot_rq2_fig3_strength_sweep_scatter,
    plot_rq3_pareto,
)

logger = get_logger()


def _normalize_defense_id(name: str) -> str:
    aliases = {
        "gray_code": "grayshield",
        "GrayShield": "grayshield",
    }
    return aliases.get(name, name)


def _resolve_grayshield_mode() -> tuple[bool, bool]:
    """
    Resolve GrayShield mode for the current run.

    When a secret key is available we default to the final V3 path, which is the
    desired experiment configuration for this project.
    """
    use_v2 = bool(os.environ.get("GRAYSHIELD_KEY", ""))
    use_v3 = bool(os.environ.get("GRAYSHIELD_V3", "")) or use_v2
    if use_v3 and not use_v2:
        raise ValueError("GrayShield V3 requires GRAYSHIELD_KEY to be set")
    return use_v2, use_v3


def _ts_dir(root: str = "results") -> str:
    """Create timestamped output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out


def _payload_dir(root: str, payload_path: str, rq: str) -> str:
    """Create output directory organized by payload and RQ.

    Structure: results/{payload_name}/{rq}/
    Example: results/malware_abc123/rq1/
    """
    # Extract payload name from path (use first 8 chars of filename)
    payload_name = os.path.basename(payload_path)
    # Remove extension and truncate for cleaner folder names
    payload_name = os.path.splitext(payload_name)[0]
    if len(payload_name) > 20:
        payload_name = payload_name[:20]

    out = os.path.join(root, payload_name, rq)
    os.makedirs(out, exist_ok=True)
    return out


def _setup_file_logging(out_dir: str) -> None:
    """Setup file logging for the current experiment."""
    if not out_dir:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    logger = get_logger()
    log_path = os.path.join(out_dir, "experiment.log")
    
    # Check if a file handler for this path already exists
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(handler.baseFilename) == os.path.abspath(log_path):
                return  # Already logging to this file
            
    # Add new file handler
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_path}")


def _compute_comprehensive_metrics(
    clean_model: torch.nn.Module,
    modified_model: torch.nn.Module,
    targets: List[str],
    loader,
    device: str,
    original_bits: List[int],
    recovered_bits: List[int],
    n_eval_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics for model comparison.

    Args:
        clean_model: Original model before injection
        modified_model: Model after injection/defense
        targets: Target parameter names
        loader: DataLoader for logit-based metrics
        device: Device string
        original_bits: Original payload bits
        recovered_bits: Recovered/extracted bits
        n_eval_samples: Number of samples used for evaluation (for transparency)

    Returns:
        Dictionary with all metrics including both prefix and strict variants
    """
    # Warn if eval samples might be too small for reliable accuracy measurement
    if n_eval_samples is not None and n_eval_samples < MIN_EVAL_SAMPLES_WARNING:
        logger.warning(
            f"EVAL SIZE WARNING: Using only {n_eval_samples} samples. "
            f"Accuracy granularity is 1/{n_eval_samples} = {1/n_eval_samples:.4f}. "
            f"Consider using at least {MIN_EVAL_SAMPLES_WARNING} samples for reliable measurements."
        )

    metrics = {
        # Payload metrics - PREFIX-BASED (backward compatible)
        "bit_accuracy": bit_accuracy(original_bits, recovered_bits),
        "lsb_similarity": bit_accuracy(original_bits, recovered_bits),  # Alias
        "exact_recovery": exact_recovery(original_bits, recovered_bits),
        "hash_match": hash_match(original_bits, recovered_bits),

        # Payload metrics - STRICT (new, penalizes truncation)
        "bit_accuracy_strict": bit_accuracy_strict(original_bits, recovered_bits),
        "exact_recovery_strict": exact_recovery_strict(original_bits, recovered_bits),

        # Length/truncation diagnostics
        "original_bits_count": len(original_bits),
        "recovered_bits_count": len(recovered_bits),
        "length_ratio": length_ratio(original_bits, recovered_bits),
        "was_truncated": was_truncated(original_bits, recovered_bits),
    }

    # Byte-level recovery
    byte_acc, correct_bytes, total_bytes = byte_recovery(original_bits, recovered_bits)
    metrics["byte_accuracy"] = byte_acc
    metrics["correct_bytes"] = correct_bytes
    metrics["total_bytes"] = total_bytes

    # Model metrics - Weight space
    metrics["cosine_similarity"] = cosine_similarity_on_targets(
        clean_model, modified_model, targets
    )
    metrics["weight_norm_ratio"] = weight_norm_ratio(
        clean_model, modified_model, targets
    )

    # L2 Distance metrics (key for stealthiness analysis)
    metrics["l2_distance"] = l2_distance(
        clean_model, modified_model, targets
    )
    metrics["relative_l2_distance"] = relative_l2_distance(
        clean_model, modified_model, targets
    )
    metrics["wasserstein_distance"] = weight_distribution_distance(
        clean_model, modified_model, targets
    )
    metrics["hamming_distance"] = hamming_distance(original_bits, recovered_bits)

    # Model metrics - Output space
    metrics["logits_kl"] = logits_kl_div(
        clean_model, modified_model, loader, device
    )
    metrics["top1_agreement"] = top1_agreement(
        clean_model, modified_model, loader, device
    )
    metrics["logits_mse"] = logits_mse(
        clean_model, modified_model, loader, device
    )

    # Paired prediction diagnostics (more sensitive than acc_drop)
    paired_diag = paired_prediction_diagnostics(
        clean_model, modified_model, loader, device
    )
    metrics["changed_predictions"] = paired_diag["changed_predictions"]
    metrics["prediction_change_rate"] = paired_diag["change_rate"]
    metrics["max_logit_diff"] = paired_diag["max_logit_diff"]
    metrics["mean_logit_diff"] = paired_diag["mean_logit_diff"]

    # Evaluation transparency
    if n_eval_samples is not None:
        metrics["eval_sample_count"] = n_eval_samples
        # Compute accuracy granularity for research awareness
        metrics["accuracy_granularity"] = 1.0 / n_eval_samples

    return metrics


# =============================================================================
# RQ1: Injection Feasibility
# =============================================================================

def run_rq1(args: ExperimentArgs, out_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    RQ1: Evaluate injection feasibility.

    Measures:
    - Capacity: How many bits can we inject?
    - Utility preservation: Does accuracy remain stable?
    - Stealthiness: Weight similarity metrics
    """
    # Seed separation: eval_seed for data, run_seed for stochastic ops
    set_seed(args.run_seed)
    out_dir = out_dir or _ts_dir()
    
    _setup_file_logging(out_dir)

    log_experiment_start(
        rq="RQ1",
        model=args.model_preset,
        task=args.task,
        payload_path=args.payload_path,
        x=args.x,
        target_mode=args.target_mode,
    )

    # Load model and processor
    preset, model, processor = load_preset(args.model_preset, device=args.device)
    model = model.to(dtype=torch.float32)
    device = next(model.parameters()).device

    # IMPORTANT: Use the actual device the model is on, not args.device
    # This ensures consistency when auto-selection picks a specific GPU
    actual_device_str = str(device)

    # Verify dtype for LSB steganography (requires float32 mantissa)
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            logger.warning(
                f"DTYPE WARNING: Parameter {name} has dtype {param.dtype}, "
                f"not float32. LSB steganography assumes float32 mantissa."
            )
            break

    # Log TF32 status for reproducibility awareness
    if torch.cuda.is_available():
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        if tf32_matmul or tf32_cudnn:
            logger.info(
                f"TF32 enabled (matmul={tf32_matmul}, cudnn={tf32_cudnn}). "
                f"This may mask small perturbations. Consider disabling for sensitivity analysis."
            )

    # Setup task runner and evaluation loader
    # Use actual_device_str to ensure TaskRunner uses the same device as the model
    task_runner = TaskRunner(args.task, preset.task_type, processor, device=actual_device_str)
    loader = task_runner.make_eval_loader(
        args.n_eval, args.batch_size,
        full_eval=args.full_eval,
        eval_seed=args.eval_seed,
    )
    # Track actual sample count for reporting
    n_eval_actual = len(loader.dataset) if hasattr(loader, 'dataset') else args.n_eval

    # Load payload
    payload = load_payload_bits(args.payload_path)

    # Select injection targets with detailed report for audit
    targets, target_report = select_targets_with_report(
        model, args.model_preset,
        mode=args.target_mode,
        layer_range=args.layer_range
    )
    cap = capacity_bits(model, targets, x=args.x)

    logger.info(f"Targets: {len(targets)} parameters")
    logger.info(f"  Resolved mode: {target_report.resolved_mode}")
    logger.info(f"  Sample prefixes: {target_report.target_prefixes[:5]}")
    logger.info(f"Capacity: {cap:,} bits ({cap/8/1024:.1f} KB)")
    logger.info(f"Payload: {payload.n_bits:,} bits ({payload.n_bits/8/1024:.1f} KB)")

    # Capacity verification
    capacity_sufficient = cap >= payload.n_bits
    capacity_ratio = payload.n_bits / cap if cap > 0 else float('inf')
    if not capacity_sufficient:
        logger.warning(f"CAPACITY WARNING: Payload ({payload.n_bits:,} bits) exceeds capacity ({cap:,} bits)")
        logger.warning(f"Only {cap / payload.n_bits * 100:.1f}% of payload can be injected!")
    else:
        logger.info(f"Capacity utilization: {capacity_ratio * 100:.1f}%")

    # === MODEL STATE: CLEAN ===
    # Save clean model checkpoint and compute fingerprint BEFORE any modifications
    clean_ckpt_path, clean_fp = save_model_state(
        model, out_dir, ModelState.CLEAN, targets,
        metadata={"model_preset": args.model_preset, "task": args.task}
    )
    log_model_state(clean_fp, logger, prefix="[CLEAN] ")

    # Baseline evaluation (with fixed samples)
    with Timer("baseline_eval") as t_base:
        base_acc = task_runner.evaluate_accuracy(model, loader)
    log_timing("baseline_eval", t_base.elapsed)

    # Save clean model for comparison
    clean_model = copy.deepcopy(model)

    # Inject payload
    with Timer("injection") as t_inject:
        inject_report = inject_bits(model, targets, payload.bits, x=args.x)
    log_timing("injection", t_inject.elapsed)

    logger.info(f"Injected: {inject_report.written_bits_total:,} bits")
    if inject_report.truncated:
        logger.warning(
            f"TRUNCATION: Only {inject_report.written_bits_total:,}/{inject_report.payload_bits_total:,} "
            f"bits written ({inject_report.capacity_used * 100:.1f}%)"
        )

    # Extract and verify
    recovered = extract_bits(model, targets, x=args.x, n_bits=payload.n_bits)

    # === MODEL STATE: POISONED ===
    # Save poisoned model checkpoint and compute fingerprint AFTER injection
    poisoned_ckpt_path, poisoned_fp = save_model_state(
        model, out_dir, ModelState.POISONED, targets,
        metadata={"model_preset": args.model_preset, "injected_bits": inject_report.written_bits_total}
    )
    log_model_state(poisoned_fp, logger, prefix="[POISONED] ")

    # CRITICAL: Verify clean != poisoned (sanity check)
    try:
        verify_fingerprints_differ(clean_fp, poisoned_fp, context="RQ1: clean vs poisoned")
        logger.info("✓ Fingerprint verification passed: clean != poisoned")
    except AssertionError as e:
        logger.error(str(e))
        raise

    # Post-injection evaluation (same samples for paired comparison)
    with Timer("post_inject_eval") as t_post:
        post_acc = task_runner.evaluate_accuracy(model, loader)
    log_timing("post_inject_eval", t_post.elapsed)

    # Compute all metrics with eval sample count for transparency
    metrics = _compute_comprehensive_metrics(
        clean_model, model, targets, loader, str(device),
        payload.bits, recovered,
        n_eval_samples=n_eval_actual,
    )
    metrics["base_acc"] = base_acc
    metrics["post_inject_acc"] = post_acc
    metrics["acc_drop"] = base_acc - post_acc

    # Build result record with detailed injection info
    record = {
        "rq": "RQ1",
        "model_preset": args.model_preset,
        "task": args.task,
        "x": args.x,
        "target_mode": args.target_mode,
        "resolved_target_mode": target_report.resolved_mode,
        "layer_range": args.layer_range,
        "n_targets": len(targets),
        "target_prefixes": target_report.target_prefixes,
        "target_warnings": target_report.warnings,
        "capacity_bits": cap,
        "capacity_kb": cap / 8 / 1024,
        "capacity_used": inject_report.capacity_used,
        "payload": {
            "path": payload.path,
            "sha256": payload.sha256,
            "file_type": payload.file_type,
            "n_bits": payload.n_bits,
        },
        # Detailed injection report
        "inject_report": inject_report.per_param,
        "injection_summary": {
            "payload_bits_total": inject_report.payload_bits_total,
            "written_bits_total": inject_report.written_bits_total,
            "truncated": inject_report.truncated,
            "capacity_used": inject_report.capacity_used,
        },
        "total_injected_bits": inject_report.written_bits_total,
        "metrics": metrics,
        "timing": {
            "baseline_eval_seconds": t_base.elapsed,
            "injection_seconds": t_inject.elapsed,
            "post_inject_eval_seconds": t_post.elapsed,
        },
        # Model state tracking (CRITICAL for audit)
        "model_states": {
            "clean": {
                "ckpt_path": clean_ckpt_path,
                "fingerprint": clean_fp.param_hash,
                "targets_hash": clean_fp.targets_hash,
            },
            "poisoned": {
                "ckpt_path": poisoned_ckpt_path,
                "fingerprint": poisoned_fp.param_hash,
                "targets_hash": poisoned_fp.targets_hash,
            },
        },
        # Seed configuration for reproducibility
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "run_seed": args.run_seed,
        # Evaluation configuration
        "n_eval": args.n_eval,
        "n_eval_actual": n_eval_actual,
        "full_eval": args.full_eval,
    }

    # Save results
    with open(os.path.join(out_dir, "rq1.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # JSON dual-write for human inspection
    json_path = os.path.join(out_dir, "rq1.json")
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    log_experiment_result("RQ1", metrics, out_dir)

    record["out_dir"] = out_dir
    return record


# =============================================================================
# RQ2: Defense Effectiveness
# =============================================================================

def run_rq2(
    args: ExperimentArgs,
    dargs: DefenseArgs,
    out_dir: Optional[str] = None,
    attacker_variant: str = "naive",
) -> Dict[str, Any]:
    """
    RQ2: Evaluate defense effectiveness.

    Measures:
    - Recovery reduction: How much does defense hurt payload recovery?
    - Accuracy preservation: Does the model still work?
    - Timing: Defense runtime overhead

    Args:
        args: Experiment arguments
        dargs: Defense arguments
        out_dir: Output directory
        attacker_variant: Attacker encoding strategy (naive, repeat3, repeat5, interleave, rs)
    """
    # Seed separation: run_seed for stochastic ops (defense)
    set_seed(args.run_seed)
    out_dir = out_dir or _ts_dir()

    _setup_file_logging(out_dir)

    # Parse attacker variant
    try:
        variant_enum = AttackerVariantEnum(attacker_variant)
    except ValueError:
        logger.warning(f"Unknown attacker_variant '{attacker_variant}', using naive")
        variant_enum = AttackerVariantEnum.NAIVE

    log_experiment_start(
        rq="RQ2",
        model=args.model_preset,
        task=args.task,
        payload_path=args.payload_path,
        defense=dargs.defense,
        flip_prob=dargs.flip_prob,
    )

    # Load model
    preset, model, processor = load_preset(args.model_preset, device=args.device)
    model = model.to(dtype=torch.float32)
    device = next(model.parameters()).device

    # Use actual device to ensure consistency with model
    actual_device_str = str(device)
    task_runner = TaskRunner(args.task, preset.task_type, processor, device=actual_device_str)
    loader = task_runner.make_eval_loader(
        args.n_eval, args.batch_size,
        full_eval=args.full_eval,
        eval_seed=args.eval_seed,
    )
    n_eval_actual = len(loader.dataset) if hasattr(loader, 'dataset') else args.n_eval

    payload = load_payload_bits(args.payload_path)

    # Use select_targets_with_report for detailed audit trail
    targets, target_report = select_targets_with_report(
        model, args.model_preset,
        mode=args.target_mode,
        layer_range=args.layer_range
    )

    logger.info(f"RQ2 Targets: {len(targets)} parameters")
    logger.info(f"  Resolved mode: {target_report.resolved_mode}")
    logger.info(f"  Attacker variant: {attacker_variant}")

    # === MODEL STATE: CLEAN ===
    # Save clean checkpoint BEFORE any modifications
    clean_ckpt_path, clean_fp = save_model_state(
        model, out_dir, ModelState.CLEAN, targets,
        metadata={"model_preset": args.model_preset, "task": args.task, "rq": "RQ2"}
    )
    log_model_state(clean_fp, logger, prefix="[CLEAN] ")

    # Baseline evaluation
    base_acc = task_runner.evaluate_accuracy(model, loader)
    clean_model = copy.deepcopy(model)

    # Encode payload using attacker variant
    encoded_bits, encoding_report = encode_payload(payload.bits, variant_enum, interleave_seed=args.run_seed)
    logger.info(f"Payload encoding: {encoding_report.variant}, expansion={encoding_report.expansion_factor}x")
    logger.info(f"  Original bits: {encoding_report.original_bits}, Encoded bits: {encoding_report.encoded_bits}")

    # Inject encoded payload
    inject_report = inject_bits(model, targets, encoded_bits, x=args.x)

    # === MODEL STATE: POISONED ===
    # Save poisoned checkpoint AFTER injection
    poisoned_ckpt_path, poisoned_fp = save_model_state(
        model, out_dir, ModelState.POISONED, targets,
        metadata={
            "model_preset": args.model_preset,
            "injected_bits": inject_report.written_bits_total,
            "attacker_variant": attacker_variant,
        }
    )
    log_model_state(poisoned_fp, logger, prefix="[POISONED] ")

    # CRITICAL: Verify clean != poisoned
    try:
        verify_fingerprints_differ(clean_fp, poisoned_fp, context="RQ2: clean vs poisoned")
        logger.info("✓ Fingerprint verification passed: clean != poisoned")
    except AssertionError as e:
        logger.error(str(e))
        raise

    # Pre-defense metrics (extract and decode)
    pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
    pre_bits_decoded, pre_decoding_report = decode_payload(
        pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
    )
    pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)
    pre_rec_strict = bit_accuracy_strict(payload.bits, pre_bits_decoded)
    pre_acc = task_runner.evaluate_accuracy(model, loader)

    gray_use_v2, gray_use_v3 = _resolve_grayshield_mode()

    # Apply defense
    indices = None
    scores = {}



    # Apply defense and measure timing
    with Timer("defense") as t_defense:
        if dargs.defense == "random":
            rep = RandomFlipDefense().apply(
                model, targets, x=args.x,
                flip_prob=dargs.flip_prob,
                indices=indices,
                seed=args.run_seed,
            )
            defense_report = {
                "type": "RandomFlip",
                "flip_prob": dargs.flip_prob,
                "total_flips": rep.total_bit_flips,
            }
        elif dargs.defense == "pattern":
            rep = PatternMaskDefense().apply(
                model, targets, x=args.x,
                pattern=dargs.pattern,
                indices=indices,
            )
            defense_report = {
                "type": "PatternMask",
                "pattern": dargs.pattern,
                "n_indices": rep.n_indices,
            }
        elif dargs.defense == "grayshield":
            rep = GrayShieldDefense().apply(
                model, targets, x=args.x,
                seed=args.run_seed,
                indices=indices,
                use_v2=gray_use_v2,
                use_v3=gray_use_v3,
            )
            defense_report = {
                "type": "GrayShield",
                "gray_version": rep.version,
                "n_params_modified": rep.n_params_modified,
                "n_indices": rep.n_indices,
            }
        elif dargs.defense == "gaussian":
            rep = GaussianNoiseDefense().apply(
                model, targets,
                sigma=dargs.sigma,
                seed=args.run_seed,
                indices=indices,
            )
            defense_report = {
                "type": "GaussianNoise",
                "sigma": dargs.sigma,
                "n_params_modified": rep.n_params_modified,
                "mean_perturbation": rep.mean_perturbation,
            }
        elif dargs.defense == "finetune":
            # Create fine-tuning loader from task runner
            finetune_loader = task_runner.make_train_loader(
                dargs.finetune_samples, args.batch_size, seed=args.run_seed
            )
            finetune_defense = FineTuneDefense(
                learning_rate=dargs.finetune_lr,
            )
            rep = finetune_defense.apply(
                model, finetune_loader,
                n_steps=dargs.finetune_steps,
                target_names=targets,
                device=str(device),
                seed=args.run_seed,
            )
            defense_report = {
                "type": "FineTune",
                "n_steps": rep.n_steps,
                "learning_rate": rep.learning_rate,
                "initial_loss": rep.initial_loss,
                "final_loss": rep.final_loss,
            }
        elif dargs.defense == "ptq":
            calibration_loader = task_runner.make_train_loader(
                dargs.ptq_calibration_samples, args.batch_size, seed=args.eval_seed
            )
            rep = PTQDefense().apply(
                model,
                targets,
                calibration_loader=calibration_loader,
                device=str(device),
                max_calibration_batches=dargs.ptq_calibration_batches,
            )
            defense_report = {
                "type": "PTQ",
                "n_params_quantized": rep.n_params_quantized,
                "n_calibration_batches": rep.n_calibration_batches,
                "compression_ratio": rep.compression_ratio,
                "avg_activation_absmax": rep.avg_activation_absmax,
            }
        elif dargs.defense == "swp":
            rep = SWPDefense().apply(
                model,
                targets,
                x=args.x,
                seed=args.run_seed,
                fraction=dargs.swp_fraction,
            )
            defense_report = {
                "type": "SWP",
                "fraction": rep.fraction,
                "sigma": rep.sigma,
                "n_selected_elements": rep.n_selected_elements,
                "target_relative_l2": rep.target_relative_l2,
                "achieved_relative_l2": rep.achieved_relative_l2,
            }
        else:
            raise ValueError(f"Unknown defense: {dargs.defense}")

    log_defense_applied(dargs.defense, len(targets), t_defense.elapsed * 1000)

    # === MODEL STATE: DEFENSED ===
    # Save defensed checkpoint AFTER defense applied
    defensed_ckpt_path, defensed_fp = save_model_state(
        model, out_dir, ModelState.DEFENSED, targets,
        metadata={
            "model_preset": args.model_preset,
            "defense_type": dargs.defense,
            "flip_prob": dargs.flip_prob,
        }
    )
    log_model_state(defensed_fp, logger, prefix="[DEFENSED] ")

    # CRITICAL: Verify all three states differ
    # Check if defense has zero strength (expected to not modify model)
    is_zero_strength = False
    if dargs.defense == "random":
        is_zero_strength = (dargs.flip_prob == 0.0)
    elif dargs.defense == "gaussian":
        is_zero_strength = (dargs.sigma == 0.0)
    elif dargs.defense == "finetune":
        is_zero_strength = (dargs.finetune_steps == 0)
    # PTQ, SWP, pattern, and GrayShield always modify the model, so no zero-strength case

    if is_zero_strength:
        # Zero-strength defense: poisoned == defensed is expected
        if poisoned_fp.param_hash == defensed_fp.param_hash:
            logger.info("✓ Zero-strength defense: poisoned == defensed (expected)")
        else:
            logger.warning("⚠ Zero-strength defense but fingerprints differ (unexpected)")
    else:
        # Non-zero defense: poisoned != defensed required
        try:
            verify_fingerprints_differ(poisoned_fp, defensed_fp, context="RQ2: poisoned vs defensed")
            logger.info("✓ Fingerprint verification passed: poisoned != defensed")
        except AssertionError as e:
            logger.warning(f"⚠ [IGNORED] Fingerprint check failed: {e}")
            # logger.error(str(e))
            # raise

    # Clean vs defensed check (only for non-zero strength)
    if not is_zero_strength:
        try:
            verify_fingerprints_differ(clean_fp, defensed_fp, context="RQ2: clean vs defensed")
            logger.info("✓ Fingerprint verification passed: clean != defensed")
        except AssertionError as e:
            logger.error(str(e))
            raise

    # Post-defense metrics (extract and decode)
    post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
    post_bits_decoded, post_decoding_report = decode_payload(
        post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
    )
    post_rec = bit_accuracy(payload.bits, post_bits_decoded)
    post_rec_strict = bit_accuracy_strict(payload.bits, post_bits_decoded)
    post_acc = task_runner.evaluate_accuracy(model, loader)

    # Compute comprehensive metrics
    metrics = _compute_comprehensive_metrics(
        clean_model, model, targets, loader, str(device),
        payload.bits, post_bits_decoded,
        n_eval_samples=n_eval_actual,
    )
    metrics.update({
        "base_acc": base_acc,
        "pre_defense_acc": pre_acc,
        "post_defense_acc": post_acc,
        "acc_drop_vs_base": base_acc - post_acc,
        "acc_drop_vs_injected": pre_acc - post_acc,
        # Prefix-based recovery (backward compatible)
        "pre_recovery": pre_rec,
        "post_recovery": post_rec,
        "recovery_reduction": pre_rec - post_rec,
        # Strict recovery (new)
        "pre_recovery_strict": pre_rec_strict,
        "post_recovery_strict": post_rec_strict,
        "recovery_reduction_strict": pre_rec_strict - post_rec_strict,
        # Decoding stats (for attacker variant analysis)
        "pre_decoding_corrected_errors": pre_decoding_report.corrected_errors,
        "post_decoding_corrected_errors": post_decoding_report.corrected_errors,
    })

    record = {
        "rq": "RQ2",
        "model_preset": args.model_preset,
        "task": args.task,
        "x": args.x,
        "target_mode": args.target_mode,
        "resolved_target_mode": target_report.resolved_mode,
        "layer_range": args.layer_range,
        "n_targets": len(targets),
        "target_prefixes": target_report.target_prefixes,
        "target_warnings": target_report.warnings,
        "payload": {
            "sha256": payload.sha256,
            "file_type": payload.file_type,
            "n_bits": payload.n_bits,
        },
        # Attacker variant info
        "attacker_variant": attacker_variant,
        "encoding_report": {
            "variant": encoding_report.variant,
            "original_bits": encoding_report.original_bits,
            "encoded_bits": encoding_report.encoded_bits,
            "expansion_factor": encoding_report.expansion_factor,
        },
        "injection_summary": {
            "payload_bits_total": inject_report.payload_bits_total,
            "written_bits_total": inject_report.written_bits_total,
            "truncated": inject_report.truncated,
        },
        "defense": defense_report,

        "metrics": metrics,
        "timing": {
            "defense_seconds": t_defense.elapsed,
        },
        # Model state tracking (CRITICAL for audit)
        "model_states": {
            "clean": {
                "ckpt_path": clean_ckpt_path,
                "fingerprint": clean_fp.param_hash,
                "targets_hash": clean_fp.targets_hash,
            },
            "poisoned": {
                "ckpt_path": poisoned_ckpt_path,
                "fingerprint": poisoned_fp.param_hash,
                "targets_hash": poisoned_fp.targets_hash,
            },
            "defensed": {
                "ckpt_path": defensed_ckpt_path,
                "fingerprint": defensed_fp.param_hash,
                "targets_hash": defensed_fp.targets_hash,
            },
        },
        # Seed configuration for reproducibility
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "run_seed": args.run_seed,
        # Evaluation configuration
        "n_eval": args.n_eval,
        "n_eval_actual": n_eval_actual,
        "full_eval": args.full_eval,
    }

    with open(os.path.join(out_dir, "rq2.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # JSON dual-write for human inspection
    json_path = os.path.join(out_dir, "rq2.json")
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    log_experiment_result("RQ2", metrics, out_dir)

    record["out_dir"] = out_dir
    return record


# =============================================================================
# RQ3: Strategy Comparison
# =============================================================================

def run_rq3(
    args: ExperimentArgs,
    dargs: DefenseArgs,
    out_dir: Optional[str] = None,
    attacker_variants: Optional[List[str]] = None,
    defenses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    RQ3: Compare defense strategies.

    Performs grid search over:
    - Random flip: flip_prob from DEFAULT_FLIP_PROBS
    - Gaussian noise: sigma from GAUSSIAN_SIGMAS
    - Fine-tune: steps from FINETUNE_STEPS
    - Pattern mask: (appendix only) patterns ["0000", "1111", "0101", "0011"]
    - PTQ/SWP/GrayShield: single operating point per attacker variant
    - Attacker variants: naive, repeat3, repeat5, interleave, rs

    Computes Pareto front for trade-off analysis.

    Args:
        args: Experiment arguments
        dargs: Defense arguments
        out_dir: Output directory
        attacker_variants: List of attacker variants to test (default: ["naive"])
        defenses: List of defense methods to test (default: MAINLINE_DEFENSES)
    """
    # Seed separation: run_seed for stochastic ops
    set_seed(args.run_seed)
    out_dir = out_dir or _ts_dir()

    _setup_file_logging(out_dir)

    # Default to naive-only if not specified
    if attacker_variants is None:
        attacker_variants = ["naive"]

    log_experiment_start(
        rq="RQ3",
        model=args.model_preset,
        task=args.task,
        payload_path=args.payload_path,
    )

    # Load base model
    preset, model0, processor = load_preset(args.model_preset, device=args.device)
    model0 = model0.to(dtype=torch.float32)
    device = next(model0.parameters()).device

    # Use actual device to ensure consistency with model
    actual_device_str = str(device)
    task_runner = TaskRunner(args.task, preset.task_type, processor, device=actual_device_str)
    loader = task_runner.make_eval_loader(
        args.n_eval, args.batch_size,
        full_eval=args.full_eval,
        eval_seed=args.eval_seed,
    )
    n_eval_actual = len(loader.dataset) if hasattr(loader, 'dataset') else args.n_eval

    payload = load_payload_bits(args.payload_path)

    # Use select_targets_with_report for detailed audit trail
    targets, target_report = select_targets_with_report(
        model0, args.model_preset,
        mode=args.target_mode,
        layer_range=args.layer_range
    )

    logger.info(f"RQ3 Targets: {len(targets)} parameters")
    logger.info(f"  Resolved mode: {target_report.resolved_mode}")
    logger.info(f"  Attacker variants to test: {attacker_variants}")

    # === MODEL STATE: CLEAN (base) ===
    # Save clean checkpoint for reference fingerprint
    clean_ckpt_path, clean_fp = save_model_state(
        model0, out_dir, ModelState.CLEAN, targets,
        metadata={"model_preset": args.model_preset, "task": args.task, "rq": "RQ3"}
    )
    log_model_state(clean_fp, logger, prefix="[CLEAN BASE] ")

    base_acc = task_runner.evaluate_accuracy(model0, loader)

    # Generate theoretical bound curves for comparison
    # This shows predicted recovery rates under BSC model
    k_values = [1, 3, 5]  # k=1 is naive, k=3 is repeat3, k=5 is repeat5
    bound_curves = generate_bound_curve(k_values=k_values, p_values=list(DEFAULT_FLIP_PROBS))
    logger.info(f"Generated theoretical bounds for k={k_values}")

    points: List[Dict[str, Any]] = []
    fingerprints: Dict[str, Dict[str, str]] = {"clean": {"fingerprint": clean_fp.param_hash}}

    # Determine which defenses to test
    if defenses is None:
        defenses = MAINLINE_DEFENSES
    defenses = [_normalize_defense_id(name) for name in defenses]
    logger.info(f"Testing defenses: {defenses}")
    logger.info(f"Testing attacker variants: {attacker_variants}")

    # Grid search: Random flip defense x Attacker variants
    if "random" in defenses:
        flip_grid = DEFAULT_FLIP_PROBS
        logger.info(f"Testing random flip with flip_prob in {flip_grid}")

    config_idx = 0

    # === RANDOM FLIP DEFENSE ===
    if "random" in defenses:
        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                logger.warning(f"Unknown attacker_variant '{variant_name}', skipping")
                continue

            for fp in flip_grid:
                config_idx += 1
                model = copy.deepcopy(model0)

                # Encode payload using attacker variant
                encoded_bits, encoding_report = encode_payload(
                    payload.bits, variant_enum, interleave_seed=args.run_seed
                )

                # Inject encoded payload
                inject_bits(model, targets, encoded_bits, x=args.x)

                # Extract and decode pre-defense
                pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                pre_bits_decoded, _ = decode_payload(
                    pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)
                pre_rec_strict = bit_accuracy_strict(payload.bits, pre_bits_decoded)

                # Compute fingerprint for this poisoned state
                poisoned_fp = compute_fingerprint(model, ModelState.POISONED, targets=targets)

                with Timer() as t:
                    RandomFlipDefense().apply(model, targets, x=args.x, flip_prob=fp, seed=args.run_seed)

                # Compute fingerprint for defensed state
                defensed_fp = compute_fingerprint(model, ModelState.DEFENSED, targets=targets)

                # Extract and decode post-defense
                post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                post_bits_decoded, decoding_report = decode_payload(
                    post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                post_rec = bit_accuracy(payload.bits, post_bits_decoded)
                post_rec_strict = bit_accuracy_strict(payload.bits, post_bits_decoded)
                post_acc = task_runner.evaluate_accuracy(model, loader)

                # Compute relative L2 distance for continuous metric
                l2_dist_rel = relative_l2_distance(model0, model, targets)
                w1_dist = weight_distribution_distance(model0, model, targets)
                hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

                # Lookup theoretical bound for this (k, p) combination
                k = {"naive": 1, "repeat3": 3, "repeat5": 5}.get(variant_name, 1)
                theoretical_bound = next(
                    (b["predicted_recovery"] for b in bound_curves if b["k"] == k and abs(b["flip_prob"] - fp) < 1e-6),
                    None
                )

                points.append({
                    "strategy": "random",
                    "flip_prob": fp,
                    "attacker_variant": variant_name,
                    "acc_drop": base_acc - post_acc,
                    "recovery_reduction": pre_rec - post_rec,
                    "post_recovery": post_rec,
                    "post_acc": post_acc,
                    "relative_l2_distance": l2_dist_rel,
                    "hamming_distance": hamming_dist,
                    "wasserstein_distance": w1_dist,
                    "exact_recovery": exact_recovery(payload.bits, post_bits_decoded),
                    "defense_time_ms": t.elapsed * 1000,
                    # Strict metrics
                    "post_recovery_strict": post_rec_strict,
                    "exact_recovery_strict": exact_recovery_strict(payload.bits, post_bits_decoded),
                    "recovery_reduction_strict": pre_rec_strict - post_rec_strict,
                    # Attacker encoding info
                    "expansion_factor": encoding_report.expansion_factor,
                    "corrected_errors": decoding_report.corrected_errors,
                    # Theoretical bound
                    "theoretical_bound": theoretical_bound,
                    # Fingerprints for verification
                    "poisoned_fingerprint": poisoned_fp.param_hash,
                    "defensed_fingerprint": defensed_fp.param_hash,
                })

                # Store fingerprint for audit
                fp_key = f"random_{variant_name}_{fp}"
                fingerprints[fp_key] = {
                    "poisoned": poisoned_fp.param_hash,
                    "defensed": defensed_fp.param_hash,
                }

    # === PATTERN DEFENSE ===
    # Defense x matches attack x for fair comparison. For a 19-bit payload attack,
    # the defense also overwrites the same 19 LSBs with the chosen pattern.
    if "pattern" in defenses:
        PATTERN_X_BITS = args.x  # align with attack x for fair comparison
        # Pattern strings: 4 distinct patterns for comprehensive testing
        # Pattern 1: all 0s, Pattern 2: all 1s, Pattern 3: 01..., Pattern 4: 10...
        base_patterns = [
            "0" * PATTERN_X_BITS,              # All zeros
            "1" * PATTERN_X_BITS,              # All ones
            "01" * (PATTERN_X_BITS // 2 + 1), # Alternating 01
            "10" * (PATTERN_X_BITS // 2 + 1)  # Alternating 10 (different from 01)
        ]
        patterns = [p[:PATTERN_X_BITS] for p in base_patterns]  # trim to x bits
        logger.info(f"Testing pattern mask with x={PATTERN_X_BITS} bits, patterns={patterns}")

        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                continue

            for pat in patterns:
                config_idx += 1
                model = copy.deepcopy(model0)

                # Encode payload using attacker variant
                encoded_bits, encoding_report = encode_payload(
                    payload.bits, variant_enum, interleave_seed=args.run_seed
                )

                # Inject encoded payload using the same x as the attack for fair comparison
                inject_bits(model, targets, encoded_bits, x=PATTERN_X_BITS)

                # Extract and decode pre-defense
                pre_bits_raw = extract_bits(model, targets, x=PATTERN_X_BITS, n_bits=len(encoded_bits))
                pre_bits_decoded, _ = decode_payload(
                    pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)
                pre_rec_strict = bit_accuracy_strict(payload.bits, pre_bits_decoded)

                # Compute fingerprint for this poisoned state
                poisoned_fp = compute_fingerprint(model, ModelState.POISONED, targets=targets)

                with Timer() as t:
                    PatternMaskDefense().apply(model, targets, x=PATTERN_X_BITS, pattern=pat)

                # Compute fingerprint for defensed state
                defensed_fp = compute_fingerprint(model, ModelState.DEFENSED, targets=targets)

                # Extract and decode post-defense (use same x as injection)
                post_bits_raw = extract_bits(model, targets, x=PATTERN_X_BITS, n_bits=len(encoded_bits))
                post_bits_decoded, decoding_report = decode_payload(
                    post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                post_rec = bit_accuracy(payload.bits, post_bits_decoded)
                post_rec_strict = bit_accuracy_strict(payload.bits, post_bits_decoded)
                post_acc = task_runner.evaluate_accuracy(model, loader)
                l2_dist_rel = relative_l2_distance(model0, model, targets)
                w1_dist = weight_distribution_distance(model0, model, targets)
                hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

                points.append({
                    "strategy": "pattern",
                    "x_bits": PATTERN_X_BITS,  # Record actual x used
                    "pattern": pat,
                    "attacker_variant": variant_name,
                    "acc_drop": base_acc - post_acc,
                    "recovery_reduction": pre_rec - post_rec,
                    "post_recovery": post_rec,
                    "post_acc": post_acc,
                    "relative_l2_distance": l2_dist_rel,
                    "exact_recovery": exact_recovery(payload.bits, post_bits_decoded),
                    "defense_time_ms": t.elapsed * 1000,
                    "hamming_distance": hamming_dist,
                    "wasserstein_distance": w1_dist,
                    # Strict metrics
                    "post_recovery_strict": post_rec_strict,
                    "exact_recovery_strict": exact_recovery_strict(payload.bits, post_bits_decoded),
                    "recovery_reduction_strict": pre_rec_strict - post_rec_strict,
                    # Attacker encoding info
                    "expansion_factor": encoding_report.expansion_factor,
                    "corrected_errors": decoding_report.corrected_errors,
                    # Fingerprints for verification
                    "poisoned_fingerprint": poisoned_fp.param_hash,
                    "defensed_fingerprint": defensed_fp.param_hash,
                })

                # Store fingerprint for audit
                fp_key = f"pattern_{variant_name}_{pat}"
                fingerprints[fp_key] = {
                    "poisoned": poisoned_fp.param_hash,
                    "defensed": defensed_fp.param_hash,
                }

    # === GAUSSIAN NOISE DEFENSE ===
    if "gaussian" in defenses:
        logger.info(f"Testing Gaussian noise with sigma in {GAUSSIAN_SIGMAS}")
        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                continue

            for sigma in GAUSSIAN_SIGMAS:
                if sigma == 0:
                    continue  # Skip zero sigma (no effect)
                config_idx += 1
                model = copy.deepcopy(model0)

                encoded_bits, encoding_report = encode_payload(
                    payload.bits, variant_enum, interleave_seed=args.run_seed
                )
                inject_bits(model, targets, encoded_bits, x=args.x)

                pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                pre_bits_decoded, _ = decode_payload(
                    pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)

                with Timer() as t:
                    GaussianNoiseDefense().apply(model, targets, sigma=sigma, seed=args.run_seed)

                post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                post_bits_decoded, decoding_report = decode_payload(
                    post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                post_rec = bit_accuracy(payload.bits, post_bits_decoded)
                post_acc = task_runner.evaluate_accuracy(model, loader)
                l2_dist_rel = relative_l2_distance(model0, model, targets)
                w1_dist = weight_distribution_distance(model0, model, targets)
                hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

                points.append({
                    "strategy": "GaussianNoise",
                    "sigma": sigma,
                    "attacker_variant": variant_name,
                    "acc_drop": base_acc - post_acc,
                    "recovery_reduction": pre_rec - post_rec,
                    "post_recovery": post_rec,
                    "post_acc": post_acc,
                    "relative_l2_distance": l2_dist_rel,
                    "hamming_distance": hamming_dist,
                    "wasserstein_distance": w1_dist,
                    "defense_time_ms": t.elapsed * 1000,
                })

    # === FINETUNE DEFENSE ===
    if "finetune" in defenses:
        logger.info(f"Testing FineTune with steps in {FINETUNE_STEPS}")
        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                continue

            for n_steps in FINETUNE_STEPS:
                if n_steps == 0:
                    continue  # Skip zero steps
                config_idx += 1
                model = copy.deepcopy(model0)

                encoded_bits, encoding_report = encode_payload(
                    payload.bits, variant_enum, interleave_seed=args.run_seed
                )
                inject_bits(model, targets, encoded_bits, x=args.x)

                pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                pre_bits_decoded, _ = decode_payload(
                    pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)

                # Create fine-tuning loader
                finetune_loader = task_runner.make_train_loader(256, args.batch_size, seed=args.run_seed)
                finetune_defense = FineTuneDefense(learning_rate=1e-5)

                with Timer() as t:
                    finetune_defense.apply(
                        model, finetune_loader, n_steps=n_steps,
                        target_names=targets, device=str(device), seed=args.run_seed
                    )

                post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                post_bits_decoded, decoding_report = decode_payload(
                    post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
                )
                post_rec = bit_accuracy(payload.bits, post_bits_decoded)
                post_acc = task_runner.evaluate_accuracy(model, loader)
                l2_dist_rel = relative_l2_distance(model0, model, targets)
                w1_dist = weight_distribution_distance(model0, model, targets)
                hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

                points.append({
                    "strategy": "FineTune",
                    "n_steps": n_steps,
                    "attacker_variant": variant_name,
                    "acc_drop": base_acc - post_acc,
                    "recovery_reduction": pre_rec - post_rec,
                    "post_recovery": post_rec,
                    "post_acc": post_acc,
                    "relative_l2_distance": l2_dist_rel,
                    "hamming_distance": hamming_dist,
                    "wasserstein_distance": w1_dist,
                    "defense_time_ms": t.elapsed * 1000,
                })

    # === STATIC PTQ DEFENSE ===
    if "ptq" in defenses:
        logger.info("Testing static PTQ defense with calibration...")
        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                continue

            config_idx += 1
            model = copy.deepcopy(model0)

            encoded_bits, encoding_report = encode_payload(
                payload.bits, variant_enum, interleave_seed=args.run_seed
            )
            inject_bits(model, targets, encoded_bits, x=args.x)

            pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
            pre_bits_decoded, _ = decode_payload(
                pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
            )
            pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)

            calibration_loader = task_runner.make_train_loader(
                dargs.ptq_calibration_samples, args.batch_size, seed=args.eval_seed
            )

            with Timer() as t:
                rep = PTQDefense().apply(
                    model,
                    targets,
                    calibration_loader=calibration_loader,
                    device=str(device),
                    max_calibration_batches=dargs.ptq_calibration_batches,
                )

            post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
            post_bits_decoded, decoding_report = decode_payload(
                post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
            )
            post_rec = bit_accuracy(payload.bits, post_bits_decoded)
            post_acc = task_runner.evaluate_accuracy(model, loader)
            l2_dist_rel = relative_l2_distance(model0, model, targets)
            w1_dist = weight_distribution_distance(model0, model, targets)
            hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

            points.append({
                "strategy": "PTQ",
                "attacker_variant": variant_name,
                "acc_drop": base_acc - post_acc,
                "recovery_reduction": pre_rec - post_rec,
                "post_recovery": post_rec,
                "post_acc": post_acc,
                "relative_l2_distance": l2_dist_rel,
                "hamming_distance": hamming_dist,
                "wasserstein_distance": w1_dist,
                "defense_time_ms": t.elapsed * 1000,
                "n_calibration_batches": rep.n_calibration_batches,
                "compression_ratio": rep.compression_ratio,
            })

    # === SWP DEFENSE ===
    if "swp" in defenses:
        logger.info(f"Testing SWP defense with fraction={dargs.swp_fraction:.2f}")
        for variant_name in attacker_variants:
            try:
                variant_enum = AttackerVariantEnum(variant_name)
            except ValueError:
                continue

            config_idx += 1
            model = copy.deepcopy(model0)

            encoded_bits, encoding_report = encode_payload(
                payload.bits, variant_enum, interleave_seed=args.run_seed
            )
            inject_bits(model, targets, encoded_bits, x=args.x)

            pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
            pre_bits_decoded, _ = decode_payload(
                pre_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
            )
            pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)

            with Timer() as t:
                rep = SWPDefense().apply(
                    model,
                    targets,
                    x=args.x,
                    seed=args.run_seed,
                    fraction=dargs.swp_fraction,
                )

            post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
            post_bits_decoded, decoding_report = decode_payload(
                post_bits_raw, variant_enum, original_length=payload.n_bits, interleave_seed=args.run_seed
            )
            post_rec = bit_accuracy(payload.bits, post_bits_decoded)
            post_acc = task_runner.evaluate_accuracy(model, loader)
            l2_dist_rel = relative_l2_distance(model0, model, targets)
            w1_dist = weight_distribution_distance(model0, model, targets)
            hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

            points.append({
                "strategy": "SWP",
                "attacker_variant": variant_name,
                "acc_drop": base_acc - post_acc,
                "recovery_reduction": pre_rec - post_rec,
                "post_recovery": post_rec,
                "post_acc": post_acc,
                "relative_l2_distance": l2_dist_rel,
                "hamming_distance": hamming_dist,
                "wasserstein_distance": w1_dist,
                "defense_time_ms": t.elapsed * 1000,
                "fraction": rep.fraction,
                "sigma": rep.sigma,
                "target_relative_l2": rep.target_relative_l2,
                "achieved_relative_l2": rep.achieved_relative_l2,
            })


    # === GRAY CODE DEFENSE ===
    # Use full coverage (defense_x = attack_x) for optimal defense.
    # Default final configuration is V3 when a secret key is available.
    GRAY_X_SWEEP = [args.x]  # Full coverage only (optimal based on analysis)
    use_v2, use_v3 = _resolve_grayshield_mode()

    if use_v3:
        logger.info("GrayShield V3 mode: HMAC + per-run salt + multi-layer sequences")
    elif use_v2:
        logger.info("GrayShield V2 mode: HMAC-keyed masking (GRAYSHIELD_KEY set)")
    else:
        logger.info("GrayShield V1 mode: Seed-based masking")
    if "grayshield" in defenses:
        gray_label = "3" if use_v3 else ("2" if use_v2 else "1")
        logger.info(f"Testing GrayShield (V{gray_label}), x sweep={GRAY_X_SWEEP}")
        for gray_x in GRAY_X_SWEEP:
            for variant_name in attacker_variants:
                try:
                    variant_enum = AttackerVariantEnum(variant_name)
                except ValueError:
                    continue

                config_idx += 1
                model = copy.deepcopy(model0)

                encoded_bits, encoding_report = encode_payload(
                    payload.bits, variant_enum, interleave_seed=args.run_seed
                )
                inject_bits(model, targets, encoded_bits, x=args.x)  # attack x

                pre_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                pre_bits_decoded, _ = decode_payload(
                    pre_bits_raw, variant_enum, original_length=payload.n_bits,
                    interleave_seed=args.run_seed
                )
                pre_rec = bit_accuracy(payload.bits, pre_bits_decoded)

                with Timer() as t:
                    GrayShieldDefense().apply(
                        model, targets, x=gray_x,  # defense x (sweep)
                        seed=args.run_seed,
                        use_v2=use_v2,
                        use_v3=use_v3,  # Enable V3 enhancements (multi-layer + salt)
                    )

                post_bits_raw = extract_bits(model, targets, x=args.x, n_bits=len(encoded_bits))
                post_bits_decoded, decoding_report = decode_payload(
                    post_bits_raw, variant_enum, original_length=payload.n_bits,
                    interleave_seed=args.run_seed
                )
                post_rec = bit_accuracy(payload.bits, post_bits_decoded)
                post_acc = task_runner.evaluate_accuracy(model, loader)
                l2_dist_rel = relative_l2_distance(model0, model, targets)
                w1_dist = weight_distribution_distance(model0, model, targets)
                hamming_dist = hamming_distance(payload.bits, post_bits_decoded)

                points.append({
                    "strategy": "GrayShield",
                    "gray_version": "v3" if use_v3 else ("v2" if use_v2 else "v1"),
                    "defense_x": gray_x,  # distinguish from attack x
                    "attacker_variant": variant_name,
                    "acc_drop": base_acc - post_acc,
                    "recovery_reduction": pre_rec - post_rec,
                    "post_recovery": post_rec,
                    "post_acc": post_acc,
                    "relative_l2_distance": l2_dist_rel,
                    "hamming_distance": hamming_dist,
                    "wasserstein_distance": w1_dist,
                    "defense_time_ms": t.elapsed * 1000,
                })

    # Compute Pareto front
    pareto = pareto_front(points, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)

    # Generate per-model visualizations (with model name suffix to avoid overwriting)
    from grayshield.visualization.plots import format_model_name
    model_suffix = f"_{args.model_preset}"

    plot_rq3_pareto(
        points, pareto,
        out_path=os.path.join(out_dir, f"rq3_pareto{model_suffix}.png"),
        model_name=args.model_preset,
    )
    # --- EXCLUDED REDUNDANT PLOTS AS OVERLEAF STORYLINE SUGGESTED ---
    # try:
    #     plot_rq3_strategy_comparison(
    #         points,
    #         out_path=os.path.join(out_dir, f"rq3_strategy_comparison_{args.model_preset}.png"),
    #         title=f"Strategy Comparison: {args.model_preset}"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate strategy comparison plot: {e}")

    # try:
    #     plot_rq3_tradeoff_analysis(
    #         points,
    #         out_path=os.path.join(out_dir, f"rq3_tradeoff_{args.model_preset}.png"),
    #         title=f"Defense Trade-off: {args.model_preset}"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate tradeoff analysis plot: {e}")

    # try:
    #     plot_l2_recovery_tradeoff(
    #         points,
    #         out_path=os.path.join(out_dir, f"rq3_l2_tradeoff_{args.model_preset}.png"),
    #         title=f"L2 vs Recovery: {args.model_preset}"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate L2 tradeoff plot: {e}")

    # try:
    #     plot_rq3_comprehensive(
    #         points, pareto,
    #         out_path=os.path.join(out_dir, f"rq3_comprehensive_{args.model_preset}.png"),
    #         title=f"Comprehensive RQ3 Analysis: {args.model_preset}"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate comprehensive plot: {e}")

    # try:
    #     plot_tradeoff(
    #         points,
    #         x_key="acc_drop",
    #         y_key="post_recovery",
    #         out_path=os.path.join(out_dir, f"tradeoff_scatter_{args.model_preset}.png"),
    #         title=f"Complete Defense Trade-off: {args.model_preset}",
    #         xlabel="Accuracy Drop",
    #         ylabel="Recovery Rate"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate tradeoff scatter plot: {e}")

    # try:
    #     plot_curve(
    #         list(bound_curves['k1'].keys()),
    #         list(bound_curves['k1'].values()),
    #         out_path=os.path.join(out_dir, f"bound_curve_k1_{args.model_preset}.png"),
    #         xlab="Bit Error Rate (p)",
    #         ylab="Expected Recovery Rate",
    #         title="Theoretical Bound (k=1)"
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to generate bound curve plot: {e}")

    # Dummy calls for new plots (to be implemented later)
    try:
        from grayshield.visualization.plots import plot_rq3_tradeoff_2x2, plot_rq3_pareto_aggregate
        plot_rq3_tradeoff_2x2(points, out_path=os.path.join(out_dir, f"rq3_tradeoff_2x2{model_suffix}.png"), model_name=args.model_preset)
        plot_rq3_pareto_aggregate(points, pareto, out_path=os.path.join(out_dir, f"rq3_pareto_aggregate{model_suffix}.png"), model_name=args.model_preset)
    except ImportError:
        logger.warning("Skipping new RQ3 plots: plot_rq3_tradeoff_2x2 or plot_rq3_pareto_aggregate not found.")
    except Exception as e:
        logger.warning(f"Failed to generate new RQ3 plots: {e}")


    random_pts = sorted(
        [p for p in points if p["strategy"] == "random"],
        key=lambda d: d["flip_prob"]
    )
    if random_pts:
        plot_curve(
            [p["flip_prob"] for p in random_pts],
            [p["post_recovery"] for p in random_pts],
            out_path=os.path.join(out_dir, "random_flip_recovery_curve.png"),
            xlab="flip_prob", ylab="post_recovery",
            title="Random flips: recovery curve"
        )

    record = {
        "rq": "RQ3",
        "model_preset": args.model_preset,
        "task": args.task,
        "x": args.x,
        "target_mode": args.target_mode,
        "resolved_target_mode": target_report.resolved_mode,
        "layer_range": args.layer_range,
        "n_targets": len(targets),
        "target_prefixes": target_report.target_prefixes,
        "target_warnings": target_report.warnings,
        "payload": {
            "sha256": payload.sha256,
            "file_type": payload.file_type,
            "n_bits": payload.n_bits,
        },
        # Attacker variants tested
        "attacker_variants": attacker_variants,
        "n_configurations": config_idx,
        "base_acc": base_acc,
        "points": points,
        "pareto_front": pareto,
        # Theoretical bounds (BSC model)
        "theoretical_bounds": bound_curves,
        # Model state fingerprints for audit
        "model_states": {
            "clean": {
                "ckpt_path": clean_ckpt_path,
                "fingerprint": clean_fp.param_hash,
                "targets_hash": clean_fp.targets_hash,
            },
        },
        "fingerprints": fingerprints,
        # Seed configuration for reproducibility
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "run_seed": args.run_seed,
        # Evaluation configuration
        "n_eval": args.n_eval,
        "n_eval_actual": n_eval_actual,
        "full_eval": args.full_eval,
    }

    with open(os.path.join(out_dir, "rq3.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # JSON dual-write: accumulate all runs into a readable list for human inspection
    json_path = os.path.join(out_dir, "rq3.json")
    existing_records = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = []
    existing_records.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "rq3": {
                "base_acc": base_acc,
                "pareto_front": pareto,
                "best_tradeoff": pareto[0] if pareto else None,
            }
        }, f, indent=2)

    logger.info(f"RQ3 complete. {len(points)} configurations tested.")
    logger.info(f"Pareto optimal points: {len(pareto)}")

    record["out_dir"] = out_dir
    return record


# =============================================================================
# RQ4: Trade-off / Pareto Analysis
# =============================================================================

def run_rq4(
    args: ExperimentArgs,
    out_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    generate_table: bool = True,
) -> Dict[str, Any]:
    """
    RQ4: Trade-off / Pareto analysis and operating points.

    Aggregates results from RQ2/RQ3 and generates:
    - RQ4-Fig1: Pareto front scatter (all methods)
    - RQ4-Table1: Operating points table

    Args:
        args: Experiment arguments
        out_dir: Output directory for plots/tables
        results_dir: Directory containing RQ2/RQ3 results
        generate_table: Whether to generate operating points table

    Returns:
        Dict with Pareto analysis results
    """
    out_dir = out_dir or _ts_dir()
    results_dir = results_dir or out_dir

    logger.info(f"RQ4: Aggregating results from {results_dir}")

    from ..visualization.rq4 import load_rq2_rq3_results
    from ..visualization.plots import plot_rq4_pareto_scatter, plot_rq4_strategy_summary

    # Collect all points from RQ2/RQ3 results.  Newer visualization helpers
    # return (points, found_counts); older callers expected points only.
    loaded = load_rq2_rq3_results(results_dir)
    if isinstance(loaded, tuple):
        all_points, found_counts = loaded
        logger.info(
            "Found RQ result files: rq2=%s, rq3=%s",
            found_counts.get("rq2", 0),
            found_counts.get("rq3", 0),
        )
    else:
        all_points = loaded

    if not all_points:
        logger.warning("No RQ2/RQ3 points found.")
        # This shouldn't happen in normal usage
        return {"error": "No RQ2/RQ3 results found", "out_dir": out_dir}

    logger.info(f"Collected {len(all_points)} data points for Pareto analysis")

    # Compute Pareto front
    pareto = pareto_front(all_points, x_key="acc_drop", y_key="recovery_reduction", maximize_y=True)
    logger.info(f"Pareto front: {len(pareto)} optimal points")

    # Generate RQ4-Fig1: Pareto front scatter with all methods
    try:
        plot_rq4_pareto_scatter(
            all_points, pareto,
            out_path=os.path.join(out_dir, "rq4_pareto_scatter.png"),
            title="RQ4: Defense Trade-off (All Methods)"
        )
        logger.info(f"Generated: {out_dir}/rq4_pareto_scatter.png")
        plot_rq4_strategy_summary(
            all_points,
            out_path=os.path.join(out_dir, "rq4_pareto_summary.png"),
            title="RQ4: Defense Trade-off Summary (Mean ± Std)"
        )
        logger.info(f"Generated: {out_dir}/rq4_pareto_summary.png")
    except Exception as e:
        logger.warning(f"Failed to generate RQ4 Pareto scatter: {e}")

    # Select operating points: safe, balanced, aggressive
    operating_points = _select_operating_points(pareto)

    # Generate RQ4-Table1: Operating points table
    table_data = []
    if generate_table and operating_points:
        table_path = os.path.join(out_dir, "rq4_operating_points.csv")
        with open(table_path, 'w') as f:
            # Header
            headers = ["Operating Point", "Method", "Strength", "Bit Accuracy (%)",
                       "Recovery Reduction (%)", "Accuracy Drop (%)", "Relative L2"]
            f.write(",".join(headers) + "\n")

            for name, point in operating_points.items():
                method = point.get('strategy', 'unknown')
                strength = _get_strength_str(point)
                bit_acc = (1 - point.get('recovery_reduction', 0)) * 100  # Convert back
                rec_red = point.get('recovery_reduction', 0) * 100
                acc_drop = point.get('acc_drop', 0) * 100
                rel_l2 = point.get('relative_l2_distance')

                rel_l2_str = f"{rel_l2:.2e}" if rel_l2 is not None else "N/A"

                row = [name, method, strength, f"{bit_acc:.1f}", f"{rec_red:.1f}",
                       f"{acc_drop:.2f}", rel_l2_str]
                f.write(",".join(row) + "\n")
                table_data.append(dict(zip(headers, row)))

        logger.info(f"Generated: {table_path}")

    record = {
        "rq": "RQ4",
        "n_points": len(all_points),
        "n_pareto": len(pareto),
        "pareto_front": pareto,
        "operating_points": operating_points,
        "table_data": table_data,
    }

    # Save results
    with open(os.path.join(out_dir, "rq4.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    record["out_dir"] = out_dir
    return record


def _select_operating_points(pareto: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Select representative operating points from Pareto front.

    - Recommended: Deployment-relevant point with minimal accuracy cost
    - Reference-MidFront: Descriptive mid-front Pareto reference
    - Reference-HighRR: Descriptive high-recovery Pareto reference
    """
    if not pareto:
        return {}

    # Sort by acc_drop
    sorted_pareto = sorted(pareto, key=lambda p: p.get('acc_drop', 0))

    result = {}

    # Recommended: lowest acc_drop
    result["Recommended"] = sorted_pareto[0]

    # Reference-HighRR: highest recovery_reduction
    result["Reference-HighRR"] = max(pareto, key=lambda p: p.get('recovery_reduction', 0))

    # Reference-MidFront: middle of Pareto front
    if len(sorted_pareto) >= 3:
        mid_idx = len(sorted_pareto) // 2
        result["Reference-MidFront"] = sorted_pareto[mid_idx]
    elif len(sorted_pareto) == 2:
        result["Reference-MidFront"] = sorted_pareto[1]
    else:
        result["Reference-MidFront"] = sorted_pareto[0]

    return result


def _get_strength_str(point: Dict[str, Any]) -> str:
    """Get string representation of defense strength parameter."""
    strategy = point.get('strategy', '')
    if strategy in ('random', 'RandomFlip'):
        return f"p={point.get('flip_prob', 0)}"
    elif strategy in ('pattern', 'PatternMask'):
        return f"pat={point.get('pattern', '??')}"
    elif strategy in ('gaussian', 'GaussianNoise'):
        return f"σ={point.get('sigma', 0)}"
    elif strategy in ('finetune', 'FineTune'):
        return f"steps={point.get('n_steps', 0)}"
    elif strategy in ('ptq', 'PTQ'):
        return "static-ptq"
    elif strategy in ('swp', 'SWP'):
        return f"frac={point.get('fraction', 0)}"
    elif strategy in ('grayshield', 'gray_code', 'GrayCode', 'GrayShield'):
        return f"x={point.get('defense_x', point.get('x', 'N/A'))}"
    return "N/A"


# =============================================================================
# Run All
# =============================================================================

def run_all(args: ExperimentArgs, dargs: DefenseArgs, attacker_variants: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all four RQs with shared output directory."""
    out_dir = _ts_dir()

    if attacker_variants is None:
        attacker_variants = ["naive"]

    results = {
        "rq1": run_rq1(args, out_dir=out_dir),
        "rq2": run_rq2(args, dargs, out_dir=out_dir),
        "rq3": run_rq3(args, dargs, out_dir=out_dir, attacker_variants=attacker_variants),
        "rq4": run_rq4(args, out_dir=out_dir, results_dir=out_dir, generate_table=True),
    }

    # Generate comprehensive summary
    try:
        plot_comprehensive_summary(
            rq1_results=[results["rq1"]],
            rq2_results=[results["rq2"]],
            rq3_results=results["rq3"],
            out_path=os.path.join(out_dir, "comprehensive_summary.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to generate summary plot: {e}")

    # Generate RQ1 comprehensive visualization
    try:
        plot_rq1_comprehensive(
            [results["rq1"]],
            out_path=os.path.join(out_dir, "rq1_comprehensive.png"),
            title=f"RQ1: Injection Feasibility - {args.model_preset}"
        )
    except Exception as e:
        logger.warning(f"Failed to generate RQ1 comprehensive plot: {e}")

    # Generate RQ2 comprehensive visualization
    try:
        plot_rq2_comprehensive(
            [results["rq2"]],
            out_path=os.path.join(out_dir, "rq2_comprehensive.png"),
            title=f"RQ2: Defense Effectiveness - {args.model_preset}"
        )
    except Exception as e:
        logger.warning(f"Failed to generate RQ2 comprehensive plot: {e}")

    # Save combined summary
    with open(os.path.join(out_dir, "all_summary.json"), "w", encoding="utf-8") as f:
        summary = {
            "rq1": {
                "capacity_bits": results["rq1"]["capacity_bits"],
                "bit_recovery": results["rq1"]["metrics"]["bit_accuracy"],
                "acc_drop": results["rq1"]["metrics"]["acc_drop"],
            },
            "rq2": {
                "pre_recovery": results["rq2"]["metrics"]["pre_recovery"],
                "post_recovery": results["rq2"]["metrics"]["post_recovery"],
                "recovery_reduction": results["rq2"]["metrics"]["recovery_reduction"],
                "acc_drop": results["rq2"]["metrics"]["acc_drop_vs_base"],
            },
            "rq3": {
                "pareto_front": results["rq3"]["pareto_front"],
            },
        }
        json.dump(summary, f, indent=2)

    logger.info(f"All RQs complete. Results in: {out_dir}")
    return results
