from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List, Dict, Any
from pathlib import Path
import os

# =============================================================================
# Project Paths Configuration
# =============================================================================

# Project root directory (parent of grayshield package)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default directories (relative to project root)
DEFAULT_PAYLOADS_DIR = PROJECT_ROOT / "data"
DEFAULT_MALWARE_DIR = DEFAULT_PAYLOADS_DIR / "malware"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Environment variable overrides
PAYLOADS_DIR = Path(os.getenv("GRAYSHIELD_PAYLOADS_DIR", DEFAULT_PAYLOADS_DIR))
MALWARE_DIR = Path(os.getenv("GRAYSHIELD_MALWARE_DIR", DEFAULT_MALWARE_DIR))
RESULTS_DIR = Path(os.getenv("GRAYSHIELD_RESULTS_DIR", DEFAULT_RESULTS_DIR))

# =============================================================================
# Target Modes and Defense Types
# =============================================================================

# Target modes for injection
# - attention: Only attention layers (Q/K/V/O projections)
# - ffn: Only feed-forward layers (intermediate/output dense)
# - embeddings: Only embedding layers (word/position embeddings)
# - encoder_only: attention + ffn within layer_range (RECOMMENDED)
# - all: DEPRECATED alias for encoder_only (prints warning)
# - full_model: Everything including embeddings and head
TargetMode = Literal["attention", "ffn", "embeddings", "encoder_only", "all", "full_model"]

# Extended defense types including new baselines
DefenseType = Literal[
    "random",           # RandomFlip - flip LSBs with probability p
    "pattern",          # PatternMask - set LSBs to specific pattern
    "gaussian",         # GaussianNoise - add N(0, sigma^2) noise to weights
    "finetune",         # FineTune - fine-tune on small clean subset
    "ptq",              # PTQ - static 8-bit projection with calibration
    "swp",              # SWP - selective weight perturbation
    "grayshield",       # GrayShield - Gray-code-guided sanitization
]

# Attacker encoding variants for adaptive attacker experiments
AttackerVariant = Literal["naive", "repeat3", "repeat5", "interleave", "rs"]

# =============================================================================
# Defense Strength Grids (paper-grade granularity)
# =============================================================================

# RandomFlip: flip probability grid
DEFAULT_FLIP_PROBS: List[float] = [
    0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1
]
# Extended grid for RQ3 sweep
FLIP_PROBS_EXTENDED: List[float] = [
    0.0, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
]

# GaussianNoise: sigma grid (standard deviation)
GAUSSIAN_SIGMAS: List[float] = [
    0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4
]

# FineTune: steps grid
FINETUNE_STEPS: List[int] = [50, 100, 200, 500]  # 0 removed (no-op baseline), focus on effective steps
FINETUNE_LR: float = 1e-5  # Default learning rate

# PTQ/SWP: lightweight paper-aligned defaults
PTQ_CALIBRATION_SAMPLES: int = 256
PTQ_CALIBRATION_BATCHES: int = 8
SWP_FRACTION: float = 0.20

# Layer range aliases for ablation studies
LAYER_RANGE_ALIASES = {
    "early": (0, 3),
    "mid": (4, 7),
    "late": (8, 11),
}

# =============================================================================
# Phase Configuration (main paper vs appendix)
# =============================================================================

# Main paper: 4 models (2 NLP + 2 CV), 2 payloads (low/high entropy)
MAIN_MODELS: List[str] = [
    "bert_sst2",        # NLP
    "roberta_sentiment", # NLP
    "vit_cifar10",      # CV
    "swin_cifar10",     # CV
]

# Appendix: all models, all payloads
ALL_MODELS: List[str] = [
    "bert_imdb",
    "bert_sst2",
    "distilbert_sst2",
    "roberta_sentiment",
    "vit_cifar10",
    "swin_cifar10",
]

# Paper payloads (SHA256 for reproducibility)
LOW_ENTROPY_PAYLOAD_SHA256 = "c37c0db91ab188c2fe01642e04e0db9186bc5bf54ad8b6b72512ad5aab921a88"
HIGH_ENTROPY_PAYLOAD_SHA256 = "5704fabda6a0851ea156d1731b4ed4383ce102ec3a93f5d7109cc2f47f8196d0"

# =============================================================================
# RQ-specific configurations
# =============================================================================

# RQ1: LSB depth sweep
RQ1_LSB_DEPTHS: List[int] = [4, 8, 16, 19, 21, 23]

# RQ2/RQ3: Mainline defense methods (bit-level perturbation focus)
MAINLINE_DEFENSES: List[str] = [
    "random",       # RandomFlip baseline
    "pattern",      # PatternMask baseline
    "gaussian",     # GaussianNoise baseline
    "finetune",     # FineTune baseline
    "ptq",          # Static PTQ baseline
    "swp",          # Selective Weight Perturbation baseline
    "grayshield",   # GrayShield
]

# RQ3: Attacker variants
RQ3_ATTACKER_VARIANTS: List[str] = ["naive", "repeat3", "repeat5", "interleave", "rs"]


@dataclass(frozen=True)
class ModelPreset:
    name: str
    task_type: Literal["text", "vision"]
    hf_model_id: str
    hf_processor_id: Optional[str] = None


@dataclass
class ExperimentArgs:
    model_preset: str
    task: str
    payload_path: str
    x: int = 2
    target_mode: TargetMode = "attention"
    layer_range: Optional[Tuple[int, int]] = None
    n_eval: int = 2048
    batch_size: int = 16
    # Seed separation for reproducibility:
    # - eval_seed: controls eval data sampling/order (fixed across runs)
    # - run_seed: controls stochastic operations (varied for mean±std)
    eval_seed: int = 42      # Fixed seed for consistent eval set
    run_seed: int = 42       # Variable seed for random ops (defense, etc.)
    seed: int = 42           # Legacy: backward compat, overrides both if set alone
    device: str = "cuda"
    # Full evaluation mode (uses complete validation/test split)
    full_eval: bool = False  # If True, ignore n_eval and use full split


@dataclass
class DefenseArgs:
    """Arguments for defense configuration."""
    defense: DefenseType = "random"
    # RandomFlip parameters
    flip_prob: float = 0.1
    # Pattern parameters (appendix only)
    pattern: str = "00"
    # GaussianNoise parameters
    sigma: float = 1e-5
    # FineTune parameters
    finetune_steps: int = 100
    finetune_lr: float = 1e-5
    finetune_samples: int = 256
    # PTQ parameters
    ptq_calibration_samples: int = PTQ_CALIBRATION_SAMPLES
    ptq_calibration_batches: int = PTQ_CALIBRATION_BATCHES
    # SWP parameters
    swp_fraction: float = SWP_FRACTION


@dataclass
class PhaseConfig:
    """Configuration for experiment phase (main paper vs appendix)."""
    phase: str = "main"  # "main" or "appendix"
    models: List[str] = field(default_factory=lambda: MAIN_MODELS.copy())
    # Payload SHA256 hashes (for paper reproducibility)
    payload_hashes: List[str] = field(default_factory=lambda: [
        LOW_ENTROPY_PAYLOAD_SHA256,
        HIGH_ENTROPY_PAYLOAD_SHA256,
    ])
    # Which defenses to run
    defenses: List[str] = field(default_factory=lambda: MAINLINE_DEFENSES.copy())
    # Include pattern defense (appendix only by default)
    include_pattern: bool = False
