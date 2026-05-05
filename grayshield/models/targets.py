"""
Target Selection for LSB Steganography

Provides consistent semantics for selecting which model parameters to target:
- attention: Only attention layers (Q/K/V/O projections)
- ffn: Only feed-forward layers (intermediate/output dense)
- embeddings: Only embedding layers (word/position embeddings)
- encoder_only: attention + ffn within layer_range (default for "all")

CRITICAL SEMANTICS (must be enforced):
1. attention + layer_range=[0,3] -> ONLY encoder.layer.{0..3}.attention.*
2. ffn + layer_range=[0,3] -> ONLY encoder.layer.{0..3}.intermediate.* and output.*
3. embeddings -> ONLY *.embeddings.*, layer_range is IGNORED
4. "all" is DEPRECATED alias for "encoder_only" (attention+ffn, NO embeddings/head)
"""
from __future__ import annotations
import re
import warnings
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import torch
from ..config import TargetMode
from ..utils.logging import get_logger

logger = get_logger()


@dataclass
class TargetSelectionReport:
    """Report of target selection for audit trail."""
    mode: str
    resolved_mode: str
    layer_range: Optional[Tuple[int, int]]
    include_embeddings: bool
    include_head: bool
    n_targets: int
    target_prefixes: List[str]  # Top-k unique prefixes for logging
    warnings: List[str]


def _is_float_weight(name: str, p: torch.nn.Parameter) -> bool:
    """Check if parameter is a float weight suitable for LSB steganography."""
    return p.data.dtype in (torch.float16, torch.float32, torch.bfloat16) and ("weight" in name)


def _detect_architecture(model: torch.nn.Module) -> Dict[str, Any]:
    """Detect model architecture from parameter names."""
    param_names = [n for n, _ in model.named_parameters()]

    # Architecture detection patterns (order: most specific first)
    archs = [
        ("distilbert", r"distilbert\.transformer\.layer\.(\d+)", "distilbert.transformer.layer", "distilbert.embeddings"),
        ("roberta", r"roberta\.encoder\.layer\.(\d+)", "roberta.encoder.layer", "roberta.embeddings"),
        ("swin", r"swin\.encoder\.layers\.(\d+)", "swin.encoder.layers", "swin.embeddings"),
        ("vit", r"vit\.encoder\.layer\.(\d+)", "vit.encoder.layer", "vit.embeddings"),
        ("bert", r"bert\.encoder\.layer\.(\d+)", "bert.encoder.layer", "bert.embeddings"),
    ]

    for name, layer_pat, encoder_kw, embed_kw in archs:
        if any(encoder_kw in n for n in param_names):
            return {
                "name": name,
                "layer_pattern": re.compile(layer_pat),
                "encoder_keyword": encoder_kw,
                "embedding_keyword": embed_kw,
                "head_keywords": ["classifier", "pooler", "head", "lm_head"],
            }

    # Fallback for unknown architectures
    return {
        "name": "generic",
        "layer_pattern": re.compile(r"encoder\.layer\.(\d+)|encoder\.layers\.(\d+)|transformer\.layer\.(\d+)"),
        "encoder_keyword": "encoder",
        "embedding_keyword": "embeddings",
        "head_keywords": ["classifier", "pooler", "head", "lm_head"],
    }


def _is_attention_param(name: str) -> bool:
    """Check if parameter belongs to attention layers."""
    lname = name.lower()
    attn_keywords = [
        "attention", "attn", "self_attn",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "q_lin", "k_lin", "v_lin", "out_lin",
        "query", "key", "value",
    ]
    # Exclude attention output LayerNorm (belongs to output block)
    if "layernorm" in lname or "layer_norm" in lname:
        return False
    return any(k in lname for k in attn_keywords)


def _is_ffn_param(name: str) -> bool:
    """Check if parameter belongs to FFN/MLP layers."""
    lname = name.lower()
    ffn_keywords = [
        "intermediate", "mlp", "ffn",
        "fc1", "fc2",
        "lin1", "lin2",
    ]
    # Include output.dense but NOT attention.output.dense
    if "output.dense" in lname and "attention" not in lname:
        return True
    if any(k in lname for k in ffn_keywords):
        return True
    return False


def _is_embedding_param(name: str, embed_keyword: str) -> bool:
    """Check if parameter belongs to embedding layers."""
    return embed_keyword in name


def _is_head_param(name: str, head_keywords: List[str]) -> bool:
    """Check if parameter belongs to classifier/head layers."""
    lname = name.lower()
    return any(k in lname for k in head_keywords)


def _extract_layer_idx(name: str, layer_pattern: re.Pattern) -> Optional[int]:
    """Extract layer index from parameter name."""
    m = layer_pattern.search(name)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return None


def select_targets(
    model: torch.nn.Module,
    preset_name: str = "",  # Kept for backward compat, architecture detected from params
    mode: TargetMode = "attention",
    layer_range: Optional[Tuple[int, int]] = None,
    include_embeddings: bool = False,
    include_head: bool = False,
) -> List[str]:
    """Select target parameters for LSB injection.

    Args:
        model: PyTorch model
        preset_name: Model preset name (for architecture detection)
        mode: Target mode - one of:
            - "attention": Only attention layers (Q/K/V/O)
            - "ffn": Only feed-forward layers
            - "embeddings": Only embedding layers (layer_range ignored)
            - "encoder_only": attention + ffn within layer_range
            - "all": DEPRECATED alias for encoder_only (prints warning)
            - "full_model": Everything including embeddings and head
        layer_range: Tuple (start, end) inclusive. Only applies to encoder layers.
        include_embeddings: Force include embeddings (overrides mode)
        include_head: Force include classifier head (overrides mode)

    Returns:
        List of parameter names to target

    Semantics enforced:
        1. attention + layer_range=[0,3] -> ONLY layer.{0..3}.attention.*
        2. ffn + layer_range=[0,3] -> ONLY layer.{0..3}.intermediate/output.*
        3. embeddings -> ONLY *.embeddings.*, layer_range IGNORED
        4. "all" -> encoder_only (attention+ffn), NO embeddings/head
    """
    warnings_list = []
    arch = _detect_architecture(model)

    # Handle mode aliases and deprecation
    resolved_mode = mode
    if mode == "all":
        warnings.warn(
            "TargetMode 'all' is deprecated. It now means 'encoder_only' "
            "(attention+ffn within layer_range, excluding embeddings/head). "
            "Use 'full_model' for truly everything.",
            DeprecationWarning,
            stacklevel=2,
        )
        warnings_list.append("'all' resolved to 'encoder_only'")
        resolved_mode = "encoder_only"

    # Embeddings mode: ignore layer_range
    if mode == "embeddings" and layer_range is not None:
        logger.warning(
            f"layer_range={layer_range} is IGNORED for mode='embeddings'. "
            "Embeddings are not organized by layers."
        )
        warnings_list.append("layer_range ignored for embeddings mode")
        layer_range = None

    targets: List[str] = []
    target_prefixes = set()

    for name, p in model.named_parameters():
        if not _is_float_weight(name, p):
            continue

        is_embed = _is_embedding_param(name, arch["embedding_keyword"])
        is_head = _is_head_param(name, arch["head_keywords"])
        is_encoder = arch["encoder_keyword"] in name

        # Check layer range for encoder params
        if is_encoder and layer_range is not None:
            layer_idx = _extract_layer_idx(name, arch["layer_pattern"])
            if layer_idx is not None:
                lo, hi = layer_range
                if not (lo <= layer_idx <= hi):
                    continue

        # Mode-specific selection
        should_include = False

        if resolved_mode == "attention":
            if is_encoder and _is_attention_param(name):
                should_include = True

        elif resolved_mode == "ffn":
            if is_encoder and _is_ffn_param(name):
                should_include = True

        elif resolved_mode == "embeddings":
            if is_embed:
                should_include = True

        elif resolved_mode == "encoder_only":
            # attention + ffn, NO embeddings/head
            if is_encoder and (_is_attention_param(name) or _is_ffn_param(name)):
                should_include = True

        elif resolved_mode == "full_model":
            # Everything
            should_include = True

        # Override flags
        if include_embeddings and is_embed:
            should_include = True
        if include_head and is_head:
            should_include = True

        # Explicit exclusions for safety
        if resolved_mode in ("attention", "ffn", "encoder_only"):
            if is_embed or is_head:
                should_include = False  # Never include for these modes

        if should_include:
            targets.append(name)
            # Extract prefix for logging (first 3 components)
            prefix = ".".join(name.split(".")[:3])
            target_prefixes.add(prefix)

    # Log selection info
    logger.info(
        f"Target selection: mode={mode} -> resolved={resolved_mode}, "
        f"layer_range={layer_range}, n_targets={len(targets)}"
    )
    if targets:
        sample_prefixes = sorted(target_prefixes)[:5]
        logger.info(f"  Sample prefixes: {sample_prefixes}")

    return targets


def select_targets_with_report(
    model: torch.nn.Module,
    preset_name: str,
    mode: TargetMode = "attention",
    layer_range: Optional[Tuple[int, int]] = None,
    include_embeddings: bool = False,
    include_head: bool = False,
) -> Tuple[List[str], TargetSelectionReport]:
    """Select targets and return detailed report for logging.

    Same as select_targets but also returns a TargetSelectionReport
    for inclusion in JSONL output.
    """
    warnings_list = []
    arch = _detect_architecture(model)

    # Handle mode aliases
    resolved_mode = mode
    if mode == "all":
        warnings_list.append("'all' resolved to 'encoder_only'")
        resolved_mode = "encoder_only"

    if mode == "embeddings" and layer_range is not None:
        warnings_list.append("layer_range ignored for embeddings mode")

    targets = select_targets(
        model, preset_name, mode, layer_range,
        include_embeddings, include_head
    )

    # Compute unique prefixes
    prefixes = set()
    for name in targets:
        prefix = ".".join(name.split(".")[:3])
        prefixes.add(prefix)

    report = TargetSelectionReport(
        mode=mode,
        resolved_mode=resolved_mode,
        layer_range=layer_range,
        include_embeddings=include_embeddings,
        include_head=include_head,
        n_targets=len(targets),
        target_prefixes=sorted(prefixes)[:10],
        warnings=warnings_list,
    )

    return targets, report
