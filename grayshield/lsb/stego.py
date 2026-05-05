from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import torch
import numpy as np
from .bits import get_low_bits

logger = logging.getLogger(__name__)


@dataclass
class InjectionReport:
    """Detailed report of payload injection operation."""
    per_param: Dict[str, int]  # {param_name: n_bits_written}
    written_bits_total: int    # Total bits actually written
    payload_bits_total: int    # Total bits in original payload
    truncated: bool            # True if some bits could not be written
    capacity_used: float       # written / payload (ratio, <= 1.0 if no truncation)


def capacity_bits(model: torch.nn.Module, target_names: List[str], x: int) -> int:
    """Compute total capacity in bits for the given targets and x LSBs."""
    named = dict(model.named_parameters())
    cap = 0
    for n in target_names:
        p = named.get(n, None)
        if p is None or p.data.dtype != torch.float32:
            continue
        cap += p.numel() * x
    return int(cap)


def inject_bits(
    model: torch.nn.Module,
    target_names: List[str],
    payload_bits: List[int],
    x: int
) -> InjectionReport:
    """
    In-place injection: overwrite low x bits with payload bits sequentially.

    Args:
        model: Model to inject into
        target_names: Parameter names to target
        payload_bits: Bits to inject
        x: Number of LSBs to use per float32

    Returns:
        InjectionReport with detailed injection statistics including:
        - per_param: {param_name: n_bits_written}
        - written_bits_total: Total bits actually written
        - payload_bits_total: Total bits in original payload
        - truncated: True if payload was truncated due to capacity limits

    Note:
        If payload exceeds model capacity, a warning will be logged and
        truncation will occur. This may affect Recovery Reduction measurements.
    """
    named = dict(model.named_parameters())
    payload_bits_total = len(payload_bits)

    # Pre-check: validate capacity before injection
    total_capacity = capacity_bits(model, target_names, x)
    if payload_bits_total > total_capacity:
        deficit = payload_bits_total - total_capacity
        logger.warning(
            f"⚠️  CAPACITY DEFICIT DETECTED: Payload requires {payload_bits_total} bits "
            f"but model capacity is only {total_capacity} bits (x={x}). "
            f"Deficit: {deficit} bits ({deficit / payload_bits_total * 100:.1f}%). "
            f"Payload will be truncated during injection."
        )

    idx = 0
    per_param: Dict[str, int] = {}

    for name in target_names:
        if idx >= payload_bits_total:
            break
        p = named.get(name, None)
        if p is None or p.data.dtype != torch.float32:
            continue
        flat = p.data.view(torch.int32).view(-1)
        n_elem = flat.numel()
        n_write = min((payload_bits_total - idx) // x, n_elem)
        if n_write <= 0:
            continue

        chunk = payload_bits[idx : idx + n_write * x]
        vals = []
        for i in range(n_write):
            v = 0
            for b in chunk[i * x : (i + 1) * x]:
                v = (v << 1) | int(b)
            vals.append(v)
        vals_t = torch.tensor(vals, device=flat.device, dtype=torch.int32)
        mask = (1 << x) - 1
        flat[:n_write] = (flat[:n_write] & ~mask) | (vals_t & mask)

        written = int(n_write * x)
        per_param[name] = written
        idx += written

    written_bits_total = sum(per_param.values())
    truncated = written_bits_total < payload_bits_total
    capacity_used = written_bits_total / payload_bits_total if payload_bits_total > 0 else 0.0

    # Warn if payload was truncated (incomplete injection)
    if truncated:
        bits_lost = payload_bits_total - written_bits_total
        logger.warning(
            f"⚠️  INCOMPLETE PAYLOAD INJECTION: "
            f"{written_bits_total}/{payload_bits_total} bits written "
            f"({bits_lost} bits truncated, {(1 - capacity_used) * 100:.1f}% lost). "
            f"Model capacity insufficient for full payload. "
            f"Recovery Reduction measurements may be inaccurate."
        )

    return InjectionReport(
        per_param=per_param,
        written_bits_total=written_bits_total,
        payload_bits_total=payload_bits_total,
        truncated=truncated,
        capacity_used=capacity_used,
    )


def inject_bits_legacy(
    model: torch.nn.Module,
    target_names: List[str],
    payload_bits: List[int],
    x: int
) -> Dict[str, int]:
    """
    Legacy wrapper for backward compatibility.
    Returns {param_name: n_bits_written} like the old inject_bits().
    """
    report = inject_bits(model, target_names, payload_bits, x)
    return report.per_param

def extract_bits(model: torch.nn.Module, target_names: List[str], x: int, n_bits: Optional[int] = None) -> List[int]:
    '''
    Extract low x bits sequentially over targets until n_bits.
    '''
    named = dict(model.named_parameters())
    out_bits: List[int] = []
    for name in target_names:
        if n_bits is not None and len(out_bits) >= n_bits:
            break
        p = named.get(name, None)
        if p is None or p.data.dtype != torch.float32:
            continue

        low = get_low_bits(p.data, x=x).view(-1).detach().cpu().numpy().astype(np.uint32)
        for v in low:
            s = format(int(v), f"0{x}b")
            out_bits.extend(int(c) for c in s)
            if n_bits is not None and len(out_bits) >= n_bits:
                return out_bits[:n_bits]
    return out_bits if n_bits is None else out_bits[:n_bits]
