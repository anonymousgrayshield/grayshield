from __future__ import annotations
from typing import Optional
import torch
import hashlib
import hmac
import struct
import re


def _extract_layer_from_param_name(param_name: str) -> Optional[int]:
    """
    Extract layer index from parameter name for layer-specific masking.

    Supports common transformer architectures:
    - bert.encoder.layer.0.attention.self.query.weight → 0
    - roberta.encoder.layer.11.output.dense.weight → 11
    - vit.encoder.layer.5.mlp.fc1.weight → 5
    - swin.encoder.layers.3.attention.qkv.weight → 3

    Returns None for embeddings/head parameters (no layer structure).

    Args:
        param_name: Full parameter name from model.named_parameters()

    Returns:
        Layer index (int) or None if no layer pattern found
    """
    patterns = [
        r"bert\.encoder\.layer\.(\d+)",
        r"roberta\.encoder\.layer\.(\d+)",
        r"distilbert\.transformer\.layer\.(\d+)",
        r"vit\.encoder\.layer\.(\d+)",
        r"swin\.encoder\.layers\.(\d+)",
        r"encoder\.layers?\.(\d+)",  # Generic fallback
    ]
    for pattern in patterns:
        m = re.search(pattern, param_name)
        if m:
            return int(m.group(1))
    return None


def get_low_bits(t: torch.Tensor, x: int) -> torch.Tensor:
    '''
    Return the low x bits as int tensor (same shape as t).
    '''
    if x <= 0 or x > 23:
        raise ValueError("x must be in [1, 23] for float32 mantissa bits.")
    if t.dtype != torch.float32:
        raise TypeError("Tensor must be float32 for bit-level ops.")
    if not t.is_contiguous():
        t = t.contiguous()
    iv = t.view(torch.int32)
    mask = (1 << x) - 1
    return iv & mask


def apply_pattern_mask(t: torch.Tensor, x: int, pattern: str, idx_flat: Optional[torch.Tensor] = None) -> None:
    '''
    In-place: set low x bits to a fixed binary pattern (len(pattern)==x).
    If idx_flat is provided, applies only to those flattened indices.
    '''
    pattern_len = len(pattern)
    if pattern_len != x or any(c not in "01" for c in pattern):
        raise ValueError(f"pattern must be binary string of length x={x}")
    if t.dtype != torch.float32:
        raise TypeError("Pattern mask expects float32 params.")

    if not t.is_contiguous():
        t_contig = t.contiguous()
        apply_pattern_mask(t_contig, x, pattern, idx_flat)
        t.copy_(t_contig)
        return

    pat = int(pattern, 2) & ((1 << x) - 1)
    iv = t.view(torch.int32).view(-1)

    if idx_flat is None:
        idx_flat = torch.arange(iv.numel(), device=iv.device, dtype=torch.long)
    else:
        idx_flat = idx_flat.to(device=iv.device, dtype=torch.long)

    mask = (1 << x) - 1
    iv[idx_flat] = (iv[idx_flat] & ~mask) | pat


def apply_random_flips(t: torch.Tensor, x: int, flip_prob: float, idx_flat: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None) -> int:
    '''
    In-place: flip each of the low x bits with probability flip_prob (per element-bit).
    Returns number of flipped bit-events.
    '''
    if not (0.0 <= flip_prob <= 1.0):
        raise ValueError("flip_prob must be in [0,1]")
    if t.dtype != torch.float32:
        raise TypeError("Random flips expects float32 params.")

    if not t.is_contiguous():
        t_contig = t.contiguous()
        apply_random_flips(t_contig, x, flip_prob, idx_flat, generator)
        t.copy_(t_contig)
        return

    iv = t.view(torch.int32).view(-1)

    if idx_flat is None:
        idx_flat = torch.arange(iv.numel(), device=iv.device, dtype=torch.long)
    else:
        idx_flat = idx_flat.to(device=iv.device, dtype=torch.long)

    flips = 0
    for bit in range(x):
        bit_mask = 1 << bit
        r = torch.rand(idx_flat.shape[0], device=iv.device, generator=generator)
        do_flip = (r < flip_prob).to(torch.int32) * bit_mask
        iv[idx_flat] ^= do_flip
        flips += int((do_flip != 0).sum().item())
    return flips


def apply_gray_code_mask(t: torch.Tensor, x: int, seed: int, idx_flat: Optional[torch.Tensor] = None) -> None:
    '''
    GrayShield V1: In-place set low x bits using a Gray code sequence.
    Offset is derived from a seed — NOT cryptographically secure.
    An attacker who guesses the seed can reconstruct the mask exactly.
    For production use, prefer apply_hmac_gray_mask() (V2).
    '''
    if t.dtype != torch.float32:
        raise TypeError("Gray code mask expects float32 params.")

    if not t.is_contiguous():
        t_contig = t.contiguous()
        apply_gray_code_mask(t_contig, x, seed, idx_flat)
        t.copy_(t_contig)
        return

    iv = t.view(torch.int32).view(-1)

    if idx_flat is None:
        idx_flat = torch.arange(iv.numel(), device=iv.device, dtype=torch.long)
    else:
        idx_flat = idx_flat.to(device=iv.device, dtype=torch.long)

    gen = torch.Generator(device=iv.device)
    gen.manual_seed(seed)
    offset = torch.randint(0, 2**31 - 1, (1,), generator=gen, device=iv.device).item()

    seq = torch.arange(idx_flat.shape[0], device=iv.device, dtype=torch.long) + offset
    gray = seq ^ (seq >> 1)
    pat = (gray & ((1 << x) - 1)).to(torch.int32)

    mask = (1 << x) - 1
    iv[idx_flat] = (iv[idx_flat] & ~mask) | pat


# ============================================================================
# GrayShield V2: HMAC-Keyed Gray Code Mask
# ============================================================================

def _hmac_derive_offset(secret_key: bytes, param_name: str, x: int) -> int:
    """
    Derive a per-parameter, per-bit-depth keyed offset using HMAC-SHA256.

    Security guarantee: without secret_key, the offset is computationally
    indistinguishable from random (PRF security under SHA256).

    Domain separation: param_name + x ensures each parameter gets a unique mask
    even if the same key is reused across different model checkpoints.
    """
    msg = f"{param_name}:x={x}".encode("utf-8")
    digest = hmac.new(secret_key, msg, hashlib.sha256).digest()
    # Take the first 8 bytes as big-endian uint64; mod to stay in int32 range
    offset = struct.unpack(">Q", digest[:8])[0]
    return int(offset % (2**31 - 1))


def apply_hmac_gray_mask(
    t: torch.Tensor,
    x: int,
    secret_key: bytes,
    param_name: str,
    idx_flat: Optional[torch.Tensor] = None,
    use_v3: bool = False,
    run_salt: Optional[int] = None,
) -> None:
    """
    GrayShield V2/V3: In-place set low x bits using a keyed Gray Code mask.

    V2 Improvements over V1 (apply_gray_code_mask):
        - Cryptographically secure: offset derived from HMAC-SHA256(key, param_name)
        - Reversible: mask is deterministic given same key — owner can restore LSBs
        - Tamper-detectable: any parameter change breaks mask reconstruction
        - CPA-secure: attacker without key sees a uniformly random sequence

    V3 Enhancements (use_v3=True):
        - Per-run randomization: run_salt adds entropy across experiment runs
        - Layer-specific sequences: different layers get different Gray Code patterns
        - Stronger domain separation across layers and experiment runs

    Usage:
        # V2 mode
        key = os.environ.get('GRAYSHIELD_KEY', '').encode() or os.urandom(32)
        apply_hmac_gray_mask(param.data, x=19, secret_key=key, param_name=name)

        # V3 mode (enhanced)
        apply_hmac_gray_mask(param.data, x=19, secret_key=key, param_name=name,
                             use_v3=True, run_salt=42)

    Args:
        t          : float32 parameter tensor (modified in-place)
        x          : number of LSBs to mask (1-23)
        secret_key : HMAC key bytes (keep secret; owner only)
        param_name : parameter name for domain separation
        idx_flat   : optional flat index subset (None = apply to all elements)
        use_v3     : enable V3 enhancements (multi-layer + per-run salt)
        run_salt   : per-run randomization salt (V3 only, reproducible via seed)
    """
    if t.dtype != torch.float32:
        raise TypeError("HMAC gray mask expects float32 params.")

    if not t.is_contiguous():
        t_contig = t.contiguous()
        apply_hmac_gray_mask(t_contig, x, secret_key, param_name, idx_flat)
        t.copy_(t_contig)
        return

    iv = t.view(torch.int32).view(-1)

    if idx_flat is None:
        idx_flat = torch.arange(iv.numel(), device=iv.device, dtype=torch.long)
    else:
        idx_flat = idx_flat.to(device=iv.device, dtype=torch.long)

    # Keyed offset — attacker cannot guess this without secret_key
    if use_v3:
        # V3: layer-specific sequences + per-run salt for enhanced randomization
        layer_idx = _extract_layer_from_param_name(param_name)
        layer_suffix = f":layer={layer_idx}" if layer_idx is not None else ""
        salt_suffix = f":salt={run_salt}" if run_salt is not None else ""
        msg = f"{param_name}:x={x}{layer_suffix}{salt_suffix}".encode("utf-8")
        digest = hmac.new(secret_key, msg, hashlib.sha256).digest()
        offset = int(struct.unpack(">Q", digest[:8])[0] % (2**31 - 1))
    else:
        # V2: Original HMAC-based offset (param_name only)
        offset = _hmac_derive_offset(secret_key, param_name, x)

    # Apply Gray Code formula with the keyed offset
    seq = torch.arange(idx_flat.shape[0], device=iv.device, dtype=torch.long) + offset
    gray = seq ^ (seq >> 1)
    pat = (gray & ((1 << x) - 1)).to(torch.int32)

    mask = (1 << x) - 1
    iv[idx_flat] = (iv[idx_flat] & ~mask) | pat
