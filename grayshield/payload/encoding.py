"""
Payload Encoding Strategies for Adaptive Attackers

Implements various encoding schemes that attackers might use to improve
payload recoverability under defense:

1. naive: No encoding (baseline)
2. repetition-k: Repeat each bit k times, decode via majority vote
3. interleave: Spread consecutive bits across positions to reduce burst errors
4. rs: Reed-Solomon RS(255,127) over GF(256), aligned with the paper

These encodings trade capacity for resilience:
- repetition-k reduces effective capacity by factor k
- interleave maintains capacity but improves burst error resilience
- RS roughly doubles payload size while correcting up to 64 byte errors/codeword
"""
from __future__ import annotations
import math
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from .reed_solomon import (
    RS_K,
    RS_N,
    pack_bits_to_bytes,
    rs_decode_bytes,
    rs_encode_bytes,
    unpack_bytes_to_bits,
)


class AttackerVariant(str, Enum):
    """Attacker encoding strategies."""
    NAIVE = "naive"
    REPEAT3 = "repeat3"
    REPEAT5 = "repeat5"
    INTERLEAVE = "interleave"
    RS = "rs"


@dataclass
class EncodingReport:
    """Report of encoding operation."""
    variant: str
    original_bits: int
    encoded_bits: int
    expansion_factor: float
    params: dict  # Encoding-specific parameters


@dataclass
class DecodingReport:
    """Report of decoding operation."""
    variant: str
    encoded_bits: int
    decoded_bits: int
    corrected_errors: int  # Estimated errors corrected (for repetition codes)
    params: dict


# =============================================================================
# Repetition Codes
# =============================================================================

def encode_repetition(bits: List[int], k: int = 3) -> Tuple[List[int], EncodingReport]:
    """Encode bits using repetition code.

    Each bit is repeated k times. Decoding uses majority vote.
    Trade-off: Reduces effective capacity by factor k, but improves
    error resilience.

    Args:
        bits: Original payload bits
        k: Repetition factor (must be odd for clean majority vote)

    Returns:
        Tuple of (encoded_bits, report)
    """
    if k < 1:
        raise ValueError(f"Repetition factor k must be >= 1, got {k}")
    if k % 2 == 0:
        # Even k works but ties are broken arbitrarily
        pass  # Allow but warn in report

    encoded = []
    for bit in bits:
        encoded.extend([bit] * k)

    report = EncodingReport(
        variant=f"repeat{k}",
        original_bits=len(bits),
        encoded_bits=len(encoded),
        expansion_factor=k,
        params={"k": k, "even_k_warning": k % 2 == 0},
    )

    return encoded, report


def decode_repetition(encoded: List[int], k: int = 3) -> Tuple[List[int], DecodingReport]:
    """Decode repetition-encoded bits using majority vote.

    Args:
        encoded: Encoded bits (length should be multiple of k)
        k: Repetition factor used during encoding

    Returns:
        Tuple of (decoded_bits, report)
    """
    if len(encoded) % k != 0:
        # Truncate to nearest multiple of k
        encoded = encoded[:len(encoded) - (len(encoded) % k)]

    decoded = []
    corrected = 0
    threshold = k // 2 + 1  # Majority threshold

    for i in range(0, len(encoded), k):
        chunk = encoded[i:i+k]
        ones = sum(chunk)
        zeros = k - ones

        if ones >= threshold:
            decoded.append(1)
            # Count "corrected" errors (bits that disagreed with majority)
            corrected += zeros
        else:
            decoded.append(0)
            corrected += ones

    # Note: corrected is an upper bound; some "errors" may be original bit flips
    # that were correctly recovered

    report = DecodingReport(
        variant=f"repeat{k}",
        encoded_bits=len(encoded),
        decoded_bits=len(decoded),
        corrected_errors=corrected,
        params={"k": k, "threshold": threshold},
    )

    return decoded, report


# =============================================================================
# Interleaving
# =============================================================================

def _generate_interleave_permutation(n: int, seed: int = 42) -> List[int]:
    """Generate deterministic permutation for interleaving."""
    rng = np.random.default_rng(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def _generate_deinterleave_permutation(n: int, seed: int = 42) -> List[int]:
    """Generate inverse permutation for deinterleaving."""
    perm = _generate_interleave_permutation(n, seed)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm


def encode_interleave(bits: List[int], seed: int = 42) -> Tuple[List[int], EncodingReport]:
    """Interleave bits to spread consecutive bits across positions.

    This improves resilience against burst errors (consecutive bit flips)
    by distributing adjacent payload bits across different positions.

    Args:
        bits: Original payload bits
        seed: Seed for deterministic permutation

    Returns:
        Tuple of (interleaved_bits, report)
    """
    n = len(bits)
    perm = _generate_interleave_permutation(n, seed)

    interleaved = [0] * n
    for i, p in enumerate(perm):
        interleaved[p] = bits[i]

    report = EncodingReport(
        variant="interleave",
        original_bits=n,
        encoded_bits=n,
        expansion_factor=1.0,
        params={"seed": seed},
    )

    return interleaved, report


def decode_interleave(encoded: List[int], seed: int = 42) -> Tuple[List[int], DecodingReport]:
    """Deinterleave bits to recover original order.

    Args:
        encoded: Interleaved bits
        seed: Same seed used during encoding

    Returns:
        Tuple of (deinterleaved_bits, report)
    """
    n = len(encoded)
    inv_perm = _generate_deinterleave_permutation(n, seed)

    deinterleaved = [0] * n
    for i, p in enumerate(inv_perm):
        deinterleaved[p] = encoded[i]

    report = DecodingReport(
        variant="interleave",
        encoded_bits=n,
        decoded_bits=n,
        corrected_errors=0,  # Interleaving doesn't correct errors
        params={"seed": seed},
    )

    return deinterleaved, report


# =============================================================================
# Combined: Repetition + Interleaving
# =============================================================================

def encode_repeat_interleave(
    bits: List[int],
    k: int = 3,
    interleave_seed: int = 42,
) -> Tuple[List[int], EncodingReport]:
    """Combine repetition and interleaving for maximum resilience.

    First applies repetition (for error correction), then interleaving
    (to spread burst errors across code words).

    Args:
        bits: Original payload bits
        k: Repetition factor
        interleave_seed: Seed for interleaving

    Returns:
        Tuple of (encoded_bits, report)
    """
    # Step 1: Repetition
    repeated, rep_report = encode_repetition(bits, k)

    # Step 2: Interleave
    interleaved, int_report = encode_interleave(repeated, interleave_seed)

    report = EncodingReport(
        variant=f"repeat{k}_interleave",
        original_bits=len(bits),
        encoded_bits=len(interleaved),
        expansion_factor=k,
        params={
            "k": k,
            "interleave_seed": interleave_seed,
        },
    )

    return interleaved, report


def decode_repeat_interleave(
    encoded: List[int],
    k: int = 3,
    interleave_seed: int = 42,
) -> Tuple[List[int], DecodingReport]:
    """Decode repetition+interleaving.

    First deinterleave, then apply majority vote.

    Args:
        encoded: Encoded bits
        k: Repetition factor
        interleave_seed: Same seed used during encoding

    Returns:
        Tuple of (decoded_bits, report)
    """
    # Step 1: Deinterleave
    deinterleaved, _ = decode_interleave(encoded, interleave_seed)

    # Step 2: Majority vote
    decoded, dec_report = decode_repetition(deinterleaved, k)

    report = DecodingReport(
        variant=f"repeat{k}_interleave",
        encoded_bits=len(encoded),
        decoded_bits=len(decoded),
        corrected_errors=dec_report.corrected_errors,
        params={
            "k": k,
            "interleave_seed": interleave_seed,
        },
    )

    return decoded, report


# =============================================================================
# Reed-Solomon: RS(255,127)
# =============================================================================

def encode_reed_solomon(bits: List[int]) -> Tuple[List[int], EncodingReport]:
    """Encode payload bits with chunked RS(255,127) over GF(256)."""
    data_bytes, dropped_tail_bits = pack_bits_to_bytes(bits)
    if dropped_tail_bits:
        raise ValueError(
            f"RS attacker requires byte-aligned payload bits, got tail={dropped_tail_bits}"
        )

    encoded_bytes, rs_report = rs_encode_bytes(data_bytes)
    encoded_bits = unpack_bytes_to_bits(encoded_bytes)
    report = EncodingReport(
        variant="rs",
        original_bits=len(bits),
        encoded_bits=len(encoded_bits),
        expansion_factor=(len(encoded_bits) / len(bits)) if bits else 1.0,
        params={
            "code": f"RS({RS_N},{RS_K})",
            "code_n": RS_N,
            "code_k": RS_K,
            "blocks_total": rs_report.blocks_total,
            "padding_bytes": rs_report.padding_bytes,
        },
    )
    return encoded_bits, report


def decode_reed_solomon(
    encoded: List[int],
    original_length: Optional[int] = None,
) -> Tuple[List[int], DecodingReport]:
    """Decode a chunked RS(255,127) payload, falling back on systematic bytes."""
    encoded_bytes, dropped_tail_bits = pack_bits_to_bytes(encoded)
    original_n_bits = original_length if original_length is not None else None
    original_n_bytes = math.ceil(original_n_bits / 8) if original_n_bits is not None else None

    decoded_bytes, rs_report = rs_decode_bytes(encoded_bytes, original_n_bytes=original_n_bytes)
    decoded_bits = unpack_bytes_to_bits(decoded_bytes, n_bits=original_n_bits)

    report = DecodingReport(
        variant="rs",
        encoded_bits=len(encoded) - dropped_tail_bits,
        decoded_bits=len(decoded_bits),
        corrected_errors=rs_report.corrected_symbol_errors,
        params={
            "code": f"RS({RS_N},{RS_K})",
            "code_n": RS_N,
            "code_k": RS_K,
            "blocks_total": rs_report.blocks_total,
            "blocks_decoded": rs_report.blocks_decoded,
            "blocks_failed": rs_report.blocks_failed,
            "blocks_truncated": rs_report.blocks_truncated,
            "padding_bytes": rs_report.padding_bytes,
            "dropped_tail_bits": dropped_tail_bits,
            "corrected_error_unit": "bytes",
        },
    )
    return decoded_bits, report


# =============================================================================
# Unified Interface
# =============================================================================

def encode_payload(
    bits: List[int],
    variant: AttackerVariant = AttackerVariant.NAIVE,
    **kwargs,
) -> Tuple[List[int], EncodingReport]:
    """Encode payload using specified attacker variant.

    Args:
        bits: Original payload bits
        variant: Encoding strategy
        **kwargs: Variant-specific parameters (k, seed, etc.)

    Returns:
        Tuple of (encoded_bits, report)
    """
    if variant == AttackerVariant.NAIVE:
        report = EncodingReport(
            variant="naive",
            original_bits=len(bits),
            encoded_bits=len(bits),
            expansion_factor=1.0,
            params={},
        )
        return bits.copy(), report

    elif variant == AttackerVariant.REPEAT3:
        return encode_repetition(bits, k=3)

    elif variant == AttackerVariant.REPEAT5:
        return encode_repetition(bits, k=5)

    elif variant == AttackerVariant.INTERLEAVE:
        seed = kwargs.get("interleave_seed", 42)
        return encode_interleave(bits, seed=seed)

    elif variant == AttackerVariant.RS:
        return encode_reed_solomon(bits)

    else:
        raise ValueError(f"Unknown attacker variant: {variant}")


def decode_payload(
    encoded: List[int],
    variant: AttackerVariant = AttackerVariant.NAIVE,
    original_length: Optional[int] = None,
    **kwargs,
) -> Tuple[List[int], DecodingReport]:
    """Decode payload using specified attacker variant.

    Args:
        encoded: Encoded/recovered bits
        variant: Encoding strategy used
        original_length: Expected original payload length (for truncation)
        **kwargs: Variant-specific parameters

    Returns:
        Tuple of (decoded_bits, report)
    """
    if variant == AttackerVariant.NAIVE:
        bits = encoded.copy()
        if original_length and len(bits) > original_length:
            bits = bits[:original_length]
        report = DecodingReport(
            variant="naive",
            encoded_bits=len(encoded),
            decoded_bits=len(bits),
            corrected_errors=0,
            params={},
        )
        return bits, report

    elif variant == AttackerVariant.REPEAT3:
        decoded, report = decode_repetition(encoded, k=3)
        if original_length and len(decoded) > original_length:
            decoded = decoded[:original_length]
            report.decoded_bits = len(decoded)
        return decoded, report

    elif variant == AttackerVariant.REPEAT5:
        decoded, report = decode_repetition(encoded, k=5)
        if original_length and len(decoded) > original_length:
            decoded = decoded[:original_length]
            report.decoded_bits = len(decoded)
        return decoded, report

    elif variant == AttackerVariant.INTERLEAVE:
        seed = kwargs.get("interleave_seed", 42)
        decoded, report = decode_interleave(encoded, seed=seed)
        if original_length and len(decoded) > original_length:
            decoded = decoded[:original_length]
            report.decoded_bits = len(decoded)
        return decoded, report

    elif variant == AttackerVariant.RS:
        return decode_reed_solomon(encoded, original_length=original_length)

    else:
        raise ValueError(f"Unknown attacker variant: {variant}")


# =============================================================================
# Theoretical Bounds (BSC Model)
# =============================================================================

def bsc_error_probability(k: int, p: float) -> float:
    """Compute word error probability for repetition-k code over BSC(p).

    Under Binary Symmetric Channel with bit flip probability p,
    a repetition-k code fails when more than k/2 bits are flipped.

    P_err(k, p) = sum_{i=ceil((k+1)/2)}^{k} C(k,i) * p^i * (1-p)^(k-i)

    Args:
        k: Repetition factor
        p: Bit flip probability

    Returns:
        Probability of decoding error per code word
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    threshold = (k + 1) // 2  # Number of flips needed to cause error
    error_prob = 0.0

    for i in range(threshold, k + 1):
        # Binomial coefficient C(k, i)
        coeff = math.comb(k, i)
        error_prob += coeff * (p ** i) * ((1 - p) ** (k - i))

    return error_prob


def predicted_recovery_rate(k: int, p: float) -> float:
    """Predict bit recovery rate for repetition-k code under flip_prob p.

    Args:
        k: Repetition factor (1 = naive)
        p: Bit flip probability

    Returns:
        Predicted fraction of bits correctly recovered
    """
    if k == 1:
        # Naive: each bit survives with probability 1-p
        return 1.0 - p

    # Repetition code: bit survives if majority vote succeeds
    return 1.0 - bsc_error_probability(k, p)


def generate_bound_curve(
    k_values: List[int] = [1, 3, 5],
    p_values: Optional[List[float]] = None,
) -> List[dict]:
    """Generate theoretical bound curves for plotting.

    Args:
        k_values: List of repetition factors to include
        p_values: Flip probabilities to evaluate (default: 0 to 0.5)

    Returns:
        List of records with {k, p, predicted_recovery}
    """
    if p_values is None:
        p_values = [i / 100 for i in range(51)]  # 0.00 to 0.50

    records = []
    for k in k_values:
        for p in p_values:
            records.append({
                "k": k,
                "flip_prob": p,
                "predicted_recovery": predicted_recovery_rate(k, p),
            })

    return records
