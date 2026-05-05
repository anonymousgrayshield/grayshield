from __future__ import annotations
from typing import List, Tuple
import hashlib


# =============================================================================
# Length ratio and truncation helpers
# =============================================================================

def length_ratio(original: List[int], recovered: List[int]) -> float:
    """
    Compute ratio of recovered length to original length.

    Args:
        original: Original bit sequence
        recovered: Recovered bit sequence

    Returns:
        len(recovered) / len(original), or 0.0 if original is empty
    """
    if len(original) == 0:
        return 0.0 if len(recovered) == 0 else float("inf")
    return len(recovered) / len(original)


def was_truncated(original: List[int], recovered: List[int]) -> bool:
    """
    Check if recovered sequence is shorter than original (truncation occurred).

    Returns:
        True if len(recovered) < len(original)
    """
    return len(recovered) < len(original)


# =============================================================================
# Bit accuracy metrics (prefix vs strict)
# =============================================================================

def bit_accuracy(a: List[int], b: List[int]) -> float:
    """
    Compute bit-level accuracy between two bit sequences (prefix-based).

    NOTE: This uses min(len(a), len(b)) - may overestimate recovery if truncated.
    For strict evaluation that penalizes truncation, use bit_accuracy_strict().

    Args:
        a: Original bit sequence
        b: Recovered bit sequence

    Returns:
        Fraction of matching bits over min-length prefix (0.0 to 1.0)
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if int(a[i]) == int(b[i])) / n


def bit_accuracy_strict(a: List[int], b: List[int]) -> float:
    """
    Compute strict bit-level accuracy that penalizes length mismatch.

    Missing bits (due to truncation) are counted as errors.
    Uses the MAXIMUM of len(a) and len(b) as denominator.

    Args:
        a: Original bit sequence
        b: Recovered bit sequence

    Returns:
        Fraction of matching bits over max-length (0.0 to 1.0)
        If lengths differ, missing bits count as errors.
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0  # Both empty = perfect match

    # Count matches only in overlapping region
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if int(a[i]) == int(b[i]))

    # Missing bits are implicitly counted as errors (not in matches)
    return matches / max_len


def hamming_distance(a: List[int], b: List[int]) -> int:
    """
    Compute Hamming distance between two bit sequences.
    
    Missing bits (due to truncation) are counted as errors.
    
    Args:
        a: Original bit sequence
        b: Recovered bit sequence
        
    Returns:
        Number of differing bits (integer)
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0

    min_len = min(len(a), len(b))
    dist = sum(1 for i in range(min_len) if int(a[i]) != int(b[i]))
    
    # Add difference in length as errors
    dist += (max_len - min_len)
    return dist


def ber(a: List[int], b: List[int]) -> float:
    """
    Compute Bit Error Rate (BER).

    Returns:
        Fraction of differing bits (0.0 to 1.0)
    """
    return 1.0 - bit_accuracy(a, b)


def exact_recovery(a: List[int], b: List[int], threshold: float = 1.0) -> bool:
    """
    Check if payload was recovered with accuracy >= threshold (prefix-based).

    NOTE: Uses prefix-based bit_accuracy which may overestimate recovery.
    For strict evaluation, use exact_recovery_strict().

    Args:
        a: Original bit sequence
        b: Recovered bit sequence
        threshold: Minimum accuracy for "successful" recovery (default 1.0 = perfect)

    Returns:
        True if bit_accuracy(a, b) >= threshold
    """
    return bit_accuracy(a, b) >= threshold


def exact_recovery_strict(a: List[int], b: List[int]) -> bool:
    """
    Strict exact recovery check: requires BOTH same length AND hash match.

    This is the definitive test for perfect payload recovery:
    - Length must be identical (no truncation)
    - All bits must match (verified via hash)

    Args:
        a: Original bit sequence
        b: Recovered bit sequence

    Returns:
        True only if len(a) == len(b) AND hash_match(a, b) is True
    """
    if len(a) != len(b):
        return False
    return hash_match(a, b)


def exact_recovery_rate(
    original_payloads: List[List[int]],
    recovered_payloads: List[List[int]],
    threshold: float = 1.0,
) -> float:
    """
    Compute the rate of exact/near-exact recovery across multiple payloads.

    Args:
        original_payloads: List of original bit sequences
        recovered_payloads: List of recovered bit sequences
        threshold: Minimum accuracy for "successful" recovery

    Returns:
        Fraction of payloads that meet the threshold
    """
    if not original_payloads:
        return 0.0
    success_count = sum(
        1 for orig, rec in zip(original_payloads, recovered_payloads)
        if exact_recovery(orig, rec, threshold)
    )
    return success_count / len(original_payloads)


def byte_recovery(a: List[int], b: List[int]) -> Tuple[float, int, int]:
    """
    Compute byte-level recovery statistics.

    Args:
        a: Original bit sequence
        b: Recovered bit sequence

    Returns:
        Tuple of (byte_accuracy, correct_bytes, total_bytes)
    """
    n_bits = min(len(a), len(b))
    n_bytes = n_bits // 8

    if n_bytes == 0:
        return 0.0, 0, 0

    correct = 0
    for i in range(n_bytes):
        start = i * 8
        orig_byte = a[start:start + 8]
        rec_byte = b[start:start + 8]
        if orig_byte == rec_byte:
            correct += 1

    return correct / n_bytes, correct, n_bytes


def hash_match(a: List[int], b: List[int]) -> bool:
    """
    Check if the SHA256 hash of bit sequences match.
    This is the strictest form of integrity verification.

    Returns:
        True if hashes match (perfect bit-for-bit recovery)
    """
    def bits_to_bytes(bits: List[int]) -> bytes:
        n_bytes = len(bits) // 8
        return bytes(
            sum(bits[i*8 + j] << (7 - j) for j in range(8))
            for i in range(n_bytes)
        )

    if len(a) != len(b):
        return False

    hash_a = hashlib.sha256(bits_to_bytes(a)).hexdigest()
    hash_b = hashlib.sha256(bits_to_bytes(b)).hexdigest()
    return hash_a == hash_b
