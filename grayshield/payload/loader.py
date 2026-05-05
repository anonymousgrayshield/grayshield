from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import logging
from ..utils.hashing import sha256_file

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Payload:
    path: str
    sha256: str
    n_bytes: int
    n_bits: int
    file_type: Optional[str]
    bits: List[int]

def _infer_file_type(path: str) -> Optional[str]:
    p = Path(path)
    return p.suffix.lstrip(".").lower() if p.suffix else None

def load_payload_bits(path: str, max_bits: Optional[int] = None) -> Payload:
    '''
    Load payload bytes from a local file and convert to bit list (0/1).
    Bytes are treated as opaque data (never executed).

    Args:
        path: Path to payload file
        max_bits: Optional maximum bits to load (for intentional truncation)

    Raises:
        FileNotFoundError: If payload file doesn't exist
        ValueError: If file is too large (>100MB) or empty

    Note:
        If max_bits is specified and file is larger, payload will be truncated
        with a warning. This is intentional for capacity testing.
    '''
    # Maximum payload size: 100 MB (prevents OOM on large files)
    MAX_PAYLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")

    # Validate file size before reading
    file_size = p.stat().st_size
    if file_size == 0:
        raise ValueError(f"Payload file is empty: {path}")
    if file_size > MAX_PAYLOAD_SIZE_BYTES:
        raise ValueError(
            f"Payload file too large: {file_size / 1024 / 1024:.2f} MB "
            f"(max: {MAX_PAYLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB). "
            f"Large payloads may cause OOM errors."
        )

    b = p.read_bytes()
    original_bits = len(b) * 8

    # Handle intentional truncation (for capacity testing)
    if max_bits is not None and max_bits < original_bits:
        logger.warning(
            f"Payload truncated: {original_bits} bits → {max_bits} bits "
            f"({(original_bits - max_bits) / original_bits * 100:.1f}% discarded). "
            f"This is intentional for max_bits={max_bits} testing."
        )
        b = b[: max_bits // 8]

    bits = [int(bit) for byte in b for bit in f"{byte:08b}"]
    sha = sha256_file(p)
    return Payload(
        path=str(p),
        sha256=sha,
        n_bytes=len(b),
        n_bits=len(bits),
        file_type=_infer_file_type(str(p)),
        bits=bits,
    )
