from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from ..lsb.bits import apply_pattern_mask

@dataclass
class PatternMaskReport:
    pattern: str
    n_params: int
    n_indices: int

class PatternMaskDefense:
    """
    Pattern-based LSB masking defense against steganographic payload injection.

    This defense overwrites the low x bits of targeted float32 parameters with
    a deterministic binary pattern (e.g., "0000", "1111", "0101"). The goal is
    to destroy hidden payloads embedded in LSBs while minimizing impact on model
    accuracy due to the structured nature of the perturbation.

    Defense Mechanism:
        - Overwrites LSBs with fixed pattern across all targeted parameters
        - Pattern repeats cyclically if needed for multi-bit masking
        - More predictable than random flipping but can be effective against
          naive steganographic attacks

    Typical Patterns (for x=4 bits):
        - "0000": All zeros (minimizes positive perturbations)
        - "1111": All ones (maximizes LSB values)
        - "0101": Alternating pattern (medium entropy)
        - "1010": Inverse alternating pattern

    Advantages:
        - Deterministic and reproducible (no seed required)
        - Low overhead (simple bit masking operation)
        - Can preserve model performance with carefully chosen patterns

    Limitations:
        - Predictable patterns may be less effective than randomized defenses
        - Adaptive attackers could potentially design patterns that survive
        - Pattern choice requires empirical validation per model architecture

    Example:
        >>> defense = PatternMaskDefense()
        >>> report = defense.apply(
        ...     model=poisoned_model,
        ...     target_names=["encoder.layer.0.attention.query.weight"],
        ...     x=4,
        ...     pattern="0101"
        ... )
        >>> print(f"Applied pattern '{report.pattern}' to {report.n_params} parameters")
    """

    def apply(self, model: torch.nn.Module, target_names: List[str], x: int, pattern: str, indices: Optional[Dict[str, torch.Tensor]] = None) -> PatternMaskReport:
        """
        Apply pattern masking defense to model parameters in-place.

        Args:
            model: PyTorch model to defend (modified in-place)
            target_names: List of parameter names to apply defense to
            x: Number of LSBs to overwrite (must match pattern length)
            pattern: Binary pattern string (e.g., "0101") of length x.
                     Must contain only '0' and '1' characters.
            indices: Optional per-parameter flat indices to target.
                     If None, all elements in each parameter are masked.
                     Format: {param_name: torch.Tensor of flat indices}

        Returns:
            PatternMaskReport containing:
                - pattern: The applied binary pattern
                - n_params: Number of parameters processed
                - n_indices: Total number of indices masked (if indices provided)

        Raises:
            ValueError: If pattern length != x or pattern contains non-binary chars

        Note:
            - Model parameters are modified in-place (no copy created)
            - Only float32 parameters are processed (others skipped)
            - If indices are provided, only those specific elements are masked
            - Pattern must be exactly x characters long

        Example:
            >>> defense = PatternMaskDefense()
            >>> # Apply alternating pattern to low 4 bits
            >>> report = defense.apply(model, targets, x=4, pattern="0101")
            >>> print(f"Masked {report.n_params} parameters")
        """
        named = dict(model.named_parameters())
        n_idx = 0
        for n in target_names:
            p = named.get(n, None)
            if p is None or p.data.dtype != torch.float32:
                continue
            idx = None if indices is None else indices.get(n, None)
            if idx is not None:
                n_idx += int(idx.numel())
            apply_pattern_mask(p.data, x=x, pattern=pattern, idx_flat=idx)
        return PatternMaskReport(pattern=pattern, n_params=len(target_names), n_indices=n_idx)
