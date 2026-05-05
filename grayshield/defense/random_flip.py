from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from ..lsb.bits import apply_random_flips

@dataclass
class RandomFlipReport:
    total_bit_flips: int
    per_param_flips: Dict[str, int]

class RandomFlipDefense:
    """
    Randomized bit-flipping defense against LSB steganographic payloads.

    This defense probabilistically flips bits in the low x LSBs of targeted
    float32 parameters. Each bit is independently flipped with probability p,
    introducing random noise that destroys hidden payloads while allowing
    control over the perturbation strength via the flip probability.

    Defense Mechanism:
        - For each targeted parameter, iterate over low x bits
        - Flip each bit independently with probability flip_prob
        - Uses seeded random generator for reproducibility
        - Expected bit flips = total_bits × flip_prob

    Key Parameters:
        - flip_prob: Controls defense strength (0.0 = no defense, 0.5 = maximum entropy)
        - x: Number of LSBs to consider (typically 4-23 bits)
        - seed: Random seed for reproducible experiments

    Theoretical Analysis:
        - Expected payload recovery = (1 - flip_prob)^n for n-bit payload
        - flip_prob = 0.01: ~99% bits survive (weak defense)
        - flip_prob = 0.1: ~90% bits survive (moderate defense)
        - flip_prob = 0.5: ~50% bits survive (strong defense)

    Advantages:
        - Tunable defense strength via flip_prob parameter
        - Probabilistically uniform coverage across all LSBs
        - Effective against naive and interleaved steganography
        - Low computational overhead

    Limitations:
        - Adaptive attackers can use error correction codes (ECC) to recover
          payloads (e.g., repeat3, repeat5 encoding with majority voting)
        - High flip_prob may degrade model accuracy
        - Randomness introduces variability in defense effectiveness

    Comparison with Pattern Masking:
        - RandomFlip: Stochastic, requires seed, harder to predict
        - PatternMask: Deterministic, no seed, more predictable

    Example:
        >>> defense = RandomFlipDefense()
        >>> report = defense.apply(
        ...     model=poisoned_model,
        ...     target_names=["encoder.layer.0.attention.query.weight"],
        ...     x=19,
        ...     flip_prob=0.1,  # Flip 10% of LSBs
        ...     seed=42
        ... )
        >>> print(f"Flipped {report.total_bit_flips} bits total")
        >>> print(f"Recovery reduction: {(1 - 0.9**n) * 100:.1f}%")
    """

    def apply(self, model: torch.nn.Module, target_names: List[str], x: int, flip_prob: float, indices: Optional[Dict[str, torch.Tensor]] = None, seed: int = 0) -> RandomFlipReport:
        """
        Apply random bit-flipping defense to model parameters in-place.

        Args:
            model: PyTorch model to defend (modified in-place)
            target_names: List of parameter names to apply defense to
            x: Number of LSBs to flip (1-23 for float32)
            flip_prob: Probability of flipping each bit (0.0-1.0).
                       Typical values: 0.01 (weak), 0.1 (moderate), 0.5 (strong)
            indices: Optional per-parameter flat indices to target.
                     If None, all elements in each parameter are processed.
                     Format: {param_name: torch.Tensor of flat indices}
            seed: Random seed for reproducibility (default: 0)

        Returns:
            RandomFlipReport containing:
                - total_bit_flips: Total number of bits flipped across all parameters
                - per_param_flips: {param_name: n_bits_flipped} for each parameter

        Raises:
            ValueError: If flip_prob not in [0.0, 1.0] or x not in [1, 23]

        Note:
            - Model parameters are modified in-place (no copy created)
            - Only float32 parameters are processed (others skipped)
            - Bit flips are independent Bernoulli trials (not exactly flip_prob * total_bits)
            - Uses same seed across all parameters for consistency
            - Expected flips = (total_bits × flip_prob) ± sqrt(total_bits × p × (1-p))

        Example:
            >>> defense = RandomFlipDefense()
            >>> # Flip 10% of low 19 bits with seed 42
            >>> report = defense.apply(model, targets, x=19, flip_prob=0.1, seed=42)
            >>> print(f"Total flips: {report.total_bit_flips}")
            >>> for param, flips in report.per_param_flips.items():
            ...     print(f"  {param}: {flips} bits flipped")
        """
        gen = torch.Generator(device=next(model.parameters()).device)
        gen.manual_seed(seed)
        named = dict(model.named_parameters())

        total = 0
        per: Dict[str, int] = {}
        for n in target_names:
            p = named.get(n, None)
            if p is None or p.data.dtype != torch.float32:
                continue
            idx = None if indices is None else indices.get(n, None)
            flips = apply_random_flips(p.data, x=x, flip_prob=flip_prob, idx_flat=idx, generator=gen)
            per[n] = flips
            total += flips
        return RandomFlipReport(total_bit_flips=total, per_param_flips=per)
