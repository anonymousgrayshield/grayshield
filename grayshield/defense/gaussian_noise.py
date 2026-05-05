"""
Gaussian Noise Defense Baseline

Adds Gaussian noise to model weights as a standard baseline defense.
This is a common approach in ML security for perturbing model weights.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class GaussianNoiseReport:
    """Report for Gaussian noise defense application."""
    sigma: float
    n_params_modified: int
    total_elements: int
    mean_perturbation: float


class GaussianNoiseDefense:
    """
    Defense that adds Gaussian noise to target parameters.

    This is a standard baseline that perturbs weights with N(0, sigma^2) noise,
    which can disrupt steganographic payloads while potentially affecting
    model utility.
    """

    def apply(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        sigma: float = 1e-5,
        seed: int = 0,
        indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> GaussianNoiseReport:
        """
        Apply Gaussian noise to target parameters.

        Args:
            model: Model to modify (in-place)
            target_names: List of parameter names to perturb
            sigma: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
            indices: Optional dict mapping param name to flat indices to perturb
                     If None, perturbs all elements

        Returns:
            GaussianNoiseReport with statistics
        """
        device = next(model.parameters()).device
        gen = torch.Generator(device='cpu')  # Generator on CPU for compatibility
        gen.manual_seed(seed)

        named = dict(model.named_parameters())
        n_params = 0
        total_elements = 0
        perturbations = []

        for name in target_names:
            param = named.get(name)
            if param is None or param.data.dtype != torch.float32:
                continue

            data = param.data

            if indices is not None and name in indices:
                # Only perturb specified indices
                idx = indices[name]
                flat = data.view(-1)
                noise = torch.randn(idx.shape[0], generator=gen) * sigma
                noise = noise.to(device)
                flat[idx] += noise
                total_elements += idx.shape[0]
                perturbations.append(noise.abs().mean().item())
            else:
                # Perturb all elements
                noise = torch.randn(data.size(), generator=gen, dtype=data.dtype) * sigma
                noise = noise.to(device)
                data.add_(noise)
                total_elements += data.numel()
                perturbations.append(noise.abs().mean().item())

            n_params += 1

        mean_perturbation = sum(perturbations) / len(perturbations) if perturbations else 0.0

        return GaussianNoiseReport(
            sigma=sigma,
            n_params_modified=n_params,
            total_elements=total_elements,
            mean_perturbation=mean_perturbation,
        )
