"""
Selective Weight Perturbation (SWP) Defense Baseline

Data-free approximation:
- Select the lowest-magnitude fraction of weights in each target tensor.
- Add Gaussian noise only on the selected subset.
- Calibrate sigma to match GrayShield's relative L2 perturbation at the same x.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import copy
import math
import os

import torch

from .gaussian_noise import GaussianNoiseDefense
from .gray_code import GrayShieldDefense
from ..metrics.model import relative_l2_distance


@dataclass
class SWPReport:
    fraction: float
    sigma: float
    n_params_modified: int
    n_selected_elements: int
    target_relative_l2: float
    achieved_relative_l2: float
    mean_perturbation: float


class SWPDefense:
    def _select_low_magnitude_indices(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        fraction: float,
    ) -> Dict[str, torch.Tensor]:
        named = dict(model.named_parameters())
        indices: Dict[str, torch.Tensor] = {}
        for name in target_names:
            param = named.get(name)
            if param is None or param.data.dtype != torch.float32:
                continue
            flat = param.data.view(-1).abs()
            k = max(1, int(math.ceil(flat.numel() * fraction)))
            if k >= flat.numel():
                indices[name] = torch.arange(flat.numel(), device=flat.device, dtype=torch.long)
                continue
            _, idx = torch.topk(flat, k, largest=False)
            indices[name] = idx
        return indices

    def _target_relative_l2(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        x: int,
        seed: int,
    ) -> float:
        ref = copy.deepcopy(model)
        use_v2 = bool(os.environ.get("GRAYSHIELD_KEY", ""))
        use_v3 = bool(os.environ.get("GRAYSHIELD_V3", ""))
        GrayShieldDefense().apply(
            ref,
            target_names,
            x=x,
            seed=seed,
            use_v2=use_v2,
            use_v3=use_v3,
        )
        return relative_l2_distance(model, ref, target_names)

    def _base_norm(self, model: torch.nn.Module, target_names: List[str]) -> float:
        named = dict(model.named_parameters())
        total_sq = 0.0
        for name in target_names:
            param = named.get(name)
            if param is None or param.data.dtype != torch.float32:
                continue
            total_sq += float(param.data.float().pow(2).sum().item())
        return math.sqrt(max(total_sq, 1e-24))

    def apply(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        x: int,
        seed: int = 0,
        fraction: float = 0.20,
    ) -> SWPReport:
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"SWP fraction must be in (0,1], got {fraction}")

        model_before = copy.deepcopy(model)
        indices = self._select_low_magnitude_indices(model, target_names, fraction)
        n_selected = sum(int(idx.numel()) for idx in indices.values())

        target_rel_l2 = self._target_relative_l2(model_before, target_names, x=x, seed=seed)
        base_norm = self._base_norm(model_before, target_names)
        sigma = max(target_rel_l2 * base_norm / math.sqrt(max(1, n_selected)), 1e-12)

        rep = GaussianNoiseDefense().apply(
            model,
            target_names,
            sigma=sigma,
            seed=seed,
            indices=indices,
        )
        achieved_rel_l2 = relative_l2_distance(model_before, model, target_names)

        return SWPReport(
            fraction=fraction,
            sigma=sigma,
            n_params_modified=rep.n_params_modified,
            n_selected_elements=n_selected,
            target_relative_l2=target_rel_l2,
            achieved_relative_l2=achieved_rel_l2,
            mean_perturbation=rep.mean_perturbation,
        )
