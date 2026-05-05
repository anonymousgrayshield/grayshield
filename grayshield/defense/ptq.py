"""
Static PTQ Defense Baseline

Implements a lightweight static PTQ-style defense that:
1. Runs a short calibration pass on clean data.
2. Projects target FP32 weights onto a symmetric 8-bit grid.
3. Dequantizes back to FP32 in-place for compatibility with existing runners.

This preserves the current experiment architecture while destroying the original
mantissa LSB channel by snapping weights to a quantized lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


@dataclass
class PTQReport:
    n_params_quantized: int
    n_calibration_batches: int
    n_activation_observers: int
    avg_activation_absmax: float
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float


class PTQDefense:
    """
    Static PTQ-style weight projection defense.

    We keep the forward path in FP32 after quantize-dequantize so the existing
    evaluation/extraction pipeline remains unchanged, while the underlying FP32
    mantissa payload is destroyed by the 8-bit projection.
    """

    def _collect_activation_stats(
        self,
        model: torch.nn.Module,
        calibration_loader: Optional[Iterable],
        device: str,
        max_batches: int,
    ) -> tuple[int, int, float]:
        if calibration_loader is None or max_batches <= 0:
            return 0, 0, 0.0

        stats: List[float] = []
        handles = []

        def hook(_module, _inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            if torch.is_tensor(out):
                stats.append(float(out.detach().abs().max().cpu()))

        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                handles.append(module.register_forward_hook(hook))

        model.eval()
        n_batches = 0
        with torch.no_grad():
            for batch in calibration_loader:
                inputs = {k: v.to(device) for k, v in batch.inputs.items()}
                model(**inputs)
                n_batches += 1
                if n_batches >= max_batches:
                    break

        for handle in handles:
            handle.remove()

        avg_absmax = sum(stats) / len(stats) if stats else 0.0
        return n_batches, len(handles), avg_absmax

    def apply(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        calibration_loader: Optional[Iterable] = None,
        device: str = "cpu",
        max_calibration_batches: int = 8,
    ) -> PTQReport:
        named = dict(model.named_parameters())
        original_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        n_calibration_batches, n_observers, avg_absmax = self._collect_activation_stats(
            model, calibration_loader, device, max_calibration_batches
        )

        n_params_quantized = 0
        quantized_bytes = 0

        for name in target_names:
            param = named.get(name)
            if param is None or param.data.dtype != torch.float32:
                continue

            data = param.data
            max_abs = float(data.abs().max().item())
            if max_abs == 0.0:
                scale = 1.0
            else:
                scale = max_abs / 127.0

            q = torch.clamp(torch.round(data / scale), -127, 127).to(torch.int8)
            data.copy_(q.to(torch.float32) * scale)

            n_params_quantized += 1
            quantized_bytes += q.numel()

        quantized_size_mb = quantized_bytes / (1024 * 1024)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0

        return PTQReport(
            n_params_quantized=n_params_quantized,
            n_calibration_batches=n_calibration_batches,
            n_activation_observers=n_observers,
            avg_activation_absmax=avg_absmax,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
        )
