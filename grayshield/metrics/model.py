from __future__ import annotations
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
import torch
import numpy as np
import math
import torch


# def cosine_similarity_on_targets(
#     model_a: torch.nn.Module,
#     model_b: torch.nn.Module,
#     target_names: List[str],
# ) -> float:
#     """
#     Compute cosine similarity between model weights on specified target parameters.

#     Args:
#         model_a: Original model
#         model_b: Modified model
#         target_names: List of parameter names to compare

#     Returns:
#         Cosine similarity (1.0 = identical, 0.0 = orthogonal)
#     """
#     da = dict(model_a.named_parameters())
#     db = dict(model_b.named_parameters())
#     xs = []
#     ys = []
#     for n in target_names:
#         if n in da and n in db:
#             xs.append(da[n].detach().flatten().float().cpu())
#             ys.append(db[n].detach().flatten().float().cpu())
#     if not xs:
#         return 1.0
#     x = torch.cat(xs)
#     y = torch.cat(ys)
#     return float(torch.nn.functional.cosine_similarity(x, y, dim=0).item())



def cosine_similarity_on_targets(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    target_names: List[str],
) -> float:
    """
    Compute cosine similarity between model weights using streaming accumulation.

    Uses float64 for numerical stability and processes parameters one at a time
    to avoid creating large concatenated tensors (memory efficient).

    Args:
        model_a: Original model
        model_b: Modified model
        target_names: List of parameter names to compare

    Returns:
        Cosine similarity in [-1.0, 1.0] (clamped for numerical stability)
    """
    da = dict(model_a.named_parameters())
    db = dict(model_b.named_parameters())

    # Use Kahan summation for improved numerical stability
    dot_sum = 0.0
    dot_c = 0.0  # Compensation for lost low-order bits
    na_sum = 0.0
    na_c = 0.0
    nb_sum = 0.0
    nb_c = 0.0

    for n in target_names:
        if n not in da or n not in db:
            continue

        # Use double precision on CPU for maximum stability
        a = da[n].detach().reshape(-1).double().cpu()
        b = db[n].detach().reshape(-1).double().cpu()

        # Kahan summation for dot product
        y = float((a * b).sum().item()) - dot_c
        t = dot_sum + y
        dot_c = (t - dot_sum) - y
        dot_sum = t

        # Kahan summation for norm_a^2
        y = float((a * a).sum().item()) - na_c
        t = na_sum + y
        na_c = (t - na_sum) - y
        na_sum = t

        # Kahan summation for norm_b^2
        y = float((b * b).sum().item()) - nb_c
        t = nb_sum + y
        nb_c = (t - nb_sum) - y
        nb_sum = t

    # Handle edge cases
    if na_sum <= 0.0 or nb_sum <= 0.0:
        # Both zero vectors = identical, one zero = undefined (return 1.0 as convention)
        return 1.0

    # Compute cosine similarity with explicit clamping
    denom = math.sqrt(na_sum) * math.sqrt(nb_sum)
    cos = dot_sum / denom

    # Clamp to valid range [-1, 1] to handle floating-point errors
    # that might cause values slightly outside this range
    cos = max(-1.0, min(1.0, cos))

    return float(cos)


def weight_norm_ratio(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    target_names: List[str],
) -> float:
    """
    Compute the ratio of weight norms: ||model_b|| / ||model_a||.

    Values close to 1.0 indicate minimal weight magnitude change.

    Returns:
        Norm ratio (ideally ~1.0)
    """
    da = dict(model_a.named_parameters())
    db = dict(model_b.named_parameters())

    norm_a = 0.0
    norm_b = 0.0

    for n in target_names:
        if n in da and n in db:
            norm_a += float(da[n].detach().float().norm().item() ** 2)
            norm_b += float(db[n].detach().float().norm().item() ** 2)

    if norm_a == 0:
        return 1.0 if norm_b == 0 else float("inf")

    return (norm_b ** 0.5) / (norm_a ** 0.5)


def l2_distance(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    target_names: List[str],
) -> float:
    """
    Compute L2 distance (Euclidean distance) between model weights.

    This metric measures the magnitude of weight perturbation caused by
    payload injection or defense application.

    For LSB steganography:
    - Low L2 distance = stealthy injection (attacker goal)
    - Low L2 distance after defense = model utility preserved (defender goal)

    Args:
        model_a: Original/clean model
        model_b: Modified model (poisoned or defended)
        target_names: List of parameter names to compare

    Returns:
        L2 distance: sqrt(sum((w_a - w_b)^2))
    """
    da = dict(model_a.named_parameters())
    db = dict(model_b.named_parameters())

    diff_sq_sum = 0.0

    for n in target_names:
        if n in da and n in db:
            diff = da[n].detach().float() - db[n].detach().float()
            diff_sq_sum += float((diff ** 2).sum().cpu().item())

    return float(diff_sq_sum ** 0.5)


def relative_l2_distance(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    target_names: List[str],
) -> float:
    """
    Compute relative L2 distance: ||w_a - w_b|| / ||w_a||.

    This normalized version is useful for comparing across different
    model sizes and architectures.

    Args:
        model_a: Original/clean model
        model_b: Modified model
        target_names: List of parameter names to compare

    Returns:
        Relative L2 distance (0 = identical, higher = more different)
    """
    da = dict(model_a.named_parameters())
    db = dict(model_b.named_parameters())

    diff_sq_sum = 0.0
    norm_a_sq = 0.0

    for n in target_names:
        if n in da and n in db:
            a = da[n].detach().float()
            b = db[n].detach().float()
            diff_sq_sum += float(((a - b) ** 2).sum().cpu().item())
            norm_a_sq += float((a ** 2).sum().cpu().item())

    if norm_a_sq == 0:
        return 0.0 if diff_sq_sum == 0 else float("inf")

    return float((diff_sq_sum ** 0.5) / (norm_a_sq ** 0.5))


def weight_distribution_distance(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    target_names: List[str],
) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between weight distributions.
    
    This measures the statistical shift in the weight distribution, tracking if a defense
    preserves the original probability distribution of weights (stealthiness).
    
    Formula for 1D W_1 distance: integral of absolute difference between inverse CDFs,
    which optimally aligns with mean absolute error of sorted arrays.
    
    Args:
        model_a: Original/clean model
        model_b: Modified model (poisoned or defended)
        target_names: List of parameter names to compare
        
    Returns:
        Wasserstein distance (W_1)
    """
    da = dict(model_a.named_parameters())
    db = dict(model_b.named_parameters())

    xs = []
    ys = []
    
    for n in target_names:
        if n in da and n in db:
            xs.append(da[n].detach().float().reshape(-1))
            ys.append(db[n].detach().float().reshape(-1))
            
    if not xs:
        return 0.0
        
    x = torch.cat(xs)
    y = torch.cat(ys)
    
    if x.numel() == 0:
        return 0.0
        
    # 1D Wasserstein distance for uniformly weighted samples is simply the MAE of their sorted values
    # Fast GPU-accelerated sorting
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    
    w1_dist = (x_sorted - y_sorted).abs().mean()
    
    return float(w1_dist.cpu().item())


@torch.no_grad()
def logits_kl_div(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: str,
) -> float:
    """
    Compute KL divergence between model output distributions.

    Direction: D_KL(P || Q) where P = model_a (reference), Q = model_b (modified)
    This measures "information lost" when approximating P with Q.

    Formula: D_KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log_P - log_Q))

    Note: KL divergence is always >= 0 by theory. Small negative values
    (e.g., -1e-9) can occur due to floating-point errors and are clamped to 0.

    Args:
        model_a: Reference model (P) - typically the clean/original model
        model_b: Modified model (Q) - typically the injected/defended model
        loader: DataLoader with evaluation batches
        device: Device to run on

    Returns:
        Average KL divergence across batches, guaranteed >= 0.0
    """
    import torch.nn.functional as F

    model_a.eval()
    model_b.eval()

    kls = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}

        la = model_a(**inputs).logits
        lb = model_b(**inputs).logits

        # Use double precision for numerical stability
        la_d = la.double()
        lb_d = lb.double()

        # Compute log probabilities with log_softmax (numerically stable)
        log_pa = F.log_softmax(la_d, dim=-1)
        log_pb = F.log_softmax(lb_d, dim=-1)

        # KL divergence: D_KL(P || Q) = sum(P * (log_P - log_Q))
        # Using exp(log_pa) is more stable than softmax when we already have log_softmax
        pa = log_pa.exp()
        kl_per_sample = torch.sum(pa * (log_pa - log_pb), dim=-1)
        kl = kl_per_sample.mean()

        # Clamp to non-negative (handles floating-point errors)
        kl_val = max(0.0, float(kl.cpu().item()))
        kls.append(kl_val)

    result = float(np.mean(kls)) if kls else 0.0

    # Final safety clamp (should be unnecessary but belt-and-suspenders)
    return max(0.0, result)


@torch.no_grad()
def top1_agreement(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: str,
) -> float:
    """
    Compute top-1 prediction agreement between two models.

    Args:
        model_a: Reference model
        model_b: Modified model
        loader: DataLoader with evaluation batches
        device: Device to run on

    Returns:
        Fraction of samples where both models predict the same class
    """
    model_a.eval()
    model_b.eval()

    agree = 0
    total = 0

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}

        pred_a = model_a(**inputs).logits.argmax(dim=-1)
        pred_b = model_b(**inputs).logits.argmax(dim=-1)

        agree += int((pred_a == pred_b).sum().item())
        total += pred_a.numel()

    return agree / total if total > 0 else 1.0


@torch.no_grad()
def paired_prediction_diagnostics(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: str,
) -> Dict[str, Any]:
    """
    Compute detailed paired prediction diagnostics for sensitivity analysis.

    This is more sensitive than simple accuracy comparison because it tracks
    exactly which predictions changed, even if overall accuracy remains similar.

    Args:
        model_a: Reference model (e.g., clean model)
        model_b: Modified model (e.g., after injection/defense)
        loader: DataLoader with evaluation batches
        device: Device to run on

    Returns:
        Dictionary with detailed diagnostics:
        - changed_predictions: Number of samples where argmax differs
        - total_samples: Total number of samples evaluated
        - change_rate: Fraction of predictions that changed
        - agreement_rate: Fraction of predictions that agree (1 - change_rate)
        - max_logit_diff: Maximum absolute logit difference observed
        - mean_logit_diff: Mean absolute logit difference
    """
    model_a.eval()
    model_b.eval()

    changed = 0
    total = 0
    max_diff = 0.0
    sum_diff = 0.0
    diff_count = 0

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}

        logits_a = model_a(**inputs).logits
        logits_b = model_b(**inputs).logits

        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)

        changed += int((pred_a != pred_b).sum().item())
        total += pred_a.numel()

        # Track logit differences
        diff = (logits_a - logits_b).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        sum_diff += float(diff.sum().item())
        diff_count += diff.numel()

    change_rate = changed / total if total > 0 else 0.0
    mean_diff = sum_diff / diff_count if diff_count > 0 else 0.0

    return {
        "changed_predictions": changed,
        "total_samples": total,
        "change_rate": change_rate,
        "agreement_rate": 1.0 - change_rate,
        "max_logit_diff": max_diff,
        "mean_logit_diff": mean_diff,
    }


@torch.no_grad()
def logits_mse(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: str,
) -> float:
    """
    Compute Mean Squared Error between model logits.

    Returns:
        Average MSE across all samples
    """
    model_a.eval()
    model_b.eval()

    mses = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}

        la = model_a(**inputs).logits
        lb = model_b(**inputs).logits

        mse = torch.mean((la - lb) ** 2)
        mses.append(float(mse.cpu().item()))

    return float(np.mean(mses)) if mses else 0.0


@dataclass
class TimingResult:
    """Timing measurement result."""
    operation: str
    elapsed_seconds: float
    n_parameters: int
    params_per_second: float


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str = ""):
        self.operation = operation
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self.start_time

    def result(self, n_parameters: int = 0) -> TimingResult:
        return TimingResult(
            operation=self.operation,
            elapsed_seconds=self.elapsed,
            n_parameters=n_parameters,
            params_per_second=n_parameters / self.elapsed if self.elapsed > 0 else 0,
        )


def measure_defense_overhead(
    defense_fn,
    model: torch.nn.Module,
    targets: List[str],
    x: int,
    n_runs: int = 5,
    **kwargs,
) -> Dict[str, float]:
    """
    Measure runtime overhead of a defense function.

    Args:
        defense_fn: Defense function to measure (e.g., RandomFlipDefense().apply)
        model: Model to apply defense to
        targets: Target parameter names
        x: Number of LSBs
        n_runs: Number of timing runs
        **kwargs: Additional arguments for defense_fn

    Returns:
        Dictionary with timing statistics
    """
    import copy

    # Count parameters
    named_params = dict(model.named_parameters())
    n_params = sum(named_params[t].numel() for t in targets if t in named_params)

    times = []
    for _ in range(n_runs):
        # Fresh copy for each run
        model_copy = copy.deepcopy(model)

        with Timer() as t:
            defense_fn(model_copy, targets, x=x, **kwargs)

        times.append(t.elapsed)

    times = np.array(times)
    return {
        "mean_seconds": float(times.mean()),
        "std_seconds": float(times.std()),
        "min_seconds": float(times.min()),
        "max_seconds": float(times.max()),
        "n_parameters": n_params,
        "params_per_second": n_params / times.mean() if times.mean() > 0 else 0,
    }


@dataclass
class ModelMetrics:
    """Comprehensive model comparison metrics."""
    cosine_similarity: float
    norm_ratio: float
    kl_divergence: float
    top1_agreement: float
    logits_mse: float
    wasserstein_distance: float


def compute_all_model_metrics(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    targets: List[str],
    loader,
    device: str,
) -> ModelMetrics:
    """
    Compute all model comparison metrics at once.

    Args:
        model_a: Reference model
        model_b: Modified model
        targets: Target parameter names
        loader: DataLoader for logit-based metrics
        device: Device to run on

    Returns:
        ModelMetrics dataclass with all metrics
    """
    return ModelMetrics(
        cosine_similarity=cosine_similarity_on_targets(model_a, model_b, targets),
        norm_ratio=weight_norm_ratio(model_a, model_b, targets),
        kl_divergence=logits_kl_div(model_a, model_b, loader, device),
        top1_agreement=top1_agreement(model_a, model_b, loader, device),
        logits_mse=logits_mse(model_a, model_b, loader, device),
        wasserstein_distance=weight_distribution_distance(model_a, model_b, targets),
    )
