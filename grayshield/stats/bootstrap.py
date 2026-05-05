from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        samp = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(samp)))
    means = np.asarray(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi
