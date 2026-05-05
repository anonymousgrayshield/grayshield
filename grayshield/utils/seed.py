from __future__ import annotations
import os
import random
from typing import Optional
import numpy as np

_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: The random seed value
        deterministic: If True, enable CUDA deterministic mode (slower but reproducible)
    """
    import torch

    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms for reproducibility
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def get_seed() -> Optional[int]:
    """Return the currently set global seed."""
    return _GLOBAL_SEED


def get_generator(seed: Optional[int] = None) -> "torch.Generator":
    """
    Get a PyTorch Generator with the specified or global seed.

    Args:
        seed: Optional specific seed. If None, uses global seed.

    Returns:
        torch.Generator initialized with the seed
    """
    import torch

    gen = torch.Generator()
    s = seed if seed is not None else (_GLOBAL_SEED or 42)
    gen.manual_seed(s)
    return gen


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker init function for reproducibility.
    Use: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    import torch

    seed = _GLOBAL_SEED or 42
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
