"""
Model State Management and Checkpointing

Provides explicit model state tracking (clean/poisoned/defensed) with:
- Checkpoint saving/loading to distinct folders
- Fingerprint computation for verification
- Sanity checks to prevent model state confusion
"""
from __future__ import annotations
import os
import hashlib
import json
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import torch


class ModelState(str, Enum):
    """Explicit model states for tracking."""
    CLEAN = "clean"
    POISONED = "poisoned"
    DEFENSED = "defensed"


@dataclass
class ModelFingerprint:
    """Fingerprint for model state verification."""
    state: str
    ckpt_path: Optional[str]
    device: str
    dtype: str
    n_params: int
    # Hash of first 1KB of a known parameter for quick verification
    param_hash: str
    # Optional: hash of all target params (more expensive)
    targets_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelFingerprint":
        return cls(**d)


def compute_param_hash(
    model: torch.nn.Module,
    n_bytes: int = 1024,
    targets: Optional[list] = None,
) -> str:
    """Compute hash from first n_bytes of model parameters.

    Args:
        model: PyTorch model
        n_bytes: Number of bytes to hash from each parameter
        targets: If provided, use the first TARGET parameter (critical for partial injection).
                 If None, uses the first parameter found (may miss injected changes).

    Returns:
        16-char hex hash string
    """
    if targets:
        # Use hash of target parameters (more reliable for partial injection)
        hasher = hashlib.sha256()
        target_set = set(targets)
        found_any = False
        for name, param in model.named_parameters():
            if name in target_set:
                data = param.detach().cpu().numpy().tobytes()[:n_bytes]
                hasher.update(data)
                found_any = True
                # Only hash first few targets for speed
                if hasher.digest_size > 0:
                    break
        if found_any:
            return hasher.hexdigest()[:16]

    # Fallback: use first parameter
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy().tobytes()[:n_bytes]
        return hashlib.sha256(data).hexdigest()[:16]
    return "no_params"


def compute_targets_hash(model: torch.nn.Module, targets: list) -> str:
    """Compute hash from all target parameters."""
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        if name in targets:
            data = param.detach().cpu().numpy().tobytes()
            hasher.update(data)
    return hasher.hexdigest()[:16]


def compute_fingerprint(
    model: torch.nn.Module,
    state: ModelState,
    ckpt_path: Optional[str] = None,
    targets: Optional[list] = None,
) -> ModelFingerprint:
    """Compute fingerprint for a model in a specific state.

    IMPORTANT: When targets is provided, param_hash is computed from TARGET
    parameters only. This is critical for detecting changes when using
    partial injection (e.g., mode=ffn with layer_range).
    """
    device = str(next(model.parameters()).device)
    dtype = str(next(model.parameters()).dtype)
    n_params = sum(p.numel() for p in model.parameters())

    # CRITICAL: Use targets for param_hash to detect partial injection changes
    param_hash = compute_param_hash(model, targets=targets)
    targets_hash = compute_targets_hash(model, targets) if targets else None

    return ModelFingerprint(
        state=state.value,
        ckpt_path=ckpt_path,
        device=device,
        dtype=dtype,
        n_params=n_params,
        param_hash=param_hash,
        targets_hash=targets_hash,
    )


def get_model_dir(run_dir: str, state: ModelState) -> str:
    """Get checkpoint directory for a specific model state."""
    return os.path.join(run_dir, "models", state.value)


def save_model_state(
    model: torch.nn.Module,
    run_dir: str,
    state: ModelState,
    targets: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ModelFingerprint]:
    """Save model checkpoint with fingerprint.

    Args:
        model: Model to save
        run_dir: Experiment output directory
        state: Model state (clean/poisoned/defensed)
        targets: Optional list of target param names for fingerprinting
        metadata: Optional additional metadata

    Returns:
        Tuple of (checkpoint_path, fingerprint)
    """
    model_dir = get_model_dir(run_dir, state)
    os.makedirs(model_dir, exist_ok=True)

    fp_path = os.path.join(model_dir, "fingerprint.json")
    ckpt_path = os.path.join(model_dir, "model.pt")

    current_fingerprint = compute_fingerprint(model, state, ckpt_path, targets)

    # Reuse an existing checkpoint only when it matches the current in-memory model.
    if os.path.exists(fp_path):
        try:
            with open(fp_path, "r") as f:
                fp_data = json.load(f)
            fingerprint_dict = {k: v for k, v in fp_data.items() if k != "metadata"}
            existing_fingerprint = ModelFingerprint.from_dict(fingerprint_dict)
            if (
                existing_fingerprint.param_hash == current_fingerprint.param_hash
                and existing_fingerprint.targets_hash == current_fingerprint.targets_hash
            ):
                print(f"[INFO] Matching fingerprint found at {fp_path}. Skipping model save.")
                return ckpt_path, existing_fingerprint
            print(f"[INFO] Existing fingerprint at {fp_path} is stale. Overwriting checkpoint.")
        except Exception as e:
            print(f"[WARN] Failed to load existing fingerprint: {e}. Re-saving model.")

    # Save model state dict
    torch.save(model.state_dict(), ckpt_path)

    # Save fingerprint for the current checkpoint
    fingerprint = current_fingerprint

    fp_data = fingerprint.to_dict()
    if metadata:
        fp_data["metadata"] = metadata
    with open(fp_path, "w") as f:
        json.dump(fp_data, f, indent=2)

    return ckpt_path, fingerprint


def load_model_state(
    model: torch.nn.Module,
    run_dir: str,
    state: ModelState,
    strict: bool = True,
) -> Tuple[torch.nn.Module, ModelFingerprint]:
    """Load model from checkpoint for a specific state.

    Args:
        model: Model instance to load weights into
        run_dir: Experiment output directory
        state: Model state to load
        strict: If True, raise error on missing state

    Returns:
        Tuple of (loaded_model, fingerprint)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist and strict=True
    """
    model_dir = get_model_dir(run_dir, state)
    ckpt_path = os.path.join(model_dir, "model.pt")

    if not os.path.exists(ckpt_path):
        if strict:
            raise FileNotFoundError(
                f"Checkpoint not found for model_state='{state.value}' at {ckpt_path}. "
                f"Ensure the model was saved before evaluation."
            )
        return model, None

    # Load state dict
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Load fingerprint
    fp_path = os.path.join(model_dir, "fingerprint.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            fp_data = json.load(f)
        fingerprint = ModelFingerprint.from_dict({
            k: v for k, v in fp_data.items() if k != "metadata"
        })
    else:
        fingerprint = compute_fingerprint(model, state, ckpt_path)

    return model, fingerprint


def verify_fingerprints_differ(
    fp1: ModelFingerprint,
    fp2: ModelFingerprint,
    context: str = "",
) -> None:
    """Assert that two fingerprints differ (sanity check).

    Raises AssertionError if fingerprints are identical when they shouldn't be.
    """
    if fp1.param_hash == fp2.param_hash:
        raise AssertionError(
            f"MODEL STATE CONFUSION: {context}\n"
            f"  State '{fp1.state}' and '{fp2.state}' have IDENTICAL fingerprints!\n"
            f"  param_hash: {fp1.param_hash}\n"
            f"  This suggests the same checkpoint is being evaluated twice.\n"
            f"  Check that model states are saved/loaded correctly."
        )


def verify_fingerprints_match(
    fp1: ModelFingerprint,
    fp2: ModelFingerprint,
    context: str = "",
) -> None:
    """Assert that two fingerprints match (verification check)."""
    if fp1.param_hash != fp2.param_hash:
        raise AssertionError(
            f"FINGERPRINT MISMATCH: {context}\n"
            f"  Expected: {fp1.param_hash} (state={fp1.state})\n"
            f"  Got: {fp2.param_hash} (state={fp2.state})\n"
            f"  The model may have been modified unexpectedly."
        )


def log_model_state(
    fingerprint: ModelFingerprint,
    logger=None,
    prefix: str = "",
) -> None:
    """Log model state information for audit trail."""
    msg = (
        f"{prefix}Model State: {fingerprint.state} | "
        f"fingerprint={fingerprint.param_hash} | "
        f"device={fingerprint.device} | dtype={fingerprint.dtype}"
    )
    if fingerprint.ckpt_path:
        msg += f" | ckpt={fingerprint.ckpt_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
