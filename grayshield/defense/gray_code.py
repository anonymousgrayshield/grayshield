from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import torch
import hashlib
from ..lsb.bits import apply_gray_code_mask, apply_hmac_gray_mask


@dataclass
class GrayCodeReport:
    n_params_modified: int
    n_indices: int
    version: str = "v1"


class GrayCodeDefense:
    """
    GrayShield defense — supports V1 (seed-based) and V2 (HMAC-keyed).

    V1 (default, backward-compat):
        Uses a seed + param_name_hash to derive the Gray Code offset.
        Deterministic but NOT cryptographically secure — attacker who knows the
        seed can reconstruct the mask.

    V2 (use_v2=True):
        Uses HMAC-SHA256(secret_key, param_name) to derive the keyed offset.
        Cryptographically secure: without the key, offset is indistinguishable
        from random (PRF security). Supports reversibility and tamper detection.

    Strength Sweep (x_range):
        By default applies the same x passed in. For Pareto-front exploration,
        call apply() multiple times with different x values (see runner.py sweep).
    """

    def apply(
        self,
        model: torch.nn.Module,
        target_names: List[str],
        x: int,
        seed: int = 0,
        indices: Optional[Dict[str, torch.Tensor]] = None,
        # V2 options
        use_v2: bool = False,
        secret_key: Optional[bytes] = None,
        # V3 options (extends V2)
        use_v3: bool = False,
    ) -> GrayCodeReport:
        """
        Apply GrayShield defense to target parameters.

        Args:
            model        : model to modify (in-place)
            target_names : list of parameter names to modify
            x            : number of LSBs to overwrite (defense strength)
            seed         : V1 seed (ignored when use_v2/use_v3=True)
            indices      : optional per-param index subsets
            use_v2       : if True, use HMAC-keyed V2 (requires secret_key)
            secret_key   : HMAC key bytes for V2/V3 (reads GRAYSHIELD_KEY env var
                           as fallback if None)
            use_v3       : if True, enable V3 enhancements (multi-layer + per-run salt)
                           Requires use_v2=True (V3 extends V2)
        """
        # V3 validation
        if use_v3:
            if not use_v2:
                raise ValueError("V3 mode requires use_v2=True (V3 extends V2)")
            use_v2 = True  # Ensure V2 path is taken

        # Generate per-run salt for V3 (deterministic from seed for reproducibility)
        run_salt = None
        if use_v3:
            gen = torch.Generator()
            gen.manual_seed(seed)
            run_salt = torch.randint(0, 2**31 - 1, (1,), generator=gen).item()

        if use_v2:
            if secret_key is None:
                env_key = os.environ.get("GRAYSHIELD_KEY", "")
                if not env_key:
                    raise ValueError(
                        "GrayShield V2/V3 requires a secret key. "
                        "Pass secret_key= or set GRAYSHIELD_KEY env var."
                    )
                secret_key = env_key.encode("utf-8")

        named = dict(model.named_parameters())
        n_idx = 0
        n_params = 0

        for n in target_names:
            p = named.get(n, None)
            if p is None or p.data.dtype != torch.float32:
                continue

            idx = None if indices is None else indices.get(n, None)
            if idx is not None:
                n_idx += int(idx.numel())
            else:
                n_idx += int(p.data.numel())

            n_params += 1

            if use_v2:
                # V2/V3: HMAC-keyed mask (cryptographically secure)
                apply_hmac_gray_mask(
                    p.data, x=x,
                    secret_key=secret_key,
                    param_name=n,
                    idx_flat=idx,
                    use_v3=use_v3,      # Enable V3 enhancements
                    run_salt=run_salt,  # Per-run randomization
                )
            else:
                # V1: seed-based (backward compat)
                name_hash = int(hashlib.md5(n.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
                param_seed = (seed + name_hash) % (2**31 - 1)
                apply_gray_code_mask(p.data, x=x, seed=param_seed, idx_flat=idx)

        return GrayCodeReport(
            n_params_modified=n_params,
            n_indices=n_idx,
            version="v3" if use_v3 else ("v2" if use_v2 else "v1"),
        )


GrayShieldReport = GrayCodeReport
GrayShieldDefense = GrayCodeDefense
