"""Global determinism helper.

`set_seed` seeds Python, NumPy, and (if installed) PyTorch + CUDA from a single
call, so a run is reproducible from one seed. torch is imported lazily so this
stays usable in numpy-only contexts (the calibration and changepoint code).
"""
import random

import numpy as np


def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> int:
    """Seed every RNG the BRE code uses. Returns the seed (handy for logging)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Best-effort determinism; safe no-ops if the backend doesn't support it.
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    return seed
