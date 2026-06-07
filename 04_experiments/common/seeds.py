"""Determinism manager. Call set_seeds() at the top of every experiment."""
from __future__ import annotations
import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> int:
    """Set seeds for Python random, NumPy, hashlib randomness, and PYTHONHASHSEED.

    Returns the seed used (callers can record it in the manifest).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: F401 - optional, only if installed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
