"""
Random number generator helpers.
"""

from typing import Optional

import numpy as np
from numpy.random import Generator


def create_rng(seed: Optional[int] = None) -> Generator:
    """
    Create a NumPy random number generator.

    Parameters
    ----------
    seed : int or None
        Seed for reproducibility. If None, uses system entropy.

    Returns
    -------
    rng : numpy.random.Generator
    """
    return np.random.default_rng(seed)