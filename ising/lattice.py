import numpy as np
from numpy.random import Generator


def random_lattice(size: int, rng: Generator) -> np.ndarray:
    """
    Create a random lattice of spins (+1 or -1) of shape (size, size).

    Parameters
    ----------
    size : int
        Linear lattice size N.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    spins : np.ndarray
        2D array of shape (size, size) with values in {-1, +1}.
    """
    # rng.choice is clear and beginner-friendly
    spins = rng.choice([-1, 1], size=(size, size))
    return spins.astype(int)


def ordered_lattice(size: int, up: bool = True) -> np.ndarray:
    """
    Create an ordered lattice of spins, all +1 (up) or all -1 (down).

    Parameters
    ----------
    size : int
        Linear lattice size N.
    up : bool
        If True, all spins +1, else all spins -1.

    Returns
    -------
    spins : np.ndarray
        2D array of shape (size, size) with values in {-1, +1}.
    """
    value = 1 if up else -1
    spins = np.full((size, size), value, dtype=int)
    return spins