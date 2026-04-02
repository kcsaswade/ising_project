import numpy as np

from .energy import energy_per_spin as _energy_per_spin


def magnetization(spins: np.ndarray) -> float:
    """
    Total magnetization (sum of spins).

    Parameters
    ----------
    spins : np.ndarray
        2D array of spins (+1 or -1), shape (N, N).

    Returns
    -------
    M : float
        Total magnetization.
    """
    return float(spins.sum())


def magnetization_per_spin(spins: np.ndarray) -> float:
    """
    Magnetization per spin, i.e. average spin.

    Parameters
    ----------
    spins : np.ndarray
        2D array of spins (+1 or -1), shape (N, N).

    Returns
    -------
    m : float
        Magnetization per spin.
    """
    # Using mean is concise and clear here
    return float(np.mean(spins))


def energy_per_spin(spins: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    """
    Convenience wrapper to compute energy per spin.

    Parameters
    ----------
    spins : np.ndarray
        2D array of spins (+1 or -1), shape (N, N).
    J : float
        Coupling constant.
    h : float
        External magnetic field.

    Returns
    -------
    e : float
        Energy per spin.
    """
    return _energy_per_spin(spins, J=J, h=h)