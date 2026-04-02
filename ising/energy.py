import numpy as np


def total_energy(spins: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    """
    Compute the total energy of a 2D Ising configuration with periodic boundaries.

    Hamiltonian:
        H = -J * sum_{<i,j>} s_i s_j - h * sum_i s_i

    We count each nearest-neighbor pair once by summing over "right" and "down"
    neighbors only.

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
    E : float
        Total energy of the configuration.
    """
    N = spins.shape[0]
    E = 0.0

    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            # Periodic neighbors: right and down only, to avoid double counting
            s_right = spins[i, (j + 1) % N]
            s_down = spins[(i + 1) % N, j]

            E -= J * s * (s_right + s_down)

    # Field term: -h * sum_i s_i
    E -= h * float(spins.sum())
    return E


def energy_per_spin(spins: np.ndarray, J: float = 1.0, h: float = 0.0) -> float:
    """
    Energy per spin.

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
    N = spins.shape[0]
    E = total_energy(spins, J=J, h=h)
    return E / float(N * N)