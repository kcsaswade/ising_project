import numpy as np
from numba import njit


@njit(nopython=True)
def sweep_kernel(spins, T, J, h, n_updates):
    N = spins.shape[0]

    for _ in range(n_updates):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        s = spins[i, j]

        up = spins[(i - 1) % N, j]
        down = spins[(i + 1) % N, j]
        left = spins[i, (j - 1) % N]
        right = spins[i, (j + 1) % N]

        neighbor_sum = up + down + left + right

        dE = 2.0 * s * (J * neighbor_sum + h)

        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            spins[i, j] = -s