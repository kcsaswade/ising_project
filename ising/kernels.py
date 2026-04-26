import numpy as np
from numba import njit
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

def create_gpu_context(N):
    threadsperblock = (16, 16)
    blockspergrid = (
        (N + threadsperblock[0] - 1) // threadsperblock[0],
        (N + threadsperblock[1] - 1) // threadsperblock[1],
    )

    rng_states = create_xoroshiro128p_states(N * N, seed=42)

    return threadsperblock, blockspergrid, rng_states


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

@cuda.jit
def sweep_kernel_checkerboard(spins, T, J, h, parity, rng_states):
    i, j = cuda.grid(2)

    N = spins.shape[0]

    if i >= N or j >= N:
        return

    # Checkerboard condition
    if (i + j) % 2 != parity:
        return

    s = spins[i, j]

    up = spins[(i - 1) % N, j]
    down = spins[(i + 1) % N, j]
    left = spins[i, (j - 1) % N]
    right = spins[i, (j + 1) % N]

    neighbor_sum = up + down + left + right

    dE = 2.0 * s * (J * neighbor_sum + h)

    # --- random number per thread ---
    idx = i * N + j
    r = xoroshiro128p_uniform_float32(rng_states, idx)

    if dE <= 0.0 or r < math.exp(-dE / T):
        spins[i, j] = -s