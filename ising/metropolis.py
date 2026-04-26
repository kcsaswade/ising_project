"""
Metropolis Monte Carlo simulation for the 2D Ising model.

Now uses external sweep kernel (ising.kernels) for core updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from turtle import mode
from typing import Optional

import numpy as np
from numpy.random import Generator

from . import lattice
from . import observables
from .kernels import sweep_kernel   # <-- NEW
from .kernels import sweep_kernel_checkerboard
from numba import cuda
#from .kernels import create_gpu_context

@dataclass
class MetropolisIsing:
    """
    2D Ising model simulation using the Metropolis algorithm.
    """
    size: int
    temperature: float = 2.5
    J: float = 1.0
    h: float = 0.0
    rng: Optional[Generator] = None
    initial_state: str = "random"

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()
        self._init_spins()
        self.backend = "None"
        self.spins_gpu = None
        self.gpu_ready = False

    # ------------------------------------------------------------------ init/reset

    def _init_spins(self) -> None:
        if self.initial_state == "ordered":
            self.spins = lattice.ordered_lattice(self.size, up=True)
        else:
            self.spins = lattice.random_lattice(self.size, self.rng)

    def _init_gpu(self):
        try:

            self.spins_gpu = cuda.to_device(self.spins)
            self.threadsperblock, self.blockspergrid, self.rng_states = create_gpu_context(self.size)

            self.gpu_ready = True
        except Exception as e:
            print("GPU init failed, falling back to CPU:", e)
            self.gpu_ready = False

    def set_backend(self, mode: str):
        self.backend = mode

        if mode == "GPU":
            self._init_gpu()

    def set_backend(self, mode: str):
        self.backend = mode

    def reset(self, initial_state: Optional[str] = None) -> None:
        if initial_state is not None:
            self.initial_state = initial_state

        if self.initial_state == "ordered":
            up = self.rng.random() < 0.5
            self.spins = lattice.ordered_lattice(self.size, up=up)
        else:
            self.spins = lattice.random_lattice(self.size, self.rng)

    def resize(self, new_size: int, initial_state: Optional[str] = None) -> None:
        self.size = int(new_size)
        self.reset(initial_state)

    # ------------------------------------------------------------------ setters

    def set_temperature(self, T: float) -> None:
        self.temperature = float(max(T, 1e-8))

    def set_field(self, h: float) -> None:
        self.h = float(h)

    # ------------------------------------------------------------------ core update

    def sweep(self, fraction=1.0) -> None:
        """
        Perform one Monte Carlo sweep via external kernel.
        """
        N2 = self.size * self.size
        n_updates = int(N2 * fraction)

        if self.backend == "CPU":
            sweep_kernel(self.spins, self.temperature, self.J, self.h, n_updates)
        if self.backend == "GPU" and self.gpu_ready:
            # RED pass
            sweep_kernel_checkerboard[self.blockspergrid, self.threadsperblock](
                self.spins_gpu, self.temperature, self.J, self.h, 0, self.rng_states
            )

            # BLACK pass
            sweep_kernel_checkerboard[self.blockspergrid, self.threadsperblock](
                self.spins_gpu, self.temperature, self.J, self.h, 1, self.rng_states
            )

            # Copy back occasionally (for now: every sweep)
            self.spins = self.spins_gpu.copy_to_host()
        else:
            # pure python fallback
            for _ in range(n_updates):
                self.step()

    def step(self) -> None:
        N = self.size

        i = self.rng.integers(0, N)
        j = self.rng.integers(0, N)

        s = self.spins[i, j]

        up = self.spins[(i - 1) % N, j]
        down = self.spins[(i + 1) % N, j]
        left = self.spins[i, (j - 1) % N]
        right = self.spins[i, (j + 1) % N]

        neighbor_sum = up + down + left + right

        dE = 2.0 * s * (self.J * neighbor_sum + self.h)

        if dE <= 0.0 or self.rng.random() < np.exp(-dE / self.temperature):
            self.spins[i, j] = -s
    # ------------------------------------------------------------------ observables

    def magnetization(self) -> float:
        return observables.magnetization_per_spin(self.spins)

    def energy_per_spin(self) -> float:
        return observables.energy_per_spin(self.spins, J=self.J, h=self.h)