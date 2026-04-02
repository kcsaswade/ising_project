"""
Metropolis Monte Carlo simulation for the 2D Ising model.

Implements:
- Single-spin flip updates with local ΔE using nearest neighbors only
- Periodic boundary conditions
- Monte Carlo sweeps (N^2 attempted updates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from . import lattice
from . import observables


@dataclass
class MetropolisIsing:
    """
    2D Ising model simulation using the Metropolis algorithm.

    Attributes
    ----------
    size : int
        Linear lattice size N (lattice has N x N spins).
    temperature : float
        Temperature T (k_B = 1).
    J : float
        Coupling constant.
    h : float
        External magnetic field.
    rng : numpy.random.Generator
        Random number generator.
    initial_state : str
        'random' or 'ordered'; how to initialize spins.
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

    # --- initialization / reset -------------------------------------------------

    def _init_spins(self) -> None:
        """Initialize the spin lattice according to the chosen initial_state."""
        if self.initial_state == "ordered":
            self.spins = lattice.ordered_lattice(self.size, up=True)
        else:
            self.spins = lattice.random_lattice(self.size, self.rng)

    def reset(self, initial_state: Optional[str] = None) -> None:
        """
        Reset the lattice.

        Parameters
        ----------
        initial_state : str or None
            If provided, overrides the current initial_state ('random' or 'ordered').
        """
        if initial_state is not None:
            self.initial_state = initial_state
        self._init_spins()

    def resize(self, new_size: int, initial_state: Optional[str] = None) -> None:
        """
        Change lattice size and reinitialize spins.

        Parameters
        ----------
        new_size : int
            New linear lattice size N.
        initial_state : str or None
            If provided, overrides current initial_state.
        """
        self.size = int(new_size)
        self.reset(initial_state)

    # --- parameter setters ------------------------------------------------------

    def set_temperature(self, T: float) -> None:
        """Set the temperature T (k_B = 1)."""
        self.temperature = float(max(T, 1e-8))  # avoid division by zero

    def set_field(self, h: float) -> None:
        """Set the external magnetic field h."""
        self.h = float(h)

    # --- core Metropolis updates ------------------------------------------------

    def _delta_energy_single_flip(self, i: int, j: int) -> float:
        """
        Compute the local energy change ΔE for flipping spin at (i, j).

        Uses only the four nearest neighbors (periodic boundaries), i.e.

            ΔE = 2 * s_ij * (J * sum_neighbors + h)

        where sum_neighbors = s_{i+1,j} + s_{i-1,j} + s_{i,j+1} + s_{i,j-1}.
        """
        N = self.size
        s = self.spins[i, j]

        # Periodic boundary conditions (wrap around with modulo N)
        up = self.spins[(i - 1) % N, j]
        down = self.spins[(i + 1) % N, j]
        left = self.spins[i, (j - 1) % N]
        right = self.spins[i, (j + 1) % N]

        neighbor_sum = up + down + left + right

        # Local energy change for flipping s -> -s
        delta_E = 2.0 * s * (self.J * neighbor_sum + self.h)
        return delta_E

    def step(self) -> None:
        """
        Perform a single Metropolis spin-flip attempt.

        Algorithm:
        1. Choose a random spin (i, j).
        2. Compute local ΔE using nearest neighbors only.
        3. If ΔE <= 0, accept the flip.
           Else accept with probability exp(-ΔE / T).
        """
        N = self.size

        # Random site selection (ergodicity)
        i = self.rng.integers(0, N)
        j = self.rng.integers(0, N)

        dE = self._delta_energy_single_flip(i, j)

        if dE <= 0.0:
            # Energy-lowering (or equal) moves are always accepted
            self.spins[i, j] *= -1
        else:
            # Accept with Boltzmann probability exp(-ΔE / T)
            if self.rng.random() < np.exp(-dE / self.temperature):
                self.spins[i, j] *= -1

    def sweep(self) -> None:
        """
        Perform one Monte Carlo sweep: N^2 attempted updates.

        Each sweep attempts, on average, one update per spin.
        """
        n_sites = self.size * self.size
        for _ in range(n_sites):
            self.step()

    # --- observables ------------------------------------------------------------

    def magnetization(self) -> float:
        """Magnetization per spin."""
        return observables.magnetization_per_spin(self.spins)

    def energy_per_spin(self) -> float:
        """Energy per spin."""
        return observables.energy_per_spin(self.spins, J=self.J, h=self.h)