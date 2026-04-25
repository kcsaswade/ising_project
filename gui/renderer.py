from __future__ import annotations

from typing import Tuple

import pygame

from utils import config


class LatticeRenderer:
    """
    Responsible for drawing the Ising lattice as colored squares.

    Spins:
      +1 -> light color (SPIN_UP_COLOR)
      -1 -> dark color (SPIN_DOWN_COLOR)
    """

    def __init__(self, rect: pygame.Rect) -> None:
        """
        Parameters
        ----------
        rect : pygame.Rect
            Area of the window reserved for the lattice visualization.
        """
        self.rect = rect

    def draw(self, surface: pygame.Surface, spins) -> None:
        """
        Draw the lattice.

        Parameters
        ----------
        surface : pygame.Surface
            Surface to draw on (usually the main screen).
        spins : np.ndarray
            2D array of spins (+1 or -1), shape (N, N).
        """
        N = spins.shape[0]
        if N <= 0:
            return

        # --- FLOAT scaling (important fix) ---
        cell_w = self.rect.width / N
        cell_h = self.rect.height / N
        cell_size = min(cell_w, cell_h)

        if cell_size <= 0:
            return

        lattice_width = cell_size * N
        lattice_height = cell_size * N

        # --- center lattice ---
        origin_x = self.rect.x + (self.rect.width - lattice_width) / 2
        origin_y = self.rect.y + (self.rect.height - lattice_height) / 2

        for i in range(N):
            for j in range(N):
                s = spins[i, j]
                color = (
                    config.SPIN_UP_COLOR if s > 0 else config.SPIN_DOWN_COLOR
                )

                x = origin_x + j * cell_size
                y = origin_y + i * cell_size

                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(int(x), int(y), int(cell_size) + 1, int(cell_size) + 1),
                )