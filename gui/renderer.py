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

        # Choose cell size so lattice fits into rect
        cell_size = min(self.rect.width // N, self.rect.height // N)
        if cell_size <= 0:
            return

        # Center the lattice inside the rect
        lattice_width = cell_size * N
        lattice_height = cell_size * N

        origin_x = self.rect.x + (self.rect.width - lattice_width) // 2
        origin_y = self.rect.y + (self.rect.height - lattice_height) // 2

        for i in range(N):
            for j in range(N):
                s = spins[i, j]
                color: Tuple[int, int, int] = (
                    config.SPIN_UP_COLOR if s > 0 else config.SPIN_DOWN_COLOR
                )
                cell_rect = pygame.Rect(
                    origin_x + j * cell_size,
                    origin_y + i * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(surface, color, cell_rect)