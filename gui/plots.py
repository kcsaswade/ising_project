from __future__ import annotations

from typing import List, Tuple

import pygame

from utils import config


class MagnetizationPlot:
    """
    Simple scrolling plot of magnetization vs. time (simulation sweeps).

    Stores a sliding window of recent magnetization values and plots them.
    """

    def __init__(self, max_points: int = 300) -> None:
        self.max_points = max_points
        self.values: List[float] = []

    def add_point(self, m: float) -> None:
        """Append a new magnetization value."""
        self.values.append(float(m))
        if len(self.values) > self.max_points:
            # Discard oldest
            self.values.pop(0)

    def clear(self) -> None:
        """Clear stored values."""
        self.values.clear()

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        # Background
        pygame.draw.rect(surface, config.PLOT_BG_COLOR, rect)

        if len(self.values) < 2:
            # Border only if not enough data
            pygame.draw.rect(surface, config.PLOT_BORDER_COLOR, rect, 1)
            return

        # Consider at most max_points, to keep drawing cost reasonable
        data = self.values[-self.max_points :]
        n = len(data)

        # Map x index -> horizontal position
        if n <= 1:
            pygame.draw.rect(surface, config.PLOT_BORDER_COLOR, rect, 1)
            return

        x0 = rect.left + 4
        x1 = rect.right - 4
        y0 = rect.bottom - 4
        y1 = rect.top + 4

        width = max(x1 - x0, 1)
        height = max(y0 - y1, 1)

        # Magnetization is in [-1, 1]; map it linearly to y
        points: List[Tuple[int, int]] = []
        for idx, m in enumerate(data):
            t = idx / float(n - 1)
            x = int(x0 + t * width)

            # m = +1 -> top, m = -1 -> bottom
            y_frac = (1.0 - (m + 1.0) / 2.0)  # 0 at top, 1 at bottom
            y = int(y1 + y_frac * height)

            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(surface, config.PLOT_LINE_COLOR, False, points, 2)

        # Border
        pygame.draw.rect(surface, config.PLOT_BORDER_COLOR, rect, 1)

class TimeSeriesPlot:
    """
    Generic scrolling time-series plot.

    Can be used for:
    - Magnetization (fixed range [-1, 1])
    - Energy (auto range)
    - |M| (fixed range [0, 1])
    - Later: C, χ

    Parameters
    ----------
    max_points : int
        Maximum number of points to store/display.
    label : str
        Optional label for the plot (not drawn yet, but useful for future).
    y_range : Optional[Tuple[float, float]]
        If provided, fixes y-axis scaling.
        If None, auto-scales based on data.
    """

    def __init__(
        self,
        max_points: int = 300,
        label: str = "",
        y_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.max_points = max_points
        self.label = label
        self.y_range = y_range

        self.values: List[float] = []

    # ------------------------------------------------------------------ data

    def add_point(self, value: float) -> None:
        """Append a new value to the plot."""
        self.values.append(float(value))
        if len(self.values) > self.max_points:
            self.values.pop(0)

    def clear(self) -> None:
        """Clear all stored values."""
        self.values.clear()

    # ------------------------------------------------------------------ helpers

    def _get_y_bounds(self) -> Tuple[float, float]:
        """
        Determine y-axis bounds.

        Returns
        -------
        (y_min, y_max)
        """
        if self.y_range is not None:
            return self.y_range

        # Auto-scale
        if not self.values:
            return (-1.0, 1.0)

        vmin = min(self.values)
        vmax = max(self.values)

        if abs(vmax - vmin) < 1e-8:
            # Avoid flat-line collapse
            return (vmin - 1.0, vmax + 1.0)

        # Add small padding
        padding = 0.05 * (vmax - vmin)
        return (vmin - padding, vmax + padding)

    # ------------------------------------------------------------------ drawing

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Draw the plot inside the given rectangle.
        """
        # Background
        pygame.draw.rect(surface, config.PLOT_BG_COLOR, rect)

        if len(self.values) < 2:
            pygame.draw.rect(surface, config.PLOT_BORDER_COLOR, rect, 1)
            return

        data = self.values[-self.max_points :]
        n = len(data)

        x0 = rect.left + 4
        x1 = rect.right - 4
        y0 = rect.bottom - 4
        y1 = rect.top + 4

        width = max(x1 - x0, 1)
        height = max(y0 - y1, 1)

        y_min, y_max = self._get_y_bounds()

        points: List[Tuple[int, int]] = []

        for idx, val in enumerate(data):
            # Horizontal mapping
            t = idx / float(n - 1)
            x = int(x0 + t * width)

            # Vertical mapping
            if y_max == y_min:
                y = (y0 + y1) // 2
            else:
                y_frac = (val - y_min) / (y_max - y_min)
                y_frac = max(0.0, min(1.0, y_frac))
                y = int(y0 - y_frac * height)

            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(surface, config.PLOT_LINE_COLOR, False, points, 2)

        # Border
        pygame.draw.rect(surface, config.PLOT_BORDER_COLOR, rect, 1)