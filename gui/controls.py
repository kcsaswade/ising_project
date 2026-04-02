from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pygame

from utils import config


@dataclass
class Button:
    """Simple clickable rectangular button."""

    rect: pygame.Rect
    text: str

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, active: bool = False) -> None:
        base_color = config.BUTTON_ACTIVE_COLOR if active else config.BUTTON_COLOR
        pygame.draw.rect(surface, base_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, config.BUTTON_BORDER_COLOR, self.rect, 1, border_radius=4)

        label = font.render(self.text, True, config.TEXT_COLOR)
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the button was clicked."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class ToggleButton(Button):
    """Button that toggles between two labels (e.g., Start / Pause)."""

    def __init__(self, rect: pygame.Rect, text_off: str, text_on: str) -> None:
        super().__init__(rect, text_off)
        self.text_off = text_off
        self.text_on = text_on
        self.toggled = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, active: bool = False) -> None:
        self.text = self.text_on if self.toggled else self.text_off
        # Use toggled state for active color as well
        super().draw(surface, font, active=self.toggled)


class Slider:
    """
    Horizontal slider for continuous parameters (e.g., temperature, field).

    The slider stores a current float value in [min_val, max_val].
    """

    def __init__(
        self,
        rect: pygame.Rect,
        min_val: float,
        max_val: float,
        initial: float,
        label: str = "",
        value_format: str = "{:.2f}",
    ) -> None:
        self.rect = rect
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.label = label
        self.value_format = value_format

        self._track_margin = 8
        self._handle_radius = 8
        self._dragging = False

        self.value = float(initial)
        self.value = max(self.min_val, min(self.value, self.max_val))

    # --- helper methods ---------------------------------------------------------

    def _pos_to_value(self, x: int) -> float:
        track_left = self.rect.x + self._track_margin
        track_right = self.rect.right - self._track_margin
        track_width = max(track_right - track_left, 1)

        t = (x - track_left) / track_width
        t = max(0.0, min(1.0, t))
        return self.min_val + t * (self.max_val - self.min_val)

    def _value_to_pos(self) -> int:
        track_left = self.rect.x + self._track_margin
        track_right = self.rect.right - self._track_margin
        track_width = max(track_right - track_left, 1)

        if self.max_val == self.min_val:
            return track_left

        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        t = max(0.0, min(1.0, t))
        return int(track_left + t * track_width)

    # --- drawing and events -----------------------------------------------------

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Label text above slider
        label_text = f"{self.label}: {self.value_format.format(self.value)}"
        label_surf = font.render(label_text, True, config.TEXT_COLOR)
        surface.blit(label_surf, (self.rect.x, self.rect.y - label_surf.get_height()))

        # Track
        track_y = self.rect.centery
        track_left = self.rect.x + self._track_margin
        track_right = self.rect.right - self._track_margin
        pygame.draw.line(
            surface,
            config.SLIDER_TRACK_COLOR,
            (track_left, track_y),
            (track_right, track_y),
            3,
        )

        # Handle
        handle_x = self._value_to_pos()
        handle_pos = (handle_x, track_y)
        pygame.draw.circle(surface, config.SLIDER_HANDLE_COLOR, handle_pos, self._handle_radius)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle mouse events.

        Returns
        -------
        changed : bool
            True if the slider value changed.
        """
        changed = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Start dragging if click is near the handle or on the track
            if self.rect.collidepoint(event.pos):
                self._dragging = True
                self.value = self._pos_to_value(event.pos[0])
                changed = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False

        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self.value = self._pos_to_value(event.pos[0])
            changed = True

        return changed


class LatticeSizeSelector:
    """
    Simple row of buttons to choose lattice size (e.g., 32, 64, 100).
    """

    def __init__(
        self,
        rect: pygame.Rect,
        sizes: Sequence[int],
        default: int,
    ) -> None:
        self.rect = rect
        self.sizes: List[int] = list(sizes)
        self.current_size: int = default if default in self.sizes else self.sizes[0]

        self.buttons: List[Tuple[int, Button]] = []
        if self.sizes:
            # Simple horizontal layout inside rect
            gap = 6
            total_gap = gap * (len(self.sizes) - 1)
            button_width = max((self.rect.width - total_gap) // len(self.sizes), 1)
            x = self.rect.x
            for size in self.sizes:
                btn_rect = pygame.Rect(x, self.rect.y, button_width, self.rect.height)
                self.buttons.append((size, Button(btn_rect, str(size))))
                x += button_width + gap

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Label above buttons
        label_surf = font.render("Lattice size N", True, config.TEXT_COLOR)
        surface.blit(label_surf, (self.rect.x, self.rect.y - label_surf.get_height() - 2))

        for size, button in self.buttons:
            active = (size == self.current_size)
            button.draw(surface, font, active=active)

    def handle_event(self, event: pygame.event.Event) -> Optional[int]:
        """
        Handle events; returns the newly selected size if changed, else None.
        """
        for size, button in self.buttons:
            if button.handle_event(event):
                self.current_size = size
                return size
        return None