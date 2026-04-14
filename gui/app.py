from __future__ import annotations

from typing import Optional

import pygame

from ising.metropolis import MetropolisIsing
from gui.controls import Button, LatticeSizeSelector, Slider, ToggleButton
from gui.renderer import LatticeRenderer
from gui.plots import TimeSeriesPlot
from utils import config
from utils import rng as rng_utils
from ising.statistics import RunningStats


class IsingApp:
    """
    Main Pygame application for the 2D Ising model.

    Responsibilities:
    - Manage the Pygame main loop
    - Own the MetropolisIsing simulation object
    - Route user input to controls
    - Update simulation in real time
    - Draw lattice, controls, and observables
    """

    def __init__(self) -> None:
        # Pygame window and timing
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption("2D Ising Model (Metropolis)")
        self.clock = pygame.time.Clock()
        self.running = True
        self.stats = RunningStats(window_size=300)

        # Fonts
        self.font = pygame.font.SysFont(None, config.FONT_SIZE)
        self.small_font = pygame.font.SysFont(None, config.SMALL_FONT_SIZE)

        # RNG shared by simulation
        self.rng = rng_utils.create_rng()

        # Simulation
        self.sim = MetropolisIsing(
            size=config.DEFAULT_LATTICE_SIZE,
            temperature=config.DEFAULT_TEMPERATURE,
            J=1.0,
            h=config.DEFAULT_FIELD,
            rng=self.rng,
            initial_state="random",
        )

        # Layout: left side = lattice, right side = control panel
        # lattice_rect = pygame.Rect(
        #     0,
        #     0,
        #     config.WINDOW_WIDTH - config.PANEL_WIDTH,
        #     config.WINDOW_HEIGHT,
        # )

        # Controls area
        #panel_left = config.WINDOW_WIDTH - config.PANEL_WIDTH
        self.plots_left = config.WINDOW_WIDTH - config.PANEL_WIDTH
        self.controls_left = config.WINDOW_WIDTH - 2 * config.PANEL_WIDTH
        padding = 10
        x = self.controls_left + padding
        y = padding
        w = config.PANEL_WIDTH - 2 * padding
        button_h = 30

        lattice_rect = pygame.Rect(
            0,
            0,
            self.controls_left,
            config.WINDOW_HEIGHT,
        )
        self.lattice_renderer = LatticeRenderer(lattice_rect)

        # Start / Pause toggle
        self.start_button = ToggleButton(
            pygame.Rect(x, y, w, button_h),
            text_off="Start",
            text_on="Pause",
        )
        y += button_h + 10

        # Reset button
        # self.reset_button = Button(
        #     pygame.Rect(x, y, w, button_h),
        #     text="Reset",
        # )
        # --- Reset buttons (split) ---

        button_w_half = (w - 10) // 2

        self.reset_random_button = Button(
            pygame.Rect(x, y, button_w_half, button_h),
            text="Reset Random",
        )

        self.reset_ordered_button = Button(
            pygame.Rect(x + button_w_half + 10, y, button_w_half, button_h),
            text="Reset Ordered",
        )

        y += button_h + 10
        #y += button_h + 20

        self.quench_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Quench",
        )

        y += button_h + 10

        self.clear_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Clear Data",
        )

        y += button_h + 20

        # Temperature slider
        self.temp_slider = Slider(
            rect=pygame.Rect(x, y, w, 20),
            min_val=config.MIN_TEMPERATURE,
            max_val=config.MAX_TEMPERATURE,
            initial=config.DEFAULT_TEMPERATURE,
            label="Temperature T",
            value_format="{:.2f}",
        )
        y += 50

        # Magnetic field slider (optional)
        self.field_slider = Slider(
            rect=pygame.Rect(x, y, w, 20),
            min_val=config.MIN_FIELD,
            max_val=config.MAX_FIELD,
            initial=config.DEFAULT_FIELD,
            label="Field h",
            value_format="{:.2f}",
        )
        y += 50

        # Steps per frame slider
        self.steps_slider = Slider(
            rect=pygame.Rect(x, y, w, 20),
            min_val=1,
            max_val=config.MAX_STEPS_PER_FRAME,
            initial=config.DEFAULT_STEPS_PER_FRAME,
            label="Steps / frame",
            value_format="{:.0f}",
        )
        y += 60

        # Lattice size selector
        selector_height = 28
        self.lattice_selector = LatticeSizeSelector(
            rect=pygame.Rect(x, y, w, selector_height),
            sizes=config.LATTICE_SIZES,
            default=config.DEFAULT_LATTICE_SIZE,
        )
        y += selector_height + 60

        self.toggle_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Show χ",
        )

        y += button_h + 20

        plot_x = self.plots_left + padding
        plot_w = config.PANEL_WIDTH - 2 * padding

        plot_height = 120
        gap = 30

        top = 20  # start from top of plots panel

        # Magnetization (top)
        self.magnetization_plot_rect = pygame.Rect(
            plot_x,
            top + 40,
            plot_w,
            plot_height,
        )

        # |M|
        self.abs_m_plot_rect = pygame.Rect(
            plot_x,
            top + 40 + plot_height + gap,
            plot_w,
            plot_height,
        )

        # Energy
        self.energy_plot_rect = pygame.Rect(
            plot_x,
            top + 40 + 2 * (plot_height + gap),
            plot_w,
            plot_height,
        )

        # Derived (C / χ)
        self.derived_plot_rect = pygame.Rect(
            plot_x,
            top + 40 + 3 * (plot_height + gap),
            plot_w,
            plot_height,
        )

        #self.plot = MagnetizationPlot(max_points=w)
        self.m_plot = TimeSeriesPlot(
            max_points=w,
            label="Magnetization",
            y_range=(-1.0, 1.0),
        )

        self.e_plot = TimeSeriesPlot(
            max_points=w,
            label="Energy",
            y_range=None,  # auto scaling
        )

        self.abs_m_plot = TimeSeriesPlot(
            max_points=w,
            label="|M|",
            y_range=(0.0, 1.0),
        )

        # --- Derived quantities (C / χ) ---

        self.c_history = []
        self.chi_history = []
        self.max_history = 1000

        self.derived_mode = "C"  # or "CHI"

        self.derived_plot = TimeSeriesPlot(
            max_points=w,
            label="Derived",
            y_range=None,
        )

        # Simulation state
        self.simulation_running = False

    # --------------------------------------------------------------------- main loop

    def run(self) -> None:
        """Run the main Pygame loop."""
        while self.running:
            self.clock.tick(config.FPS)
            self._handle_events()
            self._update_simulation()
            self._draw()

    # --------------------------------------------------------------------- event handling

    def _clear_data(self) -> None:
        """Clear all plots (and later stats)."""
        self.m_plot.clear()
        self.e_plot.clear()
        self.abs_m_plot.clear()
        self.stats.clear()
        self.c_history.clear()
        self.chi_history.clear()
        self.derived_plot.clear()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            self._handle_controls_event(event)

    def _handle_controls_event(self, event: pygame.event.Event) -> None:
        # Start / Pause toggle
        if self.start_button.handle_event(event):
            self.simulation_running = not self.simulation_running
            self.start_button.toggled = self.simulation_running

        # Reset
        # if self.reset_button.handle_event(event):
        #     self.sim.reset()
        #     self.plot.clear()
        
        # Reset Random
        if self.reset_random_button.handle_event(event):
            self.sim.reset(initial_state="random")
            self._clear_data()

        # Reset Ordered
        if self.reset_ordered_button.handle_event(event):
            self.sim.reset(initial_state="ordered")
            self._clear_data()

        # Clear Data
        if self.clear_button.handle_event(event):
            self._clear_data()

        # Quench (simple version)
        if self.quench_button.handle_event(event):
            target_T = 1.0
            self.sim.set_temperature(target_T)
            self.temp_slider.value = target_T
            self._clear_data()

        # Temperature slider
        if self.temp_slider.handle_event(event):
            self.sim.set_temperature(self.temp_slider.value)

        # Magnetic field slider
        if self.field_slider.handle_event(event):
            self.sim.set_field(self.field_slider.value)

        # Steps per frame slider (value used during update)
        self.steps_slider.handle_event(event)

        # Lattice size selector
        new_size = self.lattice_selector.handle_event(event)
        if new_size is not None and new_size != self.sim.size:
            self.sim.resize(new_size)
            #self.plot.clear()
            self.m_plot.clear()

        if self.toggle_button.handle_event(event):
            if self.derived_mode == "C":
                self.derived_mode = "CHI"
                self.toggle_button.text = "Show C"
            else:
                self.derived_mode = "C"
                self.toggle_button.text = "Show χ"

    # --------------------------------------------------------------------- simulation update

    def _update_simulation(self) -> None:
        if not self.simulation_running:
            return

        steps_per_frame = int(round(self.steps_slider.value))
        steps_per_frame = max(1, min(steps_per_frame, config.MAX_STEPS_PER_FRAME))

        for _ in range(steps_per_frame):
            self.sim.sweep()

        # Record magnetization and energy for plotting
        m = self.sim.magnetization()
        e = self.sim.energy_per_spin()
        self.m_plot.add_point(m)
        self.abs_m_plot.add_point(abs(m))  
        self.e_plot.add_point(e)
        self.stats.add(e, m)

        N_spins = self.sim.size * self.sim.size

        c = self.stats.heat_capacity(self.sim.temperature, N_spins)
        chi = self.stats.susceptibility(self.sim.temperature, N_spins)

        self.c_history.append(c)
        self.chi_history.append(chi)

        if len(self.c_history) > self.max_history:
            self.c_history.pop(0)
            self.chi_history.pop(0)

        if self.derived_mode == "C":
            self.derived_plot.values = self.c_history[-self.derived_plot.max_points:]
        else:
            self.derived_plot.values = self.chi_history[-self.derived_plot.max_points:]

    # --------------------------------------------------------------------- drawing

    def _draw(self) -> None:
        self.screen.fill(config.BG_COLOR)

        # Lattice
        self.lattice_renderer.draw(self.screen, self.sim.spins)

        # Controls panel
        controls_rect = pygame.Rect(
            self.controls_left,
            0,
            config.PANEL_WIDTH,
            config.WINDOW_HEIGHT,
        )
        pygame.draw.rect(self.screen, config.PANEL_BG_COLOR, controls_rect)

        # Plots panel
        plots_rect = pygame.Rect(
            self.plots_left,
            0,
            config.PANEL_WIDTH,
            config.WINDOW_HEIGHT,
        )
        pygame.draw.rect(self.screen, config.PANEL_BG_COLOR, plots_rect)

        # Controls
        self.start_button.draw(self.screen, self.font, active=self.start_button.toggled)
        #self.reset_button.draw(self.screen, self.font)
        self.reset_random_button.draw(self.screen, self.font)
        self.reset_ordered_button.draw(self.screen, self.font)
        self.quench_button.draw(self.screen, self.font)
        self.clear_button.draw(self.screen, self.font)
        self.temp_slider.draw(self.screen, self.font)
        self.field_slider.draw(self.screen, self.font)
        self.steps_slider.draw(self.screen, self.font)
        self.lattice_selector.draw(self.screen, self.font)
        self.toggle_button.draw(self.screen, self.font)

        N_spins = self.sim.size * self.sim.size

        c = self.stats.heat_capacity(self.sim.temperature, N_spins)
        chi = self.stats.susceptibility(self.sim.temperature, N_spins)

        text_c = self.small_font.render(f"C = {c:.3f}", True, config.TEXT_COLOR)
        text_chi = self.small_font.render(f"χ = {chi:.3f}", True, config.TEXT_COLOR)

        # self.screen.blit(text_c, (self.controls_left + 10, self.y))
        # self.screen.blit(text_chi, (self.controls_left + 10, self.y + 20))
        self.screen.blit(text_c, (self.controls_left + 10, self.lattice_selector.rect.bottom + 25))
        self.screen.blit(text_chi, (self.controls_left + 10, self.lattice_selector.rect.bottom + 50))

        # Observables (M, E, T, h)
        m = self.sim.magnetization()
        e = self.sim.energy_per_spin()
        T = self.sim.temperature
        h_val = self.sim.h

        # obs_x = panel_rect.x + 10
        # obs_y = self.lattice_selector.rect.bottom + 5

        # for text in [
        #     f"M = {m:.3f}",
        #     f"E / spin = {e:.3f}",
        #     f"T = {T:.2f}",
        #     f"h = {h_val:.2f}",
        # ]:
        #     surf = self.small_font.render(text, True, config.TEXT_COLOR)
        #     self.screen.blit(surf, (obs_x, obs_y))
        #     obs_y += surf.get_height() + 2

        # Magnetization title (left)
        label = self.font.render("Magnetization", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.magnetization_plot_rect.x, self.magnetization_plot_rect.y - 22),
        )

        # Magnetization value (right)
        value = self.font.render(f"M = {m:.3f}", True, config.TEXT_COLOR)
        value_rect = value.get_rect()
        value_rect.topright = (
            self.magnetization_plot_rect.right,
            self.magnetization_plot_rect.y - 22,
        )
        self.screen.blit(value, value_rect)

        # Plot
        self.m_plot.draw(self.screen, self.magnetization_plot_rect)

        # Absolute Magnetization title (left)
        label = self.font.render("|M|", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.abs_m_plot_rect.x, self.abs_m_plot_rect.y - 22),
        )

        # Value (right)
        value = self.font.render(f"|M| = {abs(m):.3f}", True, config.TEXT_COLOR)
        value_rect = value.get_rect()
        value_rect.topright = (
            self.abs_m_plot_rect.right,
            self.abs_m_plot_rect.y - 22,
        )
        self.screen.blit(value, value_rect)

        # Plot
        self.abs_m_plot.draw(self.screen, self.abs_m_plot_rect)

        # Energy title (left)
        label = self.font.render("Energy", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.energy_plot_rect.x, self.energy_plot_rect.y - 22),
        )

        # Energy value (right)
        value = self.font.render(f"E / spin = {e:.3f}", True, config.TEXT_COLOR)
        value_rect = value.get_rect()
        value_rect.topright = (
            self.energy_plot_rect.right,
            self.energy_plot_rect.y - 22,
        )
        self.screen.blit(value, value_rect)

        # Plot
        self.e_plot.draw(self.screen, self.energy_plot_rect)

        # Derived title
        label = self.font.render("Derived", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.derived_plot_rect.x, self.derived_plot_rect.y - 22),
        )

        # Value (right)
        if self.derived_mode == "C":
            val = self.c_history[-1] if self.c_history else 0.0
            text = f"C = {val:.3f}"
        else:
            val = self.chi_history[-1] if self.chi_history else 0.0
            text = f"χ = {val:.3f}"

        value = self.font.render(text, True, config.TEXT_COLOR)
        value_rect = value.get_rect()
        value_rect.topright = (
            self.derived_plot_rect.right,
            self.derived_plot_rect.y - 22,
        )
        self.screen.blit(value, value_rect)

        # Plot
        self.derived_plot.draw(self.screen, self.derived_plot_rect)

        pygame.display.flip()