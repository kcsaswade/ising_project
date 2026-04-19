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
import numpy as np


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
        self.mode = "LIVE"   # or "SWEEP"

        self.sweep_results = None
        self.sweep_temps = np.linspace(1.0, 4.0, 20)
        self.sweep_progress = 0.0
        self.sweep_paused = False

        self.sweeping = False
        self.sweep_index = 0
        self.sweep_phase = "equil"   # or "measure"
        self.sweep_step = 0

        self.EQUIL_STEPS = 300
        self.MEASURE_STEPS = 800
        self.SUBSAMPLE = 10

        self.sweep_results = ([], [], [])  # T, C, χ

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
        y += selector_height + 20

        self.toggle_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Show χ",
        )

        y += button_h + 20

        self.sweep_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Run Temp Sweep",
        )
        y += button_h + 10

        self.live_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Back to Live",
        )

        # --- Sweep control buttons (bottom) ---
        btn_gap = 10
        btn_w_half = (w - btn_gap) // 2

        #sweep_btn_y = config.WINDOW_HEIGHT - 70  # just above progress bar

        btn_gap = 10
        btn_w_half = (w - btn_gap) // 2

        # temporary y (will be overridden in _draw)
        self.pause_sweep_button = ToggleButton(
            pygame.Rect(x, 0, btn_w_half, button_h),
            text_off="Pause Sweep",
            text_on="Resume Sweep",
        )

        self.stop_sweep_button = Button(
            pygame.Rect(x + btn_w_half + btn_gap, 0, btn_w_half, button_h),
            text="Stop Sweep",
        )

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

        # Sweep plot 
        self.sweep_chi_plot_rect = pygame.Rect(
            self.magnetization_plot_rect.x,
            self.magnetization_plot_rect.y + self.magnetization_plot_rect.height + 40,
            self.magnetization_plot_rect.width,
            self.magnetization_plot_rect.height,
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

        self.sweep_c_plot = TimeSeriesPlot(max_points=200, label="C(T)")
        self.sweep_chi_plot = TimeSeriesPlot(max_points=200, label="χ(T)")

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

        if self.mode == "SWEEP":
            # Pause / Resume
            if self.pause_sweep_button.handle_event(event):
                self.sweep_paused = not self.sweep_paused
                self.pause_sweep_button.toggled = self.sweep_paused

            # Stop sweep (go back to LIVE cleanly)
            if self.stop_sweep_button.handle_event(event):
                self.mode = "LIVE"
                self.sweeping = False
                self.sweep_paused = False
                self.sweep_results = None
                self.sweep_progress = 0.0
                self.simulation_running = False

            # Back to live (existing)
            if self.live_button.handle_event(event):
                self.mode = "LIVE"
                self.sweeping = False
                self.sweep_paused = False
                self.sweep_results = None
                self.sweep_progress = 0.0
                self.simulation_running = False

            return
        # Start / Pause toggle
        if self.start_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            self.simulation_running = not self.simulation_running
            self.start_button.toggled = self.simulation_running
        
        # Reset Random
        if self.reset_random_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            self.sim.reset(initial_state="random")
            self._clear_data()

        # Reset Ordered
        if self.reset_ordered_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            self.sim.reset(initial_state="ordered")
            self._clear_data()

        # Clear Data
        if self.clear_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            self._clear_data()

        # Quench (simple version)
        if self.quench_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            target_T = 1.0
            self.sim.set_temperature(target_T)
            self.temp_slider.value = target_T
            self._clear_data()

        # Temperature slider
        if self.temp_slider.handle_event(event, disabled=(self.mode == "SWEEP")):
            self.sim.set_temperature(self.temp_slider.value)

        # Magnetic field slider
        if self.field_slider.handle_event(event, disabled=(self.mode == "SWEEP")):
            self.sim.set_field(self.field_slider.value)

        # Steps per frame slider (value used during update)
        self.steps_slider.handle_event(event, disabled=(self.mode == "SWEEP"))

        # Lattice size selector
        new_size = self.lattice_selector.handle_event(event, disabled=(self.mode == "SWEEP"))
        if new_size is not None and new_size != self.sim.size:
            self.sim.resize(new_size)
            #self.plot.clear()
            self.m_plot.clear()

        if self.toggle_button.handle_event(event, disabled=(self.mode == "SWEEP")):
            if self.derived_mode == "C":
                self.derived_mode = "CHI"
                self.toggle_button.text = "Show C"
            else:
                self.derived_mode = "C"
                self.toggle_button.text = "Show χ"
        
        # Run sweep
        if self.sweep_button.handle_event(event):
            self.simulation_running = False
            self.mode = "SWEEP"
            self.sweep_paused = False
            self.pause_sweep_button.toggled = False
            self.sweeping = True
            self.sweep_index = 0
            self.sweep_phase = "equil"
            self.sweep_step = 0
            self.sweep_results = ([], [], [])
            self.sweep_progress = 0.0
        
        # Back to live mode
        if self.live_button.handle_event(event):
            self.mode = "LIVE"
            self.simulation_running = False
            self.sweep_results = None
            self.sweep_progress = 0.0

    # --------------------------------------------------------------------- simulation update

    def _update_sweep(self):
        if self.sweep_paused:
            return
        
        T_vals, C_vals, chi_vals = self.sweep_results

        if self.sweep_index >= len(self.sweep_temps):
            self.sweeping = False
            return

        T = self.sweep_temps[self.sweep_index]

        if 2.0 < T < 2.6:
            equil_steps = 600
        else:
            equil_steps = 300

        if self.sweep_phase == "equil":
            if self.sweep_step == 0:
                self.sim.set_temperature(T)

                # Only reset ONCE at the start of sweep
                if self.sweep_index == 0:
                    self.sim.reset("random")

            self.sim.sweep()
            self.sweep_step += 1

            if self.sweep_step >= equil_steps:
                self.sweep_phase = "measure"
                self.sweep_step = 0
                self.stats.clear()

        elif self.sweep_phase == "measure":
            self.sim.sweep()

            if self.sweep_step % self.SUBSAMPLE == 0:
                e = self.sim.energy_per_spin()
                m = self.sim.magnetization()
                self.stats.add(e, m)

            self.sweep_step += 1

            if self.sweep_step >= self.MEASURE_STEPS:
                N = self.sim.size * self.sim.size
                C = self.stats.heat_capacity(T, N)
                chi = self.stats.susceptibility(T, N)

                T_vals.append(T)
                C_vals.append(C)
                chi_vals.append(chi)

                self.sweep_index += 1
                self.sweep_phase = "equil"
                self.sweep_step = 0

                self.sweep_progress = self.sweep_index / len(self.sweep_temps)

    def _update_simulation(self) -> None:
        if self.mode == "SWEEP" and self.sweeping:
            self._update_sweep()
            return
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

    def run_temperature_sweep(self):
        results_T = []
        results_C = []
        results_chi = []

        EQUIL_STEPS = 300
        MEASURE_STEPS = 800
        SUBSAMPLE = 10

        total_T = len(self.sweep_temps)

        for i, T in enumerate(self.sweep_temps):
            self.sim.set_temperature(T)
            self.sim.reset("random")
            if 2.0 < T < 2.6:
                EQUIL_STEPS = 600
            else:
                EQUIL_STEPS = 300

            # --- Equilibration ---
            for _ in range(EQUIL_STEPS):
                self.sim.sweep()

            # --- Measurement ---
            self.stats.clear()

            for step in range(MEASURE_STEPS):
                self.sim.sweep()

                # Subsampling (VERY important)
                if step % SUBSAMPLE == 0:
                    e = self.sim.energy_per_spin()
                    m = self.sim.magnetization()
                    self.stats.add(e, m)

            N = self.sim.size * self.sim.size
            C = self.stats.heat_capacity(T, N)
            chi = self.stats.susceptibility(T, N)

            results_T.append(T)
            results_C.append(C)
            results_chi.append(chi)

            # --- Progress update ---
            self.sweep_progress = (i + 1) / total_T

        return results_T, results_C, results_chi
    # --------------------------------------------------------------------- drawing

    def _draw_sweep_plots(self):
        T_vals, C_vals, chi_vals = self.sweep_results

        # --- Feed data ---
        self.sweep_c_plot.values = C_vals
        self.sweep_chi_plot.values = chi_vals

        # --- C(T) title ---
        label = self.font.render("C(T)", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.magnetization_plot_rect.x, self.magnetization_plot_rect.y - 22),
        )

        # --- C(T) plot ---
        self.sweep_c_plot.draw(self.screen, self.magnetization_plot_rect)

        # --- χ(T) title ---
        label = self.font.render("χ(T)", True, config.TEXT_COLOR)
        self.screen.blit(
            label,
            (self.sweep_chi_plot_rect.x, self.sweep_chi_plot_rect.y - 22),
        )

        # --- χ(T) plot ---
        self.sweep_chi_plot.draw(self.screen, self.sweep_chi_plot_rect)

    def _draw(self) -> None:
        controls_disabled = (self.mode == "SWEEP")
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
        self.start_button.draw(self.screen, self.font, active=self.start_button.toggled, disabled=controls_disabled)
        self.reset_random_button.draw(self.screen, self.font, disabled=controls_disabled)
        self.reset_ordered_button.draw(self.screen, self.font, disabled=controls_disabled)
        self.quench_button.draw(self.screen, self.font, disabled=controls_disabled)
        self.clear_button.draw(self.screen, self.font, disabled=controls_disabled)
        self.temp_slider.draw(self.screen, self.font, disabled=controls_disabled)
        self.field_slider.draw(self.screen, self.font, disabled=controls_disabled)
        self.steps_slider.draw(self.screen, self.font, disabled=controls_disabled)
        self.lattice_selector.draw(self.screen, self.font, disabled=controls_disabled)
        self.toggle_button.draw(self.screen, self.font, disabled=controls_disabled)
        self.sweep_button.draw(self.screen, self.font, disabled=controls_disabled)

        N_spins = self.sim.size * self.sim.size

        c = self.stats.heat_capacity(self.sim.temperature, N_spins)
        chi = self.stats.susceptibility(self.sim.temperature, N_spins)

        # Observables (M, E, T, h)
        m = self.sim.magnetization()
        e = self.sim.energy_per_spin()
        T = self.sim.temperature
        h_val = self.sim.h

        if self.mode == "SWEEP":
            if self.sweep_results is not None:
                self._draw_sweep_plots()

            # --- Progress bar ---
            bar_x = self.plots_left + 10
            bar_y = config.WINDOW_HEIGHT - 30
            bar_w = config.PANEL_WIDTH - 20
            bar_h = 10

            # --- Position sweep buttons ABOVE progress bar (aligned with bar) ---
            btn_gap = 10
            btn_w_half = (bar_w - btn_gap) // 2

            button_y = bar_y - 60  # 40 pixels above bar

            self.pause_sweep_button.rect.topleft = (bar_x, button_y)
            self.stop_sweep_button.rect.topleft = (bar_x + btn_w_half + btn_gap, button_y)

            # also update widths (important if panel sizes differ)
            self.pause_sweep_button.rect.width = btn_w_half
            self.stop_sweep_button.rect.width = btn_w_half

            # Draw sweep control buttons
            self.pause_sweep_button.draw(
                self.screen,
                self.font,
                active=self.pause_sweep_button.toggled,
            )

            self.stop_sweep_button.draw(self.screen, self.font)

            # Draw progress bar background
            pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))

            fill_w = int(bar_w * self.sweep_progress)
            pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_y, fill_w, bar_h))

            # Optional label
            pct = int(self.sweep_progress * 100)
            text = self.small_font.render(f"Sweep: {pct}%", True, config.TEXT_COLOR)
            self.screen.blit(text, (bar_x, bar_y - 20))

            self.live_button.draw(self.screen, self.font)

            pygame.display.flip()
            return

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