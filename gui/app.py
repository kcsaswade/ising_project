from __future__ import annotations

from typing import Optional

import pygame

from ising.metropolis import MetropolisIsing
from gui.controls import Button, LatticeSizeSelector, RadioButtonGroup, Slider, ToggleButton
from gui.renderer import LatticeRenderer
from gui.plots import TimeSeriesPlot
from utils import config
from utils import rng as rng_utils
from ising.statistics import RunningStats
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import csv
import sys
import threading


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

        self.saved_sweeps = []
        self.selected_sweeps = set()
        self.sweep_counter = 0

        # simple color cycle for plots
        self.sweep_colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 180, 255),
            (255, 200, 100),
            (200, 120, 255),
        ]

        # Simulation
        self.sim = MetropolisIsing(
            size=config.DEFAULT_LATTICE_SIZE,
            temperature=config.DEFAULT_TEMPERATURE,
            J=1.0,
            h=config.DEFAULT_FIELD,
            rng=self.rng,
            initial_state="random",
        )

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
        y += selector_height + 10

        # --- Optimization selector ---
        self.optim_label_pos = (x, y)

        y += 25

        self.optim_selector = RadioButtonGroup(
            ["None", "CPU", "GPU"],
            position=(x + 5, y),
            spacing=80,
            default=0,
        )

        self.optim_mode = "None"

        y += 20

        self.toggle_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Show χ",
        )

        y += button_h + 10

        self.sweep_button = Button(
            pygame.Rect(x, y, w, button_h),
            text="Run Temp Sweep",
        )
        y += button_h + 20

        self.scroll_offset = 0
        self.scroll_speed = 20  # pixels per scroll
        self.list_item_height = 25

        # Define list box rect
        self.sweep_list_rect = pygame.Rect(
            self.controls_left + 10,
            self.lattice_selector.rect.bottom + 140,
            config.PANEL_WIDTH - 20,
            100,  # visible height (adjust later)
        )

        self.export_button = Button(
            pygame.Rect(
                self.sweep_list_rect.x,
                self.sweep_list_rect.bottom + 10,
                self.sweep_list_rect.width,
                30,
            ),
            text="Export Selected (CSV)",
        )

        # --- Sweep control buttons (bottom) ---
        btn_gap = 10
        btn_w_half = (w - btn_gap) // 2

        #sweep_btn_y = config.WINDOW_HEIGHT - 70  # just above progress bar

        btn_gap = 10
        btn_w_half = (w - btn_gap) // 2

        self.save_button = Button(
            pygame.Rect(x, 0, btn_w_half, button_h),
            text="Save & Back",
        )

        self.discard_button = Button(
            pygame.Rect(x + btn_w_half + btn_gap, 0, btn_w_half, button_h),
            text="Discard & Back",
        )

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
        
        self.sweep_live_plot = TimeSeriesPlot(
            max_points=120,
            label="E(t)",
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

    def _get_save_path(self):
    
        if sys.platform == "darwin":
            # --- macOS: use osascript ---
            import subprocess

            script = '''
            set filePath to POSIX path of (choose file name with prompt "Save sweep data as CSV" default name "sweep_data.csv")
            return filePath
            '''

            try:
                path = subprocess.check_output(
                    ["osascript", "-e", script]
                ).decode("utf-8").strip()
                return path
            except subprocess.CalledProcessError:
                return None

        else:
            # --- Windows / Linux: use tkinter safely ---

            root = tk.Tk()
            root.withdraw()           # hide main window
            root.attributes("-topmost", True)  # bring dialog to front

            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save sweep data"
            )

            root.destroy()
            return path if path else None

    def _export_selected_sweeps(self):

        file_path = self._get_save_path()
        if not file_path:
            return

        selected = [s for s in self.saved_sweeps if s["id"] in self.selected_sweeps]

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sweep_label", "temperature", "C", "chi",
                "N", "J", "h", "equil_steps", "measure_steps", "subsample"
            ])

            for sweep in selected:
                meta = sweep["meta"]

                for T, C, chi in zip(sweep["T"], sweep["C"], sweep["chi"]):
                    writer.writerow([
                        sweep["label"],
                        T,
                        C,
                        chi,
                        meta["N"],
                        meta["J"],
                        meta["h"],
                        meta["equil_steps"],
                        meta["measure_steps"],
                        meta["subsample"],
                    ])

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

            # Save & Back
            if self.save_button.handle_event(event, disabled=self.sweeping):
                sweep_data = {
                    "id": time.time(),
                    "T": list(self.sweep_results[0]),
                    "C": list(self.sweep_results[1]),
                    "chi": list(self.sweep_results[2]),
                    "label": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "color": self.sweep_colors[self.sweep_counter % len(self.sweep_colors)],
                    # --- METADATA ---
                    "meta": {
                        "N": self.sim.size,
                        "J": self.sim.J,
                        "h": self.sim.h,
                        "equil_steps": self.EQUIL_STEPS,
                        "measure_steps": self.MEASURE_STEPS,
                        "subsample": self.SUBSAMPLE,
                    }
                }

                self.saved_sweeps.append(sweep_data)
                self.sweep_counter += 1

                self.mode = "LIVE"

            # Discard & Back
            if self.discard_button.handle_event(event, disabled=self.sweeping):
                self.mode = "LIVE"

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

        choice = self.optim_selector.handle_event(event, disabled=(self.mode == "SWEEP"))
        if choice is not None:
            self.optim_mode = choice
            self.sim.set_backend(choice)   # we’ll define this next

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
        
        if self.mode == "LIVE":
            if event.type == pygame.MOUSEWHEEL:
                self.scroll_offset -= event.y * self.scroll_speed

                max_offset = max(0, len(self.saved_sweeps) * self.list_item_height - self.sweep_list_rect.height)
                self.scroll_offset = max(0, min(self.scroll_offset, max_offset))
            
        if self.mode == "LIVE" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.sweep_list_rect.collidepoint(event.pos):

                rel_y = event.pos[1] - self.sweep_list_rect.y + self.scroll_offset
                index = rel_y // self.list_item_height

                if 0 <= index < len(self.saved_sweeps):
                    sweep = self.saved_sweeps[index]

                    if sweep["id"] in self.selected_sweeps:
                        self.selected_sweeps.remove(sweep["id"])
                    else:
                        self.selected_sweeps.add(sweep["id"])

        if self.mode == "LIVE":
            if self.export_button.handle_event(event, disabled=(len(self.selected_sweeps) == 0)):
                self._export_selected_sweeps()

    # --------------------------------------------------------------------- simulation update

    def _update_sweep(self):
        if self.sweep_paused:
            return

        T_vals, C_vals, chi_vals = self.sweep_results

        if self.sweep_index >= len(self.sweep_temps):
            self.sweeping = False
            return

        T = self.sweep_temps[self.sweep_index]

        # --- Adaptive schedule ---
        if 2.0 < T < 2.6:
            equil_steps = 600
            measure_steps = 1000
            subsample = 10
        else:
            equil_steps = 150
            measure_steps = 300
            subsample = 20

        # --- Equilibration phase ---
        if self.sweep_phase == "equil":
            if self.sweep_step == 0:
                self._temp_start_time = time.time()
                self.sim.set_temperature(T)

                # Only reset at very beginning of full sweep
                if self.sweep_index == 0:
                    self.sim.reset("random")


                self.sweep_live_plot.clear()

            self.sim.sweep(fraction=1.0)
            self.sweep_live_plot.add_point(self.sim.energy_per_spin())
            self.sweep_step += 1

            if self.sweep_step >= equil_steps:
                self.sweep_phase = "measure"
                self.sweep_step = 0
                self.stats.clear()

        # --- Measurement phase ---
        elif self.sweep_phase == "measure":
            self.sim.sweep(fraction=1.0)
            self.sweep_live_plot.add_point(self.sim.energy_per_spin())

            if self.sweep_step % subsample == 0:
                e = self.sim.energy_per_spin()
                m = self.sim.magnetization()
                self.stats.add(e, m)

            self.sweep_step += 1

            if self.sweep_step >= measure_steps:
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

                elapsed = time.time() - self._temp_start_time
                print(f"T={T:.2f} took {elapsed:.2f}s")

    def _update_simulation(self) -> None:
        if self.mode == "SWEEP" and self.sweeping:
            self._update_sweep()
            return
        if not self.simulation_running:
            return

        steps_per_frame = int(round(self.steps_slider.value))
        steps_per_frame = max(1, min(steps_per_frame, config.MAX_STEPS_PER_FRAME))

        for _ in range(steps_per_frame):
            self.sim.sweep(fraction=1.0)

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
                self.sim.sweep(fraction=1.0)

            # --- Measurement ---
            self.stats.clear()

            for step in range(MEASURE_STEPS):
                self.sim.sweep(fraction=1.0)

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

    def _draw_saved_sweeps(self):
        selected = [s for s in self.saved_sweeps if s["id"] in self.selected_sweeps]
        if not selected:
            return

        label = self.font.render("C(T)", True, config.TEXT_COLOR)
        self.screen.blit(label, (self.magnetization_plot_rect.x, self.magnetization_plot_rect.y - 22))

        label = self.font.render("χ(T)", True, config.TEXT_COLOR)
        self.screen.blit(label, (self.sweep_chi_plot_rect.x, self.sweep_chi_plot_rect.y - 22))
        
        # --- draw plot backgrounds + borders ---
        pygame.draw.rect(self.screen, config.PLOT_BG_COLOR, self.magnetization_plot_rect)
        pygame.draw.rect(self.screen, config.PLOT_BORDER_COLOR, self.magnetization_plot_rect, 1)

        pygame.draw.rect(self.screen, config.PLOT_BG_COLOR, self.sweep_chi_plot_rect)
        pygame.draw.rect(self.screen, config.PLOT_BORDER_COLOR, self.sweep_chi_plot_rect, 1)

        # --- global scaling ---
        all_C = [v for s in selected for v in s["C"]]
        all_chi = [v for s in selected for v in s["chi"]]

        self._draw_multi_curve(selected, "C", self.magnetization_plot_rect, all_C)
        self._draw_multi_curve(selected, "chi", self.sweep_chi_plot_rect, all_chi)

    def _draw_multi_curve(self, sweeps, key, rect, all_values):
        min_val = min(all_values)
        max_val = max(all_values)
        span = max(max_val - min_val, 1e-6)

        for sweep in sweeps:
            data = sweep[key]
            if len(data) < 2:
                continue

            points = []
            for i, val in enumerate(data):
                x = rect.x + i * rect.width / len(data)
                y = rect.bottom - (val - min_val) / span * rect.height
                points.append((x, y))

            pygame.draw.lines(self.screen, sweep["color"], False, points, 2)

    def _draw_curve(self, data, rect, color):
        if len(data) < 2:
            return

        min_val = min(data)
        max_val = max(data)
        span = max(max_val - min_val, 1e-6)

        points = []
        for i, val in enumerate(data):
            x = rect.x + i * rect.width / len(data)
            y = rect.bottom - (val - min_val) / span * rect.height
            points.append((x, y))

        pygame.draw.lines(self.screen, color, False, points, 2)

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

    def _draw_sweep_list(self):
        rect = self.sweep_list_rect

        # --- Background ---
        pygame.draw.rect(self.screen, (20, 20, 40), rect)
        pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # --- Clipping (prevents drawing outside box) ---
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(rect)

        start_y = rect.y - self.scroll_offset

        for i, sweep in enumerate(self.saved_sweeps):
            y = start_y + i * self.list_item_height

            # Skip if outside visible area (optimization)
            if y < rect.y - self.list_item_height or y > rect.bottom:
                continue

            x = rect.x + 5

            if sweep["id"] in self.selected_sweeps:
                pygame.draw.rect(
                    self.screen,
                    (60, 60, 100),
                    pygame.Rect(rect.x, y, rect.width, self.list_item_height),
                )

            # checkbox
            box_rect = pygame.Rect(x, y + 5, 14, 14)
            pygame.draw.rect(self.screen, (200, 200, 200), box_rect, 1)

            if sweep["id"] in self.selected_sweeps:
                pygame.draw.line(self.screen, (200, 200, 200),
                                (box_rect.left, box_rect.top),
                                (box_rect.right, box_rect.bottom), 2)
                pygame.draw.line(self.screen, (200, 200, 200),
                                (box_rect.right, box_rect.top),
                                (box_rect.left, box_rect.bottom), 2)

            # --- COLOR SWATCH ---
            swatch_size = 10
            swatch_x = box_rect.right + 6
            swatch_y = y + 7

            pygame.draw.rect(
                self.screen,
                sweep["color"],
                pygame.Rect(swatch_x, swatch_y, swatch_size, swatch_size),
            )

            # optional border (makes it pop on dark bg)
            pygame.draw.rect(
                self.screen,
                (200, 200, 200),
                pygame.Rect(swatch_x, swatch_y, swatch_size, swatch_size),
                1
            )

            # label
            label = self.font.render(sweep["label"], True, config.TEXT_COLOR)
            self.screen.blit(label, (swatch_x + swatch_size + 6, y + 3))

        # restore clipping
        self.screen.set_clip(prev_clip)

        content_height = len(self.saved_sweeps) * self.list_item_height

        if content_height > rect.height:
            scrollbar_width = 6

            bar_height = rect.height * (rect.height / content_height)
            bar_y = rect.y + (self.scroll_offset / content_height) * rect.height

            pygame.draw.rect(
                self.screen,
                (120, 180, 220),
                pygame.Rect(rect.right - scrollbar_width, bar_y, scrollbar_width, bar_height),
            )

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

        # Optimization label
        label = self.font.render("Optimization", True, config.TEXT_COLOR)
        self.screen.blit(label, self.optim_label_pos)

        # Radio buttons
        self.optim_selector.draw(
            self.screen,
            self.font,
            disabled=(self.mode == "SWEEP"),
        )

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

            live_plot_rect = pygame.Rect(
                bar_x,
                bar_y - 300,   # sits above buttons/status
                bar_w,
                100,
            )

            # --- Position sweep buttons ABOVE progress bar (aligned with bar) ---
            btn_gap = 10
            btn_w_half = (bar_w - btn_gap) // 2

            button_y = bar_y - 60  # 40 pixels above bar

            save_y = button_y - 45

            self.save_button.rect.topleft = (bar_x, save_y)
            self.discard_button.rect.topleft = (bar_x + btn_w_half + btn_gap, save_y)

            self.pause_sweep_button.rect.topleft = (bar_x, button_y)
            self.stop_sweep_button.rect.topleft = (bar_x + btn_w_half + btn_gap, button_y)

            # also update widths (important if panel sizes differ)
            self.pause_sweep_button.rect.width = btn_w_half
            self.stop_sweep_button.rect.width = btn_w_half

            sweep_done = not self.sweeping
            self.save_button.draw(self.screen, self.font, disabled=not sweep_done)
            self.discard_button.draw(self.screen, self.font, disabled=not sweep_done)

            self.pause_sweep_button.draw(
                self.screen,
                self.font,
                active=self.pause_sweep_button.toggled,
                disabled=sweep_done,
            )

            self.stop_sweep_button.draw(self.screen, self.font, disabled=sweep_done)

            # Draw progress bar background
            pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))

            fill_w = int(bar_w * self.sweep_progress)
            pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_y, fill_w, bar_h))

            # --- Mini live trace ---
            label = self.font.render("E(t) at current T", True, config.TEXT_COLOR)
            self.screen.blit(label, (live_plot_rect.x, live_plot_rect.y - 18))

            self.sweep_live_plot.draw(self.screen, live_plot_rect)

            # --- Sweep status text (temperature + index) ---
            if self.sweeping and self.sweep_index < len(self.sweep_temps):
                current_T = self.sweep_temps[self.sweep_index]
                step = self.sweep_index + 1
                total = len(self.sweep_temps)
                phase = self.sweep_phase  # "equil" or "measure"

                status_str = f"T = {current_T:.2f}   |   {phase}   |   {step}/{total}"
            else:
                status_str = "Sweep complete"

            status_text = self.font.render(status_str, True, config.TEXT_COLOR)

            # Position ABOVE buttons (which are above the bar)
            status_y = button_y - 65
            self.screen.blit(status_text, (bar_x, status_y))

            # Optional label
            pct = int(self.sweep_progress * 100)
            text = self.font.render(f"Sweep: {pct}%", True, config.TEXT_COLOR)
            self.screen.blit(text, (bar_x, bar_y - 20))

            # self.live_button.draw(self.screen, self.font)

            pygame.display.flip()
            return

        self._draw_sweep_list()

        export_disabled = (len(self.selected_sweeps) == 0)

        self.export_button.draw(
            self.screen,
            self.font,
            disabled=export_disabled
        )

        if self.selected_sweeps:
            # --- CLEAR ONLY PLOTS PANEL ---
            pygame.draw.rect(
                self.screen,
                config.PANEL_BG_COLOR,
                pygame.Rect(self.plots_left, 0, config.PANEL_WIDTH, config.WINDOW_HEIGHT),
            )

            # --- Draw saved sweeps ONLY ---
            self._draw_saved_sweeps()

        else:
            # Draw current live data plots
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