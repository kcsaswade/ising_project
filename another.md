# 2D Ising Model with Metropolis Algorithm and Pygame GUI

This project implements a 2D Ising model on a square lattice using the Metropolis Monte Carlo algorithm, with an interactive real-time visualization built in Pygame.

The focus is on correctness, clarity, and modularity rather than on advanced features or performance tricks.

---

## Physical Model

We simulate a 2D square lattice of spins $s_i = \pm 1$ with periodic boundary conditions. The Hamiltonian is

$$
H = -J \sum_{\langle i, j \rangle} s_i s_j - h \sum_i s_i
$$

where the first sum runs over nearest-neighbor pairs and the second over all spins.

The Metropolis algorithm evolves the system by repeatedly:

1. Choosing a random spin.
2. Computing the local energy change $\Delta E$ caused by flipping that spin using only its four neighbors.
3. Accepting the flip if $\Delta E \le 0$, or with probability $\exp(-\Delta E / T)$ otherwise.

We set the Boltzmann constant $k_B = 1$, so the temperature $T$ is in energy units.

At low temperatures you should observe ordered (ferromagnetic) configurations, while at high temperatures the lattice becomes disordered.

---

## Project Structure

The repository is organized to clearly separate physics/simulation logic, GUI rendering, and utilities:

```text
ising_project/
│
├── main.py
│
├── ising/
│   ├── __init__.py
│   ├── lattice.py
│   ├── energy.py
│   ├── metropolis.py
│   ├── observables.py
│
├── gui/
│   ├── __init__.py
│   ├── app.py
│   ├── renderer.py
│   ├── controls.py
│   ├── plots.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── rng.py
│
└── README.md
```

### `main.py`

Entry point of the application. It initializes Pygame and starts the `IsingApp` from `gui/app.py`.

### `ising/` (Physics and Simulation)

- `lattice.py`  
  Functions to create:
  - `random_lattice(N, rng)`: random configuration of $\pm 1$ spins.
  - `ordered_lattice(N, up=True)`: fully ordered lattice (all +1 or all -1).

- `energy.py`  
  Functions to compute:
  - `total_energy(spins, J, h)`: full energy with periodic boundary conditions.
  - `energy_per_spin(spins, J, h)`: energy per spin.

- `observables.py`  
  Functions for physical observables:
  - `magnetization(spins)`: total magnetization.
  - `magnetization_per_spin(spins)`: magnetization per spin.
  - `energy_per_spin(spins, J, h)`: convenience wrapper.

- `metropolis.py`  
  Contains the `MetropolisIsing` dataclass that manages:
  - The current lattice of spins.
  - Parameters: `size`, `temperature`, `J`, and `h`.
  - Methods:
    - `step()`: single Metropolis spin-flip attempt using local $\Delta E$.
    - `sweep()`: one Monte Carlo sweep (N² attempted updates).
    - `reset(initial_state="random"|"ordered")`: reinitialize lattice.
    - `resize(new_size, initial_state=None)`: change lattice size and reinitialize.
    - `set_temperature(T)` and `set_field(h)`.
    - `magnetization()` and `energy_per_spin()` for observables.

There is no GUI code inside this package.

### `gui/` (Visualization and Controls)

- `app.py`  
  The main application class `IsingApp`:
  - Owns the `MetropolisIsing` instance.
  - Implements the Pygame main loop (`run` method).
  - Handles events, updates the simulation when running, and redraws the screen.
  - Coordinates:
    - `LatticeRenderer` for drawing the spin lattice.
    - Control widgets (`Button`, `ToggleButton`, `Slider`, `LatticeSizeSelector`).
    - `MagnetizationPlot` for the real-time magnetization graph.

- `renderer.py`  
  `LatticeRenderer` draws the Ising lattice as a grid of colored squares:
  - +1 spins: light color.
  - -1 spins: dark color.
  It uses the left part of the window, scaling the cell size to fit the lattice.

- `controls.py`  
  Basic GUI widgets for user interaction:
  - `Button`: clickable rectangular button.
  - `ToggleButton`: a button that toggles state (e.g., Start/Pause).
  - `Slider`: horizontal slider for continuous parameters (e.g., temperature, field).
  - `LatticeSizeSelector`: row of buttons to choose lattice size (e.g., 32, 64, 100).

- `plots.py`  
  `MagnetizationPlot` keeps a sliding window of magnetization values and draws a simple scrolling line plot in the control panel.

### `utils/` (Utilities)

- `config.py`  
  Centralized configuration:
  - Window size and layout (`WINDOW_WIDTH`, `WINDOW_HEIGHT`, `PANEL_WIDTH`).
  - Simulation parameters (temperature range, field range, steps per frame).
  - Lattice size options.
  - Color definitions (background, spins, panel, buttons, plot).
  - Font sizes.

- `rng.py`  
  Helper to create a NumPy random generator:
  - `create_rng(seed=None)`.

---

## How to Run

1. **Install dependencies**

   From your environment (e.g., virtualenv or conda):

   ```bash
   pip install pygame numpy
   ```

2. **Run the application**

   From the `ising_project/` directory:

   ```bash
   python main.py
   ```

A window will open, showing:

- A large lattice visualization on the left.
- A control and info panel on the right.

---

## Controls

All controls are located in the right-hand panel.

- **Start / Pause toggle**
  - Button labeled "Start" / "Pause".
  - Click to start the Metropolis sweeps.
  - Click again to pause the simulation.

- **Reset**
  - Resets the lattice to a new initial configuration (currently random).
  - Clears the magnetization plot.

- **Temperature slider (T)**
  - Labeled "Temperature T".
  - Range: approximately 0.5 to 5.0 (see `MIN_TEMPERATURE`, `MAX_TEMPERATURE`).
  - Drag horizontally to change T in real time.
  - Lower T → more ordered, higher T → more disordered.

- **Magnetic field slider (h)**
  - Labeled "Field h".
  - Range: specified by `MIN_FIELD` and `MAX_FIELD` (e.g., -2.0 to 2.0).
  - Adjusts the uniform external field term in the Hamiltonian.

- **Steps per frame slider**
  - Labeled "Steps / frame".
  - Controls how many Monte Carlo sweeps are performed per rendered frame.
  - Higher values speed up evolution but increase CPU usage.

- **Lattice size selector**
  - Row of buttons labeled with N (e.g., 32, 64, 100).
  - Click to change the lattice size.
  - The lattice is reinitialized and the magnetization plot is cleared.

---

## Observables Display

In the right-hand panel (below the lattice size selector) you will see:

- `M` – Magnetization per spin.
- `E / spin` – Energy per spin.
- `T` – Current temperature.
- `h` – Current external magnetic field.

These values update as the simulation runs.

---

## Magnetization Plot

At the bottom of the control panel is a simple real-time plot:

- **Horizontal axis**: simulation time (recent sweeps).
- **Vertical axis**: magnetization per spin $M$ in the range $[-1, 1]$.
- The plot scrolls as new values are added, showing recent dynamics of the system.

---

## Notes and Extensibility

- Periodic boundary conditions are implemented explicitly in the Metropolis update via modulo indexing of neighbors.
- The energy change $\Delta E$ is computed locally using only four nearest neighbors; there is no full lattice recomputation during Metropolis steps.
- Global observables (energy per spin, magnetization per spin) are computed from the full lattice when needed, which is sufficient for lattices up to around $100 \times 100$ on a typical machine.
- The modular structure should make it straightforward to:
  - Add measurements (e.g., specific heat, susceptibility).
  - Implement alternative update schemes (e.g., Glauber dynamics).
  - Replace or extend the GUI without touching the core physics modules.