"""
GUI package for the Ising model project.

Separated into:
- app.py      : main application and Pygame loop
- renderer.py : drawing the lattice
- controls.py : buttons, sliders, and lattice size selector
- plots.py    : simple real-time magnetization plot
"""

from .app import IsingApp  # noqa: F401