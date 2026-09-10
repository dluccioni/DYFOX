"""Takagi-Taupin diffraction in transmission (Laue geometry).

    from laue import LaueGrid, solve_dislocation, WaveCache

    grid = LaueGrid(xtal, t_crystal=273e-6, dx=0.25e-6, beam_width=120e-6)
    stack = solve_dislocation(segs, s_dev, grid, xtal)

`grid` fixes the sampling and the beam, `solver` marches the equations,
`cache` remembers the answers, and `kernels` holds the CUDA that does
the work. Importing this package does not touch the GPU; the first
solve does.
"""

from laue.cache import WaveCache
from laue.grid import LaueGrid
from laue.solver import solve_dislocation, solve_pristine

__all__ = ["LaueGrid", "WaveCache", "solve_dislocation", "solve_pristine"]
