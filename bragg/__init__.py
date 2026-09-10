"""Takagi-Taupin diffraction in reflection (Bragg geometry).

    from bragg import BraggConfig, bragg_g_vector, solve_surface_batch

    g = bragg_g_vector(xtal)
    cfg = BraggConfig(xtal, t_crystal=30e-6, dx=0.05e-6)
    surface, info = solve_surface_batch(segs, s_dev, cfg, ys, xtal, g)

This is a different problem from the transmission case, not a variant of
it. The diffracted beam leaves through the entrance surface, so

    D_0(z = 0) = incident        at the front
    D_g(z = t) = 0               nothing enters from the back

is a two-point boundary-value problem, and no single march solves it.
See `bragg.solver` for what does.

Everything here runs in NumPy. Pass CuPy if you have it and the grid is
large enough to pay for the trip, but nothing requires it, and importing
this package never touches a GPU.

Conventions are the same as the transmission side, deliberately, so that
a sign slip cannot quietly flip a Burgers-vector assignment:

    dD/ds = sigma D,  sigma = -i r_e lambda F / V_cell
    couplings carry sigma_h e^{-iH} and sigma_hbar e^{+iH},  H = 2 pi g.u
    alpha = -2 pi s_dev
    the physical exit wave restores the factored phase: D_g exp(-i g.u)

Validation lives in spc_dfxm/bragg/validate_bragg.py.
"""

from bragg.geometry import (BraggConfig, bragg_g_vector, edge_segment,
                            threading_segment)
from bragg.reference import (darwin_width_urad, reflectivity_riccati,
                             reflectivity_semi_infinite)
from bragg.solver import (perfect_symbol, solve_bragg, solve_bragg_stable,
                          solve_surface_batch)

__all__ = ["BraggConfig", "bragg_g_vector", "edge_segment",
           "threading_segment", "darwin_width_urad", "reflectivity_riccati",
           "reflectivity_semi_infinite", "perfect_symbol", "solve_bragg",
           "solve_bragg_stable", "solve_surface_batch"]
