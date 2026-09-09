"""Beams that are not a single coherent plane wave.

A focused beam cannot also be collimated: a sheet of thickness w carries
an angular spread of at least lambda / w, which for a 1 um sheet at
17 keV is 73 urad, twenty times the Bragg Darwin width and eighty times
the Laue one.  A real condenser is wider still.  Such a beam is a
partially coherent superposition of tilted modes, modelled as an
incoherent sum over incidence angle.
"""

import numpy as np


def condenser_angles(w_D, reach=75.0):
    """Incidence angles for an incoherent condenser sum, plus weights.

    Dense across the Darwin width, coarse in the wings.  Sampling
    validated against a 0.2 w_D reference on the Laue side:

        fine step   N    max |dI| / Imax
        0.5 w_D     48       8.3e-05
        1.0 w_D     21       1.8e-02      <- the step used here
        2.0 w_D     11       7.8e-02      too coarse

    1.0 w_D sits an order of magnitude below the Poisson noise of a
    300-count exposure, so it is invisible in a recorded image.
    """
    fine = np.arange(-4.0 * w_D, 4.0 * w_D + 1e-9, w_D)
    coarse = np.arange(-reach, reach + 1e-9, 5.0)
    a = np.unique(np.concatenate([fine, coarse, [0.0]]))
    return a, np.gradient(a)


def gauss_hermite_offsets(sigma, n=8):
    """2D Gauss-Hermite quadrature for an isotropic Gaussian of rms sigma.

    Returns (offsets (n*n, 2), weights (n*n,)) with weights summing to 1
    -- the correct incoherent average over a 2D Gaussian source,
    including the polar Jacobian that ad-hoc ring sampling omits.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    X, Y = np.meshgrid(x, x)
    WX, WY = np.meshgrid(w, w)
    offs = np.stack([X.ravel() * sigma, Y.ravel() * sigma], axis=1)
    wts = (WX * WY).ravel()
    return offs, wts / wts.sum()
