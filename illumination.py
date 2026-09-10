"""Beams that are not a single coherent plane wave.

You cannot focus a beam and collimate it at the same time. A sheet of
thickness w carries an angular spread of at least lambda / w, which for
a 1 um sheet at 17 keV is 73 urad: twenty Bragg Darwin widths, or eighty
Laue ones. A real condenser is wider still. So the beam is a partially
coherent pile of tilted modes, and the way to handle it is an incoherent
sum over incidence angle.

`condenser_angles` samples that spread for a focused beam.
`gauss_hermite_offsets` does the same for an extended source, in two
dimensions. `convergence_bandwidth_nodes` adds the energy band, which
is not independent of the angle: both move the deviation parameter, so
they have to be sampled together.
"""

import numpy as np

from units import param


def condenser_angles(w_D, reach=75.0):
    """Incidence angles for an incoherent condenser sum, and their weights.

    Dense across the Darwin width, coarse out in the wings. Checked on
    the Laue side against a 0.2 w_D reference, which is cheap enough to
    compute:

        fine step   N    max |dI| / Imax
        0.5 w_D     48       8.3e-05
        1.0 w_D     21       1.8e-02      <- what we use
        2.0 w_D     11       7.8e-02      too coarse

    At 1.0 w_D the sampling error sits an order of magnitude under the
    shot noise of a 300-count exposure, so no recorded image can see it.
    """
    fine = np.arange(-4.0 * w_D, 4.0 * w_D + 1e-9, w_D)
    coarse = np.arange(-reach, reach + 1e-9, 5.0)
    a = np.unique(np.concatenate([fine, coarse, [0.0]]))
    return a, np.gradient(a)


def gauss_hermite_offsets(sigma, n=8):
    """Nodes and weights for averaging over a Gaussian source of rms sigma.

    n*n offsets, and weights that sum to one. This is the average an
    extended incoherent source really produces. Sampling a few rings by
    hand instead drops the polar Jacobian and leaves far too much weight
    sitting on the unblurred centre.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    X, Y = np.meshgrid(x, x)
    WX, WY = np.meshgrid(w, w)
    offs = np.stack([X.ravel() * sigma, Y.ravel() * sigma], axis=1)
    wts = (WX * WY).ravel()
    return offs, wts / wts.sum()


def convergence_bandwidth_nodes(xtal, sig_psi, sig_delta, n_psi, psi_max,
                                n_delta, del_max, *, matched=None,
                                vignette_f=None):
    """Quadrature nodes over a beam's angular spread and its energy band.

    A real beam is spread in both at once, and the two are not
    independent: an incidence angle psi and a relative energy offset
    delta reach the same deviation parameter along the line
    psi + tan(theta_B) delta = const. So the grid is over both, and the
    weights are the product of two Gaussians.

    `matched` describes an instrument that deliberately correlates them.
    "psi" is a dispersing optic that tilts each energy back onto the
    Bragg condition; "delta" is the degenerate case where the angular
    spread is entirely the band's doing. Neither is the default.

    `vignette_f` drops nodes whose chromatic carrier has walked outside
    the objective before anything is solved. Their weight stays out of
    the normalisation, which is right: that light really is lost.

    Returns the deviation parameters, the chromatic arguments and the
    weights, sorted from the Bragg condition outward, plus a dict
    describing what was dropped.
    """
    lam = param(xtal, "lam")
    theta_B = param(xtal, "theta_B")
    sin_2tB = param(xtal, "sin_2tB")
    tan2t = np.tan(2 * theta_B)
    psi = np.linspace(-psi_max, psi_max, n_psi)
    del_ = np.linspace(-del_max, del_max, n_delta)
    w_psi = np.exp(-0.5 * (psi / sig_psi) ** 2)
    w_del = (np.exp(-0.5 * (del_ / sig_delta) ** 2) if sig_delta > 0
             else np.ones_like(del_))
    PSI, DEL = np.meshgrid(psi, del_, indexing="ij")
    if matched == "psi":
        PSI = PSI - np.tan(theta_B) * DEL
    if matched == "delta":
        DEL = -PSI / np.tan(theta_B)
    W = np.outer(w_psi, w_del)
    W /= W.sum()
    S_DEV = (PSI + np.tan(theta_B) * DEL) * sin_2tB / lam
    D_ARG = DEL + PSI / tan2t              # equivalent chromatic argument

    nodes = np.argsort(np.abs(S_DEV).ravel())   # solve near-Bragg first
    S_flat, D_flat, W_flat = (S_DEV.ravel()[nodes], D_ARG.ravel()[nodes],
                              W.ravel()[nodes])
    info = {"n_total": len(S_flat), "n_kept": len(S_flat),
            "band_fraction": 1.0}
    if vignette_f is not None:
        keep = np.abs(sin_2tB * D_flat / lam) <= vignette_f
        if not keep.all():
            S_flat, D_flat, W_flat = S_flat[keep], D_flat[keep], W_flat[keep]
            info["n_kept"] = int(keep.sum())
            info["band_fraction"] = float(W_flat.sum())
    return S_flat, D_flat, W_flat, info
