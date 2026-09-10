"""Sampling a beam that is not a single coherent plane wave.

    condenser_angles             incidence angles and weights for a
                                 focused beam
    gauss_hermite_offsets        n*n offsets and weights for a Gaussian
                                 source
    convergence_bandwidth_nodes  joint quadrature over incidence angle
                                 and energy band
"""

import numpy as np

from units import param


def condenser_angles(w_D, reach=75.0):
    """Incidence angles for an incoherent condenser sum, and their weights.

    Sampled every w_D across +-4 w_D, then every 5 units out to `reach`.
    Weights are the node spacing.
    """
    fine = np.arange(-4.0 * w_D, 4.0 * w_D + 1e-9, w_D)
    coarse = np.arange(-reach, reach + 1e-9, 5.0)
    a = np.unique(np.concatenate([fine, coarse, [0.0]]))
    return a, np.gradient(a)


def gauss_hermite_offsets(sigma, n=8):
    """Nodes and weights for averaging over a Gaussian source of rms sigma.

    Returns n*n offsets and weights summing to one.
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

    An n_psi x n_delta grid with Gaussian weights of width `sig_psi` and
    `sig_delta`. `matched="psi"` tilts each energy back onto the Bragg
    condition; `matched="delta"` ties the angular spread entirely to the
    band. `vignette_f` drops nodes whose chromatic carrier lies outside
    the objective, leaving their weight out of the normalisation.

    Returns deviation parameters, chromatic arguments and weights,
    sorted from the Bragg condition outward, plus a dict recording how
    many nodes were kept.
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
