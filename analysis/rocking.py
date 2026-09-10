"""Rocking curves.

    planewave_rocking  the closed-form two-beam plane-wave solution
    com_maps           runs a scan and reduces it to one number per
                       pixel, the angle that pixel's intensity is
                       centred on
"""

import numpy as np

from backend import FP64, cupy
from dislocations import analytic_line_params, compute_H_grid
from units import param, urad_from_deviation


def planewave_rocking(s_dev, xtal, t_crystal, absorption=True):
    """|D_g|^2 against deviation parameter for a perfect crystal.

        |sigma_h|^2 / gamma^2 * sin^2(gamma s) * exp(2 Re(sigma_0) s)

    with gamma^2 = |sigma_h sigma_hbar| + alpha^2 / 4 and
    alpha = -2 pi s_dev. Pass absorption=False to drop the exponential
    and return the bare envelope.
    """
    s_dev = np.asarray(s_dev)
    sig_h = param(xtal, "sig_h")
    sig_hbar = param(xtal, "sig_hbar")
    alpha = -2 * np.pi * s_dev
    gam = np.sqrt(np.abs(sig_h * sig_hbar) + alpha ** 2 / 4)
    spath = t_crystal / param(xtal, "cos_tB")
    rc = np.abs(sig_h) ** 2 / gam ** 2 * np.sin(gam * spath) ** 2
    if absorption:
        rc = rc * np.exp(2 * param(xtal, "sig0").real * spath)
    return rc


def com_maps(seg_list, s_dev_arr, grid, xtal, optics, ells=(0, 2), *,
             batch=16, precision=FP64):
    """Where in the rocking curve each pixel sits, per pupil charge.

        COM(x, y) = sum_s phi(s) I(x, y, s) / sum_s I(x, y, s)

    A pixel's centre of mass is the rocking angle its intensity is
    centred on, so the map is a picture of local lattice tilt. Comparing
    the open-aperture map against the spiral one is what separates tilt
    from the vortex.

    Accumulated in float64 on the GPU across batched solves, so the
    whole scan never has to be held at once. Returns the maps in
    microradians, the integrated rocking curve for each charge, and the
    summed intensity that weighted them.
    """
    from laue.solver import solve_dislocation
    cp = cupy()
    s_dev_arr = np.asarray(s_dev_arr, dtype=float)
    Nx, Ny, _, _, _ = grid.geometry()
    # In-kernel closed form where the geometry allows it, H grid otherwise.
    H_gpu = (None if analytic_line_params(seg_list, xtal) is not None
             else compute_H_grid(seg_list, grid, xtal, precision=precision))
    phi_urad = urad_from_deviation(s_dev_arr, xtal)
    sums_wI = {ell: cp.zeros((Ny, Nx), dtype=cp.float64) for ell in ells}
    sums_I = {ell: cp.zeros((Ny, Nx), dtype=cp.float64) for ell in ells}
    rock = {ell: np.zeros(len(s_dev_arr)) for ell in ells}
    for bs in range(0, len(s_dev_arr), batch):
        be = min(bs + batch, len(s_dev_arr))
        stack = solve_dislocation(seg_list, s_dev_arr[bs:be], grid, xtal,
                                  H_gpu=H_gpu, precision=precision)
        for ell in ells:
            I = optics.image(stack, grid.dx, ell=ell)
            I64 = I.astype(cp.float64)
            w = cp.asarray(phi_urad[bs:be])[:, None, None]
            sums_wI[ell] += (I64 * w).sum(axis=0)
            sums_I[ell] += I64.sum(axis=0)
            rock[ell][bs:be] = cp.asnumpy(I64.sum(axis=(1, 2)))
            del I, I64
        del stack
    if H_gpu is not None:
        del H_gpu
    cp.get_default_memory_pool().free_all_blocks()
    com = {ell: cp.asnumpy(sums_wI[ell] / cp.maximum(sums_I[ell], 1e-300))
           for ell in ells}
    weight = {ell: cp.asnumpy(sums_I[ell]) for ell in ells}
    return com, rock, weight
