"""The transmission (Laue) solver.

Both amplitudes travel into the crystal and out the far face, so this is
an initial-value problem: start at the entrance surface and march to the
exit. That is what makes Laue cheap compared to reflection, where the
diffracted beam comes back out the way it went in and the problem
becomes a boundary-value one.

What comes back is the physical exit wave, not an envelope. The
displacement phase is folded into the off-diagonal couplings as
exp(-+iH) inside the kernel, so there is no envelope approximation and
no phase to restore afterwards.

    stack = solve_dislocation(segs, s_dev, grid, xtal)

`s_dev` may be an array, and then one launch gives you a whole rocking
scan sharing a single displacement grid. Building that grid is what a
solve mostly costs, so scanning this way rather than in a Python loop is
worth roughly the length of the scan.

Everything returned is a CuPy array of shape (n_angles, Ny, Nx). Call
`backend.to_numpy` at the point where you want it back on the host.
"""

import numpy as np

from backend import BLOCK, FP64, cupy
from dislocations import analytic_line_params, compute_H_grid
from laue import kernels


def _alloc(grid, xtal, n_e, precision):
    cp = cupy()
    Nx, Ny, dz, ds, spx = grid.geometry()
    Dg_re = cp.empty(n_e * Ny * Nx, dtype=precision.cp_real)
    Dg_im = cp.empty(n_e * Ny * Nx, dtype=precision.cp_real)
    smem = kernels.shared_bytes(Nx, precision)
    return Nx, Ny, dz, ds, spx, Dg_re, Dg_im, smem


def solve_dislocation(seg_list, s_dev_arr, grid, xtal, *, H_gpu=None,
                      batch=64, precision=FP64):
    """Exit wave of a dislocation field, at each deviation parameter given.

    Pass `H_gpu` to reuse a displacement grid you already built, which
    is what to do when the same dislocations are solved at several
    angles or thicknesses.

    Geometries that are a single straight line along x_lab take the
    closed-form path instead, evaluating the phase inside the kernel
    with no grid at all. Nothing in the paper's geometry qualifies; the
    path exists so the two can be checked against each other.
    """
    cp = cupy()
    DTYPE = precision.scalar
    s_dev_arr = np.atleast_1d(np.asarray(s_dev_arr, dtype=float))
    n_e = len(s_dev_arr)
    Nx, Ny, dz, ds, spx, _, _, smem = _alloc(grid, xtal, 1, precision)

    ap = None if H_gpu is not None else analytic_line_params(seg_list, xtal)
    if ap is None and H_gpu is None:
        h_bytes = Nx * Ny * grid.Nz * precision.itemsize
        free_mem = cp.cuda.Device().mem_info[0]
        if h_bytes > 0.7 * free_mem:
            raise MemoryError(f"H grid needs {h_bytes/1e9:.1f} GB, "
                              f"{free_mem/1e9:.1f} GB free")
        H_gpu = compute_H_grid(seg_list, grid, xtal, precision=precision)

    out = cp.empty((n_e, Ny, Nx), dtype=precision.cp_complex)
    common = (
        DTYPE(xtal.sig0.real), DTYPE(xtal.sig0.imag),
        DTYPE(xtal.sig_h.real), DTYPE(xtal.sig_h.imag),
        DTYPE(xtal.sig_hbar.real), DTYPE(xtal.sig_hbar.imag))
    for bs in range(0, n_e, batch):
        be = min(bs + batch, n_e)
        nb = be - bs
        Dg_re = cp.empty(nb * Ny * Nx, dtype=precision.cp_real)
        Dg_im = cp.empty(nb * Ny * Nx, dtype=precision.cp_real)
        sd_gpu = cp.asarray(s_dev_arr[bs:be].astype(precision.real))
        tail = (sd_gpu, np.int32(nb),
                DTYPE(dz), DTYPE(spx), DTYPE(ds), DTYPE(grid.dx),
                np.int32(grid.Nz), np.int32(Nx), np.int32(Ny),
                DTYPE(grid.sheet_arg()),
                DTYPE(grid.beam_width / 2.0))
        if ap is not None:
            kernels.prepare(precision.kernel("analytic"), smem)(
                (Ny * nb,), (BLOCK,), (
                    Dg_re, Dg_im, *common, *tail,
                    DTYPE(ap["y0"]), DTYPE(ap["z0"]),
                    DTYPE(ap["e1y"]), DTYPE(ap["e1z"]),
                    DTYPE(ap["e2y"]), DTYPE(ap["e2z"]),
                    DTYPE(ap["C_atan"]), DTYPE(ap["ge1"]), DTYPE(ap["ge2"]),
                    DTYPE(ap["b_perp"]), DTYPE(ap["nu"]), DTYPE(ap["a2"])),
                shared_mem=smem)
        else:
            kernels.prepare(precision.kernel("fullfield"), smem)(
                (Ny * nb,), (BLOCK,), (
                    Dg_re, Dg_im, *common, *tail, H_gpu), shared_mem=smem)
        out[bs:be] = (Dg_re.reshape(nb, Ny, Nx)
                      + 1j * Dg_im.reshape(nb, Ny, Nx)
                      ).astype(precision.cp_complex)
        del Dg_re, Dg_im
    return out


def solve_pristine(s_dev_arr, grid, xtal, *, precision=FP64):
    """Exit wave of a perfect crystal, at each deviation parameter given.

    The rocking curve of this is the reference every strained solve is
    read against, and it has a closed form to check it with. See
    `analysis.rocking.planewave_rocking`.
    """
    DTYPE = precision.scalar
    cp = cupy()
    s_dev_arr = np.atleast_1d(np.asarray(s_dev_arr, dtype=float))
    n_e = len(s_dev_arr)
    Nx, Ny, dz, ds, spx, Dg_re, Dg_im, smem = _alloc(grid, xtal, n_e,
                                                     precision)
    sd_gpu = cp.asarray(s_dev_arr.astype(precision.real))
    kernels.prepare(precision.kernel("pristine"), smem)((Ny * n_e,), (BLOCK,), (
        Dg_re, Dg_im,
        DTYPE(xtal.sig0.real), DTYPE(xtal.sig0.imag),
        DTYPE(xtal.sig_h.real), DTYPE(xtal.sig_h.imag),
        DTYPE(xtal.sig_hbar.real), DTYPE(xtal.sig_hbar.imag),
        sd_gpu, np.int32(n_e),
        DTYPE(dz), DTYPE(spx), DTYPE(ds), DTYPE(grid.dx),
        np.int32(grid.Nz), np.int32(Nx), np.int32(Ny),
        DTYPE(grid.sheet_arg()), DTYPE(grid.beam_width / 2.0)),
        shared_mem=smem)
    return (Dg_re.reshape(n_e, Ny, Nx)
            + 1j * Dg_im.reshape(n_e, Ny, Nx)).astype(precision.cp_complex)
