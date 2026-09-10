"""Azimuthal power spectrum of an exit wave: which OAM channel it carries.

Two estimators of the power in each azimuthal channel m:

    oam_spectrum      bilinear sampling on rings, then a discrete
                      azimuthal transform of each
    oam_spectrum_fft  polar resampling and one FFT per ring

Fields are indexed (y, x), and the ring coordinates are passed to
map_coordinates in that order.
"""

import numpy as np

from backend import to_numpy


def oam_spectrum(field, dx, nx, ny, r_max_um=10.0, n_r=40, n_th=360,
                 ms=range(-5, 6)):
    """P_m over rings from 0.5 um out to r_max_um about the grid centre.

    `dx` is the pitch in metres and `nx`, `ny` are the full grid
    dimensions, so the rings are centred at (ny/2, nx/2) the way the
    solver lays a field out. Weighted by radius, normalised to one.
    """
    img = to_numpy(field)
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    rs = np.linspace(0.5, r_max_um, n_r) * 1e-6
    P = np.zeros(len(list(ms)))
    for r in rs:
        xs = r * np.cos(th) / dx + nx / 2
        ys = r * np.sin(th) / dx + ny / 2
        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        f = (img[y0, x0] * (1 - fx) * (1 - fy)
             + img[y0, x0 + 1] * fx * (1 - fy)
             + img[y0 + 1, x0] * (1 - fx) * fy
             + img[y0 + 1, x0 + 1] * fx * fy)
        for i, mm in enumerate(ms):
            C = np.mean(f * np.exp(-1j * mm * th))
            P[i] += np.abs(C) ** 2 * r
    return P / P.sum()


def oam_spectrum_fft(field, dx, m_max=5, r_max_um=8.0, n_th=512):
    """(m, P_m) by polar resampling and one FFT per ring.

    Rings stop at the first radius that would leave the array.  The
    centre is the geometric centre of the patch, (n - 1) / 2.
    """
    from scipy.ndimage import map_coordinates
    ew = to_numpy(field)
    n = ew.shape[0]
    c = (n - 1) / 2.0
    ms = np.arange(-m_max, m_max + 1)
    th = np.arange(n_th) * 2 * np.pi / n_th
    P = np.zeros(len(ms))
    for rp in np.arange(0.5, r_max_um, 0.25) * 1e-6 / dx:
        xs, ys = c + rp * np.cos(th), c + rp * np.sin(th)
        if max(xs.max(), ys.max()) > n - 1 or min(xs.min(), ys.min()) < 0:
            break
        co = [ys, xs]                                  # (y, x) storage order
        f = (map_coordinates(ew.real, co, order=1, mode="nearest")
             + 1j * map_coordinates(ew.imag, co, order=1, mode="nearest"))
        C = np.fft.fft(f) / n_th
        for i, m in enumerate(ms):
            P[i] += rp * abs(C[m % n_th]) ** 2
    return ms, P / P.sum()
