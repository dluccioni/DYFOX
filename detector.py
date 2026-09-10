"""From a simulated image to what a pixel detector records.

    bin_pixels           square binning by an integer factor
    area_average         exact area average onto any anisotropic grid
    object_pixel_size    detector pitch referred to the object plane
    foreshortened_pixel  the same, along the foreshortened axis

Poisson noise is applied by the caller, not here.
"""

import numpy as np


def bin_pixels(img, factor):
    """Sum `factor` x `factor` blocks; trailing partial blocks dropped."""
    if factor == 1:
        return img
    ny, nx = img.shape
    ny -= ny % factor
    nx -= nx % factor
    return img[:ny, :nx].reshape(ny // factor, factor,
                                 nx // factor, factor).sum(axis=(1, 3))


def object_pixel_size(pixel_m, magnification):
    """Detector pitch referred to the object, in um, unforeshortened."""
    return pixel_m / magnification * 1e6


def foreshortened_pixel(pix_obj_um, theta_B, geometry):
    """Object-plane pitch along x when the exit surface is viewed obliquely.

    Laue gives pitch / cos(theta_B), Bragg pitch / sin(theta_B). y is
    unforeshortened in both.
    """
    if geometry == "Laue":
        return pix_obj_um / np.cos(theta_B)
    if geometry == "Bragg":
        return pix_obj_um / np.sin(theta_B)
    raise ValueError(f"geometry must be Laue or Bragg, not {geometry!r}")


def area_average(img, dx_um, pix_y_um, pix_x_um):
    """Area-average onto a detector grid of any pitch.

    Integrates the image once, then differences the bilinearly
    interpolated integral at the pixel edges, so pixel boundaries need
    not fall on simulation samples. `dx_um` is the simulation pitch in
    micrometres.
    """
    n = img.shape[0]
    C = np.zeros((n + 1, n + 1))
    C[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    span = n * dx_um

    def edges(pix):
        k = int(np.floor(span / pix))
        off = 0.5 * (span - k * pix)
        return (off + np.arange(k + 1) * pix) / dx_um

    ey, ex = edges(pix_y_um), edges(pix_x_um)
    gy = np.interp(ey, np.arange(n + 1), np.arange(n + 1))
    iy = np.clip(gy, 0, n)
    ix = np.clip(np.interp(ex, np.arange(n + 1), np.arange(n + 1)), 0, n)

    def bilin(a, b):
        a0 = np.floor(a).astype(int)
        b0 = np.floor(b).astype(int)
        a1 = np.clip(a0 + 1, 0, n)
        b1 = np.clip(b0 + 1, 0, n)
        fa = (a - a0)[:, None]
        fb = (b - b0)[None, :]
        return ((1 - fa) * (1 - fb) * C[np.ix_(a0, b0)]
                + fa * (1 - fb) * C[np.ix_(a1, b0)]
                + (1 - fa) * fb * C[np.ix_(a0, b1)]
                + fa * fb * C[np.ix_(a1, b1)])

    S = bilin(iy, ix)
    return S[1:, 1:] - S[:-1, 1:] - S[1:, :-1] + S[:-1, :-1]
