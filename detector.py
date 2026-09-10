"""From a simulated image to what a pixel detector records.

Two samplings, because the objective views the object plane obliquely:

    bin_pixels     square binning, when the detector pitch is an exact
                   multiple of the simulation grid (the paper's 3x3 of
                   0.25 um -> 0.75 um)
    area_average   exact area average onto an arbitrary, anisotropic
                   pixel grid, for the general case

Poisson noise is not applied here. The expected counts are
`img * dose / reference`, that expression is not associative, and the
random stream belongs to whoever is drawing from it, so the scripts do
both parts themselves.
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

    The objective looks along the diffracted beam, so it does not sample
    the object plane isotropically. In Laue the exit-face normal is z and
    k_g sits theta_B away from it, giving pitch / cos(theta_B). In Bragg
    the diffracted beam leaves at 90 - theta_B from the surface normal,
    giving pitch / sin(theta_B), which at 17 keV is 2.4x coarser. Square
    binning gets that badly wrong. y is unforeshortened either way.
    """
    if geometry == "Laue":
        return pix_obj_um / np.cos(theta_B)
    if geometry == "Bragg":
        return pix_obj_um / np.sin(theta_B)
    raise ValueError(f"geometry must be Laue or Bragg, not {geometry!r}")


def area_average(img, dx_um, pix_y_um, pix_x_um):
    """Area-average onto a detector grid of any pitch.

    Exact even where the pixel boundaries fall between simulation
    samples: integrate the image once, then difference the bilinearly
    interpolated integral at the pixel edges. `dx_um` is the simulation
    pitch in micrometres.
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
