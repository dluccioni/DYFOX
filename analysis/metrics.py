"""Numbers you can put on a simulated image.

None of it needs a GPU. Hand these a CuPy array or a NumPy one; they
work on a host copy either way.
"""

import numpy as np

from backend import to_numpy


def crop_slices(Nx, Ny, dx, half_um=25.0):
    """Slices that crop an (Ny, Nx) grid to +/- half_um about the centre.

    The extent that comes back is in um and describes the TRANSPOSED
    image, x_lab vertical and y_lab horizontal, which is how these
    fields are displayed throughout.
    """
    hpx = int(round(half_um * 1e-6 / dx))
    sy = slice(Ny // 2 - hpx, Ny // 2 + hpx)
    sx = slice(Nx // 2 - hpx, Nx // 2 + hpx)
    ext = [-half_um, half_um, -half_um, half_um]
    return sy, sx, ext


def radial_profile(img, dx, center_px=None):
    """Azimuthally averaged radial intensity profile, (r_um, profile)."""
    img = to_numpy(img)
    Ny, Nx = img.shape
    cy, cx = (Ny // 2, Nx // 2) if center_px is None else center_px
    Y, X = np.ogrid[:Ny, :Nx]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    rmax = min(cy, cx)
    mask = R < rmax
    counts = np.bincount(R[mask].ravel(), minlength=rmax)
    sums = np.bincount(R[mask].ravel(), weights=img[mask].ravel(),
                       minlength=rmax)
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)[:rmax]
    return np.arange(rmax) * dx * 1e6, prof


def ssim_pair(img_a, img_b):
    """SSIM between two images normalised to their joint maximum."""
    from skimage.metrics import structural_similarity
    a = to_numpy(img_a)
    b = to_numpy(img_b)
    mx = max(a.max(), b.max(), 1e-30)
    return float(structural_similarity(a / mx, b / mx, data_range=1.0))


def contrast_ratio(img, floor_frac=1e-4):
    """Peak over background, taking the image median as the background.

    The median is floored at `floor_frac` of the peak. A real crystal
    has a diffuse floor from surface roughness, residual strain and
    diffuse scattering; a simulated one has whatever floating point
    leaves behind, and without the floor this cheerfully reports
    contrast ratios of 1e8.
    """
    img = to_numpy(img)
    peak = float(np.max(img))
    bg = max(float(np.median(img)), floor_frac * peak)
    return peak / bg
