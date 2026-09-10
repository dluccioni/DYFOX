"""Finding the phase singularities in an exit wave.

    winding_number  loop integral of grad(arg field) about a given
                    point, at a given radius
    vortex_census   every singularity in the frame and its charge, from
                    the plaquette curl
"""

import numpy as np

from backend import to_numpy


def winding_number(field, dx, r_m=1.0e-6, n_samples=720, center=(0.0, 0.0)):
    """n = (1/2pi) times the closed-loop integral of grad(arg field) . dl.

    Samples the complex field bilinearly on a circle of radius r_m about
    `center` (metres, lab frame), takes the angles, unwraps and closes
    the loop. Sampling the field itself rather than the wrapped phase is
    what keeps the branch cut from leaving artefacts.
    """
    img = to_numpy(field)
    Ny, Nx = img.shape
    ang = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = (center[0] + r_m * np.cos(ang)) / dx + Nx / 2
    ys = (center[1] + r_m * np.sin(ang)) / dx + Ny / 2
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    x0 = np.clip(x0, 0, Nx - 2)
    y0 = np.clip(y0, 0, Ny - 2)
    f = (img[y0, x0] * (1 - fx) * (1 - fy)
         + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy
         + img[y0 + 1, x0 + 1] * fx * fy)
    phi = np.angle(f)
    dphi = np.diff(np.unwrap(phi))
    closing = np.angle(np.exp(1j * (phi[0] - phi[-1])))
    return float((np.sum(dphi) + closing) / (2 * np.pi))


def vortex_census(field, dx):
    """(x_um, y_um, q) for every phase singularity, from the plaquette curl.

    Coordinates run from the grid centre and land on plaquette centres,
    so a vortex reported at (0, 0) is sitting on the central pixel
    corner.
    """
    ph = np.angle(to_numpy(field))
    Ny, Nx = ph.shape

    def w(d):
        return np.angle(np.exp(1j * d))

    ddx = w(np.diff(ph, axis=1))
    ddy = w(np.diff(ph, axis=0))
    curl = ddx[:-1, :] + ddy[:, 1:] - ddx[1:, :] - ddy[:, :-1]
    q = np.round(curl / (2 * np.pi)).astype(int)
    ii, jj = np.nonzero(q)
    x_um = (jj + 0.5 - Nx / 2) * dx * 1e6
    y_um = (ii + 0.5 - Ny / 2) * dx * 1e6
    return np.stack([x_um, y_um, q[ii, jj]], axis=1)
