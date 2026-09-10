"""Geometry for symmetric Bragg: the grid, and dislocations to put in it.

The diffracting planes lie parallel to the surface, so g runs along the
surface normal and the two characteristics share a lateral direction
while running opposite ways in depth:

    s_0 = ( cos t_B, 0,  sin t_B)      into the crystal
    s_g = ( cos t_B, 0, -sin t_B)      back out

The depth step is dz = dx tan(theta_B), one pixel of lateral transport
per step. `bragg_g_vector` returns g along the surface normal;
`threading_segment` and `edge_segment` build dislocations to put in it.
"""

import numpy as np


def bragg_g_vector(xtal):
    """g along the surface normal, with the magnitude of the reflection."""
    return np.array([0.0, 0.0, float(np.linalg.norm(xtal.g_vec))])


class BraggConfig:
    """Grid for a symmetric-Bragg solve.

    dz is tied to dx so the lateral walk is exactly one pixel per depth step.
    """

    def __init__(self, xtal, t_crystal=30e-6, dx=0.05e-6, half_x=12e-6,
                 beam_thickness=None, beam_centre=0.0):
        self.xtal = xtal
        self.dx = float(dx)
        self.dz = float(dx) * xtal.tan_tB     # one pixel of walk per step
        self.Nz = int(round(float(t_crystal) / self.dz)) + 1
        self.t_crystal = (self.Nz - 1) * self.dz
        self.half_x = float(half_x)
        # pad laterally by the total walk so the field of interest is never
        # contaminated by the roll-around of the index shift
        # each march walks Nz pixels, and the round trip doubles that; the
        # lateral wrap must not reach the region of interest
        self.Nx = int(2 * (round(half_x / self.dx) + 2 * self.Nz + 16))
        self.ds = self.dz / xtal.sin_tB
        # None = plane wave (floods the full entrance face, the default the
        # validation checks assume).  A finite value is a sheet beam: a top
        # hat in x at z = 0, as the Laue side does.
        self.beam_thickness = beam_thickness
        self.beam_centre = float(beam_centre)

    def incident(self, xp=np, n=1):
        """D_0 on the entrance face, shape (n, Nx)."""
        if self.beam_thickness is None:
            return xp.ones((n, self.Nx), dtype=xp.complex128)
        x = (xp.arange(self.Nx) - self.Nx / 2) * self.dx
        top = (xp.abs(x - self.beam_centre)
               < 0.5 * self.beam_thickness).astype(xp.complex128)
        return xp.broadcast_to(top, (n, self.Nx)).copy()

    def x_axis(self):
        return (np.arange(self.Nx) - self.Nx / 2) * self.dx

    def z_axis(self):
        return np.arange(self.Nz) * self.dz

    def __repr__(self):
        return (f"BraggConfig(t={self.t_crystal*1e6:.2f}um, "
                f"dx={self.dx*1e9:.0f}nm, dz={self.dz*1e9:.0f}nm, "
                f"Nz={self.Nz}, Nx={self.Nx})")

def edge_segment(sign, cfg, b_mag, depth=None):
    """Edge dislocation, line along y (normal to the plane of incidence).

    b lies in the (x, z) plane at 45 deg to the surface normal so that
    g.b = |g||b| cos45 = 2, matching the paper's diamond(400) case.
    """
    depth = cfg.t_crystal * 0.25 if depth is None else depth
    xi = np.array([0.0, 1.0, 0.0])
    b_hat = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    bv = sign * b_mag * b_hat
    L = 400e-6
    core = np.array([0.0, 0.0, depth])
    return [(core - L * xi, core + L * xi, bv)]

def threading_segment(kind, sign, b_mag, g_mag, psi=np.pi / 4,
                      y0=0.0, x0=0.0, azimuth_deg=0.0):
    """A dislocation threading the surface at angle psi from the normal.

    At psi = 45 degrees both the pure edge and the pure screw reach
    g.b = 2 for g along the normal, so a quartet of them can share one
    line direction the way the transmission case does.

    `azimuth_deg` rotates the line's in-plane projection, taking it out
    of the plane of incidence. That is what the line-orientation study
    sweeps, and it is the case the two-dimensional treatment is least
    obviously entitled to.
    """
    xi = np.array([0.0, np.sin(psi), np.cos(psi)])
    if azimuth_deg:
        c, s = np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))
        xi = np.array([c * xi[0] - s * xi[1], s * xi[0] + c * xi[1], xi[2]])
    bz = 2.0 / g_mag
    if kind == "Screw":
        bv = sign * b_mag * xi
    else:
        by = -bz / np.tan(psi)
        bx = np.sqrt(max(b_mag ** 2 - bz ** 2 - by ** 2, 0.0))
        bv = sign * np.array([bx, by, bz])
    L = 300e-6
    core = np.array([x0, y0, 0.0])
    return [(core - L * xi, core + L * xi, bv)]
