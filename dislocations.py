"""Dislocations, and the displacement phase they impose on the beam.

A dislocation is a straight segment, in lab-frame metres:

    [(r_start, r_end, b), ...]

The displacement phase H = 2*pi*g.u follows from isotropic Volterra
elasticity, with the arctan term exact and the smooth edge terms
regularised inside a core radius. Three routines evaluate it:

    compute_H_grid        the full (Ny, Nz, Nx) grid, on the GPU
    H_plane               one (Nz, Nx) plane, in NumPy or CuPy
    analytic_line_params  kernel arguments for the closed form, or None
                          when the geometry does not qualify
"""

import numpy as np

from backend import FP64, cupy


def rotate_about_z(v, deg):
    """Rotate a lab-frame vector about the surface normal."""
    if not deg:
        return v
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def line_direction(U_lab, line_cryst, tilt_deg=0.0):
    """Unit line direction in the lab frame, optionally tilted about z.

    The tilt turns the line; the Burgers vector is unchanged.
    """
    xi = U_lab @ np.asarray(line_cryst, float)
    xi = xi / np.linalg.norm(xi)
    return rotate_about_z(xi, tilt_deg)


def burgers_vector(U_lab, b_cryst, b_mag, sign=+1):
    """Lab-frame Burgers vector of magnitude `b_mag` and the given sign."""
    b_hat = U_lab @ np.asarray(b_cryst, float)
    b_hat = b_hat / np.linalg.norm(b_hat)
    return sign * b_mag * b_hat


def g_dot_b(g_vec, b):
    """g.b, which is the topological charge of the exit-wave vortex."""
    return float(np.dot(np.asarray(g_vec, float), np.asarray(b, float)))


def beam_column_center(z0, y0, tan_tB):
    """Segment centre on the forward characteristic at depth z0.

    Returns (-z0 tan(theta_B), y0, z0).
    """
    return np.array([-z0 * tan_tB, y0, z0])


def straight_segment(xi, b, center, half_length):
    """One segment, as the one-element list the solvers take."""
    xi = np.asarray(xi, float)
    center = np.asarray(center, float)
    return [(center - half_length * xi, center + half_length * xi,
             np.asarray(b, float))]


def volterra_frame(xi, bv):
    """Split b into screw and edge parts and build the frame for them.

    Returns (b_screw, b_perp, e1, e2): the screw component along the
    line, the magnitude of the edge component, a unit vector along it,
    and the third axis. For pure screw, e1 is whichever axis is furthest
    from the line.
    """
    b_screw = float(np.dot(bv, xi))
    b_perp_vec = bv - b_screw * xi
    b_perp = np.linalg.norm(b_perp_vec)
    if b_perp > 1e-20:
        e1 = b_perp_vec / b_perp
    elif abs(xi[2]) > 0.9:
        e1 = np.array([1., 0., 0.])
    elif abs(xi[0]) < 0.9:
        e1 = np.cross(xi, [1, 0, 0])
        e1 /= np.linalg.norm(e1)
    else:
        e1 = np.cross(xi, [0, 1, 0])
        e1 /= np.linalg.norm(e1)
    e2 = np.cross(xi, e1)
    return b_screw, b_perp, e1, e2


#  Chunk-sized float64 arrays alive at the widest point of one y-chunk,
#  counting the temporaries each binary operation allocates.
_CHUNK_WORKING_ARRAYS = 12


def _fit_chunk(chunk_y, Ny, Nz, Nx, floor=4):
    """Shrink the y-chunk until its working set fits in free memory.

    Never smaller than `floor` rows. Chunk size does not affect the
    result.
    """
    cp = cupy()
    free = cp.cuda.Device().mem_info[0]
    budget = free - Ny * Nz * Nx * 8            # what the result itself takes
    per_row = _CHUNK_WORKING_ARRAYS * Nz * Nx * 8
    while chunk_y > floor and chunk_y * per_row > 0.7 * budget:
        chunk_y //= 2
    return max(chunk_y, 1)


def compute_H_grid(seg_list, grid, xtal, precision=FP64, chunk_y=32):
    """H = 2*pi*g.u over the whole (Ny, Nz, Nx) volume, on the GPU.

    Built in y-chunks, which shrink when the working set would not
    otherwise fit. Segments are summed in list order.
    """
    cp = cupy()
    Nx, Ny, dz, _, _ = grid.geometry()
    Nz = grid.Nz
    dxv = grid.dx
    gv = xtal.g_vec
    nu = xtal.nu
    a_core = xtal.a_core
    CHUNK_Y = _fit_chunk(chunk_y, Ny, Nz, Nx)
    x_1d = (cp.arange(Nx, dtype=cp.float64) - Nx / 2) * dxv
    z_1d = cp.arange(Nz, dtype=cp.float64) * dz
    H = cp.zeros((Ny, Nz, Nx), dtype=precision.cp_real)
    for (ra, rb, bv) in seg_list:
        ra, rb, bv = (np.asarray(v, float) for v in (ra, rb, bv))
        t = rb - ra
        L = np.linalg.norm(t)
        if L < 1e-20:
            continue
        xi = t / L
        core = (ra + rb) / 2.0
        b_screw, b_perp, e1, e2 = volterra_frame(xi, bv)
        onu = 1.0 - nu
        C_atan = np.dot(gv, e1) * b_perp + np.dot(gv, xi) * b_screw
        ge1, ge2 = np.dot(gv, e1), np.dot(gv, e2)
        Xb = (x_1d - core[0])[None, None, :]
        Zb = (z_1d - core[2])[None, :, None]
        for ys in range(0, Ny, CHUNK_Y):
            ye = min(ys + CHUNK_Y, Ny)
            Yb = ((cp.arange(ys, ye, dtype=cp.float64) - Ny / 2) * dxv
                  - core[1])[:, None, None]
            eta = Xb * xi[0] + Yb * xi[1] + Zb * xi[2]
            Px = Xb - eta * xi[0]
            Py = Yb - eta * xi[1]
            Pz = Zb - eta * xi[2]
            del eta
            d1 = Px * e1[0] + Py * e1[1] + Pz * e1[2]
            d2 = Px * e2[0] + Py * e2[1] + Pz * e2[2]
            del Px, Py, Pz
            r2 = d1 ** 2 + d2 ** 2
            r2c = cp.maximum(r2, a_core ** 2)
            Hc = C_atan * cp.arctan2(d2, d1)
            if b_perp > 1e-20:
                c1 = b_perp / (2 * np.pi)
                sm_ux = c1 * d1 * d2 / (2 * onu * r2c)
                sm_uy = -c1 * ((1 - 2 * nu) / (4 * onu) * cp.log(r2c)
                               + (d1 ** 2 - d2 ** 2) / (4 * onu * r2c))
                Hc += 2 * np.pi * (ge1 * sm_ux + ge2 * sm_uy)
            del d1, d2, r2, r2c
            H[ys:ye] += Hc.astype(precision.cp_real)
            del Hc
    return H


def H_plane(seg_list, cfg, g_vec, nu, a_core, xp=np, y_plane=0.0):
    """H on the (Nz, Nx) plane of incidence, in NumPy or CuPy.

    The same expressions as `compute_H_grid` with y held fixed.
    """
    x_1d = xp.asarray((np.arange(cfg.Nx) - cfg.Nx / 2) * cfg.dx)
    z_1d = xp.asarray(np.arange(cfg.Nz) * cfg.dz)
    H = xp.zeros((cfg.Nz, cfg.Nx), dtype=xp.float64)
    gv = np.asarray(g_vec, float)
    for (ra, rb, bv) in seg_list:
        ra, rb, bv = (np.asarray(v, float) for v in (ra, rb, bv))
        t = rb - ra
        L = np.linalg.norm(t)
        if L < 1e-20:
            continue
        xi = t / L
        core = (ra + rb) / 2.0
        b_screw, b_perp, e1, e2 = volterra_frame(xi, bv)
        b_perp = float(b_perp)
        onu = 1.0 - nu
        C_atan = float(np.dot(gv, e1)) * b_perp + float(np.dot(gv, xi)) * b_screw
        ge1, ge2 = float(np.dot(gv, e1)), float(np.dot(gv, e2))

        Xb = (x_1d - core[0])[None, :]
        Zb = (z_1d - core[2])[:, None]
        Yb = y_plane - core[1]
        eta = Xb * xi[0] + Yb * xi[1] + Zb * xi[2]
        d1 = ((Xb - eta * xi[0]) * e1[0] + (Yb - eta * xi[1]) * e1[1]
              + (Zb - eta * xi[2]) * e1[2])
        d2 = ((Xb - eta * xi[0]) * e2[0] + (Yb - eta * xi[1]) * e2[1]
              + (Zb - eta * xi[2]) * e2[2])
        r2c = xp.maximum(d1 ** 2 + d2 ** 2, a_core ** 2)
        Hc = C_atan * xp.arctan2(d2, d1)
        if b_perp > 1e-20:
            c1 = b_perp / (2 * np.pi)
            sm_u1 = c1 * d1 * d2 / (2 * onu * r2c)
            sm_u2 = -c1 * ((1 - 2 * nu) / (4 * onu) * xp.log(r2c)
                           + (d1 ** 2 - d2 ** 2) / (4 * onu * r2c))
            Hc = Hc + 2 * np.pi * (ge1 * sm_u1 + ge2 * sm_u2)
        H += Hc
    return H


def analytic_line_params(seg_list, xtal):
    """Kernel arguments for evaluating H in closed form, or None.

    Returns None, meaning the caller should build an H grid, unless
    `seg_list` is a single line along x_lab whose displacement frame has
    no x component. The frame is built exactly as `compute_H_grid`
    builds it.
    """
    if len(seg_list) != 1:
        return None
    ra, rb, bv = (np.asarray(v, float) for v in seg_list[0])
    t = rb - ra
    L = np.linalg.norm(t)
    if L < 1e-20:
        return None
    xi = t / L
    if abs(abs(xi[0]) - 1.0) > 1e-9:
        return None
    core = (ra + rb) / 2.0
    gv = xtal.g_vec
    b_screw, b_perp, e1, e2 = volterra_frame(xi, bv)
    if abs(e1[0]) > 1e-9 or abs(e2[0]) > 1e-9:
        return None
    C_atan = np.dot(gv, e1) * b_perp + np.dot(gv, xi) * b_screw
    return dict(y0=core[1], z0=core[2],
                e1y=e1[1], e1z=e1[2], e2y=e2[1], e2z=e2[2],
                C_atan=C_atan, ge1=np.dot(gv, e1), ge2=np.dot(gv, e2),
                b_perp=b_perp, nu=xtal.nu,
                a2=xtal.a_core ** 2)
