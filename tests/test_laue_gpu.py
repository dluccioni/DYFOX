"""The transmission solver, on a device.

Skipped without CuPy and a GPU, so the rest of the suite still runs on a
laptop. Each test is small enough to finish in a second or two; the one
that is not is marked `slow`.
"""

import numpy as np
import pytest

from analysis.rocking import planewave_rocking
from backend import FP32, FP64, to_numpy
from dislocations import (burgers_vector, compute_H_grid, g_dot_b,
                          straight_segment, volterra_frame)
from laue import LaueGrid, solve_dislocation, solve_pristine
from optics import Optics, apply_pupils
from units import deviation_from_urad

pytestmark = pytest.mark.gpu


@pytest.fixture
def lens(diamond400):
    return Optics(1e-4, 0.274, diamond400)


def mixed_on_x(xtal, grid, sign=+1):
    """A dislocation along x_lab, the one geometry the in-kernel closed
    form applies to, with a Burgers vector that is actually allowed.

    The choice of b matters more than it looks. Along x_lab with g also
    along x_lab, b = (a/2)[110] gives g.b = 2 and mixed character, so
    both the arctan term and the smooth edge terms are exercised.
    b = (a/2)[100] would give g.b = 2 sqrt(2), which is not an integer
    and therefore not a dislocation at all: exp(-iH) is then
    discontinuous across the arctan branch cut, the two kernels put the
    cut in different places, and they disagree by 14 percent for a
    reason that has nothing to do with either being wrong.

    The core sits half a pixel off the sampled rows, as
    `config.make_segment` places it. On a row that passes exactly
    through the core, atan2 is evaluated at its own singularity, the
    two kernels land on opposite sides of it in the last bit, and the
    march amplifies that into a completely different row.
    """
    b = burgers_vector(np.eye(3), [1, 1, 0], xtal.b_mag, sign)
    return straight_segment([1.0, 0.0, 0.0], b,
                            [0.0, grid.dx / 2, grid.t_crystal / 2], 400e-6)


def geometry_n(xtal, grid, sign=+1):
    """The paper's inclined [101] line, which needs the H grid."""
    xi = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    b = burgers_vector(np.eye(3), [1, 0, -1], xtal.b_mag, sign)
    z0 = grid.t_crystal / 2
    return straight_segment(xi, b, [-z0 * xtal.tan_tB, 0.0, z0], 400e-6)


def test_pristine_matches_the_analytic_rocking_curve(gpu, diamond400):
    """A wide beam, at the centre of the fan, is a plane wave."""
    grid = LaueGrid(diamond400, t_crystal=45e-6 / diamond400.tan_tB,
                    dx=0.25e-6, beam_width=20e-6, beam_thickness=120e-6,
                    Nz=801)
    Ny, Nx = grid.shape
    s = np.linspace(-6, 6, 25) / diamond400.xi_g
    stack = solve_pristine(s, grid, diamond400)
    got = to_numpy(abs(stack[:, Ny // 2, Nx // 2]) ** 2)
    want = planewave_rocking(s, diamond400, grid.t_crystal)
    resid = got / got.max() - want / want.max()
    assert float(np.sqrt(np.mean(resid ** 2))) < 0.02


def test_the_coupling_sign_is_anchored_on_a_uniform_shear(gpu, diamond400):
    """Impose H = q y; the exit wave must carry phase -q y.

    This pins the exp(-iH) convention without reference to any
    dislocation, so a sign slip cannot hide behind a Burgers vector.
    """
    grid = LaueGrid(diamond400, t_crystal=20e-6, dx=0.25e-6,
                    beam_width=40e-6, beam_thickness=1e-6, Nz=101)
    Ny, Nx = grid.shape
    q = 2 * np.pi / 10e-6
    y = (gpu.arange(Ny, dtype=gpu.float64) - Ny / 2) * grid.dx
    H = gpu.broadcast_to((q * y)[:, None, None],
                         (Ny, grid.Nz, Nx)).astype(gpu.float64).copy()
    stack = solve_dislocation([], [0.0], grid, diamond400, H_gpu=H)
    rows = slice(Ny // 2 - 60, Ny // 2 + 60)
    phi = np.unwrap(np.angle(to_numpy(stack[0])[rows, Nx // 2]))
    slope = np.polyfit(np.arange(len(phi)) * grid.dx, phi, 1)[0]
    assert slope == pytest.approx(-q, rel=0.05)


def test_displacement_phase_winds_by_two_pi_g_dot_b(gpu, diamond400,
                                                    small_grid):
    """The topological statement the whole method rests on."""
    for sign in (+1, -1):
        seg = geometry_n(diamond400, small_grid, sign)
        H = to_numpy(compute_H_grid(seg, small_grid, diamond400))
        Ny_, Nz_, Nx_ = H.shape
        plane = H[:, :, Nx_ // 2].astype(float)
        (ra, rb, bv), = seg
        xi = (rb - ra) / np.linalg.norm(rb - ra)
        core = (ra + rb) / 2.0
        t_par = (0.0 - core[0]) / xi[0]
        cy = Ny_ / 2 + (core[1] + t_par * xi[1]) / small_grid.dx
        cz = (core[2] + t_par * xi[2]) / small_grid.geometry()[2]
        ang = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
        ys = np.clip((cy + 20 * np.cos(ang)).astype(int), 0, Ny_ - 1)
        zs = np.clip((cz + 20 * np.sin(ang)).astype(int), 0, Nz_ - 1)
        p = plane[ys, zs]
        wind = (np.sum(np.angle(np.exp(1j * np.diff(p))))
                + np.angle(np.exp(1j * (p[0] - p[-1])))) / (2 * np.pi)
        assert wind == pytest.approx(g_dot_b(diamond400.g_vec, bv), abs=0.05)


def test_a_core_on_a_sampled_row_is_not_the_same_problem(gpu, diamond400,
                                                        small_grid):
    """Why segments are placed half a pixel off the grid.

    Put the core exactly on a sampled row and that row evaluates
    atan2 at its own singularity. The two kernels reach it by different
    arithmetic, disagree in the last bit, and the depth march turns that
    into a row that is completely different. Every other row stays
    identical, so this is a sampling artefact rather than an instability.
    """
    from dislocations import analytic_line_params
    b = burgers_vector(np.eye(3), [1, 1, 0], diamond400.b_mag, +1)
    on_row = straight_segment([1.0, 0.0, 0.0], b,
                              [0.0, 0.0, small_grid.t_crystal / 2], 400e-6)
    assert analytic_line_params(on_row, diamond400) is not None
    s = [float(deviation_from_urad(117.5, diamond400))]
    a = to_numpy(solve_dislocation(on_row, s, small_grid, diamond400))[0]
    H = compute_H_grid(on_row, small_grid, diamond400)
    c = to_numpy(solve_dislocation(on_row, s, small_grid, diamond400,
                                   H_gpu=H))[0]
    differing = int((np.abs(a - c).max(axis=1) > 1e-12).sum())
    assert differing == 1
    assert np.abs(a - c).max(axis=1).argmax() == a.shape[0] // 2


def test_a_zero_burgers_vector_leaves_no_phase(gpu, diamond400, small_grid):
    seg = geometry_n(diamond400, small_grid)
    dead = [(seg[0][0], seg[0][1], np.zeros(3))]
    H = compute_H_grid(dead, small_grid, diamond400)
    assert float(abs(H).max()) == 0.0


def test_the_two_kernels_agree_on_a_line_along_x(gpu, diamond400,
                                                 small_grid):
    """The check the paper's geometry cannot make.

    An inclined line does not qualify for the closed-form in-kernel
    phase, so on the published geometry both paths run the same kernel
    and comparing them says nothing. A line along x_lab does qualify,
    and then the two are genuinely independent formulations.
    """
    from dislocations import analytic_line_params
    seg = mixed_on_x(diamond400, small_grid)
    assert analytic_line_params(seg, diamond400) is not None
    charge = g_dot_b(diamond400.g_vec, seg[0][2])
    assert charge == pytest.approx(2.0, abs=1e-9)   # visible, and integer
    s = [float(deviation_from_urad(117.5, diamond400))]
    closed_form = solve_dislocation(seg, s, small_grid, diamond400)
    H = compute_H_grid(seg, small_grid, diamond400)
    on_a_grid = solve_dislocation(seg, s, small_grid, diamond400, H_gpu=H)
    rel = float(gpu.sqrt(gpu.mean(gpu.abs(closed_form - on_a_grid) ** 2))
                / gpu.sqrt(gpu.mean(gpu.abs(on_a_grid) ** 2)))
    assert rel < 1e-6                # different code, the same answer


def test_the_paper_geometry_does_not_qualify_for_the_closed_form(
        diamond400, small_grid):
    from dislocations import analytic_line_params
    assert analytic_line_params(geometry_n(diamond400, small_grid),
                                diamond400) is None


def test_chunk_size_changes_nothing(gpu, diamond400, small_grid):
    """`compute_H_grid` adapts its y-chunk to available memory, so this
    has to be exactly true rather than nearly."""
    seg = geometry_n(diamond400, small_grid)
    ref = to_numpy(compute_H_grid(seg, small_grid, diamond400, chunk_y=32))
    for ck in (1, 7, 16, 512):
        got = to_numpy(compute_H_grid(seg, small_grid, diamond400,
                                      chunk_y=ck))
        assert got.tobytes() == ref.tobytes()


def test_the_plate_charge_reads_out_the_burgers_sign(gpu, diamond400,
                                                    small_grid, lens):
    """The measurement the method exists to make.

    The exit wave carries m = -g.b, so a plate of charge l = +g.b
    cancels the winding and the core fills in. Reverse the Burgers
    vector and the same plate doubles the winding instead, so which of
    l = +2 and l = -2 is brighter says which sign is present.

    The margin here is about a factor of two, on a 40 um crystal in a
    beam 8 um wide. The paper's much larger ratio needs the production
    geometry, which is far too slow to put in a unit test.
    """
    Ny, Nx = small_grid.shape
    s = [float(deviation_from_urad(117.5, diamond400))]
    c = 2                                   # +-0.5 um about the core
    core = np.s_[Ny // 2 - c:Ny // 2 + c, Nx // 2 - c:Nx // 2 + c]
    I = {}
    for sign in (+1, -1):
        stack = solve_dislocation(geometry_n(diamond400, small_grid, sign),
                                  s, small_grid, diamond400)
        pups = [lens.spiral(Nx, Ny, small_grid.dx, e) for e in (+2, -2)]
        for e, img in zip((+2, -2), to_numpy(apply_pupils(stack[0], pups))):
            I[(sign, e)] = float(img[core].mean())

    assert I[(+1, +2)] > 1.5 * I[(+1, -2)]      # +b prefers the +2 plate
    assert I[(-1, -2)] > 1.5 * I[(-1, +2)]      # -b prefers the -2 plate
    assert I[(+1, +2)] > 1.5 * I[(-1, +2)]      # and one plate separates them


def test_the_aperture_can_only_remove_energy(gpu, diamond400, small_grid,
                                             lens):
    Ny, Nx = small_grid.shape
    stack = solve_dislocation(geometry_n(diamond400, small_grid), [0.0],
                              small_grid, diamond400)
    e_in = float(gpu.sum(gpu.abs(stack[0]) ** 2))
    e_open = float(apply_pupils(
        stack[0], [lens.spiral(Nx, Ny, small_grid.dx, 0)]).sum())
    e_spiral = float(apply_pupils(
        stack[0], [lens.spiral(Nx, Ny, small_grid.dx, 2)]).sum())
    assert e_open <= e_in * 1.001
    assert e_spiral <= e_open * 1.001


def test_batching_a_rocking_scan_does_not_change_it(gpu, diamond400,
                                                    small_grid):
    seg = geometry_n(diamond400, small_grid)
    s = np.linspace(-3e5, 3e5, 9)
    H = compute_H_grid(seg, small_grid, diamond400)
    a = to_numpy(solve_dislocation(seg, s, small_grid, diamond400,
                                   H_gpu=H, batch=64))
    b = to_numpy(solve_dislocation(seg, s, small_grid, diamond400,
                                   H_gpu=H, batch=2))
    assert a.tobytes() == b.tobytes()


def test_single_precision_is_close_but_not_equal(gpu, diamond400,
                                                 small_grid, lens):
    """FP32 is a draft mode: good to a few percent, not to the bit."""
    seg = geometry_n(diamond400, small_grid)
    s = [float(deviation_from_urad(117.5, diamond400))]
    a = to_numpy(solve_dislocation(seg, s, small_grid, diamond400,
                                   precision=FP64))
    b = to_numpy(solve_dislocation(seg, s, small_grid, diamond400,
                                   precision=FP32))
    assert a.dtype == np.complex128 and b.dtype == np.complex64
    rel = np.sqrt(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(a) ** 2))
    assert 0 < rel < 0.05


def test_the_wave_cache_returns_what_it_stored(gpu, diamond400, small_grid,
                                               tmp_cache):
    """A hit comes back as complex64 while a miss computes complex128.

    That asymmetry is deliberate and load-bearing: it is why the
    regression gate always starts from an empty cache.
    """
    seg = geometry_n(diamond400, small_grid)
    first = tmp_cache.solve(seg, [0.0], small_grid, diamond400, tag="edge")
    second = tmp_cache.solve(seg, [0.0], small_grid, diamond400, tag="edge")
    assert first.dtype == np.complex128
    assert second.dtype == np.complex64
    assert to_numpy(second) == pytest.approx(
        to_numpy(first).astype(np.complex64))


def test_the_cache_key_separates_different_physics(diamond400, small_grid,
                                                   tmp_cache):
    seg_p = geometry_n(diamond400, small_grid, +1)
    seg_m = geometry_n(diamond400, small_grid, -1)
    k = tmp_cache.key(seg_p, [0.0], small_grid, diamond400)
    assert tmp_cache.key(seg_m, [0.0], small_grid, diamond400) != k
    assert tmp_cache.key(seg_p, [1.0], small_grid, diamond400) != k
    assert tmp_cache.key(seg_p, [0.0],
                         small_grid.replace(grid_pad=14), diamond400) != k
    assert tmp_cache.key(seg_p, [0.0], small_grid,
                         diamond400.at_energy(12.5)) != k
    assert tmp_cache.key(seg_p, [0.0], small_grid, diamond400,
                         precision=FP32) != k
    assert tmp_cache.key(seg_p, [0.0], small_grid, diamond400) == k


def test_com_maps_reads_local_tilt(gpu, diamond400, small_grid, lens):
    from analysis.rocking import com_maps
    seg = geometry_n(diamond400, small_grid)
    s = np.linspace(-4e5, 4e5, 7)
    com, rock, weight = com_maps(seg, s, small_grid, diamond400, lens,
                                 ells=(0, 2))
    Ny, Nx = small_grid.shape
    assert com[0].shape == (Ny, Nx)
    assert len(rock[0]) == len(s)
    assert np.isfinite(com[0]).all()
    assert weight[0].sum() > 0


@pytest.mark.slow
def test_the_production_grid_winds_by_two_pi_g_dot_b(gpu, diamond400):
    """The same topological check at the size the figures were made at.

    Two and a bit gigabytes of displacement phase, which is the reason
    this one is marked slow rather than run with the rest.
    """
    from spc_dfxm.paper import config
    grid = LaueGrid(diamond400, t_crystal=config.T_CRYSTAL, dx=config.DX,
                    beam_width=config.BEAM_WIDTH_Y, beam_thickness=1e-6,
                    grid_pad=config.GRID_PAD)
    assert grid.shape == (544, 1056) and grid.Nz == 491
    seg = config.make_segment("Edge", +1, grid, diamond400)
    H = to_numpy(compute_H_grid(seg, grid, diamond400))
    Ny_, Nz_, Nx_ = H.shape
    plane = H[:, :, Nx_ // 2].astype(float)
    del H
    gpu.get_default_memory_pool().free_all_blocks()

    (ra, rb, bv), = seg
    xi = (rb - ra) / np.linalg.norm(rb - ra)
    core = (ra + rb) / 2.0
    t_par = (0.0 - core[0]) / xi[0]
    cy = Ny_ / 2 + (core[1] + t_par * xi[1]) / grid.dx
    cz = (core[2] + t_par * xi[2]) / grid.geometry()[2]
    ang = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
    ys = np.clip((cy + 40 * np.cos(ang)).astype(int), 0, Ny_ - 1)
    zs = np.clip((cz + 40 * np.sin(ang)).astype(int), 0, Nz_ - 1)
    p = plane[ys, zs]
    wind = (np.sum(np.angle(np.exp(1j * np.diff(p))))
            + np.angle(np.exp(1j * (p[0] - p[-1])))) / (2 * np.pi)
    assert wind == pytest.approx(g_dot_b(diamond400.g_vec, bv), abs=0.05)
