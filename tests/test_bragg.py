"""The reflection solver, checked against closed forms.

Everything here runs in NumPy, which is the point: the Bragg side needs
no GPU, and these tests would fail to import if it ever started needing
one.
"""

import numpy as np
import pytest

from bragg import (BraggConfig, bragg_g_vector, darwin_width_urad,
                   reflectivity_riccati, reflectivity_semi_infinite,
                   solve_bragg, solve_bragg_stable, solve_surface_batch,
                   threading_segment)
from dislocations import g_dot_b
from units import deviation_from_urad


@pytest.fixture(scope="module")
def g(diamond400):
    return bragg_g_vector(diamond400)


@pytest.fixture(scope="module")
def cfg(diamond400):
    return BraggConfig(diamond400, t_crystal=30e-6, dx=0.05e-6, half_x=12e-6)


def test_g_lies_along_the_surface_normal(diamond400, g):
    """Symmetric Bragg diffracts off planes parallel to the surface."""
    assert g[0] == 0.0 and g[1] == 0.0
    assert g[2] == pytest.approx(np.linalg.norm(diamond400.g_vec), rel=1e-15)


def test_reusing_the_laue_g_would_make_the_dislocation_invisible(
        diamond400, g):
    """Why bragg_g_vector exists rather than xtal.g_vec.

    The Laue vector lies in the surface, and a dislocation threading
    that surface has a Burgers vector entirely out of it, so g.b comes
    out at zero: no contrast at all. Reflection needs g along the
    normal, which is what bragg_g_vector gives.
    """
    for kind in ("Edge", "Screw"):
        seg = threading_segment(kind, +1, diamond400.b_mag, float(g[2]))
        assert g_dot_b(g, seg[0][2]) == pytest.approx(2.0, abs=1e-9)
        assert g_dot_b(diamond400.g_vec, seg[0][2]) == pytest.approx(
            0.0, abs=1e-9)


def test_threading_segments_have_integer_charge(diamond400, g):
    for kind in ("Edge", "Screw"):
        for sign in (+1, -1):
            seg = threading_segment(kind, sign, diamond400.b_mag,
                                    float(g[2]))
            assert g_dot_b(g, seg[0][2]) == pytest.approx(2.0 * sign,
                                                          abs=1e-9)


def test_depth_step_advects_exactly_one_pixel(diamond400, cfg):
    """dz = dx tan(theta_B) in reflection, the mirror of the Laue rule."""
    assert cfg.dz == pytest.approx(cfg.dx * diamond400.tan_tB, rel=1e-15)
    assert cfg.ds == pytest.approx(cfg.dz / diamond400.sin_tB, rel=1e-15)


def test_grid_is_padded_past_the_round_trip_walk(cfg):
    """The lateral roll wraps; the pad keeps the wrap out of the frame."""
    walk_px = 2 * cfg.Nz
    assert cfg.Nx > 2 * (cfg.half_x / cfg.dx + walk_px)


def test_darwin_width(diamond400):
    assert darwin_width_urad(diamond400) == pytest.approx(3.588, abs=0.01)


def test_reflectivity_never_exceeds_one(diamond400):
    s = deviation_from_urad(np.linspace(-30, 20, 401), diamond400)
    R = reflectivity_semi_infinite(s, diamond400)
    assert np.abs(R).max() <= 1.0 + 1e-12


def test_total_reflection_plateau_is_where_it_should_be(diamond400):
    """A domain one Darwin width wide, centred on the refraction shift."""
    th = np.linspace(-14, 4, 3601)
    R = np.abs(reflectivity_semi_infinite(
        deviation_from_urad(th, diamond400), diamond400))
    top = th[R > 0.99 * R.max()]
    width = top.max() - top.min()
    assert width == pytest.approx(darwin_width_urad(diamond400), rel=0.25)
    centre = 0.5 * (top.max() + top.min())
    assert centre == pytest.approx(-6.764, abs=0.2)
    assert centre < 0                      # refraction pushes it negative


def test_riccati_approaches_the_semi_infinite_limit(diamond400):
    """Thick enough, the finite crystal reflects like a bulk one."""
    s = deviation_from_urad(np.array([-6.7644]), diamond400)
    bulk = np.abs(reflectivity_semi_infinite(s, diamond400))[0]
    thin = np.abs(reflectivity_riccati(s, 5e-6, diamond400))[0]
    thick = np.abs(reflectivity_riccati(s, 60e-6, diamond400))[0]
    assert abs(thick - bulk) < abs(thin - bulk)
    assert thick == pytest.approx(bulk, abs=0.02)


def test_twenty_microns_is_not_semi_infinite(diamond400):
    """Which is why the Bragg scene uses 40 um."""
    s = deviation_from_urad(np.array([-6.7644]), diamond400)
    bulk = np.abs(reflectivity_semi_infinite(s, diamond400))[0]
    at20 = np.abs(reflectivity_riccati(s, 20e-6, diamond400))[0]
    assert abs(at20 - bulk) > 1e-3


def test_grid_solver_matches_the_riccati_reference(diamond400, g):
    """The perfect-crystal case, solved on the grid and in closed form."""
    cfg = BraggConfig(diamond400, t_crystal=8e-6, dx=0.1e-6, half_x=4e-6)
    for urad in (-6.5, -3.0, 1.0):
        sd = float(deviation_from_urad(urad, diamond400))
        ew, info = solve_bragg([], sd, cfg, diamond400, g, xp=np,
                               n_iter=600, tol=1e-13)
        mine = abs(complex(ew[cfg.Nx // 2]))
        ref = abs(reflectivity_riccati(np.array([sd]), cfg.t_crystal,
                                       diamond400)[0])
        assert info["converged"]
        assert mine == pytest.approx(ref, rel=2e-3)


def test_the_crystal_cannot_amplify(diamond400, g):
    cfg = BraggConfig(diamond400, t_crystal=8e-6, dx=0.1e-6, half_x=4e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    ew, _ = solve_bragg([], sd, cfg, diamond400, g, xp=np, n_iter=600,
                        tol=1e-13)
    assert np.abs(ew).max() <= 1.0 + 1e-9


def test_shooting_agrees_with_the_fixed_point_where_both_work(diamond400, g):
    """Thin enough that the fixed point still converges, so the two
    genuinely independent methods can be compared.

    They agree to about 1e-3, and no tighter, however long the fixed
    point is left to run. That is not a convergence failure: the round
    trip and the down-march are different discretisations of the same
    equation, so they have different truncation error on the same grid.
    Agreeing at all is the point.
    """
    cfg = BraggConfig(diamond400, t_crystal=4e-6, dx=0.1e-6, half_x=3e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    a, it = solve_bragg([], sd, cfg, diamond400, g, xp=np, n_iter=800,
                        tol=1e-13)
    b, info = solve_bragg_stable([], sd, cfg, diamond400, g, xp=np)
    assert it["converged"]
    assert info["residual"] < 1e-8
    assert np.abs(a - b).max() < 2e-3 * np.abs(a).max()


def test_batched_surface_solve_matches_one_plane_at_a_time(diamond400, g):
    cfg = BraggConfig(diamond400, t_crystal=6e-6, dx=0.1e-6, half_x=3e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    seg = threading_segment("Edge", +1, diamond400.b_mag, float(g[2]))
    ys = np.array([-0.2e-6, 0.0, 0.3e-6])
    batched, info = solve_surface_batch(seg, sd, cfg, ys, diamond400, g,
                                        xp=np)
    assert info["residual"] < 1e-7
    for j, y in enumerate(ys):
        one, _ = solve_bragg_stable(seg, sd, cfg, diamond400, g, xp=np,
                                    y_plane=float(y))
        assert np.abs(batched[j] - one).max() < 1e-7 * np.abs(one).max()


def test_a_zero_burgers_vector_is_a_perfect_crystal(diamond400, g):
    cfg = BraggConfig(diamond400, t_crystal=6e-6, dx=0.1e-6, half_x=3e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    seg = threading_segment("Edge", +1, diamond400.b_mag, float(g[2]))
    dead = [(seg[0][0], seg[0][1], np.zeros(3))]
    with_zero, _ = solve_bragg_stable(dead, sd, cfg, diamond400, g, xp=np)
    pristine, _ = solve_bragg_stable([], sd, cfg, diamond400, g, xp=np)
    assert np.abs(with_zero - pristine).max() < 1e-12


def test_a_big_chunk_gives_the_same_answer_as_a_small_one(diamond400, g):
    """Batching is an optimisation and must not change the answer.

    It nearly did. One Krylov space serves every plane in a chunk, so a
    large chunk starves it, and both SciPy and CuPy then report success
    with a residual of a few percent. In the paper's Bragg scene that
    produced visible stripes at every chunk boundary.

    This test cannot reproduce that on its own: the failure needs the
    production-size field, hundreds of planes of strongly strained
    crystal, which is far too slow for a unit test. A grid this small
    converges whatever you do to it. What is checked here is the
    invariance itself, and the regression gate covers the size where it
    broke.
    """
    cfg = BraggConfig(diamond400, t_crystal=8e-6, dx=0.15e-6, half_x=3e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    seg = threading_segment("Edge", +1, diamond400.b_mag, float(g[2]),
                            y0=-1.5e-6)
    ys = (np.arange(16) - 8) * cfg.dx
    big, info_b = solve_surface_batch(seg, sd, cfg, ys, diamond400, g,
                                      xp=np, chunk=16)
    one, info_1 = solve_surface_batch(seg, sd, cfg, ys, diamond400, g,
                                      xp=np, chunk=1)
    assert info_b["converged"] and info_1["converged"]
    assert info_b["residual"] < 1e-7 and info_1["residual"] < 1e-7
    rel = np.abs(big - one).max() / np.abs(one).max()
    assert rel < 1e-6


def test_the_solver_reports_how_hard_it_had_to_work(diamond400, g):
    """`converged` is the field to check, not the return code."""
    cfg = BraggConfig(diamond400, t_crystal=6e-6, dx=0.2e-6, half_x=2e-6)
    sd = float(deviation_from_urad(-6.5, diamond400))
    _, info = solve_surface_batch([], sd, cfg, np.zeros(4), diamond400, g,
                                  xp=np, chunk=4)
    for key in ("residual", "iterations", "retries", "splits",
                "unconverged_chunks", "converged"):
        assert key in info
    assert info["converged"] is True
    assert info["unconverged_chunks"] == 0
