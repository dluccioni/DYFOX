"""The sampling grid and the imaging stage, on the CPU.

The grid tests are about exact advection, which is the single choice
that most affects what the solver produces. The optics tests are about
the pupil conventions, which are easy to get subtly wrong and hard to
see in a picture.
"""

import numpy as np
import pytest

from crystal import DIAMOND, CrystalParams
from laue import LaueGrid
from optics import (Optics, apply_pupils, image_from_spectrum,
                    object_spectrum)


# --------------------------------------------------------------- the grid
def test_depth_step_gives_exactly_one_pixel_per_step(diamond400):
    grid = LaueGrid(diamond400, t_crystal=273.4e-6, dx=0.25e-6,
                    beam_width=120e-6)
    _, _, dz, _, shift_px = grid.geometry()
    assert grid.exact
    assert shift_px == 1.0                       # exactly, not approximately
    assert dz == pytest.approx(grid.dx / diamond400.tan_tB, rel=1e-9)


def test_the_paper_grid_is_the_expected_size(diamond400):
    grid = LaueGrid(diamond400, t_crystal=273.4037251665087e-6, dx=0.25e-6,
                    beam_width=120e-6)
    assert grid.Nz == 491
    assert grid.shape == (544, 1056)


def test_refining_dx_refines_dz_with_it(diamond400):
    coarse = LaueGrid(diamond400, t_crystal=40e-6, dx=0.25e-6,
                      beam_width=8e-6)
    fine = coarse.replace(dx=0.125e-6)
    assert fine.exact
    assert fine.geometry()[4] == 1.0
    assert fine.Nz > coarse.Nz
    assert fine.geometry()[2] == pytest.approx(coarse.geometry()[2] / 2,
                                               rel=1e-3)


def test_forcing_Nz_breaks_exact_advection(diamond400):
    grid = LaueGrid(diamond400, t_crystal=40e-6, dx=0.25e-6,
                    beam_width=8e-6, Nz=301)
    assert not grid.exact
    assert grid.geometry()[4] != 1.0


def test_thickness_snaps_to_a_whole_number_of_steps(diamond400):
    """The requested thickness is met to within one depth step."""
    grid = LaueGrid(diamond400, t_crystal=273.4e-6, dx=0.25e-6,
                    beam_width=120e-6)
    dz = grid.geometry()[2]
    assert abs(grid.t_crystal - 273.4e-6) < dz


def test_grid_widens_with_the_borrmann_fan(diamond400):
    thin = LaueGrid(diamond400, t_crystal=40e-6, dx=0.25e-6, beam_width=8e-6)
    thick = LaueGrid(diamond400, t_crystal=273e-6, dx=0.25e-6,
                     beam_width=8e-6)
    assert thick.shape[1] > thin.shape[1]        # more x, same y
    assert thick.shape[0] == thin.shape[0]


def test_grid_dimensions_are_multiples_of_32(diamond400):
    for t in (20e-6, 100e-6, 273e-6):
        Ny, Nx = LaueGrid(diamond400, t_crystal=t, dx=0.25e-6,
                          beam_width=53e-6).shape
        assert Nx % 32 == 0 and Ny % 32 == 0


def test_grid_key_separates_what_matters(diamond400):
    a = LaueGrid(diamond400, t_crystal=40e-6, dx=0.25e-6, beam_width=8e-6)
    assert a.key() != a.replace(beam_thickness=2e-6).key()
    assert a.key() != a.replace(grid_pad=14).key()
    assert a.key() == a.replace().key()


# ------------------------------------------------------------- the optics
@pytest.fixture
def lens(diamond400):
    return Optics(1.0e-4, 0.274, diamond400, xp=np)


def test_open_pupil_is_a_disc_of_the_right_radius(lens, diamond400):
    p = lens.pupil(256, 256, 0.25e-6, ell=0)
    f = np.fft.fftshift(np.fft.fftfreq(256, 0.25e-6))
    FX, FY = np.meshgrid(f, f)
    inside = np.hypot(FX, FY) <= 1.0e-4 / diamond400.lam
    assert np.array_equal(p != 0, inside)
    assert set(np.unique(p.real)) <= {0.0, 1.0}


def test_spiral_pupil_winds_by_ell(lens):
    p = lens.spiral(256, 256, 0.25e-6, ell=2)
    n = 256
    r = 6                                      # bins, inside the NA disc
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ys = np.round(n // 2 + r * np.sin(th)).astype(int)
    xs = np.round(n // 2 + r * np.cos(th)).astype(int)
    ph = np.unwrap(np.angle(p[ys, xs]))
    assert (ph[-1] - ph[0]) / (2 * np.pi) == pytest.approx(2.0, abs=0.05)


def test_centred_spiral_zeroes_the_dc_bin(lens):
    """The bin-average of exp(i l phi) over the singular bin is zero.

    Leaving it alone would let the object's mean field through the
    plate undiffracted, which is exactly what the spiral is there to
    stop.
    """
    p = lens.spiral(256, 128, 0.25e-6, ell=2)
    assert p[64, 128] == 0


def test_displaced_spiral_keeps_its_dc_bin(lens):
    """Off centre, the origin bin is an ordinary one."""
    p = lens.spiral(256, 128, 0.25e-6, ell=2, offset_bfp=(2e-6, 0.0))
    assert p[64, 128] != 0


def test_open_pupil_never_zeroes_the_dc_bin(lens):
    assert lens.pupil(256, 128, 0.25e-6, ell=0)[64, 128] == 1


def test_chromatic_pupil_walks_the_aperture(lens, diamond400):
    """A relative energy offset slides the whole disc, and can vignette."""
    centred = lens.chromatic(512, 512, 0.25e-6, 0, 0.0)
    walked = lens.chromatic(512, 512, 0.25e-6, 0, 3e-4)
    assert walked.sum() != centred.sum() or not np.array_equal(walked, centred)
    f = np.fft.fftshift(np.fft.fftfreq(512, 0.25e-6))
    dfx = diamond400.sin_2tB * 3e-4 / diamond400.lam
    cx = np.argmin(np.abs(f - dfx))
    assert walked[512 // 2, cx] != 0


def test_plate_quantisation_takes_discrete_phases(lens):
    p = lens.pupil(256, 256, 0.25e-6, ell=2, levels=8)
    inside = p != 0
    ph = np.angle(p[inside]) % (2 * np.pi)
    assert len(np.unique(np.round(ph, 9))) <= 8


def test_pupil_axis_order_on_a_non_square_grid(lens):
    """Fields are stored (y, x); a pupil must agree, or m flips sign."""
    p = lens.spiral(256, 128, 0.25e-6, ell=2)
    assert p.shape == (128, 256)


def test_pupils_are_cached_but_not_confused(lens):
    a = lens.spiral(128, 128, 0.25e-6, 2)
    assert lens.spiral(128, 128, 0.25e-6, 2) is a
    assert lens.spiral(128, 128, 0.25e-6, -2) is not a
    assert lens.spiral(128, 128, 0.125e-6, 2) is not a
    lens.clear_cache()
    assert lens.spiral(128, 128, 0.25e-6, 2) is not a


def test_a_wider_aperture_resolves_more(diamond400):
    narrow = Optics(1e-4, 0.274, diamond400, xp=np)
    wide = Optics(4e-4, 0.274, diamond400, xp=np)
    assert wide.resolution() < narrow.resolution()
    assert wide.aperture_f > narrow.aperture_f


# ------------------------------------------------------------- the imaging
def test_round_trip_through_an_all_pass_pupil_is_the_identity(lens, rng):
    field = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    ones = np.ones((64, 64), complex)
    out = image_from_spectrum(object_spectrum(field), ones)
    assert out == pytest.approx(np.abs(field) ** 2, rel=1e-10, abs=1e-12)


def test_parseval_holds_through_the_transform(lens, rng):
    field = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    spec = object_spectrum(field)
    assert (np.abs(spec) ** 2).sum() == pytest.approx(
        (np.abs(field) ** 2).sum() * field.size, rel=1e-10)


def test_an_aperture_can_only_remove_energy(lens, rng):
    field = (rng.normal(size=(128, 128))
             + 1j * rng.normal(size=(128, 128))).astype(complex)
    e_in = (np.abs(field) ** 2).sum()
    e_out = image_from_spectrum(object_spectrum(field),
                                lens.pupil(128, 128, 0.25e-6, 0)).sum()
    assert e_out <= e_in * (1 + 1e-9)


def test_apply_pupils_agrees_with_one_at_a_time(lens, rng):
    field = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64)))
    pups = [lens.spiral(64, 64, 0.25e-6, e) for e in (0, 2, -2)]
    many = apply_pupils(field, pups)
    for i, p in enumerate(pups):
        one = image_from_spectrum(object_spectrum(field), p)
        assert many[i] == pytest.approx(one, rel=1e-12, abs=1e-15)


def test_a_matched_plate_fills_in_a_vortex(diamond400):
    """The point of the whole method, on a synthetic vortex.

    A field carrying exp(i m phi) imaged through a plate of charge
    l = -m has its winding cancelled, so the core fills in. The same
    plate on the opposite charge doubles the winding and the core
    stays dark.
    """
    lens = Optics(3e-3, 0.274, diamond400, xp=np)
    n, dx = 128, 0.25e-6
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (2 * (3e-6) ** 2))
    for m in (+2, -2):
        field = env * np.exp(1j * m * np.arctan2(Y, X))
        matched = image_from_spectrum(
            object_spectrum(field), lens.spiral(n, n, dx, -m))
        opposed = image_from_spectrum(
            object_spectrum(field), lens.spiral(n, n, dx, +m))
        c = slice(n // 2 - 2, n // 2 + 2)
        assert matched[c, c].max() > 20 * opposed[c, c].max()
