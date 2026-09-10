"""What the analysis functions must get right about a known field.

Every check here is on a synthetic field whose answer is known in
advance, so a failure says the estimator is wrong rather than that the
physics moved.
"""

import numpy as np
import pytest

from analysis.features import features, locate, loo_accuracy, rich_features
from analysis.metrics import crop_slices, radial_profile
from analysis.oam import oam_spectrum, oam_spectrum_fft
from analysis.rocking import planewave_rocking
from analysis.topology import vortex_census, winding_number

DX = 0.25e-6


def vortex_field(nx=256, ny=256, m=2, dx=DX, sigma_um=6.0, centre=(0.0, 0.0)):
    """exp(i m phi) about `centre`, with a Gaussian envelope.

    Stored (y, x), as every field in this codebase is, and phi measured
    from the +x axis.
    """
    x = (np.arange(nx) - nx / 2) * dx - centre[0]
    y = (np.arange(ny) - ny / 2) * dx - centre[1]
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    return (np.exp(1j * m * np.arctan2(Y, X))
            * np.exp(-r ** 2 / (2 * (sigma_um * 1e-6) ** 2)))


# ---------------------------------------------------------------- topology
@pytest.mark.parametrize("m", [-3, -2, -1, 1, 2, 3])
def test_winding_number_reads_the_charge(m):
    field = vortex_field(m=m)
    assert winding_number(field, DX, r_m=3e-6) == pytest.approx(m, abs=0.01)


def test_winding_number_is_radius_independent():
    field = vortex_field(m=2)
    got = [winding_number(field, DX, r_m=r) for r in (1e-6, 3e-6, 6e-6)]
    assert all(g == pytest.approx(2, abs=0.01) for g in got)


def test_vortex_census_finds_one_vortex_of_the_right_charge():
    field = vortex_field(m=2)
    vs = vortex_census(field, DX)
    near = vs[np.hypot(vs[:, 0], vs[:, 1]) < 2.0]
    assert near[:, 2].sum() == pytest.approx(2)


def test_vortex_census_locates_an_off_centre_vortex():
    field = vortex_field(m=1, centre=(5e-6, -3e-6))
    vs = vortex_census(field, DX)
    strongest = vs[np.argmax(np.abs(vs[:, 2]))]
    assert strongest[0] == pytest.approx(5.0, abs=0.5)
    assert strongest[1] == pytest.approx(-3.0, abs=0.5)


# --------------------------------------------------------------------- OAM
@pytest.mark.parametrize("m", [-2, 1, 2, 3])
def test_oam_spectrum_peaks_at_the_imposed_charge(m):
    nx = ny = 256
    field = vortex_field(nx=nx, ny=ny, m=m)
    P = oam_spectrum(field, DX, nx, ny, r_max_um=8.0)
    channels = list(range(-5, 6))
    assert channels[int(np.argmax(P))] == m
    assert P[channels.index(m)] > 0.9


def test_oam_spectrum_normalised():
    P = oam_spectrum(vortex_field(m=2), DX, 256, 256)
    assert P.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("m", [-2, 2])
def test_oam_spectrum_fft_agrees_on_the_dominant_channel(m):
    field = vortex_field(m=m, sigma_um=8.0)
    ms, P = oam_spectrum_fft(field, DX, r_max_um=6.0)
    assert ms[int(np.argmax(P))] == m


def test_oam_axis_order_is_not_mirrored():
    """A mirrored sampling maps m to -m; this is the guard against it."""
    P = oam_spectrum(vortex_field(m=2), DX, 256, 256)
    channels = list(range(-5, 6))
    assert P[channels.index(2)] > 10 * P[channels.index(-2)]


# ---------------------------------------------------------------- features
def test_locate_finds_a_displaced_blob():
    ny = nx = 200
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.exp(-((yy - 130) ** 2 + (xx - 70) ** 2) / (2 * 8.0 ** 2))
    cy, cx = locate(img)
    assert (cy, cx) == pytest.approx((130, 70), abs=2)


def test_features_are_translation_invariant():
    """The moments are taken about the located centre, so moving the
    pattern must not change them."""
    ny = nx = 240
    yy, xx = np.mgrid[0:ny, 0:nx]

    def blob(cy, cx):
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return np.exp(-r2 / (2 * 6.0 ** 2)) + 0.05 * np.exp(-r2 / 400.0)

    a = features(blob(120, 120), DX)
    b = features(blob(132, 108), DX)
    assert np.allclose(a, b, atol=0.02)


def test_features_separate_a_bright_core_from_a_dark_one():
    ny = nx = 240
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - 120, xx - 120)
    bright = np.exp(-r ** 2 / (2 * 5.0 ** 2)) + 0.02
    dark = (1 - np.exp(-r ** 2 / (2 * 5.0 ** 2))) * np.exp(
        -r ** 2 / (2 * 30.0 ** 2)) + 0.02
    assert features(bright, DX)[0] > features(dark, DX)[0]


def test_rich_features_length_and_finiteness():
    ny = nx = 240
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.exp(-((yy - 120) ** 2 + (xx - 120) ** 2) / (2 * 7.0 ** 2)) + 0.01
    f = rich_features(img, 0.25)
    assert f.shape == (11,)
    assert np.all(np.isfinite(f))


def test_loo_accuracy_is_perfect_on_separated_classes(rng):
    F = np.stack([rng.normal(loc=c * 8.0, scale=0.2, size=(24, 5))
                  for c in range(4)])
    assert loo_accuracy(F, lambda c: c) == 1.0


def test_loo_accuracy_is_chance_on_identical_classes(rng):
    F = rng.normal(size=(4, 24, 5))
    assert loo_accuracy(F, lambda c: c) < 0.5


# ----------------------------------------------------------------- metrics
def test_crop_slices_are_square_and_centred():
    sy, sx, ext = crop_slices(1056, 544, DX, half_um=25.0)
    assert sy.stop - sy.start == sx.stop - sx.start == 200
    assert (sy.start + sy.stop) // 2 == 544 // 2
    assert ext == [-25.0, 25.0, -25.0, 25.0]


def test_radial_profile_of_a_flat_image_is_flat():
    r_um, prof = radial_profile(np.ones((200, 200)), DX)
    assert np.allclose(prof, 1.0)
    assert r_um[1] == pytest.approx(DX * 1e6)


def test_radial_profile_recovers_a_gaussian_width():
    ny = nx = 256
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma_px = 12.0
    img = np.exp(-((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2)
                 / (2 * sigma_px ** 2))
    r_um, prof = radial_profile(img, DX)
    half = np.argmin(np.abs(prof - 0.5))
    assert half * 1.0 == pytest.approx(sigma_px * np.sqrt(2 * np.log(2)),
                                       rel=0.05)


# ----------------------------------------------------------------- rocking
def test_planewave_rocking_peaks_at_zero_deviation():
    xtal = {"sig_h": -1.2e4 - 3.4e2j, "sig_hbar": -1.2e4 + 3.4e2j,
            "sig0": -72.45 - 5.9e4j, "cos_tB": 0.91264}
    s = np.linspace(-4, 4, 401) / 54.47e-6
    rc = planewave_rocking(s, xtal, 45e-6)
    assert np.argmax(rc) == pytest.approx(len(s) // 2, abs=1)


def test_planewave_rocking_absorption_only_attenuates():
    xtal = {"sig_h": -1.2e4 - 3.4e2j, "sig_hbar": -1.2e4 + 3.4e2j,
            "sig0": -72.45 - 5.9e4j, "cos_tB": 0.91264}
    s = np.linspace(-4, 4, 201) / 54.47e-6
    with_abs = planewave_rocking(s, xtal, 273.4e-6)
    without = planewave_rocking(s, xtal, 273.4e-6, absorption=False)
    assert np.all(with_abs <= without + 1e-30)
    assert with_abs.max() < without.max()
