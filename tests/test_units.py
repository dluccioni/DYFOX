"""The deviation-parameter conversions, and why there are three of them."""

import numpy as np
import pytest

import units

# diamond (400) at 17 keV, derived here rather than quoted, so the test
# stays true if the constants move
A_DIAMOND = 3.567e-10
HC_EV_M = 12398.419e-10
LAM = HC_EV_M / (17.0e3)
D_400 = A_DIAMOND / 4.0
THETA_B = np.arcsin(LAM / (2 * D_400))
XTAL = {"sin_2tB": np.sin(2 * THETA_B), "lam": LAM}
XI_G = 54.472e-6                      # extinction length, from the engine


def test_round_trip():
    s = np.linspace(-10, 10, 81) / 54.47e-6
    back = units.deviation_from_urad(units.urad_from_deviation(s, XTAL),
                                     XTAL)
    assert np.allclose(back, s, rtol=1e-12)


def test_rad_and_urad_agree_to_rounding():
    phi_urad = np.linspace(-150.0, 150.0, 121)
    a = units.deviation_from_rad(phi_urad * 1e-6, XTAL)
    b = units.deviation_from_urad(phi_urad, XTAL)
    assert np.allclose(a, b, rtol=1e-12)


def test_folding_the_constant_changes_values():
    """`phi * sin_2tB / lam` is not `phi * (sin_2tB / lam)`.

    This is the reason the conversion is a function with a fixed
    expression rather than a multiplication by a precomputed scale.
    """
    phi = np.linspace(-150e-6, 150e-6, 121)
    exact = units.deviation_from_rad(phi, XTAL)
    folded = phi * (XTAL["sin_2tB"] / XTAL["lam"])
    assert np.allclose(exact, folded, rtol=1e-12)
    assert np.count_nonzero(exact != folded) > 0


def test_accepts_a_params_object():
    """The conversions work with a mapping or an object."""
    class Xtal:
        sin_2tB = XTAL["sin_2tB"]
        lam = XTAL["lam"]

    assert units.deviation_from_rad(1e-5, Xtal) == \
        units.deviation_from_rad(1e-5, XTAL)


def test_weak_beam_operating_point():
    """117.5 urad is the operating point the paper selects."""
    s = units.deviation_from_rad(117.5e-6, XTAL)
    assert units.urad_from_deviation(s, XTAL) == pytest.approx(117.5)
    assert s * XI_G == pytest.approx(65.5, abs=0.2)
