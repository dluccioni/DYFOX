"""The crystal numbers, derived from first principles and checked.

Every expected value here is worked out independently of the code, from
published constants, so a change in the engine that moves any of them
shows up as a failing test rather than as a differently wrong figure.
"""

import numpy as np
import pytest

from crystal import DIAMOND, CrystalParams, Material, lab_frame
from units import HC_EV_M, R_E

A_DIAMOND = 3.567e-10


def test_wavelength_matches_hc_over_e(diamond400):
    assert diamond400.lam == pytest.approx(HC_EV_M / 17.0e3, rel=1e-15)
    assert diamond400.lam * 1e10 == pytest.approx(0.72932, abs=5e-6)


def test_bragg_angle_from_braggs_law(diamond400):
    d = A_DIAMOND / 4.0
    expect = np.arcsin(diamond400.lam / (2 * d))
    assert diamond400.theta_B == pytest.approx(expect, rel=1e-15)
    assert np.degrees(diamond400.theta_B) == pytest.approx(24.137, abs=1e-3)


def test_structure_factor_of_the_diamond_basis(diamond400):
    """S_g = 8 for (400): every atom of the basis scatters in phase."""
    S_g = np.sum(np.exp(2j * np.pi * DIAMOND.basis @ np.array([4, 0, 0])))
    assert abs(S_g) == pytest.approx(8.0, abs=1e-12)
    assert abs(diamond400.F_g) == pytest.approx(8 * abs(1.589 + 0.003 - 0.002j),
                                                rel=1e-12)
    assert abs(diamond400.F_g) == pytest.approx(12.736, abs=1e-3)


def test_220_is_allowed_and_400_forbidden_partner():
    """(220) is allowed; (200) is a forbidden diamond reflection."""
    for hkl, expect in (((2, 2, 0), 8.0), ((4, 0, 0), 8.0), ((2, 0, 0), 0.0)):
        S = np.sum(np.exp(2j * np.pi * DIAMOND.basis @ np.array(hkl)))
        assert abs(S) == pytest.approx(expect, abs=1e-12)


def test_absorption_attenuates(diamond400):
    """The solver evolves exp(sigma s), so Re(sigma_0) must be negative.

    With the opposite sign convention the crystal amplifies, by about
    1.8 % over the paper's 273 um path, which is easy to miss in a plot.
    """
    assert diamond400.sig0.real < 0
    assert diamond400.sig0.real == pytest.approx(-72.45, abs=0.1)
    path = 273.4e-6 / diamond400.cos_tB
    assert np.exp(2 * diamond400.sig0.real * path) < 1.0


def test_extinction_length(diamond400):
    assert diamond400.xi_g == pytest.approx(np.pi / abs(diamond400.sig_h),
                                            rel=1e-15)
    assert diamond400.xi_g * 1e6 == pytest.approx(54.47, abs=0.02)


def test_sigma_from_classical_electron_radius(diamond400):
    """sigma = -i r_e lambda F / V, rebuilt from the definition."""
    V = A_DIAMOND ** 3
    F0 = (6.0 + 0.003 - 0.002j) * 8
    assert diamond400.sig0 == pytest.approx(
        -1j * R_E * diamond400.lam * F0 / V, rel=1e-14)


def test_sigma_hbar_conjugates_only_the_geometric_sum(diamond400):
    """F_{-g} = f conj(S_g), not conj(F_g).

    Conjugating the atomic factor too would make sigma_h sigma_hbar
    exactly real, which quietly turns an approximate symmetry of the
    open-aperture centre of mass into an exact one.
    """
    assert diamond400.sig_h != np.conj(diamond400.sig_hbar)
    prod = diamond400.sig_h * diamond400.sig_hbar
    assert abs(prod.imag / prod.real) > 1e-6


def test_thickness_in_extinction_lengths(diamond400):
    """Snapping 270 um to the nearest Pendelloesung maximum gives 273.404 um.

    Half a period is xi_g cos(theta_B), and the snap keeps the plate on
    a maximum (n = 5, so 5.5 half-periods). The result is 5.019
    extinction lengths and follows from the crystal numbers alone.
    """
    half_pend = diamond400.xi_g * diamond400.cos_tB
    t_crystal = (round(270e-6 / half_pend - 0.5) + 0.5) * half_pend
    assert t_crystal * 1e6 == pytest.approx(273.404, abs=0.002)
    assert t_crystal / diamond400.xi_g == pytest.approx(5.019, abs=0.005)


def test_energy_override_keeps_the_reflection(diamond400):
    other = diamond400.at_energy(12.5, f_prime=0.003, f_dprime=0.002)
    assert other.hkl == diamond400.hkl
    assert other.lam > diamond400.lam        # lower energy, longer wave
    assert other.theta_B > diamond400.theta_B


def test_g_vector_has_no_two_pi(diamond400):
    """|g| = 1/d, the crystallographic convention this code uses."""
    assert np.linalg.norm(diamond400.g_vec) == pytest.approx(
        1.0 / diamond400.d_hkl, rel=1e-14)


def test_lab_frame_is_right_handed_and_puts_g_in_the_surface():
    U = lab_frame((0, 0, 1), (1, 1, 0))
    assert np.linalg.det(U) == pytest.approx(1.0, abs=1e-12)
    assert U @ U.T == pytest.approx(np.eye(3), abs=1e-12)
    g = U @ (np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1, 1, 0]))
    assert abs(g[2]) < 1e-12                 # g has no component along z


def test_lab_frame_refuses_a_bragg_geometry():
    with pytest.raises(ValueError, match="Bragg"):
        lab_frame((0, 0, 1), (0, 0, 4))


def test_with_frame_turns_g_and_leaves_the_scalars(diamond400):
    """Reorienting the crystal moves g and changes nothing else.

    The frame here has x along [110], so the (400) vector no longer
    lies along x and picks up a y component. A frame whose x is still
    parallel to g would leave g_vec alone, correctly.
    """
    U = lab_frame((0, 0, 1), (1, 1, 0))
    turned = diamond400.with_frame(U)
    assert turned.lam == diamond400.lam
    assert turned.sig_h == diamond400.sig_h
    assert turned.xi_g == diamond400.xi_g
    assert not np.array_equal(turned.g_vec, diamond400.g_vec)
    assert turned.g_vec[1] != 0.0
    assert np.linalg.norm(turned.g_vec) == pytest.approx(
        np.linalg.norm(diamond400.g_vec), rel=1e-14)


def test_with_frame_leaves_g_alone_when_x_stays_parallel_to_it(diamond400):
    """A (011) plate still has (400) along its surface x, so g does not move."""
    turned = diamond400.with_frame(lab_frame((0, 1, 1), (4, 0, 0)))
    assert np.array_equal(turned.g_vec, diamond400.g_vec)


def test_missing_form_factor_is_a_clear_error():
    with pytest.raises(KeyError, match="no f0"):
        CrystalParams.from_reflection(DIAMOND, (1, 1, 1), 17.0)


def test_a_user_can_supply_another_material():
    """Silicon, entirely from the caller's own numbers."""
    silicon = Material(name="silicon", a_lat=5.431e-10,
                       basis=DIAMOND.basis, Z=14.0, nu=0.22,
                       f0={(4, 0, 0): 8.66})
    si = CrystalParams.from_reflection(silicon, (4, 0, 0), 17.0)
    assert si.a_lat == 5.431e-10
    assert si.nu == 0.22
    assert si.b_mag == pytest.approx(5.431e-10 * np.sqrt(2) / 2)
    # Longer lattice parameter, so a smaller Bragg angle at fixed energy.
    assert si.theta_B < 24.0 * np.pi / 180


def test_key_identifies_the_crystal(diamond400):
    k = diamond400.key()
    assert k["material"] == "diamond" and k["hkl"] == [4, 0, 0]
    assert k["E_keV"] == 17.0
    assert diamond400.at_energy(12.5).key() != k


def test_item_access_mirrors_attributes(diamond400):
    assert diamond400["lam"] == diamond400.lam
