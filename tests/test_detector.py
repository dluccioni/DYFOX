"""Recording an image on a pixel grid must conserve what was there."""

import numpy as np
import pytest

import detector
import illumination

DX_UM = 0.25


def test_binning_conserves_the_total(rng):
    img = rng.random((414, 414))
    assert detector.bin_pixels(img, 3).sum() == pytest.approx(img.sum(),
                                                              rel=1e-12)


def test_binning_by_one_is_the_identity(rng):
    img = rng.random((100, 100))
    assert detector.bin_pixels(img, 1) is img


def test_binning_drops_the_partial_block(rng):
    img = rng.random((100, 100))
    out = detector.bin_pixels(img, 3)
    assert out.shape == (33, 33)
    assert out.sum() == pytest.approx(img[:99, :99].sum(), rel=1e-12)


def test_area_average_matches_binning_on_a_commensurate_grid(rng):
    """A pixel three times the simulation pitch is exactly a 3x3 sum."""
    img = rng.random((120, 120))
    binned = detector.bin_pixels(img, 3)
    area = detector.area_average(img, DX_UM, 3 * DX_UM, 3 * DX_UM)
    assert area.shape == binned.shape
    assert np.allclose(area, binned, rtol=1e-9)


def test_area_average_conserves_a_uniform_image():
    img = np.ones((200, 200))
    out = detector.area_average(img, DX_UM, 0.75, 1.87)
    interior = out[1:-1, 1:-1]
    assert np.allclose(interior, interior.flat[0], rtol=1e-9)
    assert interior.flat[0] == pytest.approx(0.75 * 1.87 / DX_UM ** 2,
                                             rel=1e-6)


def test_area_average_handles_anisotropic_pixels(rng):
    img = rng.random((200, 200))
    out = detector.area_average(img, DX_UM, 0.75, 1.87)
    assert out.shape[0] > out.shape[1]        # coarser along x
    assert out.sum() < img.sum() * 1.001


def test_object_pixel_size():
    assert detector.object_pixel_size(13e-6, 17.0) == pytest.approx(0.7647,
                                                                    rel=1e-3)


def test_foreshortening_is_worse_in_bragg():
    theta_B = np.deg2rad(24.1374)
    pitch = detector.object_pixel_size(13e-6, 17.0)
    laue = detector.foreshortened_pixel(pitch, theta_B, "Laue")
    bragg = detector.foreshortened_pixel(pitch, theta_B, "Bragg")
    assert laue == pytest.approx(pitch / np.cos(theta_B))
    assert bragg == pytest.approx(pitch / np.sin(theta_B))
    assert bragg > 2 * laue


def test_unknown_geometry_is_rejected():
    with pytest.raises(ValueError):
        detector.foreshortened_pixel(0.75, 0.42, "Sideways")


# ------------------------------------------------------------ illumination
def test_gauss_hermite_weights_sum_to_one():
    _, w = illumination.gauss_hermite_offsets(1.0, 8)
    assert w.sum() == pytest.approx(1.0)


def test_gauss_hermite_reproduces_the_variance():
    sigma = 3.5e-6
    offs, w = illumination.gauss_hermite_offsets(sigma, 8)
    assert (w * offs[:, 0]).sum() == pytest.approx(0.0, abs=1e-18)
    assert (w * offs[:, 0] ** 2).sum() == pytest.approx(sigma ** 2, rel=1e-9)
    assert (w * offs[:, 1] ** 2).sum() == pytest.approx(sigma ** 2, rel=1e-9)


def test_condenser_angles_straddle_zero_and_reach_the_wings():
    a, w = illumination.condenser_angles(0.90, reach=75.0)
    assert 0.0 in a
    assert a.min() == pytest.approx(-75.0)
    assert a.max() == pytest.approx(75.0)
    assert np.all(np.diff(a) > 0)
    assert len(w) == len(a)


def test_condenser_sampling_is_dense_across_the_darwin_width():
    a, _ = illumination.condenser_angles(0.90)
    core = a[np.abs(a) <= 3.6]
    assert np.max(np.diff(core)) <= 0.90 + 1e-9
