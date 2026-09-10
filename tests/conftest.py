"""Test configuration: put the repository root on the path.

There is nothing to install, so the tests reach the engine the same way
the scripts do.  Tests that need a GPU are marked `gpu` and skipped when
CuPy or a device is missing, so the CPU suite runs anywhere.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CuPy and a CUDA device")
    config.addinivalue_line("markers", "slow: more than about ten seconds")
    config.addinivalue_line("markers",
                            "paper: reproduces a published figure")


@pytest.fixture(scope="session")
def gpu():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:                       # driver missing, etc.
        pytest.skip(f"no usable CUDA device: {exc}")
    return cp


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(20260909)


@pytest.fixture(scope="session")
def diamond400():
    """The paper's crystal: diamond (400) at 17 keV."""
    from crystal import DIAMOND, CrystalParams
    return CrystalParams.from_reflection(DIAMOND, (4, 0, 0), 17.0,
                                         f_prime=0.003, f_dprime=0.002,
                                         label="Diamond(400)")


@pytest.fixture(scope="session")
def small_grid(diamond400):
    """A grid small enough to solve in a second, still exactly advecting."""
    from laue import LaueGrid
    return LaueGrid(diamond400, t_crystal=40e-6, dx=0.25e-6,
                    beam_width=8e-6, beam_thickness=1e-6)


@pytest.fixture(params=["fp64", "fp32"])
def precision(request):
    """Both precisions, for tests that must hold in either."""
    from backend import FP32, FP64
    return FP64 if request.param == "fp64" else FP32


@pytest.fixture
def tmp_cache(tmp_path):
    from laue import WaveCache
    return WaveCache(str(tmp_path / "waves"))


@pytest.fixture(autouse=True)
def _free_gpu_pool():
    """Hand the device memory back between tests, so a slow one cannot
    starve the next."""
    yield
    from backend import free_gpu
    free_gpu()
