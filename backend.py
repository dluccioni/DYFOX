"""Array-library helpers, and the two floating-point precisions.

`cupy`, `to_numpy`, `array_module` and `free_gpu` cover the difference
between CuPy and NumPy arrays. `FP64` and `FP32` carry their dtypes and
the CUDA kernels compiled for them, and are passed to the solvers:

    waves = solve_dislocation(segs, s_dev, grid, xtal, precision=FP32)

FP64 is the default. Nothing in this module imports CuPy at module
level.
"""

import numpy as np

BLOCK = 256          # CUDA threads per block, one per x-column


def cupy():
    """The CuPy module, or an ImportError naming what to install."""
    try:
        import cupy
    except ImportError as exc:
        raise ImportError(
            "this needs CuPy and a CUDA GPU (pip install cupy-cuda12x, or "
            "pick the wheel matching your CUDA). The Bragg solver, the "
            "crystal parameters and everything in analysis/ run without "
            "one.") from exc
    return cupy


def to_numpy(a):
    """Host copy of `a`, device array or not."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def array_module(a):
    """The library `a` belongs to, so a function can work on either."""
    return cupy() if type(a).__module__.startswith("cupy") else np


def free_gpu():
    """Release CuPy's memory pool. Does nothing if CuPy is not loaded."""
    import sys
    cp = sys.modules.get("cupy")
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


class Precision:
    """One floating-point precision, and the kernels built for it.

    `real`/`complex` are the NumPy dtypes, `cp_real`/`cp_complex` the
    CuPy ones, fetched on demand. `scalar(x)` casts a float for a kernel
    argument list. Kernels are memoised per instance. Two instances
    exist: `FP64` and `FP32`.
    """

    def __init__(self, name):
        if name not in ("fp64", "fp32"):
            raise ValueError(f"precision is 'fp64' or 'fp32', not {name!r}")
        self.name = name
        self.is_fp32 = name == "fp32"
        self.real = np.float32 if self.is_fp32 else np.float64
        self.complex = np.complex64 if self.is_fp32 else np.complex128
        self.itemsize = 4 if self.is_fp32 else 8
        self._kernels = {}

    @property
    def cp_real(self):
        return cupy().float32 if self.is_fp32 else cupy().float64

    @property
    def cp_complex(self):
        return cupy().complex64 if self.is_fp32 else cupy().complex128

    def scalar(self, x):
        """Cast a scalar for a kernel argument list."""
        return self.real(x)

    def kernel(self, name):
        """The named CUDA kernel for this precision, compiled on first use."""
        if name not in self._kernels:
            from laue import kernels
            self._kernels[name] = kernels.build(name, self)
        return self._kernels[name]

    def __repr__(self):
        return f"Precision({self.name!r})"


FP64 = Precision("fp64")
FP32 = Precision("fp32")


def precision(name):
    """FP64 or FP32 by name, for turning a flag into an object."""
    if isinstance(name, Precision):
        return name
    return FP32 if str(name).lower() in ("fp32", "32", "single") else FP64
