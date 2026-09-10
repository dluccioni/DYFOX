"""Talking to whichever array library the caller happens to be using.

The Laue solver runs on CuPy. Everything else works the same on CuPy or
NumPy arrays, so instead of each module working out for itself how to
get a host copy, they ask here.

This is also where precision lives. `FP64` and `FP32` are the two
objects the solvers take; each knows its dtypes and holds the kernels
compiled for it. Pass one around rather than setting a global:

    from backend import FP64, FP32
    waves = solve_dislocation(segs, s_dev, grid, xtal, precision=FP32)

FP64 is the right default. On an RTX 4090 the march costs about 0.15 s
per rocking angle in double against 0.01 s in single, which is nothing
beside building the displacement grid, and single leaves a 5 % noise
floor in the spiral-suppressed background that the centre-of-mass ratio
maps are measured against. FP32 is for drafts.

Nothing in this file imports CuPy at module level, and it needs to stay
that way: it has to be importable on a machine with no CUDA at all,
which is what makes the rest of the engine useful there.
"""

import numpy as np

BLOCK = 256          # CUDA threads per block, one per x-column


def cupy():
    """The CuPy module, or a readable error if it is not installed.

    Import this way rather than at the top of a file. Half the engine
    runs on a laptop, and only the code that actually launches a kernel
    should be the code that insists on a GPU.
    """
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
    """Hand CuPy's memory pool back, if CuPy is even loaded.

    One solve holds a displacement-phase grid of a few gigabytes, so the
    scripts drop the pool between configurations. Does nothing when
    there is no GPU in play.
    """
    import sys
    cp = sys.modules.get("cupy")
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


class Precision:
    """One floating-point precision, and the kernels built for it.

    `real` and `complex` are the NumPy dtypes; `cp_real` and `cp_complex`
    the CuPy ones, fetched on demand so that naming a precision does not
    require a GPU. `scalar(x)` casts a Python float on its way into a
    kernel argument list.

    There are exactly two instances, `FP64` and `FP32`. Construct no
    others: kernels are memoised per instance, so a third would compile
    the same code again.
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
        """The named CUDA kernel, compiled for this precision and kept.

        Compiling is a second or two, so it happens on first use and not
        at import.
        """
        if name not in self._kernels:
            from laue import kernels
            self._kernels[name] = kernels.build(name, self)
        return self._kernels[name]

    def __repr__(self):
        return f"Precision({self.name!r})"


FP64 = Precision("fp64")
FP32 = Precision("fp32")


def precision(name):
    """FP64 or FP32 by name, for turning a command-line flag into an object."""
    if isinstance(name, Precision):
        return name
    return FP32 if str(name).lower() in ("fp32", "32", "single") else FP64
