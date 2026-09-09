"""Where the engine meets the array library.

The Laue solver is CuPy; everything else works on either CuPy or NumPy
arrays.  Rather than have each module decide how to get a host copy,
they ask here.

This module never imports CuPy.  It cannot: it must be importable on a
machine that has no CUDA, which is most of what makes the engine usable
outside the Laue solver.
"""

import numpy as np


def to_numpy(a):
    """A host copy of `a`, whether it is a CuPy or a NumPy array.

    `cupy.asnumpy` is exactly `a.get()` for a device array, so this
    reproduces it without importing CuPy to find out.
    """
    return a.get() if hasattr(a, "get") else np.asarray(a)


def free_gpu():
    """Release CuPy's memory pool, if CuPy is loaded.

    The solvers hold multi-gigabyte displacement-phase grids, and the
    scripts free the pool between configurations.  A no-op when there is
    no GPU in play.
    """
    import sys
    cp = sys.modules.get("cupy")
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
