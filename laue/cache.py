"""Keeping solved exit waves on disk so separate runs can share them.

A solve is expensive and completely determined by its inputs, so the
answer is worth writing down. Point a cache at a directory and ask it
for a solve; it hashes everything that went in and either loads the
answer or computes and stores it.

    waves = WaveCache("output/data/waves")
    stack = waves.solve(segs, s_dev, grid, xtal, tag="edge_+1")

One thing to know before trusting a result: the cache stores complex64
to keep the files a sensible size, while a fresh solve in FP64 hands
back complex128. So a cache hit is not bit-identical to a miss, and
which of the two you got is part of your numbers. If that matters for
what you are doing, and for reproducing a published figure it does,
start from an empty directory and let every run take the same path
through it. Passing `store_dtype=np.complex128` removes the asymmetry
at four times the disk.
"""

import hashlib
import json
import os

import numpy as np

from backend import FP64, cupy
from laue.solver import solve_dislocation

VERSION = 9


class WaveCache:
    """A directory of solved exit waves, keyed by what produced them."""

    def __init__(self, directory, version=VERSION,
                 store_dtype=np.complex64):
        self.directory = directory
        self.version = version
        self.store_dtype = store_dtype

    def key(self, seg_list, s_dev_arr, grid, xtal, precision=FP64):
        """The hash under which this solve is filed.

        Everything the answer depends on goes in, which is the point:
        change a Burgers vector, an angle, the grid pad or the
        precision, and you get a different file rather than a wrong one.
        """
        segs = [[[float(v) for v in np.asarray(part, float).ravel()]
                 for part in seg] for seg in seg_list]
        payload = dict(segments=segs,
                       s_dev=[float(s) for s in np.atleast_1d(s_dev_arr)],
                       grid=grid.key(), crystal=xtal.key(),
                       precision=precision.name,
                       store=np.dtype(self.store_dtype).name,
                       version=self.version)
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def path(self, seg_list, s_dev_arr, grid, xtal, precision=FP64, tag=""):
        """Where such a solve would live. The tag is only for reading by eye."""
        key = self.key(seg_list, s_dev_arr, grid, xtal, precision)
        stem = f"{tag}_{key}" if tag else key
        return os.path.join(self.directory, f"{precision.name}_{stem}.npy")

    def solve(self, seg_list, s_dev_arr, grid, xtal, *, precision=FP64,
              H_gpu=None, batch=64, tag="", use_cache=True):
        """The exit-wave stack, from disk if it is there and from the GPU if not."""
        cp = cupy()
        if not use_cache:
            return solve_dislocation(seg_list, s_dev_arr, grid, xtal,
                                     H_gpu=H_gpu, batch=batch,
                                     precision=precision)
        path = self.path(seg_list, s_dev_arr, grid, xtal, precision, tag)
        if os.path.exists(path):
            return cp.asarray(np.load(path))
        stack = solve_dislocation(seg_list, s_dev_arr, grid, xtal,
                                  H_gpu=H_gpu, batch=batch,
                                  precision=precision)
        os.makedirs(self.directory, exist_ok=True)
        np.save(path, cp.asnumpy(stack).astype(self.store_dtype))
        return stack

    def __repr__(self):
        return f"WaveCache({self.directory!r}, v{self.version})"
