"""Solving the Bragg boundary-value problem.

Three routines, and which one to use is not a matter of taste:

    solve_surface_batch  what to actually call. Solves every y-plane of
                         a chunk in one GMRES.
    solve_bragg_stable   one plane at a time. The same method, simpler
                         to read, and what the batched version was
                         checked against.
    solve_bragg          fixed-point iteration. Diverges for any crystal
                         thick enough to be interesting, and is kept
                         only as an independent reference on thin ones.

Why the fixed point fails is worth knowing, because it is the reason
this file is more complicated than the Laue solver. Marching D_0 down
and D_g back up in turn is Richardson iteration on (I - M)v = c, and it
converges only while the round-trip operator's spectral radius stays
under one. That holds until the crystal approaches total reflection,
which here is about 10 um, and a converged Bragg reflection needs more
like 30. But the problem is linear, so the answer is to stop iterating
and solve it.

The shooting form: march both amplitudes down from the surface with
D_g(0) = v unknown. The back-face condition D_g(t) = 0 is then a linear
system A v = -d, where A v is the homogeneous response and d the
particular one, and GMRES does not care about the spectral radius. The
perfect crystal has a closed-form Fourier symbol, which makes an
excellent preconditioner for the strained problem.

One subtlety: the lateral roll is periodic rather than truncated.
Zeroing the wrapped column seeds an edge that walks Nz pixels inward and
eats the field of view, so the grid is padded instead. See
`BraggConfig.Nx`.
"""

import numpy as np

from dislocations import H_plane


def _march_down(D0_0, Dg_0, Ch, Cb, e0, eg, ds, Nz, xp):
    """Propagate both amplitudes from z = 0 to z = t.  Returns D_g(t)."""
    D0, Dg = D0_0, Dg_0
    for k in range(Nz - 1):
        D0n = e0 * xp.roll(D0 + ds * Cb[k] * Dg, 1, axis=-1)
        Dgn = xp.roll((Dg - ds * Ch[k] * D0) / eg, -1, axis=-1)
        D0, Dg = D0n, Dgn
    return Dg


def perfect_symbol(e0, eg, ds, Nz, Nx, xtal):
    """Fourier symbol of the perfect-crystal shooting operator.

    With H = 0 the down-march is shift-invariant, so each spatial frequency
    evolves independently through a 2x2 chain.  Returns T21, T22 such that
    D_g(t) = T21 D_0(0) + T22 D_g(0) mode by mode -- the exact inverse of the
    unstrained problem, and therefore an excellent preconditioner for the
    strained one.
    """
    sh, sb = complex(xtal.sig_h), complex(xtal.sig_hbar)
    m = np.fft.fftfreq(Nx) * Nx
    p = np.exp(-2j * np.pi * m / Nx)          # roll(+1)
    pi = np.conj(p)                           # roll(-1)
    T11 = np.ones(Nx, complex); T12 = np.zeros(Nx, complex)
    T21 = np.zeros(Nx, complex); T22 = np.ones(Nx, complex)
    for _ in range(Nz - 1):
        a11 = e0 * p
        a12 = e0 * p * ds * sb
        a21 = -pi * ds * sh / eg
        a22 = pi / eg
        T11, T12, T21, T22 = (a11 * T11 + a12 * T21, a11 * T12 + a12 * T22,
                              a21 * T11 + a22 * T21, a21 * T12 + a22 * T22)
    return T21, T22


def solve_bragg(seg_list, s_dev, cfg, xtal, g_vec, xp=None,
                n_iter=400, tol=1e-11, verbose=False):
    """Reflected amplitude at the entrance surface.

    Returns (Dg_surface, info).  Dg_surface is the PHYSICAL exit wave,
    D_g * exp(-i g.u), sampled on the x grid at z = 0.
    """
    if xp is None:
        try:
            import cupy as cp
            xp = cp
        except Exception:
            xp = np
    s0 = complex(xtal.sig0)
    sh = complex(xtal.sig_h)
    sb = complex(xtal.sig_hbar)
    alpha = -2.0 * np.pi * float(s_dev)
    ds = cfg.ds

    if seg_list:
        H = H_plane(seg_list, cfg, g_vec, xtal.nu, xtal.a_core, xp=xp)
    else:
        H = xp.zeros((cfg.Nz, cfg.Nx), dtype=xp.float64)
    Ch = (sh * xp.exp(-1j * H)).astype(xp.complex128)       # sigma_h e^{-iH}
    Cb = (sb * xp.exp(+1j * H)).astype(xp.complex128)       # sigma_hbar e^{+iH}

    # one-step propagators for the homogeneous parts
    e0 = complex(np.exp(s0 * ds))                    # D_0 along s_0
    eg = complex(np.exp((s0 + 1j * alpha) * ds))     # D_g along s_g

    D0 = xp.zeros((cfg.Nz, cfg.Nx), dtype=xp.complex128)
    Dg = xp.zeros((cfg.Nz, cfg.Nx), dtype=xp.complex128)
    D0[0] = 1.0 + 0.0j                               # uniform incident wave

    prev = None
    info = {"iterations": 0, "residual": np.nan, "converged": False}
    # The lateral roll is periodic, not truncated: zeroing the wrapped column
    # seeds an edge that propagates Nz pixels inward and corrupts the field of
    # view.  The grid is padded instead (see BraggConfig.Nx).
    for it in range(n_iter):
        # ---- forward march of D_0 (z increasing), D_g frozen as a source.
        # The integrating factor multiplies the START of the step, which makes
        # the trapezoid second order; applying it to the average is only first.
        for k in range(cfg.Nz - 1):
            D0[k + 1] = (e0 * xp.roll(D0[k], 1)
                         + 0.5 * ds * (e0 * xp.roll(Cb[k] * Dg[k], 1)
                                       + Cb[k + 1] * Dg[k + 1]))
        # ---- backward march of D_g (z decreasing), D_0 frozen as a source
        Dg[cfg.Nz - 1] = 0.0
        for k in range(cfg.Nz - 2, -1, -1):
            Dg[k] = (eg * xp.roll(Dg[k + 1], 1)
                     + 0.5 * ds * (eg * xp.roll(Ch[k + 1] * D0[k + 1], 1)
                                   + Ch[k] * D0[k]))
        cur = Dg[0].copy()
        if prev is not None:
            num = float(xp.abs(cur - prev).max())
            den = float(xp.abs(cur).max()) + 1e-300
            info["residual"] = num / den
            if verbose and it % 20 == 0:
                print(f"    iter {it:4d}  residual {num/den:.3e}")
            if num / den < tol:
                info["converged"] = True
                info["iterations"] = it + 1
                break
        prev = cur
    else:
        info["iterations"] = n_iter

    exit_wave = Dg[0] * xp.exp(-1j * H[0])
    return exit_wave, info


def solve_bragg_stable(seg_list, s_dev, cfg, xtal, g_vec, xp=None,
                       tol=1e-10, y_plane=0.0, restart=40,
                       maxiter=20):
    """Bragg exit wave (physical, phase restored), shooting + preconditioned
    GMRES.  Unconditionally stable in crystal thickness."""
    from scipy.sparse.linalg import LinearOperator, gmres
    if xp is None:
        xp = np
    s0, sh, sb = (complex(xtal.sig0), complex(xtal.sig_h),
                  complex(xtal.sig_hbar))
    alpha = -2.0 * np.pi * float(s_dev)
    ds = cfg.ds
    H = (H_plane(seg_list, cfg, g_vec, xtal.nu, xtal.a_core, xp=xp,
                 y_plane=y_plane)
         if seg_list else xp.zeros((cfg.Nz, cfg.Nx), dtype=xp.float64))
    Ch = (sh * xp.exp(-1j * H)).astype(xp.complex128)
    Cb = (sb * xp.exp(+1j * H)).astype(xp.complex128)
    e0 = complex(np.exp(s0 * ds))
    eg = complex(np.exp((s0 + 1j * alpha) * ds))

    zero = xp.zeros(cfg.Nx, dtype=xp.complex128)
    ones = cfg.incident(xp=xp, n=1)[0]
    d = _march_down(ones, zero, Ch, Cb, e0, eg, ds, cfg.Nz, xp)

    def matvec(v):
        return _march_down(zero, xp.asarray(v.astype(np.complex128)),
                           Ch, Cb, e0, eg, ds, cfg.Nz, xp)

    _, T22 = perfect_symbol(e0, eg, ds, cfg.Nz, cfg.Nx, xtal)
    floor = 1e-8 * np.abs(T22).max()
    T22s = np.where(np.abs(T22) < floor, floor, T22)

    def precon(w):
        return np.fft.ifft(np.fft.fft(w) / T22s)

    A = LinearOperator((cfg.Nx, cfg.Nx), matvec=matvec, dtype=np.complex128)
    M = LinearOperator((cfg.Nx, cfg.Nx), matvec=precon, dtype=np.complex128)
    nit = [0]
    v, code = gmres(A, -np.asarray(d), M=M, rtol=tol, restart=restart,
                    maxiter=maxiter, callback=lambda _: nit.__setitem__(0, nit[0] + 1),
                    callback_type="pr_norm")
    res = float(np.linalg.norm(A.matvec(v) + np.asarray(d))
                / (np.linalg.norm(np.asarray(d)) + 1e-300))
    return xp.asarray(v) * xp.exp(-1j * H[0]), {
        "gmres_code": code, "iterations": nit[0], "residual": res}


def solve_surface_batch(seg_list, s_dev, cfg, y_values, xtal, g_vec,
                        xp=None, chunk=32, tol=1e-10, restart=40,
                        maxiter=25, verbose=False, gpu_gmres=True,
                        accept=1e-7, grow=4, tries=3,
                        restart_cap=256, max_depth=6, per_plane=6,
                        chunk_cap=32):
    """Exit wave on the (ny, Nx) surface grid.

    Transport is confined to the plane of incidence, so each y-plane is
    an independent boundary-value problem. Solving them one at a time is
    launch-bound, so a chunk of planes goes into a single GMRES, which
    does ny_chunk times the work per kernel launch and is what makes a
    fine grid affordable.

    Batching has a cost the solver will not tell you about. One Krylov
    space now has to serve `chunk` independent blocks, so with the
    default restart of 40 a 32-plane chunk gets barely one direction
    per plane. Where the strain is strong that is not enough, and both
    SciPy and CuPy return their `code = 0` for success while the true
    residual sits at a few percent. In the paper's five-dislocation
    Bragg scene that showed up as stripes at every chunk boundary
    beyond y = +9 um.

    So the residual is measured rather than believed. A chunk whose
    true relative residual exceeds `accept` is solved again, warm
    started, with the Krylov space `grow` times larger, up to `tries`
    attempts or `restart_cap` directions, whichever comes first. If it
    still will not converge, the chunk is split in half and each half
    solved separately: the planes are independent, so halving them
    doubles the Krylov directions each one gets, and the recursion
    bottoms out at a single plane, which always converges.

    That also means a large `chunk=` cannot quietly defeat the solver.
    It is capped where the Krylov space can still cover it several
    times over, and the split handles the rest.

    Chunks that were already converged cost nothing extra.
    `info` reports `converged`, the worst `residual`, and how many
    `retries` and `splits` it took.
    """
    from scipy.sparse.linalg import LinearOperator, gmres
    if xp is None:
        try:
            import cupy as cp
            xp = cp
        except Exception:
            xp = np
    to_np = (lambda a: a) if xp is np else xp.asnumpy
    s0, sh, sb = (complex(xtal.sig0), complex(xtal.sig_h),
                  complex(xtal.sig_hbar))
    alpha = -2.0 * np.pi * float(s_dev)
    ds = cfg.ds
    e0 = complex(np.exp(s0 * ds))
    eg = complex(np.exp((s0 + 1j * alpha) * ds))
    _, T22 = perfect_symbol(e0, eg, ds, cfg.Nz, cfg.Nx, xtal)
    floor = 1e-8 * np.abs(T22).max()
    T22s = np.where(np.abs(T22) < floor, floor, T22)

    ny = len(y_values)
    out = np.zeros((ny, cfg.Nx), dtype=np.complex128)
    worst, its, retries, splits, unconverged = 0.0, 0, 0, 0, 0
    # One Krylov space has to serve every plane in a chunk, so a very
    # large chunk cannot converge at any affordable restart, and the
    # split below would then thrash.  32 is where both of the paper's
    # geometries converge first time, measured on the hardest region of
    # each: 1.2 s per plane and no splits, against 6 to 8 s per plane
    # for 8 or 16, which get a smaller absolute space and split
    # repeatedly.  Bigger is not better either; 42 already splits.
    chunk = max(1, min(chunk, chunk_cap))

    def solve_block(j0, j1, depth=0):
        """Solve planes [j0, j1) into `out`, splitting if they resist."""
        nonlocal worst, its, retries, splits, unconverged
        nyc = j1 - j0
        H = xp.stack([H_plane(seg_list, cfg, g_vec, xtal.nu, xtal.a_core,
                              xp=xp, y_plane=float(y))
                      for y in y_values[j0:j1]], axis=1) if seg_list else \
            xp.zeros((cfg.Nz, nyc, cfg.Nx), dtype=xp.float64)
        Ch = (sh * xp.exp(-1j * H)).astype(xp.complex128)
        Cb = (sb * xp.exp(+1j * H)).astype(xp.complex128)
        zero = xp.zeros((nyc, cfg.Nx), dtype=xp.complex128)
        inc = cfg.incident(xp=xp, n=nyc)
        d = to_np(_march_down(inc, zero, Ch, Cb, e0, eg, ds, cfg.Nz,
                              xp)).ravel()

        def matvec(v):
            vv = xp.asarray(v.astype(np.complex128).reshape(nyc, cfg.Nx))
            return to_np(_march_down(zero, vv, Ch, Cb, e0, eg, ds,
                                     cfg.Nz, xp)).ravel()

        def precon(w):
            W = w.reshape(nyc, cfg.Nx)
            return np.fft.ifft(np.fft.fft(W, axis=-1) / T22s[None, :],
                               axis=-1).ravel()

        N = nyc * cfg.Nx
        cnt = [0]
        # A block of nyc planes is nyc independent problems sharing
        # one Krylov space, so the space has to scale with it. Six
        # directions per plane converges the paper's hardest chunks
        # first time; starting at 40 and growing costs several
        # wasted solves to reach the same place.
        cap = max(restart, min(restart_cap, 8 * nyc))
        start = max(restart, min(cap, per_plane * nyc))
        if gpu_gmres and xp is not np:
            # Krylov vectors stay on the device: the orthogonalisation is
            # then memory-bound instead of 58 ms of CPU per iteration.
            import cupyx.scipy.sparse.linalg as csl
            T22g = xp.asarray(T22s)
            dg = xp.asarray(d).reshape(nyc, cfg.Nx)

            def mv_g(vv):
                return _march_down(zero, vv.reshape(nyc, cfg.Nx), Ch, Cb,
                                   e0, eg, ds, cfg.Nz, xp).ravel()

            def pc_g(w):
                W = w.reshape(nyc, cfg.Nx)
                return xp.fft.ifft(xp.fft.fft(W, axis=-1) / T22g[None, :],
                                   axis=-1).ravel()

            Ag = csl.LinearOperator((N, N), matvec=mv_g,
                                    dtype=xp.complex128)
            Mg = csl.LinearOperator((N, N), matvec=pc_g,
                                    dtype=xp.complex128)
            nb = float(xp.linalg.norm(dg)) + 1e-300
            vg, x0, r = None, None, start
            for _ in range(tries):
                vg, code = csl.gmres(Ag, -dg.ravel(), x0=x0, M=Mg, tol=tol,
                                     restart=r, maxiter=maxiter,
                                     callback=lambda _: cnt.__setitem__(
                                         0, cnt[0] + 1),
                                     callback_type="pr_norm")
                res = float(xp.linalg.norm(mv_g(vg) + dg.ravel()) / nb)
                if res <= accept or r >= cap:
                    break
                retries += 1
                x0, r = vg, min(r * grow, cap)
                if verbose:
                    print(f"      y {j0}-{j1}: residual {res:.1e} above "
                          f"{accept:.0e}, retrying with restart {r}",
                          flush=True)
            v = to_np(vg)
        else:
            A = LinearOperator((N, N), matvec=matvec, dtype=np.complex128)
            M = LinearOperator((N, N), matvec=precon, dtype=np.complex128)
            nb = np.linalg.norm(d) + 1e-300
            v, x0, r = None, None, start
            for _ in range(tries):
                v, code = gmres(A, -d, x0=x0, M=M, rtol=tol, restart=r,
                                maxiter=maxiter,
                                callback=lambda _: cnt.__setitem__(
                                    0, cnt[0] + 1),
                                callback_type="pr_norm")
                res = float(np.linalg.norm(A.matvec(v) + d) / nb)
                if res <= accept or r >= cap:
                    break
                retries += 1
                x0, r = v, min(r * grow, cap)
                if verbose:
                    print(f"      y {j0}-{j1}: residual {res:.1e} above "
                          f"{accept:.0e}, retrying with restart {r}",
                          flush=True)

        if res > accept and nyc > 1 and depth < max_depth:
            # A wider Krylov space is not affordable, but half as many
            # planes need half as much of it.  The planes are
            # independent, so splitting changes nothing but the cost.
            del H, Ch, Cb
            if xp is not np:
                xp.get_default_memory_pool().free_all_blocks()
            splits += 1
            if verbose:
                print(f"      y {j0}-{j1}: residual {res:.1e} still above "
                      f"{accept:.0e} at restart {cap}, splitting",
                      flush=True)
            mid = j0 + nyc // 2
            solve_block(j0, mid, depth + 1)
            solve_block(mid, j1, depth + 1)
            return

        unconverged += res > accept
        worst = max(worst, res)
        its = max(its, cnt[0])
        ewc = xp.asarray(v.reshape(nyc, cfg.Nx)) * xp.exp(-1j * H[0])
        out[j0:j1] = to_np(ewc)
        del H, Ch, Cb
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"      y {j0}-{j1}: {cnt[0]} its, res {res:.1e}",
                  flush=True)

    for j0 in range(0, ny, chunk):
        solve_block(j0, min(j0 + chunk, ny))
    return out, {"residual": worst, "iterations": its,
                 "retries": retries, "splits": splits,
                 "unconverged_chunks": unconverged,
                 "converged": unconverged == 0}
