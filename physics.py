"""GPU physics engine for the SPC-DFXM figure set.

Takagi--Taupin dynamical diffraction of dislocation exit waves in
diamond(400) at 17 keV, plus the Fourier-optics imaging stage with an
optional spiral phase plate at the back focal plane.

The solver is the *full-field* formulation used for the published figures:
the displacement phase H = 2*pi*g.u (Volterra) is precomputed on a 3D
(Ny, Nz, Nx) grid and folded into the off-diagonal couplings as
exp(+/- iH), so the kernel integrates the physical wave directly -- no
envelope/gradient approximation and no exit-phase restoration step.  An
envelope-formulation kernel (local deviation parameter from the
non-singular Cai gradient) is retained for cross-validation.

GEOMETRY N (2026-08-18): production geometry is a consistent symmetric
Laue frame -- (001) diamond plate, g = (400) || x_lab, dislocation line
[101] inclined 45 deg in the diffraction plane, edge b = (a/2)[10-1],
screw b = (a/2)[101], segments centred on the line/direct-beam crossing
(make_segment).  All production solves therefore use the 3D H-grid
kernel; the closed-form in-kernel H path applies only to lines exactly
along x_lab (the (220) screw of the geometry studies).

Design changes vs the monolithic spc_dfxm_figures.py this replaces:

* No mutable module globals: `SimConfig` carries the grid/beam settings and
  is passed explicitly.  Optics are requested per (grid, charge, offset).
* Batched s_dev solves reuse ONE H grid per dislocation configuration
  (the old script recomputed the ~3 GB H grid for every rocking angle).
* One TT solve serves every pupil: `apply_pupils` FFTs the exit wave once
  and applies all requested pupils (open, spiral charges, misaligned
  spirals) from the cached spectrum.
* Exit waves are cached on disk (data/waves/) keyed by a content hash of
  the physics parameters, so independent figure scripts share solves.
* Misaligned spiral pupils take an explicit (dx, dy) BFP offset in metres.
  (The old fig-8 code offset both frequency axes by the same amount, so a
  quoted offset delta was actually applied as |delta|*sqrt(2) diagonally.)

All arrays returned by solvers are CuPy unless noted; call `to_cpu` at the
edge of the pipeline.
"""

import hashlib
import json
import os

import numpy as np
import cupy as cp

# ---------------------------------------------------------------------------
#  Precision policy: FP64 kernels by default.  On an RTX 4090 the TT march
#  costs ~0.15 s per rocking angle in FP64 (vs 0.01 s FP32) -- negligible
#  against the H-grid build -- and double precision removes the ~5% FP32
#  noise floor in the spiral-suppressed dark background that feeds the COM
#  ratio maps.  Set SPC_FP32=1 for quick drafts.
# ---------------------------------------------------------------------------
USE_FP32 = os.environ.get("SPC_FP32", "0") == "1"
DTYPE = np.float32 if USE_FP32 else np.float64
CP_DTYPE = cp.float32 if USE_FP32 else cp.float64
CP_COMPLEX = cp.complex64 if USE_FP32 else cp.complex128
BLOCK = 256

# Exit waves are cached on disk so that independent scripts share
# solves.  WHERE they are cached is the caller's decision, not the
# engine's, and importing the engine must not create directories:
# scripts/paper/config.py points this at the run's output tree.
WAVE_DIR = None


def set_cache_dir(path):
    """Choose where exit_waves() stores solves.  Created on first write."""
    global WAVE_DIR
    WAVE_DIR = path
    return WAVE_DIR

# ===========================================================================
#  Crystal parameters -- diamond (400) at 17 keV
# ===========================================================================
R_E = 2.8179403262e-15          # classical electron radius (m)
HC_EV_M = 12398.419e-10         # hc (eV * m)

# Diamond cubic basis (8 atoms)
ATOM_POS = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                     [.25, .25, .25], [.75, .75, .25],
                     [.75, .25, .75], [.25, .75, .75]])

A_DIAMOND = 3.567e-10           # lattice parameter (m)
E_CENTER = 17.0                 # photon energy (keV)
# Isotropic (Voigt-Reuss-Hill) Poisson ratio of diamond from
# C11 = 1079, C12 = 124, C44 = 578 GPa.  (The script this replaces -- and
# manuscript Table 1 -- used 0.28, which is not a diamond value; the
# topological arctan winding is nu-independent, but the edge smooth-field
# prefactors 1/(1-nu) and (1-2nu)/(4(1-nu)) change at the ~30% level.)
NU_POISSON = 0.069
# Anomalous scattering corrections for C at 17 keV
F_PRIME, F_DPRIME = 0.003, 0.002

# Reflection setups.  U_lab rows = [x_lab, y_lab, z_lab] in crystal coords:
#   x_lab: in diffraction plane, along crystal surface
#   y_lab: out of diffraction plane
#   z_lab: surface normal (beam propagation / depth direction)
# f0: Cromer-Mann f0(sin(theta)/lambda) for carbon at the reflection's q.
REFLECTIONS = {
    "400": dict(
        g_hkl=np.array([4, 0, 0]),
        # GEOMETRY N (2026-08-18 referee fix).  The kernel marches a
        # SYMMETRIC Laue geometry (both characteristics at +-theta_B
        # about z_lab), which requires g PERPENDICULAR to the surface
        # normal: g || x_lab.  The previous frame (z || [110],
        # g 45 deg out of the surface) was internally inconsistent --
        # no symmetric-Laue experiment realises it, and the edge's
        # visibility came entirely from the inconsistent out-of-surface
        # g component (with line || x and b perpendicular to the line,
        # a consistent frame forces g.b = 0).  The consistent frame is
        # the cube frame of a standard (001) diamond plate:
        #   x || [100] (= g-hat),  y || [010],  z || [001] (normal).
        # The dislocation line moves to [101], 45 deg inclined in the
        # diffraction plane (see DISLOC_TYPES).  det U = +1.
        U_lab=np.eye(3),
        f0=1.589,
        label="Diamond(400)",
    ),
    "220": dict(
        g_hkl=np.array([2, 2, 0]),
        # Same (001) plate, g || [110]: hosts the maximal-contrast
        # screw (line = b = [110] || x, g.b = 2) and the end-on
        # threading edge (line [001], b = 1/2[110], g.b = 2).  Kept for
        # the geometry studies; production figures use "400".
        U_lab=np.array([[1, 1, 0], [-1, 1, 0], [0, 0, np.sqrt(2)]])
        / np.sqrt(2),
        f0=1.961,
        label="Diamond(220)",
    ),
}


def make_xtal_params(reflection="400", E_keV=E_CENTER):
    """Crystal diffraction parameters for symmetric Laue geometry.

    Returns a plain dict (JSON-serialisable scalars plus small arrays) so it
    can participate in the wave-cache content hash.
    """
    refl = REFLECTIONS[reflection]
    a_lat = A_DIAMOND
    g_hkl = refl["g_hkl"]
    # NEGATIVE imaginary part: the solver evolves D ~ exp(sigma * s) with
    # sigma = -i C F, so absorption (Re sigma < 0) requires Im F < 0 in
    # this wave convention.  (The script this replaces used +i f'', which
    # made the crystal *amplify* by ~1.8% over the 245 um path.)
    f_atom = refl["f0"] + F_PRIME - 1j * F_DPRIME
    lam = HC_EV_M / (E_keV * 1e3)
    b_mag = a_lat * np.sqrt(2) / 2
    d_hkl = a_lat / np.sqrt(np.sum(g_hkl ** 2))
    theta_B = np.arcsin(lam / (2 * d_hkl))
    g_vec = refl["U_lab"] @ (g_hkl / a_lat)     # lab frame, units 1/m (no 2pi)
    V_cell = a_lat ** 3
    # Forward scattering uses f(0) = Z + f' + i f'', NOT the reflection's
    # f0(q).  (The script this replaces used f0(q) for F_0 as well, which
    # understates refraction; absorption, set by f'', was unaffected.)
    f_forward = 6.0 + F_PRIME - 1j * F_DPRIME
    F_0 = f_forward * len(ATOM_POS)
    S_g = np.sum(np.exp(2j * np.pi * ATOM_POS @ g_hkl))
    F_g = f_atom * S_g
    # F_{-g} = f * conj(S_g): only the geometric sum conjugates -- the
    # complex atomic factor does not.  (The script this replaces used
    # conj(F_g), which conjugated f'' as well; that incidentally made
    # sigma_h*sigma_hbar exactly real, so the open-aperture COM
    # antisymmetry was exact there and is exact only up to anomalous-
    # absorption terms ~1e-3 here.)
    F_mg = f_atom * np.conj(S_g)
    sig0 = -1j * R_E * lam * F_0 / V_cell
    sig_h = -1j * R_E * lam * F_g / V_cell
    sig_hbar = -1j * R_E * lam * F_mg / V_cell
    xi_g = np.pi / abs(sig_h)
    return dict(lam=lam, d_hkl=d_hkl, theta_B=theta_B, F_g=F_g,
                sig0=sig0, sig_h=sig_h, sig_hbar=sig_hbar, xi_g=xi_g,
                sin_tB=np.sin(theta_B), cos_tB=np.cos(theta_B),
                tan_tB=np.tan(theta_B), sin_2tB=np.sin(2 * theta_B),
                label=refl["label"], E_keV=E_keV, a_lat=a_lat,
                nu=NU_POISSON, g_vec=g_vec, b_mag=b_mag, a_core=b_mag,
                U_lab=refl["U_lab"])


# Optional energy override for facility studies (e.g. LCLS self-seeding
# tops out at 12.5 keV).  Everything downstream -- T_CRYSTAL snapping,
# S_DEV_WB, SimConfig's exact-advection dz, the wave-cache key (E_keV) --
# follows from P, so an import-time override keeps the pipeline coherent.
E_CENTER = float(os.environ.get("SPC_E_KEV", E_CENTER))
P = make_xtal_params(E_keV=E_CENTER)

# ===========================================================================
#  Simulation configuration
# ===========================================================================
# Crystal thickness: snap to the Pendelloesung maximum nearest the
# manuscript's t ~ 5*xi_g = 270 um, so the pristine rocking-curve peak sits
# at a thickness-oscillation maximum (n = 5 -> t = 5.5 half-periods).
# The snap always uses the 17 keV production parameters: the plate is a
# physical sample, so an energy override (SPC_E_KEV) changes the beam,
# not the crystal.  SimConfig re-derives Nz for exact advection at the
# active energy, reproducing this thickness to within one dz.
_P17 = make_xtal_params(E_keV=17.0)
_HALF_PEND = _P17["xi_g"] * _P17["cos_tB"]
T_CRYSTAL = (round(270e-6 / _HALF_PEND - 0.5) + 0.5) * _HALF_PEND

DX = 0.25e-6                    # pixel size (m)
# Grid pad beyond the Borrmann fan.  Overridable for off-nominal energies
# where the wider fan would push the kernel over the 99 KB shared-memory
# ceiling (the pad region carries only the decayed fan tail).
GRID_PAD_DEFAULT = 30
BEAM_WIDTH_Y = 120e-6           # sheet-beam FOV in y (manuscript Sec. 3.1)
GRID_PAD = int(os.environ.get("SPC_GRID_PAD", GRID_PAD_DEFAULT))

# DFXM optics (Table 1).  Objective matched to the standard hard-X-ray
# DFXM configuration (Be CRL stack, f = 274 mm at 17 keV) rather than to
# a hypothetical long-arm instrument: d2 = 4.93 m fits a real hutch,
# where the earlier f = 500 mm / M = 100x choice implied a 50.5 m arm.
# Only F_LENS enters the physics -- through R_BFP = NA*f, which sets the
# metre-to-frequency conversion for a plate displacement in make_pupil.
# M_DFXM, D1_LENS and D2_LENS are documentation of the imaging geometry.
M_DFXM = 17.0
F_LENS = 0.274                  # m
NA_DFXM = 1.0e-4
D1_LENS = F_LENS * (M_DFXM + 1) / M_DFXM
D2_LENS = D1_LENS * F_LENS / (D1_LENS - F_LENS)
R_BFP = NA_DFXM * F_LENS        # BFP radius (m)

# Weak-beam operating point.  For a 1 um sheet the integrated response
# is filled in by the beam's own angular envelope (lambda/w = 73 urad),
# so genuine background suppression requires Delta_phi beyond it (the
# TEM weak-beam prescription translated to sheet-beam DFXM).  The
# operating-point scan (figV6) selects Delta_phi* = 117.5 urad
# (s* = 65.5/xi_g): pristine per-area background 6.7e-4 of strong beam,
# peak/background contrast ~2e4.  (The manuscript's original point was
# s = 3/xi_g = 5.4 urad, which gives no integrated suppression.)
PHI_WB = 117.5e-6               # rad
S_DEV_WB = PHI_WB * P["sin_2tB"] / P["lam"]


def sdev_to_urad(s_dev, params=P):
    """Rocking angle Delta_phi (urad) for a deviation parameter s_dev (1/m)."""
    return np.asarray(s_dev) * params["lam"] / params["sin_2tB"] * 1e6


class SimConfig:
    """Grid + beam settings for one solve (immutable value object).

    By default the depth step is TIED to exact characteristic advection:
    dz = dx / tan(theta_B), so the Borrmann-fan transport shifts the
    field by exactly one pixel per step and the interpolation is
    error-free.  Sub-pixel linear-interpolation transport (the old
    pipeline's 0.36 px/step at Nz = 1501) acts as a compounded low-pass
    filter that suppresses the um-scale weak-beam dislocation signal by
    more than an order of magnitude (verified: 65x peak loss vs the
    exact-advection solution).  Resolution is refined through dx, which
    refines dz with it.  Pass Nz explicitly only for validation studies
    of the interpolating transport.
    """

    def __init__(self, beam_thickness=1.0e-6, t_crystal=T_CRYSTAL, Nz=None,
                 dx=DX, beam_width=BEAM_WIDTH_Y, beam_sigma=0.0):
        # beam_sigma > 0 replaces the top-hat sheet with a Gaussian
        # AMPLITUDE profile of that sigma (metres); intensity FWHM is
        # then 2*sqrt(ln 2)*beam_sigma.  Zero keeps the top-hat.
        self.beam_thickness = float(beam_thickness)
        self.beam_sigma = float(beam_sigma)
        self.dx = float(dx)
        self.beam_width = float(beam_width)
        if Nz is None:
            dz = self.dx / P["tan_tB"]
            self.Nz = int(round(float(t_crystal) / dz)) + 1
            self.t_crystal = (self.Nz - 1) * dz
            self.exact = True
        else:
            self.Nz = int(Nz)
            self.t_crystal = float(t_crystal)
            dz = self.t_crystal / (self.Nz - 1)
            self.exact = abs(P["tan_tB"] * dz / self.dx - 1.0) < 1e-9

    def sheet_arg(self):
        """Kernel beam parameter: +half-width (top-hat) or -sigma_A."""
        if self.beam_sigma > 0:
            return -self.beam_sigma
        return self.beam_thickness / 2.0

    def grid(self, params=P):
        """(Nx, Ny, dz, ds, shift_px) for this configuration."""
        borrmann = self.t_crystal * params["tan_tB"]
        pad = GRID_PAD * self.dx
        eff_thick = (6.0 * self.beam_sigma if self.beam_sigma > 0
                     else self.beam_thickness)
        extent_x = eff_thick + 2 * borrmann + 2 * pad
        Nx = max(int(np.ceil(extent_x / self.dx)), 128)
        Nx = int(np.ceil(Nx / 32)) * 32
        Ny = max(int(np.ceil((self.beam_width + 2 * pad) / self.dx)), 64)
        Ny = int(np.ceil(Ny / 32)) * 32
        dz = self.t_crystal / (self.Nz - 1)
        ds = dz / params["cos_tB"]
        shift_px = params["tan_tB"] * dz / self.dx
        if abs(shift_px - round(shift_px)) < 1e-9:
            shift_px = float(round(shift_px))
        return Nx, Ny, dz, ds, shift_px

    def key(self):
        return dict(beam_thickness=self.beam_thickness,
                    beam_sigma=self.beam_sigma,
                    t_crystal=self.t_crystal, Nz=self.Nz, dx=self.dx,
                    beam_width=self.beam_width)


DEFAULT_CONFIG = SimConfig()

# ===========================================================================
#  Dislocation geometry
# ===========================================================================
# GEOMETRY N crystal-frame definitions (converted to lab via U_lab = I):
#   edge : line [101], b (a/2)[10-1] -> pure edge,  g.b = 2, slip (010)
#   screw: line [101], b (a/2)[101]  -> pure screw, g.b = 2
# Both types share the line direction [101]: in the diffraction (x-z)
# plane, 45 deg from the surface, so the Borrmann fan maps the
# displacement winding into a 2D image vortex.  This is the only
# shared-<110>-line arrangement in a consistent (400) symmetric-Laue
# frame in which BOTH pure types are visible (geometry study 2026-08-18:
# matched/mismatched core ratio 14x for both).
DISLOC_TYPES = {
    "Edge": dict(line_cryst=np.array([1., 0., 1.]),
                 b_cryst=np.array([1., 0., -1.])),
    "Screw": dict(line_cryst=np.array([1., 0., 1.]),
                  b_cryst=np.array([1., 0., 1.])),
}
VARIANTS = [("Edge", +1), ("Edge", -1), ("Screw", +1), ("Screw", -1)]


def variant_name(dtype, sign):
    return f"{dtype} {'+' if sign > 0 else '-'}b"


def make_segment(dtype, sign, config=DEFAULT_CONFIG, params=P, y0=None,
                 z0=None):
    """Single straight dislocation segment through the gauge volume.

    The line passes through the WALKING DIRECT-BEAM COLUMN at depth z0:
    the s_0 characteristic advects at -tan(theta_B) per unit depth, so
    the column sits at x = -z0 tan(theta_B) there, and the crossing
    images at x_img = -2 (z0 - t/2) tan(theta_B) (the usual depth
    mapping).  For the inclined Geometry-N line this crossing is where
    the core contrast forms; centring the segment anywhere else probes
    only the strain tail (geometry study round 3).  Returns
    [(ra, rb, b_vec)] in lab-frame metres; the segment is long enough
    to act as an infinite line.
    """
    d = DISLOC_TYPES[dtype]
    xi = params["U_lab"] @ d["line_cryst"]
    xi = xi / np.linalg.norm(xi)
    b_hat = params["U_lab"] @ d["b_cryst"]
    b_hat = b_hat / np.linalg.norm(b_hat)
    bv = sign * params["b_mag"] * b_hat
    extent = 2 * (config.t_crystal
                  + config.beam_thickness
                  + 2 * config.t_crystal * params["tan_tB"])
    if y0 is None:
        y0 = config.dx / 2
    if z0 is None:
        z0 = config.t_crystal / 2.0
    center = np.array([-z0 * params["tan_tB"], y0, z0])
    return [(center - extent * xi, center + extent * xi, bv)]


def g_dot_b(dtype, sign=+1, params=P):
    d = DISLOC_TYPES[dtype]
    b_hat = params["U_lab"] @ d["b_cryst"]
    b_hat = b_hat / np.linalg.norm(b_hat)
    return float(np.dot(params["g_vec"], sign * params["b_mag"] * b_hat))


# ===========================================================================
#  Displacement phase grid  H = 2*pi*g.u  (Volterra, isotropic elasticity)
# ===========================================================================
def compute_H_grid(seg_list, config, params=P):
    """Pre-compute H = 2*pi*g.u on the 3D (Ny, Nz, Nx) grid (GPU, DTYPE).

    The arctan (topological) term is exact; the smooth edge terms are
    core-regularised at r = a_core.  Computed in y-chunks to bound the
    FP64 intermediates.
    """
    Nx, Ny, dz, _, _ = config.grid(params)
    Nz = config.Nz
    dxv = config.dx
    gv = params["g_vec"]
    nu = params["nu"]
    a_core = params["a_core"]
    CHUNK_Y = 32
    x_1d = (cp.arange(Nx, dtype=cp.float64) - Nx / 2) * dxv
    z_1d = cp.arange(Nz, dtype=cp.float64) * dz
    H = cp.zeros((Ny, Nz, Nx), dtype=CP_DTYPE)
    for (ra, rb, bv) in seg_list:
        ra, rb, bv = (np.asarray(v, float) for v in (ra, rb, bv))
        t = rb - ra
        L = np.linalg.norm(t)
        if L < 1e-20:
            continue
        xi = t / L
        core = (ra + rb) / 2.0
        b_screw = float(np.dot(bv, xi))
        b_perp_vec = bv - b_screw * xi
        b_perp = np.linalg.norm(b_perp_vec)
        if b_perp > 1e-20:
            e1 = b_perp_vec / b_perp
        elif abs(xi[2]) > 0.9:
            e1 = np.array([1., 0., 0.])
        elif abs(xi[0]) < 0.9:
            e1 = np.cross(xi, [1, 0, 0])
            e1 /= np.linalg.norm(e1)
        else:
            e1 = np.cross(xi, [0, 1, 0])
            e1 /= np.linalg.norm(e1)
        e2 = np.cross(xi, e1)
        onu = 1.0 - nu
        C_atan = np.dot(gv, e1) * b_perp + np.dot(gv, xi) * b_screw
        ge1, ge2 = np.dot(gv, e1), np.dot(gv, e2)
        Xb = (x_1d - core[0])[None, None, :]
        Zb = (z_1d - core[2])[None, :, None]
        for ys in range(0, Ny, CHUNK_Y):
            ye = min(ys + CHUNK_Y, Ny)
            Yb = ((cp.arange(ys, ye, dtype=cp.float64) - Ny / 2) * dxv
                  - core[1])[:, None, None]
            eta = Xb * xi[0] + Yb * xi[1] + Zb * xi[2]
            Px = Xb - eta * xi[0]
            Py = Yb - eta * xi[1]
            Pz = Zb - eta * xi[2]
            del eta
            d1 = Px * e1[0] + Py * e1[1] + Pz * e1[2]
            d2 = Px * e2[0] + Py * e2[1] + Pz * e2[2]
            del Px, Py, Pz
            r2 = d1 ** 2 + d2 ** 2
            r2c = cp.maximum(r2, a_core ** 2)
            Hc = C_atan * cp.arctan2(d2, d1)
            if b_perp > 1e-20:
                c1 = b_perp / (2 * np.pi)
                sm_ux = c1 * d1 * d2 / (2 * onu * r2c)
                sm_uy = -c1 * ((1 - 2 * nu) / (4 * onu) * cp.log(r2c)
                               + (d1 ** 2 - d2 ** 2) / (4 * onu * r2c))
                Hc += 2 * np.pi * (ge1 * sm_ux + ge2 * sm_uy)
            del d1, d2, r2, r2c
            H[ys:ye] += Hc.astype(CP_DTYPE)
            del Hc
    return H


# ===========================================================================
#  CUDA kernels
# ===========================================================================
def _typed(src, fname):
    """Instantiate a DTYPE-templated kernel source at the module precision."""
    if USE_FP32:
        src = (src.replace("DTYPE*", "float*").replace("DTYPE ", "float ")
               .replace("(DTYPE)", "(float)")
               .replace("FABS(", "fabsf(").replace("SQRT(", "sqrtf(")
               .replace("COS(", "cosf(").replace("SIN(", "sinf(")
               .replace("EXP(", "expf(").replace("FLOOR(", "floorf(")
               .replace("3.14159265358979323846", "3.14159265358979f"))
    else:
        src = (src.replace("DTYPE*", "double*").replace("DTYPE ", "double ")
               .replace("(DTYPE)", "(double)")
               .replace("FABS(", "fabs(").replace("SQRT(", "sqrt(")
               .replace("COS(", "cos(").replace("SIN(", "sin(")
               .replace("EXP(", "exp(").replace("FLOOR(", "floor("))
    return cp.RawKernel(src, fname)


# --- Full-field batched kernel: exp(+/- iH) in the off-diagonal coupling ---
_FULLFIELD_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_fullfield_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy,
    DTYPE* H_grid)   /* (Ny, Nz, Nx): pre-computed 2*pi*g.u */
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }

    /* D_0 = sheet-beam top hat in x, D_g = 0 */
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE shm2 = shr*shr + shi*shi;
    DTYPE alpha = -2.0*PI*s_dev_val;      /* strain lives in exp(iH), not alpha */
    DTYPE gam = SQRT(shm2 + alpha*alpha*0.25);
    DTYPE exp_pr = EXP(s0r * ds);

    for (int k = 0; k < Nz-1; k++){
        int nx = 1 - curr;
        DTYPE gds_ = gam*ds, cg, sg;
        if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
        DTYPE P11r=cg, P11i=-alpha*0.5*sg, P22r=cg, P22i=alpha*0.5*sg;
        DTYPE pi_=(s0i+alpha*0.5)*ds;
        DTYPE epr=exp_pr*COS(pi_), epi=exp_pr*SIN(pi_);
        DTYPE M11r=epr*P11r-epi*P11i, M11i=epr*P11i+epi*P11r;
        DTYPE M22r=epr*P22r-epi*P22i, M22i=epr*P22i+epi*P22r;

        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE H_val = H_grid[iy * Nz * Nx_ + k * Nx_ + ix];
            DTYPE cH = COS(H_val), sH = SIN(H_val);
            /* Deformed-crystal couplings: chi(r - u) modulates the g
               component by exp(-iH) and the -g component by exp(+iH)
               (H = 2 pi g.u, amplitudes on e^{+i k.r} carriers -- the
               convention anchored by the refraction sign of sigma_0).
               sigma_h * exp(-iH) feeds D_g; sigma_hbar * exp(+iH)
               feeds D_0.  (The script this replaces used the opposite
               signs together with an improper U_lab; the two flips
               cancel for the topological content, but neither was
               derivable.) */
            DTYPE P12r=(sbr*cH-sbi*sH)*sg, P12i=(sbi*cH+sbr*sH)*sg;
            DTYPE P21r=(shr*cH+shi*sH)*sg, P21i=(shi*cH-shr*sH)*sg;
            DTYPE M12r=epr*P12r-epi*P12i, M12i=epr*P12i+epi*P12r;
            DTYPE M21r=epr*P21r-epi*P21i, M21i=epr*P21i+epi*P21r;

            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""
kernel_fullfield = _typed(_FULLFIELD_SRC, "tt3d_fullfield_batch")

# --- Analytic-H batched kernel for a straight line along x_lab ---------
# H = 2*pi*g.u is x-independent for a line parallel to x, so it is
# evaluated in closed form once per (row, depth step): no 3D H grid (no
# memory ceiling, no depth aliasing), exact at every z sample.
_ANALYTIC_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_analytic_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy,
    DTYPE y0, DTYPE z0,
    DTYPE e1y, DTYPE e1z, DTYPE e2y, DTYPE e2z,
    DTYPE C_atan, DTYPE ge1, DTYPE ge2,
    DTYPE b_perp, DTYPE nu, DTYPE a2)
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE shm2 = shr*shr + shi*shi;
    DTYPE alpha = -2.0*PI*s_dev_val;
    DTYPE gam = SQRT(shm2 + alpha*alpha*0.25);
    DTYPE exp_pr = EXP(s0r * ds);
    DTYPE dy = y_pix - y0;
    DTYPE onu = 1.0 - nu;
    DTYPE c1 = b_perp / (2.0*PI);

    for (int k = 0; k < Nz-1; k++){
        int nx = 1 - curr;
        DTYPE gds_ = gam*ds, cg, sg;
        if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
        DTYPE P11r=cg, P11i=-alpha*0.5*sg, P22r=cg, P22i=alpha*0.5*sg;
        DTYPE pi_=(s0i+alpha*0.5)*ds;
        DTYPE epr=exp_pr*COS(pi_), epi=exp_pr*SIN(pi_);
        DTYPE M11r=epr*P11r-epi*P11i, M11i=epr*P11i+epi*P11r;
        DTYPE M22r=epr*P22r-epi*P22i, M22i=epr*P22i+epi*P22r;

        /* H(y, z_k): x-independent for a line along x_lab */
        DTYPE dzv = k*dz - z0;
        DTYPE d1 = dy*e1y + dzv*e1z;
        DTYPE d2 = dy*e2y + dzv*e2z;
        DTYPE r2 = d1*d1 + d2*d2;
        DTYPE r2c = (r2 > a2) ? r2 : a2;
        DTYPE H_val = C_atan * ATAN2(d2, d1);
        if (b_perp > 0.0) {
            DTYPE sm_ux = c1*d1*d2/(2.0*onu*r2c);
            DTYPE sm_uy = -c1*((1.0-2.0*nu)/(4.0*onu)*LOG(r2c)
                               + (d1*d1 - d2*d2)/(4.0*onu*r2c));
            H_val += 2.0*PI*(ge1*sm_ux + ge2*sm_uy);
        }
        DTYPE cH = COS(H_val), sH = SIN(H_val);
        /* sigma_h e^{-iH} feeds D_g; sigma_hbar e^{+iH} feeds D_0 */
        DTYPE P12r=(sbr*cH-sbi*sH)*sg, P12i=(sbi*cH+sbr*sH)*sg;
        DTYPE P21r=(shr*cH+shi*sH)*sg, P21i=(shi*cH-shr*sH)*sg;
        DTYPE M12r=epr*P12r-epi*P12i, M12i=epr*P12i+epi*P12r;
        DTYPE M21r=epr*P21r-epi*P21i, M21i=epr*P21i+epi*P21r;

        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }

    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""


def _typed_analytic(src):
    src = src.replace("ATAN2(", "ATAN2X(").replace("LOG(", "LOGX(")
    if USE_FP32:
        src = src.replace("ATAN2X(", "atan2f(").replace("LOGX(", "logf(")
    else:
        src = src.replace("ATAN2X(", "atan2(").replace("LOGX(", "log(")
    return _typed(src, "tt3d_analytic_batch")


kernel_analytic = _typed_analytic(_ANALYTIC_SRC)

# --- Pristine batched kernel (rocking-curve reference) ---
_PRISTINE_SRC = r"""
#define PI 3.14159265358979323846
extern "C" __global__ void tt3d_pristine_batch(
    DTYPE* Dg_re, DTYPE* Dg_im,
    DTYPE s0r, DTYPE s0i, DTYPE shr, DTYPE shi, DTYPE sbr, DTYPE sbi,
    DTYPE* s_dev_arr, int n_e,
    DTYPE dz, DTYPE shift_px, DTYPE ds, DTYPE dx_,
    int Nz, int Nx_, int Ny_,
    DTYPE sheet_hw, DTYPE fov_hy)
{
    extern __shared__ DTYPE smem[];
    int iy = blockIdx.x / n_e;
    int ie = blockIdx.x % n_e;
    int tid = threadIdx.x, nthreads = blockDim.x;
    DTYPE y_pix = (iy - Ny_/2)*dx_;
    DTYPE s_dev_val = s_dev_arr[ie];
    int out_off = ie * Ny_ * Nx_ + iy * Nx_;
    int curr = 0;
    #define NX_PAD (Nx_ + 1)
    #define SM(buf,arr,i) smem[(buf)*4*NX_PAD + (arr)*NX_PAD + (i)]

    if (FABS(y_pix) >= fov_hy) {
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            Dg_re[out_off+ix] = 0.0; Dg_im[out_off+ix] = 0.0;
        }
        return;
    }
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        DTYPE x_pos = (ix - Nx_/2)*dx_;
        /* sheet_hw > 0: top-hat half-width; sheet_hw < 0: Gaussian
           amplitude profile with sigma_A = -sheet_hw (ID03-style
           condensed line focus). */
        DTYPE beam_val = (sheet_hw > (DTYPE)0.0)
            ? ((FABS(x_pos) < sheet_hw) ? 1.0 : 0.0)
            : EXP(-x_pos*x_pos/(2.0*sheet_hw*sheet_hw));
        SM(0,0,ix)=beam_val; SM(0,1,ix)=0.0f; SM(0,2,ix)=0.0f; SM(0,3,ix)=0.0f;
    }
    __syncthreads();

    DTYPE alpha = -2.0*PI*s_dev_val;
    DTYPE shm2=shr*shr+shi*shi;
    DTYPE gam=SQRT(shm2+alpha*alpha*0.25);
    DTYPE gds_=gam*ds, cg, sg;
    if(gam>1e-30){cg=COS(gds_);sg=SIN(gds_)/gam;} else{cg=1;sg=ds;}
    DTYPE P11r=cg,P11i=-alpha*0.5*sg, P22r=cg,P22i=alpha*0.5*sg;
    DTYPE P12r=sbr*sg,P12i=sbi*sg, P21r=shr*sg,P21i=shi*sg;
    DTYPE pr=s0r*ds, pi_=(s0i+alpha*0.5)*ds;
    DTYPE epr=EXP(pr)*COS(pi_), epi=EXP(pr)*SIN(pi_);
    DTYPE M11r=epr*P11r-epi*P11i,M11i=epr*P11i+epi*P11r;
    DTYPE M12r=epr*P12r-epi*P12i,M12i=epr*P12i+epi*P12r;
    DTYPE M21r=epr*P21r-epi*P21i,M21i=epr*P21i+epi*P21r;
    DTYPE M22r=epr*P22r-epi*P22i,M22i=epr*P22i+epi*P22r;

    for (int k=0; k<Nz-1; k++){
        int nx = 1 - curr;
        for (int ix = tid; ix < Nx_; ix += nthreads) {
            DTYPE src0=(DTYPE)ix+shift_px;
            int i0=(int)FLOOR(src0); DTYPE f0=src0-(DTYPE)i0;
            DTYPE d0r,d0i;
            if(i0>=0&&i0+1<Nx_){d0r=(1-f0)*SM(curr,0,i0)+f0*SM(curr,0,i0+1);
                d0i=(1-f0)*SM(curr,1,i0)+f0*SM(curr,1,i0+1);}
            else if(i0>=0&&i0<Nx_){d0r=(1-f0)*SM(curr,0,i0);d0i=(1-f0)*SM(curr,1,i0);}
            else{d0r=0;d0i=0;}
            DTYPE srcg=(DTYPE)ix-shift_px;
            int ig=(int)FLOOR(srcg); DTYPE fg=srcg-(DTYPE)ig;
            DTYPE dgr,dgi;
            if(ig>=0&&ig+1<Nx_){dgr=(1-fg)*SM(curr,2,ig)+fg*SM(curr,2,ig+1);
                dgi=(1-fg)*SM(curr,3,ig)+fg*SM(curr,3,ig+1);}
            else if(ig>=0&&ig<Nx_){dgr=(1-fg)*SM(curr,2,ig);dgi=(1-fg)*SM(curr,3,ig);}
            else{dgr=0;dgi=0;}
            SM(nx,0,ix)=M11r*d0r-M11i*d0i+M12r*dgr-M12i*dgi;
            SM(nx,1,ix)=M11r*d0i+M11i*d0r+M12r*dgi+M12i*dgr;
            SM(nx,2,ix)=M21r*d0r-M21i*d0i+M22r*dgr-M22i*dgi;
            SM(nx,3,ix)=M21r*d0i+M21i*d0r+M22r*dgi+M22i*dgr;
        }
        __syncthreads(); curr=nx;
    }
    for (int ix = tid; ix < Nx_; ix += nthreads) {
        Dg_re[out_off+ix]=(DTYPE)SM(curr,2,ix);
        Dg_im[out_off+ix]=(DTYPE)SM(curr,3,ix);
    }
    #undef NX_PAD
    #undef SM
}
"""
kernel_pristine = _typed(_PRISTINE_SRC, "tt3d_pristine_batch")


# ===========================================================================
#  Solvers
# ===========================================================================
def _grid_alloc(config, n_e, params=P):
    Nx, Ny, dz, ds, spx = config.grid(params)
    Dg_re = cp.empty(n_e * Ny * Nx, dtype=CP_DTYPE)
    Dg_im = cp.empty(n_e * Ny * Nx, dtype=CP_DTYPE)
    smem = 2 * 4 * (Nx + 1) * (4 if USE_FP32 else 8)
    return Nx, Ny, dz, ds, spx, Dg_re, Dg_im, smem


def _prep_kernel(kernel, smem):
    """Opt in to the large dynamic shared-memory carveout when needed."""
    if smem > 48 * 1024:
        limit = cp.cuda.Device().attributes.get(
            "MaxSharedMemoryPerBlockOptin", 99 * 1024)
        if smem > limit:
            raise MemoryError(
                f"kernel needs {smem/1024:.0f} KB shared memory per block "
                f"but the device allows {limit/1024:.0f} KB; reduce the "
                f"x-grid (beam width / crystal thickness) or use FP32")
        kernel.max_dynamic_shared_size_bytes = smem
    return kernel


def _analytic_params(seg_list, params=P):
    """Closed-form H parameters for a single straight line along x_lab.

    Returns None if the geometry does not qualify (then the H-grid path
    is used).  Frame construction mirrors compute_H_grid exactly.
    """
    if len(seg_list) != 1:
        return None
    ra, rb, bv = (np.asarray(v, float) for v in seg_list[0])
    t = rb - ra
    L = np.linalg.norm(t)
    if L < 1e-20:
        return None
    xi = t / L
    if abs(abs(xi[0]) - 1.0) > 1e-9:
        return None
    core = (ra + rb) / 2.0
    gv = params["g_vec"]
    b_screw = float(np.dot(bv, xi))
    b_perp_vec = bv - b_screw * xi
    b_perp = np.linalg.norm(b_perp_vec)
    if b_perp > 1e-20:
        e1 = b_perp_vec / b_perp
    elif abs(xi[2]) > 0.9:
        e1 = np.array([1., 0., 0.])
    elif abs(xi[0]) < 0.9:
        e1 = np.cross(xi, [1, 0, 0])
        e1 /= np.linalg.norm(e1)
    else:
        e1 = np.cross(xi, [0, 1, 0])
        e1 /= np.linalg.norm(e1)
    e2 = np.cross(xi, e1)
    if abs(e1[0]) > 1e-9 or abs(e2[0]) > 1e-9:
        return None
    C_atan = np.dot(gv, e1) * b_perp + np.dot(gv, xi) * b_screw
    return dict(y0=core[1], z0=core[2],
                e1y=e1[1], e1z=e1[2], e2y=e2[1], e2z=e2[2],
                C_atan=C_atan, ge1=np.dot(gv, e1), ge2=np.dot(gv, e2),
                b_perp=b_perp, nu=params["nu"],
                a2=params["a_core"] ** 2)


def solve_dislocation(seg_list, s_dev_arr, config=DEFAULT_CONFIG, params=P,
                      H_gpu=None, batch=64):
    """Full-field TT solve for a dislocation at each s_dev in s_dev_arr.

    For the paper's geometry (single straight line along x_lab) the
    displacement phase is evaluated in closed form inside the kernel --
    no 3D H grid, no depth aliasing.  Other geometries fall back to the
    precomputed H grid.  Returns (Dg_stack, Nx, Ny) with Dg_stack
    complex (n_e, Ny, Nx) on GPU -- the *physical* exit wave.
    """
    s_dev_arr = np.atleast_1d(np.asarray(s_dev_arr, dtype=float))
    n_e = len(s_dev_arr)
    Nx, Ny, dz, ds, spx, _, _, smem = _grid_alloc(config, 1, params)

    ap = None if H_gpu is not None else _analytic_params(seg_list, params)
    if ap is None and H_gpu is None:
        h_bytes = Nx * Ny * config.Nz * (4 if USE_FP32 else 8)
        free_mem = cp.cuda.Device().mem_info[0]
        if h_bytes > 0.7 * free_mem:
            raise MemoryError(f"H grid needs {h_bytes/1e9:.1f} GB, "
                              f"{free_mem/1e9:.1f} GB free")
        H_gpu = compute_H_grid(seg_list, config, params)

    out = cp.empty((n_e, Ny, Nx), dtype=CP_COMPLEX)
    common = (
        DTYPE(params["sig0"].real), DTYPE(params["sig0"].imag),
        DTYPE(params["sig_h"].real), DTYPE(params["sig_h"].imag),
        DTYPE(params["sig_hbar"].real), DTYPE(params["sig_hbar"].imag))
    for bs in range(0, n_e, batch):
        be = min(bs + batch, n_e)
        nb = be - bs
        Dg_re = cp.empty(nb * Ny * Nx, dtype=CP_DTYPE)
        Dg_im = cp.empty(nb * Ny * Nx, dtype=CP_DTYPE)
        sd_gpu = cp.asarray(s_dev_arr[bs:be].astype(DTYPE))
        tail = (sd_gpu, np.int32(nb),
                DTYPE(dz), DTYPE(spx), DTYPE(ds), DTYPE(config.dx),
                np.int32(config.Nz), np.int32(Nx), np.int32(Ny),
                DTYPE(config.sheet_arg()),
                DTYPE(config.beam_width / 2.0))
        if ap is not None:
            _prep_kernel(kernel_analytic, smem)((Ny * nb,), (BLOCK,), (
                Dg_re, Dg_im, *common, *tail,
                DTYPE(ap["y0"]), DTYPE(ap["z0"]),
                DTYPE(ap["e1y"]), DTYPE(ap["e1z"]),
                DTYPE(ap["e2y"]), DTYPE(ap["e2z"]),
                DTYPE(ap["C_atan"]), DTYPE(ap["ge1"]), DTYPE(ap["ge2"]),
                DTYPE(ap["b_perp"]), DTYPE(ap["nu"]), DTYPE(ap["a2"])),
                shared_mem=smem)
        else:
            _prep_kernel(kernel_fullfield, smem)((Ny * nb,), (BLOCK,), (
                Dg_re, Dg_im, *common, *tail, H_gpu), shared_mem=smem)
        out[bs:be] = (Dg_re.reshape(nb, Ny, Nx)
                      + 1j * Dg_im.reshape(nb, Ny, Nx)).astype(CP_COMPLEX)
        del Dg_re, Dg_im
    return out, Nx, Ny


def solve_pristine(s_dev_arr, config=DEFAULT_CONFIG, params=P):
    """Pristine-crystal TT solve, batched over s_dev.  Returns (stack, Nx, Ny)."""
    s_dev_arr = np.atleast_1d(np.asarray(s_dev_arr, dtype=float))
    n_e = len(s_dev_arr)
    Nx, Ny, dz, ds, spx, Dg_re, Dg_im, smem = _grid_alloc(config, n_e, params)
    sd_gpu = cp.asarray(s_dev_arr.astype(DTYPE))
    _prep_kernel(kernel_pristine, smem)((Ny * n_e,), (BLOCK,), (
        Dg_re, Dg_im,
        DTYPE(params["sig0"].real), DTYPE(params["sig0"].imag),
        DTYPE(params["sig_h"].real), DTYPE(params["sig_h"].imag),
        DTYPE(params["sig_hbar"].real), DTYPE(params["sig_hbar"].imag),
        sd_gpu, np.int32(n_e),
        DTYPE(dz), DTYPE(spx), DTYPE(ds), DTYPE(config.dx),
        np.int32(config.Nz), np.int32(Nx), np.int32(Ny),
        DTYPE(config.sheet_arg()), DTYPE(config.beam_width / 2.0)),
        shared_mem=smem)
    stack = (Dg_re.reshape(n_e, Ny, Nx)
             + 1j * Dg_im.reshape(n_e, Ny, Nx)).astype(CP_COMPLEX)
    return stack, Nx, Ny


# ===========================================================================
#  Exit-wave disk cache
# ===========================================================================
def _wave_key(dtype, sign, s_dev_arr, config, params):
    payload = dict(dtype=dtype, sign=sign,
                   s_dev=[float(s) for s in np.atleast_1d(s_dev_arr)],
                   config=config.key(),
                   E_keV=params["E_keV"], reflection=params["label"],
                   precision="fp32" if USE_FP32 else "fp64",
                   version=8)   # v8: Geometry N ((001) plate, line [101])
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def exit_waves(dtype, sign, s_dev_arr, config=DEFAULT_CONFIG, params=P,
               cache=True, H_gpu=None):
    """Physical exit waves for a dislocation variant at each s_dev.

    Disk-cached in data/waves/ so independent figure scripts share solves.
    Returns (Dg_stack complex64 GPU (n_e, Ny, Nx), Nx, Ny).
    """
    if cache and WAVE_DIR is None:
        raise RuntimeError(
            "no exit-wave cache directory: call set_cache_dir(path), or "
            "pass cache=False to solve without caching")
    key = _wave_key(dtype, sign, s_dev_arr, config, params)
    path = (os.path.join(WAVE_DIR, f"{dtype.lower()}_{sign:+d}_{key}.npy")
            if cache else None)
    if cache and os.path.exists(path):
        arr = np.load(path)
        return cp.asarray(arr), arr.shape[2], arr.shape[1]
    seg = make_segment(dtype, sign, config, params)
    stack, Nx, Ny = solve_dislocation(seg, s_dev_arr, config, params,
                                      H_gpu=H_gpu)
    if cache:
        os.makedirs(WAVE_DIR, exist_ok=True)
        np.save(path, cp.asnumpy(stack).astype(np.complex64))
    return stack, Nx, Ny


def to_cpu(a):
    return cp.asnumpy(a)


# ===========================================================================
#  Fourier-optics imaging stage
# ===========================================================================
_pupil_cache = {}


def make_pupil(Nx, Ny, ell=0, offset_bfp=(0.0, 0.0), dx=DX, params=P):
    """Pupil on the fftshifted frequency grid.

    ell = 0 gives the open NA-limited aperture; ell != 0 multiplies in the
    spiral exp(i*ell*phi).  `offset_bfp` displaces the spiral singularity by
    (dx, dy) metres in the back focal plane (x_bfp = lambda*f*fx); the
    aperture edge stays centred, matching a translated plate whose clear
    aperture is much larger than the NA disc.
    """
    key = (Nx, Ny, ell, tuple(np.round(np.asarray(offset_bfp) * 1e9)), dx)
    if key in _pupil_cache:
        return _pupil_cache[key]
    fx = cp.fft.fftshift(cp.fft.fftfreq(Nx, dx))
    fy = cp.fft.fftshift(cp.fft.fftfreq(Ny, dx))
    FX, FY = cp.meshgrid(fx, fy)
    circ = (cp.sqrt(FX ** 2 + FY ** 2)
            <= NA_DFXM / params["lam"]).astype(CP_DTYPE)
    if ell == 0:
        pupil = circ.astype(CP_COMPLEX)
    else:
        dfx = offset_bfp[0] / (params["lam"] * F_LENS)
        dfy = offset_bfp[1] / (params["lam"] * F_LENS)
        phi = cp.arctan2(FY - dfy, FX - dfx)
        pupil = (circ * cp.exp(1j * ell * phi)).astype(CP_COMPLEX)
        if offset_bfp[0] == 0.0 and offset_bfp[1] == 0.0:
            # The frequency bin containing the singularity averages
            # exp(i*ell*phi) to ~0 over the bin; arctan2(0,0)=0 would
            # instead pass the object's mean field untouched.
            pupil[Ny // 2, Nx // 2] = 0.0
    _pupil_cache[key] = pupil
    return pupil


def make_chromatic_pupil(Nx, Ny, ell, delta_E, dx=DX, params=P):
    """Pupil seen by an energy offset delta_E = dE/E in the diffracted beam.

    At fixed incidence the diffracted carrier direction walks by
    sin(2 theta_B) * delta_E, so the whole exit-wave spectrum is displaced
    in the fixed BFP by Delta_f = sin(2 theta_B) * delta_E / lambda; in the
    envelope gauge (spectrum pinned to the axis) this is equivalent to
    displacing BOTH the aperture stop and the spiral singularity by
    -Delta_f.  This models the worst case of non-dispersed illumination:
    energies whose walk exceeds the NA disc vignette away entirely (the
    lens acts as its own ~1.3e-4 bandwidth filter).
    """
    dfx = params["sin_2tB"] * delta_E / params["lam"]
    key = (Nx, Ny, ell, "chrom", round(float(dfx) * 1e-3), dx)
    if key in _pupil_cache:
        return _pupil_cache[key]
    fx = cp.fft.fftshift(cp.fft.fftfreq(Nx, dx))
    fy = cp.fft.fftshift(cp.fft.fftfreq(Ny, dx))
    FX, FY = cp.meshgrid(fx, fy)
    circ = (cp.sqrt((FX - dfx) ** 2 + FY ** 2)
            <= NA_DFXM / params["lam"]).astype(CP_DTYPE)
    if ell == 0:
        pupil = circ.astype(CP_COMPLEX)
    else:
        phi = cp.arctan2(FY, FX - dfx)
        pupil = (circ * cp.exp(1j * ell * phi)).astype(CP_COMPLEX)
    _pupil_cache[key] = pupil
    return pupil


def gauss_hermite_offsets(sigma, n=8):
    """2D Gauss-Hermite quadrature for an isotropic Gaussian of rms sigma.

    Returns (offsets (n*n, 2), weights (n*n,)) with weights summing to 1 --
    the correct incoherent average over a 2D Gaussian source, including
    the polar Jacobian that ad-hoc ring sampling omits.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    X, Y = np.meshgrid(x, x)
    WX, WY = np.meshgrid(w, w)
    offs = np.stack([X.ravel() * sigma, Y.ravel() * sigma], axis=1)
    wts = (WX * WY).ravel()
    return offs, wts / wts.sum()


def apply_pupils(Dg, pupils):
    """Image intensities for one exit wave through many pupils.

    Dg: (Ny, Nx) complex GPU exit wave.  pupils: list of (Ny, Nx) pupils on
    the fftshifted grid.  The object spectrum is computed once; each pupil
    costs one batched IFFT slice.  Returns (n_pupils, Ny, Nx) float32 GPU.
    """
    spec = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(Dg)))
    stack = cp.stack([spec * pup for pup in pupils])
    img = cp.fft.ifftshift(stack, axes=(-2, -1))
    img = cp.fft.ifft2(img, axes=(-2, -1))
    img = cp.fft.fftshift(img, axes=(-2, -1))
    return cp.abs(img) ** 2


def image_intensity(Dg_stack, ell=0, offset_bfp=(0.0, 0.0), dx=DX, params=P):
    """Batched imaging of an exit-wave stack through one pupil.

    Dg_stack: (n_e, Ny, Nx) complex GPU.  Returns (n_e, Ny, Nx) intensity.
    """
    n_e, Ny, Nx = Dg_stack.shape
    pup = make_pupil(Nx, Ny, ell, offset_bfp, dx, params)
    sh = cp.fft.fftshift(Dg_stack, axes=(-2, -1))
    spec = cp.fft.fftshift(cp.fft.fft2(sh, axes=(-2, -1)), axes=(-2, -1))
    img = cp.fft.ifftshift(spec * pup[None], axes=(-2, -1))
    img = cp.fft.ifft2(img, axes=(-2, -1))
    img = cp.fft.fftshift(img, axes=(-2, -1))
    return cp.abs(img) ** 2


# ===========================================================================
#  Analysis metrics
# ===========================================================================
def winding_number(Dg, dx=DX, r_m=1.0e-6, n_samples=720, center=(0.0, 0.0)):
    """Phase winding n = (1/2pi) * closed-loop integral of grad(arg Dg) . dl.

    Samples the *complex field* bilinearly on a circle of radius r_m about
    `center` (metres, lab frame), takes angles, unwraps, and closes the
    loop.  Bilinear sampling of the field (not the wrapped phase) avoids
    the branch-cut artefacts of nearest-pixel phase sampling.
    """
    img = cp.asnumpy(Dg) if isinstance(Dg, cp.ndarray) else np.asarray(Dg)
    Ny, Nx = img.shape
    ang = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = (center[0] + r_m * np.cos(ang)) / dx + Nx / 2
    ys = (center[1] + r_m * np.sin(ang)) / dx + Ny / 2
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    x0 = np.clip(x0, 0, Nx - 2)
    y0 = np.clip(y0, 0, Ny - 2)
    f = (img[y0, x0] * (1 - fx) * (1 - fy)
         + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy
         + img[y0 + 1, x0 + 1] * fx * fy)
    phi = np.angle(f)
    dphi = np.diff(np.unwrap(phi))
    closing = np.angle(np.exp(1j * (phi[0] - phi[-1])))
    return float((np.sum(dphi) + closing) / (2 * np.pi))


def radial_profile(img, dx=DX, center_px=None):
    """Azimuthally averaged radial intensity profile.

    Returns (r_um, profile).  Vectorised bincount, CPU numpy.
    """
    img = cp.asnumpy(img) if isinstance(img, cp.ndarray) else np.asarray(img)
    Ny, Nx = img.shape
    cy, cx = (Ny // 2, Nx // 2) if center_px is None else center_px
    Y, X = np.ogrid[:Ny, :Nx]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    rmax = min(cy, cx)
    mask = R < rmax
    counts = np.bincount(R[mask].ravel(), minlength=rmax)
    sums = np.bincount(R[mask].ravel(), weights=img[mask].ravel(),
                       minlength=rmax)
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)[:rmax]
    return np.arange(rmax) * dx * 1e6, prof


def ssim_pair(img_a, img_b):
    """SSIM between two images normalised to their joint maximum."""
    from skimage.metrics import structural_similarity
    a = cp.asnumpy(img_a) if isinstance(img_a, cp.ndarray) else np.asarray(img_a)
    b = cp.asnumpy(img_b) if isinstance(img_b, cp.ndarray) else np.asarray(img_b)
    mx = max(a.max(), b.max(), 1e-30)
    return float(structural_similarity(a / mx, b / mx, data_range=1.0))


def com_maps(dtype, sign, s_dev_arr, ells=(0, 2), config=DEFAULT_CONFIG,
             params=P, batch=16):
    """Centre-of-mass maps of the rocking curve, one per pupil charge.

    COM(x, y) = sum_s [ Delta_phi(s) I_ell(x, y, s) ] / sum_s I_ell(x, y, s)
    accumulated in FP64 on GPU across batched TT solves.  Returns
    {ell: numpy (Ny, Nx) map in urad}, plus the integrated rocking curve
    {ell: numpy (n_s,)} for reuse.
    """
    s_dev_arr = np.asarray(s_dev_arr, dtype=float)
    seg = make_segment(dtype, sign, config, params)
    Nx, Ny, _, _, _ = config.grid(params)
    # analytic in-kernel H when the geometry qualifies; H grid otherwise
    H_gpu = (None if _analytic_params(seg, params) is not None
             else compute_H_grid(seg, config, params))
    phi_urad = sdev_to_urad(s_dev_arr, params)
    sums_wI = {ell: cp.zeros((Ny, Nx), dtype=cp.float64) for ell in ells}
    sums_I = {ell: cp.zeros((Ny, Nx), dtype=cp.float64) for ell in ells}
    rock = {ell: np.zeros(len(s_dev_arr)) for ell in ells}
    for bs in range(0, len(s_dev_arr), batch):
        be = min(bs + batch, len(s_dev_arr))
        stack, _, _ = solve_dislocation(seg, s_dev_arr[bs:be], config, params,
                                        H_gpu=H_gpu)
        for ell in ells:
            I = image_intensity(stack, ell=ell, dx=config.dx, params=params)
            I64 = I.astype(cp.float64)
            w = cp.asarray(phi_urad[bs:be])[:, None, None]
            sums_wI[ell] += (I64 * w).sum(axis=0)
            sums_I[ell] += I64.sum(axis=0)
            rock[ell][bs:be] = cp.asnumpy(I64.sum(axis=(1, 2)))
            del I, I64
        del stack
    if H_gpu is not None:
        del H_gpu
    cp.get_default_memory_pool().free_all_blocks()
    com = {ell: cp.asnumpy(sums_wI[ell] / cp.maximum(sums_I[ell], 1e-300))
           for ell in ells}
    weight = {ell: cp.asnumpy(sums_I[ell]) for ell in ells}
    return com, rock, weight


def crop_slices(Nx, Ny, dx=DX, half_um=25.0):
    """Slices cropping the (Ny, Nx) grid to +/- half_um about the centre,
    plus the imshow extent [y0, y1, x0, x1] in um for the transposed image
    (x_lab vertical, y_lab horizontal -- the display convention of the
    figure set)."""
    hpx = int(round(half_um * 1e-6 / dx))
    sy = slice(Ny // 2 - hpx, Ny // 2 + hpx)
    sx = slice(Nx // 2 - hpx, Nx // 2 + hpx)
    ext = [-half_um, half_um, -half_um, half_um]
    return sy, sx, ext


def report_setup():
    """Print the derived physics constants (mirrors manuscript Table 1)."""
    print(f"{P['label']} at {P['E_keV']} keV")
    print(f"  lambda   = {P['lam']*1e10:.4f} A")
    print(f"  theta_B  = {np.degrees(P['theta_B']):.2f} deg")
    print(f"  xi_g     = {P['xi_g']*1e6:.1f} um")
    print(f"  |F_g|    = {abs(P['F_g']):.3f}")
    cfg = DEFAULT_CONFIG
    _, _, dz, _, spx = cfg.grid()
    print(f"  t        = {cfg.t_crystal*1e6:.1f} um "
          f"({cfg.t_crystal/P['xi_g']:.2f} xi_g)")
    print(f"  Nz       = {cfg.Nz},  dz = {dz*1e9:.1f} nm "
          f"(exact advection: {spx:.3f} px/step)")
    print(f"  WB s_dev = 3/xi_g -> Delta_phi = "
          f"{sdev_to_urad(S_DEV_WB):.2f} urad")
    print(f"  NA = {NA_DFXM:.1e}, delta_res = {0.61*P['lam']/NA_DFXM*1e6:.2f} um, "
          f"R_BFP = {R_BFP*1e6:.0f} um")
    for dt in DISLOC_TYPES:
        print(f"  g.b({dt.lower():5s}) = {g_dot_b(dt):+.3f}")


if __name__ == "__main__":
    report_setup()
