"""Materials, and the diffraction parameters a reflection implies.

A `Material` is a substance: lattice parameter, basis, atomic number,
Poisson ratio, form factors. `CrystalParams` is a material plus a
reflection plus a photon energy, and carries the wavelength, Bragg
angle, susceptibility couplings and extinction length the solvers use.

    xtal = CrystalParams.from_reflection(DIAMOND, (4, 0, 0), 17.0)

The formulae assume a cubic lattice and a symmetric geometry.
"""

from dataclasses import dataclass, replace

import numpy as np

from units import HC_EV_M, R_E

# The diamond-cubic basis, eight atoms in the conventional cell.
DIAMOND_CUBIC = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                          [.25, .25, .25], [.75, .75, .25],
                          [.75, .25, .75], [.25, .75, .75]])


@dataclass(frozen=True)
class Material:
    """A substance, independent of any beam or reflection.

    `f0` holds the Cromer-Mann form factor per reflection, keyed by
    (h, k, l). `burgers_mag` defaults to the 1/2<110> Burgers vector of
    fcc and diamond-cubic; `core_radius`, where the Volterra field is
    regularised, defaults to one Burgers vector.
    """

    name: str
    a_lat: float                       # lattice parameter (m)
    basis: np.ndarray                  # fractional coordinates, (N, 3)
    Z: float                           # atomic number, for forward scattering
    nu: float                          # isotropic Poisson ratio
    f0: dict                           # {(h, k, l): f0(q)}
    burgers_mag: float = None
    core_radius: float = None

    def burgers(self):
        if self.burgers_mag is not None:
            return self.burgers_mag
        return self.a_lat * np.sqrt(2) / 2

    def core(self):
        return self.core_radius if self.core_radius is not None \
            else self.burgers()


# nu is the Voigt-Reuss-Hill value from C11 = 1079, C12 = 124,
# C44 = 578 GPa.
DIAMOND = Material(
    name="diamond",
    a_lat=3.567e-10,
    basis=DIAMOND_CUBIC,
    Z=6.0,
    nu=0.069,
    f0={(4, 0, 0): 1.589, (2, 2, 0): 1.961},
)


def lab_frame(normal_hkl, g_hkl):
    """Rows x, y, z of a lab frame with z along `normal_hkl`, x along g.

    z is the surface normal, x is g projected into the surface, y
    completes a right-handed set. Raises if g is parallel to z.
    """
    z = np.asarray(normal_hkl, float)
    z = z / np.linalg.norm(z)
    g = np.asarray(g_hkl, float)
    x = g - np.dot(g, z) * z
    if np.linalg.norm(x) < 1e-12:
        raise ValueError("g is parallel to the surface normal: that is a "
                         "Bragg geometry, not a symmetric Laue one")
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


@dataclass(frozen=True)
class CrystalParams:
    """A material, a reflection and an energy, worked through.

    Attribute names are the ones the solvers use: `lam`, `theta_B`,
    `sig0`, `sig_h`, `sig_hbar`, `xi_g`, and the trig shorthands
    `sin_tB`, `cos_tB`, `tan_tB`, `sin_2tB`.
    """

    material: Material
    hkl: tuple
    E_keV: float
    U_lab: np.ndarray
    label: str
    lam: float
    d_hkl: float
    theta_B: float
    F_g: complex
    sig0: complex
    sig_h: complex
    sig_hbar: complex
    xi_g: float
    sin_tB: float
    cos_tB: float
    tan_tB: float
    sin_2tB: float
    a_lat: float
    nu: float
    g_vec: np.ndarray
    b_mag: float
    a_core: float

    @classmethod
    def from_reflection(cls, material, hkl, E_keV, U_lab=None, f0=None,
                        f_prime=0.0, f_dprime=0.0, label=None):
        """Diffraction parameters for one reflection at one energy.

        `f_prime` and `f_dprime` are the anomalous corrections, supplied
        by the caller and entering as `f + f' - i f''`. `U_lab` sets the
        lab frame, `f0` overrides the material's tabulated value.
        """
        hkl = tuple(int(v) for v in hkl)
        g_hkl = np.array(hkl)
        if U_lab is None:
            U_lab = np.eye(3)
        if f0 is None:
            if hkl not in material.f0:
                raise KeyError(
                    f"{material.name} has no f0 for {hkl}; pass f0=... or "
                    f"add it to the material (have {sorted(material.f0)})")
            f0 = material.f0[hkl]

        a_lat = material.a_lat
        f_atom = f0 + f_prime - 1j * f_dprime
        lam = HC_EV_M / (E_keV * 1e3)
        b_mag = material.burgers()
        d_hkl = a_lat / np.sqrt(np.sum(g_hkl ** 2))
        theta_B = np.arcsin(lam / (2 * d_hkl))
        g_vec = U_lab @ (g_hkl / a_lat)      # lab frame, 1/m, no 2 pi
        V_cell = a_lat ** 3
        # Forward scattering factor f(0), not the reflection's f0(q).
        f_forward = material.Z + f_prime - 1j * f_dprime
        F_0 = f_forward * len(material.basis)
        S_g = np.sum(np.exp(2j * np.pi * material.basis @ g_hkl))
        F_g = f_atom * S_g
        # F_{-g} = f conj(S_g): the geometric sum conjugates, the
        # atomic factor does not.
        F_mg = f_atom * np.conj(S_g)
        sig0 = -1j * R_E * lam * F_0 / V_cell
        sig_h = -1j * R_E * lam * F_g / V_cell
        sig_hbar = -1j * R_E * lam * F_mg / V_cell
        xi_g = np.pi / abs(sig_h)
        if label is None:
            label = f"{material.name.capitalize()}({''.join(str(v) for v in hkl)})"
        return cls(material=material, hkl=hkl, E_keV=float(E_keV),
                   U_lab=U_lab, label=label, lam=lam, d_hkl=d_hkl,
                   theta_B=theta_B, F_g=F_g, sig0=sig0, sig_h=sig_h,
                   sig_hbar=sig_hbar, xi_g=xi_g,
                   sin_tB=np.sin(theta_B), cos_tB=np.cos(theta_B),
                   tan_tB=np.tan(theta_B), sin_2tB=np.sin(2 * theta_B),
                   a_lat=a_lat, nu=material.nu, g_vec=g_vec,
                   b_mag=b_mag, a_core=material.core())

    def at_energy(self, E_keV, **kw):
        """The same crystal and reflection at a different photon energy."""
        return CrystalParams.from_reflection(
            self.material, self.hkl, E_keV, U_lab=self.U_lab,
            label=self.label, **kw)

    def with_frame(self, U_lab):
        """The same crystal mounted in a different lab frame.

        Only `U_lab` and `g_vec` change; the scalars are unchanged.
        """
        U_lab = np.asarray(U_lab, float)
        g_hkl = np.array(self.hkl)
        return replace(self, U_lab=U_lab,
                       g_vec=U_lab @ (g_hkl / self.a_lat))

    def key(self):
        """The identity of this crystal, for a cache key."""
        return {"material": self.material.name, "hkl": list(self.hkl),
                "E_keV": self.E_keV, "label": self.label}

    def __getitem__(self, name):
        """Attribute access by name, so `xtal["lam"]` also works."""
        return getattr(self, name)

    def report(self):
        """The derived numbers, the way Table 1 lists them."""
        lines = [f"{self.label} at {self.E_keV} keV",
                 f"  lambda   = {self.lam * 1e10:.4f} A",
                 f"  theta_B  = {np.degrees(self.theta_B):.2f} deg",
                 f"  xi_g     = {self.xi_g * 1e6:.1f} um",
                 f"  |F_g|    = {abs(self.F_g):.3f}",
                 f"  Re sig0  = {self.sig0.real:+.1f} /m"]
        return "\n".join(lines)
