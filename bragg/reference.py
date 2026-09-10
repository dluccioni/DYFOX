"""Perfect-crystal answers in closed form, to check the solver against.

Two of them, computed differently on purpose. The semi-infinite
reflectivity is the stationary point of the Riccati equation and falls
out of a quadratic; the finite-thickness one integrates that same
equation numerically with SciPy. Neither shares code or a library with
the grid solver, so agreement between them and it means something.

`darwin_width_urad` gives the width of the total-reflection domain,
which is the natural angular unit for anything in reflection geometry.
"""

import numpy as np

def reflectivity_semi_infinite(s_dev, xtal):
    """Amplitude reflectivity of a semi-infinite perfect crystal.

    Stationary point of the Riccati equation
        dR/dz = -(1/sin tB) [ sigma_h + (2 sigma_0 + i alpha) R
                              + sigma_hbar R^2 ],
    i.e. the root of the quadratic with |R| <= 1.
    """
    s_dev = np.atleast_1d(np.asarray(s_dev, float))
    s0, sh, sb = complex(xtal.sig0), complex(xtal.sig_h), complex(xtal.sig_hbar)
    alpha = -2.0 * np.pi * s_dev
    B = 2.0 * s0 + 1j * alpha
    disc = np.sqrt(B ** 2 - 4.0 * sh * sb + 0j)
    r1 = (-B + disc) / (2.0 * sb)
    r2 = (-B - disc) / (2.0 * sb)
    return np.where(np.abs(r1) <= np.abs(r2), r1, r2)


def reflectivity_riccati(s_dev, t_crystal, xtal, rtol=1e-11,
                         atol=1e-14):
    """Finite-thickness reference: integrate the Riccati ODE with scipy.

    Independent of the grid solver in both discretisation and library,
    which is what makes agreement between them worth anything.

    The equation is quadratic in R, so far out in the tails an adaptive
    integrator will probe a step where R runs away before rejecting it.
    Those probes overflow harmlessly and the accepted steps are
    unaffected, so the warnings they raise are suppressed rather than
    shown to the caller as if something had gone wrong.
    """
    from scipy.integrate import solve_ivp
    s0, sh, sb = complex(xtal.sig0), complex(xtal.sig_h), complex(xtal.sig_hbar)
    out = []
    for sd in np.atleast_1d(np.asarray(s_dev, float)):
        alpha = -2.0 * np.pi * float(sd)

        def rhs(z, R):
            R = R[0] + 1j * R[1]
            dR = -(sh + (2.0 * s0 + 1j * alpha) * R + sb * R * R) / xtal.sin_tB
            return [dR.real, dR.imag]

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sol = solve_ivp(rhs, [t_crystal, 0.0], [0.0, 0.0],
                            rtol=rtol, atol=atol, dense_output=False,
                            method="DOP853")
        out.append(sol.y[0, -1] + 1j * sol.y[1, -1])
    return np.array(out)


def darwin_width_urad(xtal):
    """Full width of the total-reflection domain, symmetric Bragg.

    w = 2 |C sqrt(chi_h chi_hbar)| / sin(2 theta_B), expressed through the
    solver's own sigma (sigma = -i r_e lambda F / V = i pi chi / lambda).
    """
    sh, sb = complex(xtal.sig_h), complex(xtal.sig_hbar)
    chi_prod = np.sqrt(sh * sb) * xtal.lam / (1j * np.pi)
    return float(2.0 * abs(chi_prod) / xtal.sin_2tB * 1e6)
