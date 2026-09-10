"""Perfect-crystal answers in closed form, to check the solver against.

    reflectivity_semi_infinite  stationary point of the Riccati
                                equation, from a quadratic
    reflectivity_riccati        that equation integrated to finite
                                thickness with SciPy
    darwin_width_urad           width of the total-reflection domain

Neither reflectivity shares code with the grid solver.
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

    Overflow warnings from the integrator's rejected trial steps are
    suppressed; accepted steps, and every returned value, are
    unaffected.
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
