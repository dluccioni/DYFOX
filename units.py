"""Rocking angle to deviation parameter.

    deviation_from_rad    angle in radians      phi * sin_2tB / lam
    deviation_from_urad   angle in microradians urad * 1e-6 * sin_2tB / lam
    urad_from_deviation   the inverse           s * lam / sin_2tB * 1e6

The three expressions are kept separate.
"""

import numpy as np

# Shared with the crystal calculation.
R_E = 2.8179403262e-15          # classical electron radius (m)
HC_EV_M = 12398.419e-10         # hc (eV * m)


def param(xtal, key):
    """Crystal parameter by name, from a mapping or an object."""
    try:
        return xtal[key]
    except (TypeError, IndexError, KeyError):
        return getattr(xtal, key)


def deviation_from_rad(phi, xtal):
    """Deviation parameter (1/m) for a rocking angle in radians."""
    return np.asarray(phi) * param(xtal, "sin_2tB") / param(xtal, "lam")


def deviation_from_urad(phi_urad, xtal):
    """Deviation parameter (1/m) for a rocking angle in microradians."""
    return (np.asarray(phi_urad) * 1e-6 * param(xtal, "sin_2tB")
            / param(xtal, "lam"))


def urad_from_deviation(s_dev, xtal):
    """Rocking angle (urad) for a deviation parameter in 1/m."""
    return np.asarray(s_dev) * param(xtal, "lam") / param(xtal, "sin_2tB") * 1e6
