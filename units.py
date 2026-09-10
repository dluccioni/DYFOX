"""Rocking angle to deviation parameter.

Three functions rather than one, because the pipeline does this
conversion three different ways and the differences survive into the
figures:

    phi * sin_2tB / lam           angle already in radians
    urad * 1e-6 * sin_2tB / lam   angle in microradians
    s * lam / sin_2tB * 1e6       going back the other way

Folding sin_2tB / lam into a single constant is the obvious tidy-up and
it changes roughly half the values on a rocking grid in the last bit,
because floating-point multiplication is not associative. The published
figures were made with the forms above, so keep them apart.
"""

import numpy as np

# Constants the crystal calculation shares.
R_E = 2.8179403262e-15          # classical electron radius (m)
HC_EV_M = 12398.419e-10         # hc (eV * m)


def param(xtal, key):
    """Crystal parameter by name, from a mapping or from an object."""
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
