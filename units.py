"""Rocking angle and deviation parameter.

Three conversions, and they are deliberately three rather than one with a
scale factor, because floating-point multiplication is not associative:

    phi * sin_2tB / lam        !=  phi * (sin_2tB / lam)
    (urad * 1e-6) * sin_2tB / lam  !=  (urad * sin_2tB / lam) * 1e-6

for many inputs.  The pipeline uses the first form for angles already in
radians and the second for angles in microradians, and the published
figures depend on which one ran.  Each function below reproduces one call
site's expression exactly; do not fold them together.

    deviation_from_rad     rocking angle in rad     -> s_dev in 1/m
    deviation_from_urad    rocking angle in urad    -> s_dev in 1/m
    urad_from_deviation    s_dev in 1/m             -> rocking angle in urad
"""

import numpy as np

# Physical constants shared by the crystal calculation.
R_E = 2.8179403262e-15          # classical electron radius (m)
HC_EV_M = 12398.419e-10         # hc (eV * m)


def param(xtal, key):
    """Read a crystal parameter from a mapping or from an object.

    The engine moves from a parameter dict to a CrystalParams object in
    two steps; accepting both keeps every call site written the same way
    across the move.
    """
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
