"""Volatility swap fair strike via convexity adjustment.

Uses the Brockhaus-Long approximation to adjust the fair variance
strike into a fair volatility strike:

.. math::

    K_{\\text{vol}} \\approx \\sqrt{K_{\\text{var}}^2}
    \\left(1 - \\frac{\\nu^2 T}{8}\\right)

where *ν* is the vol-of-vol parameter.  For a flat-vol surface (ν = 0)
this collapses to *K_vol* = *σ*.
"""

import math


def fair_vol_strike(k_var_sq: float, nu: float, T: float) -> float:
    """Brockhaus-Long convexity-adjusted fair volatility swap strike.

    Parameters
    ----------
    k_var_sq:
        Fair variance strike (annualised variance rate), e.g. the
        ``strike_var`` field of a ``FairVarianceSwap``.
    nu:
        Vol-of-vol parameter.
    T:
        Swap maturity in years.

    Returns
    -------
    float
        Fair volatility strike (annualised).
    """
    sq = math.sqrt(k_var_sq)
    return sq * (1.0 - (nu * nu * T) / 8.0)
