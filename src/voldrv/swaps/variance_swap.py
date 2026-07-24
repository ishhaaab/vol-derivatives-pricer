"""Fair variance strike via log-contract static replication.

Implements the Demeterfi et al. replication formula for a fair variance
swap strike using a strip of OTM option prices pulled from the surface.

Reference
---------
Demeterfi, Derman, Kamal, Zou — *More Than You Ever Wanted to Know
About Volatility Swaps* (1999).
"""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from voldrv.surface_bridge import SurfaceBridge


# ---------------------------------------------------------------------------
# FairVarianceSwap result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FairVarianceSwap:
    """Result of a fair variance strike calculation.

    Attributes
    ----------
    strike_var:
        The annualised fair variance K_var².
    forward:
        Forward price *F* at expiry *T*.
    expiry:
        Swap maturity in years.
    n_strikes:
        Number of strike grid points used in the integration.
    pi_integral:
        The Π term — integral of option prices weighted by 1/K².
    """

    strike_var: float
    forward: float
    expiry: float
    n_strikes: int
    pi_integral: float



# Variance swap fair strike
def fair_variance_strike(
    bridge: SurfaceBridge,
    T: float,
    n_points: int = 4000,
    K_min: float | None = None,
    K_max: float | None = None,
) -> FairVarianceSwap:
    """Compute the fair annualised variance strike via static replication.

    The simplest (and dividend-blind) form of the Demeterfi formula:

    .. math::

        K_{\\text{var}}^2 = \\frac{2}{T} \\, e^{rT} \\, \\Pi

    where

    .. math::

        \\Pi = \\int_0^F \\frac{1}{K^2} P(K,T)\\,dK
               + \\int_F^\\infty \\frac{1}{K^2} C(K,T)\\,dK

    and *F* = S₀·exp((r - q)·T) is the forward price.

    This is derived from the log-contract replication:

    ln(S_T/S₀) = ln(F/S₀) - Π_T  (after removing the zero-cost forward)

    and Ito's lemma:

    E[ln(S_T/S₀)] = ln(F/S₀) - σ²T/2 → σ² = (2/T)·e^(rT)·Π.

    For a flat-vol surface with σ = 0.2, T = 1.0, r = 0.05, q = 0.0,
    the formula should produce K_var² ≈ 0.04 (= σ²).

    Parameters
    ----------
    bridge:
        Surface bridge wrapping a calibrated fitted surface.
    T:
        Swap maturity in years (must be within the surface range).
    n_points:
        Number of strike grid points.  Default 4000.
    K_min:
        Lowest strike in the grid.  If ``None``, derived from the surface
        (max(0.01, S·0.2)).
    K_max:
        Highest strike in the grid.  If ``None``, defaults to S·5.

    Returns
    -------
    FairVarianceSwap
        Frozen dataclass containing the fair variance strike.
    """
    S0 = bridge.spot
    r = bridge.risk_free
    q = bridge.div_yield

    # Forward price
    F = S0 * exp((r - q) * T)

    # Default grid bounds
    if K_min is None:
        K_min = max(0.01, S0 * 0.2)
    if K_max is None:
        K_max = S0 * 5.0

    # Build strike grid: two segments split at F so the integration
    # cleanly separates puts (K <= F) from calls (K >= F).
    # We place roughly half the points on each side.
    n_half = max(n_points // 2, 10)

    # Log spaced strikes below F (descending then reversed to make ascending)
    log_min = log(K_min)
    log_F = log(F)
    strikes_low = np.logspace(log_min, log_F, n_half, base=exp(1.0))

    # Log-spaced strikes above F
    log_max = log(K_max)
    strikes_high = np.logspace(log_F, log_max, n_half + 1, base=exp(1.0))

    # Skip the first point of strikes_high to avoid double-counting F
    # (strikes_low's last point is F, strikes_high's first point is F)
    strikes_high = strikes_high[1:]

    # Full unique sorted grid
    strikes = np.concatenate([strikes_low, strikes_high])
    # Remove any duplicate F (near-exact duplicates from floating point)
    strikes = np.unique(strikes)

    # Evaluate option prices via vectorised calls (OTM puts on K <= F,
    # OTM calls on K > F).  The IV lookup per strike still iterates but
    # the BS pricing is fully vectorised.
    mask_put = strikes <= F
    mask_call = strikes > F
    prices = np.empty_like(strikes)
    if np.any(mask_put):
        prices[mask_put] = bridge.get_option_prices(strikes[mask_put], T, "P")
    if np.any(mask_call):
        prices[mask_call] = bridge.get_option_prices(strikes[mask_call], T, "C")

    # Integrand: g(K) = price / K²
    integrand = prices / (strikes * strikes)

    # Trapezoidal integration for Π
    pi_integral = float(np.trapezoid(integrand, strikes))

    # Demeterfi formula: K_var² = (2/T) * e^(rT) * Π
    erT = exp(r * T)
    k_var_sq = (2.0 / T) * erT * pi_integral

    return FairVarianceSwap(
        strike_var=k_var_sq,
        forward=F,
        expiry=T,
        n_strikes=len(strikes),
        pi_integral=pi_integral,
    )
