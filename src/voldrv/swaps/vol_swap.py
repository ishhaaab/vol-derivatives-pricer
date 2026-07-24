"""Volatility swap fair strike via convexity adjustment.

Uses the Brockhaus-Long approximation to adjust the fair variance
strike into a fair volatility strike:

.. math::

    K_{\\text{vol}} \\approx \\sqrt{K_{\\text{var}}^2}
    \\left(1 - \\frac{\\nu^2 T}{8}\\right)

where *ν* is the vol-of-vol parameter.  For a flat-vol surface (ν = 0)
this collapses to *K_vol* = *σ*.

This module also provides a surface-driven vol-of-vol estimator
(:func:`nu_from_surface`) and an end-to-end pricing convenience function
(:func:`fair_vol_strike_from_surface`).
"""

import math
from dataclasses import dataclass

import numpy as np

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.variance_swap import fair_variance_strike


# ---------------------------------------------------------------------------
# Low-level Brockhaus-Long
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Surface-driven vol-of-vol estimator + end-to-end pricer
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FairVolSwap:
    """Result of an end-to-end vol swap pricing from the surface.

    Attributes
    ----------
    strike_vol:
        The fair volatility strike (annualised).
    k_var_sq:
        Underlying fair variance strike K_var² (annualised variance).
    nu:
        Vol-of-vol estimated from the surface.
    expiry:
        Swap maturity in years.
    """

    strike_vol: float
    k_var_sq: float
    nu: float
    expiry: float


def nu_from_surface(
    bridge: SurfaceBridge,
    T: float,
    K_range_moneyness: tuple[float, float] = (0.85, 1.15),
    n_points: int = 50,
) -> float:
    """Estimate the vol-of-vol parameter nu from the smile curvature.

    Samples the implied vol on a log-spaced strike grid covering
    moneyness ``[K_range_moneyness[0] * F, K_range_moneyness[1] * F]``
    where *F* = S · exp((r - q) · T).  Fits a quadratic
    σ(k) ≈ s₀ + s₁·k + s₂·k² in log-moneyness *k* = ln(K/F).
    Returns ν = 4 · |s₂| · √T as a curvature-based proxy for
    vol-of-vol.

    On a flat smile the quadratic coefficient s₂ ≈ 0, so ν ≈ 0 and
    :func:`fair_vol_strike` collapses to √(K_var²).

    Parameters
    ----------
    bridge:
        Surface bridge wrapping a fitted surface.
    T:
        Expiry in years; must be within the surface range
        ``[bridge.min_expiry, bridge.max_expiry]``.
    K_range_moneyness:
        Lower and upper moneyness fractions relative to the forward.
        Default ``(0.85, 1.15)``.
    n_points:
        Number of strike grid points for the quadratic fit.  Default 50.

    Returns
    -------
    float
        Estimated vol-of-vol ν (non-negative).

    Raises
    ------
    ValueError
        If *T* is outside the surface expiry range.
    """
    if T < bridge.min_expiry or T > bridge.max_expiry:
        raise ValueError(
            f"nu_from_surface: T={T} outside surface range "
            f"[{bridge.min_expiry}, {bridge.max_expiry}]"
        )

    F = bridge.forward(T)
    K_lo, K_hi = K_range_moneyness

    # Log-spaced strikes in real space using np.linspace + np.exp
    lo = math.log(K_lo * F)
    hi = math.log(K_hi * F)
    strikes = np.exp(np.linspace(lo, hi, n_points))

    # Evaluate implied vol at each strike
    ivs = np.array([bridge.get_iv(float(k), T) for k in strikes])

    # Log-moneyness
    k = np.log(strikes / F)

    # Quadratic fit: polyfit returns highest-degree coefficient first
    coeffs = np.polyfit(k, ivs, 2)
    s2 = float(coeffs[0])  # k² coefficient

    nu = 4.0 * abs(s2) * math.sqrt(T)

    # Clamp to non-negative (should always be true after abs)
    return max(nu, 0.0)


def fair_vol_strike_from_surface(
    bridge: SurfaceBridge,
    T: float,
    n_points_var: int = 4000,
) -> FairVolSwap:
    """End-to-end vol swap pricing from the surface.

    Composes :func:`~voldrv.swaps.variance_swap.fair_variance_strike` and
    :func:`nu_from_surface`, then applies the Brockhaus-Long convexity
    adjustment.

    Parameters
    ----------
    bridge:
        Surface bridge wrapping a fitted surface.
    T:
        Swap maturity in years.
    n_points_var:
        Number of strike grid points for the variance swap integration.
        Default 4000.

    Returns
    -------
    FairVolSwap
        Frozen dataclass containing the fair volatility strike, the
        underlying variance strike, the estimated vol-of-vol, and the
        expiry.
    """
    var = fair_variance_strike(bridge, T, n_points=n_points_var)
    nu = nu_from_surface(bridge, T)
    k_vol = fair_vol_strike(var.strike_var, nu, T)

    return FairVolSwap(
        strike_vol=k_vol,
        k_var_sq=var.strike_var,
        nu=nu,
        expiry=T,
    )
