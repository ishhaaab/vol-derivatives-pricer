"""Shark fin structured note pricing -- single upper-barrier version.

A shark fin note pays a participating coupon if the underlying stays
within a corridor AND never hits a knock-out barrier.  This module
implements the single-upper-barrier version (the corridor lower bound
L is a documented stretch).  The note pays:

  - ``coupon_rate`` at maturity T if S_t never hits barrier B (> S0)
  - ``refund`` (default 0.0) at maturity T if S_t hits B before T

PV = exp(-rT) * (coupon_rate * P_no_touch + refund * (1 - P_no_touch))

Two pricing methods are provided:

  method="bs"   -- Closed-form Black-Scholes via the reflection
                   principle for a single upper barrier.  Uses the
                   ATM vol sigma = bridge.get_iv(spot, T).  Valid
                   reference value, ignores the smile.
  method="mc"   -- Euler-Maruyama Monte Carlo under the surface's
                   Dupire local vol.  Tracks max(S_t) per path to
                   flag a barrier hit.  On a flat smile this matches
                   the BS closed form (modulo MC noise); on a skewed
                   smile it picks up the vol-of-vol / skew premium
                   that the BS formula ignores.

The BS closed form is derived from the reflection principle for
Brownian motion with drift.  See the module-level docstring of
``_prob_no_touch_bs`` for the exact formula.

Cross-check (flat surface):
  BS method and MC method agree within ~3 MC standard errors on a
  flat smile (where local vol = ATM vol = sigma).

Cross-check (skewed surface):
  BS method and MC method DIFFER beyond MC noise -- the divergence is
  the smile premium, the price a smile-aware pricing model pays for
  accounting for the skew that the flat-BS closed form ignores.

Stretch (NOT yet implemented):
  - Double-barrier (corridor floor L): would require a double
    reflection-principle extension or a 2D image-method formula.
  - "Pays participating coupon alpha * max(S_T - K, 0) if no hit,
    fixed digital D if hit" decomposition (the more general
    participating shark fin, with alpha and K as parameters).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from voldrv.surface_bridge import SurfaceBridge
# Reuse the local-vol precomputation from the range-accrual module.
# _precompute_lv_grid is a private helper but the structured-product
# modules are tightly coupled (shark-fin MC reuses the same
# Euler-Maruyama + 2D cached local-vol machinery as range_accrual).
from voldrv.structured.range_accrual import _precompute_lv_grid, _N_GRID  # pyright: ignore[reportPrivateImportUsage]


# ---------------------------------------------------------------------------
# BS closed-form helper
# ---------------------------------------------------------------------------
def _prob_no_touch_bs(
    S0: float, B: float, T: float, r: float, q: float, sigma: float,
) -> float:
    """Reflection-principle no-touch probability for a single upper barrier.

    For GBM S_t = S0 * exp(nu*t + sigma*W_t) with nu = r - q - sigma^2/2,
    upper barrier B > S0, expiry T:

        P(no touch above B by T) =
            Phi((a - nu*T) / (sigma*sqrt(T)))
            - exp(2*nu*a / sigma^2) * Phi(-(a + nu*T) / (sigma*sqrt(T)))

    where a = ln(B / S0), Phi is the standard normal CDF.

    Parameters
    ----------
    S0: starting spot
    B: upper barrier (> S0)
    T: expiry in years
    r: risk-free rate
    q: dividend yield
    sigma: BS volatility (CONSTANT for the BS world)

    Returns
    -------
    float
        Probability the path never touches B in [0, T].  In [0, 1].
    """
    if B <= S0:
        # barrier at or below starting spot: trivially hit
        return 0.0
    if sigma <= 0.0 or T <= 0.0:
        # no vol or no time: cannot hit
        return 1.0
    a = math.log(B / S0)            # > 0
    nu = r - q - 0.5 * sigma * sigma  # drift of log-spot
    sqrt_T = math.sqrt(T)
    z1 = (a - nu * T) / (sigma * sqrt_T)
    z2 = -(a + nu * T) / (sigma * sqrt_T)
    p_no_touch = float(norm.cdf(z1) - math.exp(2.0 * nu * a / (sigma * sigma)) * norm.cdf(z2))
    # numerical safety: clip to [0, 1] in case of FP edge
    return min(1.0, max(0.0, p_no_touch))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SharkFinResult:
    """Result of a shark-fin pricing calculation.

    Attributes
    ----------
    pv: present value per unit notional
    p_no_touch: probability the barrier was never hit
    method: "bs" or "mc"
    pv_std: standard error of the MC PV estimate (None for BS)
    n_paths: number of MC paths (None for BS)
    barrier: B used
    expiry: T used
    coupon_rate: total coupon paid at T if no hit
    refund: total paid at T if hit
    """
    pv: float
    p_no_touch: float
    method: str
    pv_std: float | None
    n_paths: int | None
    barrier: float
    expiry: float
    coupon_rate: float
    refund: float


# ---------------------------------------------------------------------------
# Public pricer
# ---------------------------------------------------------------------------
def price_shark_fin(
    bridge: SurfaceBridge,
    T: float,
    B: float,
    coupon_rate: float,
    refund: float = 0.0,
    method: str = "bs",
    n_paths: int = 10_000,
    n_steps_per_year: int = 252,
    rng_seed: int | None = None,
) -> SharkFinResult:
    """Price a single-upper-barrier shark-fin note.

    Payoff: pays coupon_rate at T if S never hits B, else refund.
    PV = exp(-rT) * (coupon_rate * P_no_touch + refund * (1 - P_no_touch)).

    Parameters
    ----------
    bridge: SurfaceBridge wrapping a calibrated fitted surface.
    T: maturity in years.
    B: upper knock-out barrier (> S0).
    coupon_rate: total coupon paid at maturity if no hit.
    refund: total paid at maturity if hit (default 0).
    method: "bs" (closed-form, uses ATM vol) or "mc" (Euler-Maruyama
        under local vol).
    n_paths, n_steps_per_year, rng_seed: MC parameters (ignored for "bs").
    """
    S0 = bridge.spot
    r = bridge.risk_free
    q = bridge.div_yield
    if B <= S0:
        raise ValueError(
            f"shark-fin barrier B={B} must be > spot S0={S0} (got B <= S0)"
        )

    discount = math.exp(-r * T)

    if method == "bs":
        sigma_atm = bridge.get_iv(S0, T)
        p_no_touch = _prob_no_touch_bs(S0, B, T, r, q, sigma_atm)
        pv = discount * (coupon_rate * p_no_touch + refund * (1.0 - p_no_touch))
        return SharkFinResult(
            pv=pv, p_no_touch=p_no_touch, method="bs",
            pv_std=None, n_paths=None,
            barrier=B, expiry=T, coupon_rate=coupon_rate, refund=refund,
        )

    if method != "mc":
        raise ValueError(f"method must be 'bs' or 'mc'; got {method!r}")

    # --- MC path ---
    # Log-Euler discretisation: d(ln S) = (r-q-σ²/2)dt + σ dW.
    # Exact for GBM with constant σ; first-order weak approximation for
    # local-vol SDE at finite dt.  Combined with Brownian-bridge
    # interpolation between adjacent step points to recover the
    # *continuous* barrier-hit probability — without the bridge the
    # discrete monitoring creates an upward bias in P_no_touch.
    dt = 1.0 / n_steps_per_year
    n_steps = max(int(round(T * n_steps_per_year)), 1)
    rng = np.random.default_rng(rng_seed)

    # Pre-compute local vol on a 2D (K, t) grid (same machinery as
    # range_accrual).  _precompute_lv_grid clamps t to [min_expiry,
    # max_expiry] with a RuntimeWarning if exceeded; for a 1y note
    # this should not fire if the surface is reasonably calibrated.
    S_min_safe = max(S0 * 0.1, 1e-4)
    S_max_safe = S0 * 10.0
    grid_strikes = np.logspace(
        np.log10(max(S_min_safe, 1e-8)),
        np.log10(S_max_safe),
        _N_GRID,
    )
    step_times = np.array([k * dt for k in range(n_steps)])
    lv_grid = _precompute_lv_grid(bridge, grid_strikes, step_times, S0)

    log_S = np.full((n_paths,), math.log(S0), dtype=float)
    hit = np.zeros(n_paths, dtype=bool)  # True if continuous path ever hit B
    sqrt_dt = math.sqrt(dt)

    for k in range(n_steps):
        S_prev = np.exp(log_S)

        sigma = np.interp(
            np.clip(S_prev, S_min_safe, S_max_safe),
            grid_strikes,
            lv_grid[k],
        )
        Z = rng.standard_normal(n_paths)
        nu_drift = (r - q - 0.5 * sigma * sigma) * dt
        log_S = log_S + nu_drift + sigma * sqrt_dt * Z
        S = np.exp(log_S)

        # --- step-point crossing: the terminal state reached B ---
        hit = hit | (S >= B)

        # --- intra-step crossing via Brownian-bridge interpolation ---
        # Paths that haven't yet hit AND have both endpoints below B
        # might have crossed B between t_{k-1} and t_k.
        needs_check = (~hit) & (S_prev < B) & (S < B)
        if needs_check.any():
            # a = ln(B/S_prev) > 0, b = ln(B/S) > 0
            a = np.log(B / S_prev[needs_check])
            b = np.log(B / S[needs_check])
            # Brownian-bridge crossing probability under σ = const on [t_{k-1}, t_k]
            p_cross = np.exp(-2.0 * a * b / (sigma[needs_check] ** 2 * dt))
            p_cross = np.clip(p_cross, 0.0, 1.0)
            U = rng.uniform(size=needs_check.sum())
            new_hits = U < p_cross
            idx = np.where(needs_check)[0][new_hits]
            hit[idx] = True

    # No-touch indicator: continuous barrier was never hit
    no_touch_per_path = (~hit).astype(float)
    p_no_touch = float(np.mean(no_touch_per_path))
    pv_per_path = discount * (coupon_rate * no_touch_per_path + refund * (1.0 - no_touch_per_path))
    pv = float(np.mean(pv_per_path))
    pv_std = float(np.std(pv_per_path, ddof=1) / math.sqrt(n_paths))

    return SharkFinResult(
        pv=pv, p_no_touch=p_no_touch, method="mc",
        pv_std=pv_std, n_paths=n_paths,
        barrier=B, expiry=T, coupon_rate=coupon_rate, refund=refund,
    )
