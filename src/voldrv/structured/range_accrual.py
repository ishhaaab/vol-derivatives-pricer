"""Monte Carlo pricing of range-accrual structured notes under local vol.

Daily accrual: coupon accrues on each observation day where the spot
stays within [lower, upper].  Discounted to present and averaged across
paths.

SDE: dS = (r q) S dt + σ_loc(S, t) S dW   (Euler-Maruyama)

Performance note
----------------
``local_vol_at`` is called on a pre-computed strike grid at each time
step (~200 grid points), then the per-path local vols are obtained via
linear interpolation.  This avoids a full O(paths × steps) evaluation
while maintaining accuracy.
"""

from dataclasses import dataclass

import numpy as np

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.local_vol import local_vol_at


# Constants
_N_GRID: int = 200  # number of grid points for pre-computed local vol
_DEFAULT_FALLBACK_VOL: float = 0.2  # ultimate safety net for ATM IV fallback


# RangeAccrualResult
@dataclass(frozen=True, slots=True)
class RangeAccrualResult:
    """Result of a range-accrual Monte Carlo simulation.

    Attributes
   ----------
    pv:
        Present value per unit notional.
    n_paths:
        Number of simulated paths.
    n_steps:
        Total number of daily observation steps.
    accrual_mean:
        Mean fraction of days where the spot was in-range (pre-discount).
    accrual_std:
        Standard deviation of the per-path accrual fraction.
    """

    pv: float
    n_paths: int
    n_steps: int
    accrual_mean: float
    accrual_std: float


# ATM fallback value
def _atm_iv_fallback(bridge: SurfaceBridge, t: float)-> float:
    """Compute a safe ATM implied vol at time *t* for fallback use.

    Falls back through three layers:
      1. Try *t* directly.
      2. Try ``min_expiry`` as a safe anchor.
      3. Hard-coded flat-vol default (``_DEFAULT_FALLBACK_VOL``).
    """
    try:
        return bridge.get_iv(bridge.spot, t)
    except Exception:
        pass
    try:
        return bridge.get_iv(bridge.spot, bridge.min_expiry)
    except Exception:
        pass
    return _DEFAULT_FALLBACK_VOL  # final safety net


# Safe local-vol via pre-computed grid + interpolation
def _safe_local_vol(bridge: SurfaceBridge, S: np.ndarray, t: float,
                    S0: float)-> np.ndarray:
    """Evaluate local vol per path using a pre-computed grid.

    Steps
   ----
    1. Clamp each path's spot to ``[S_min_safe, S_max_safe]``.
    2. Build a log-spaced grid of ``_N_GRID`` strikes covering the safe range.
    3. Evaluate ``local_vol_at`` at each grid point.
    4. Linearly interpolate each path's local vol from the grid.
    5. If the grid evaluation fails entirely, fall back to ATM IV.
    """
    # Clamp t to the surface's minimum expiry so we don't
    # request a time slice before the earliest fitted slice.
    t = max(t, bridge.min_expiry)

    S_min_safe = max(S0 * 0.1, 1e-4)
    S_max_safe = S0 * 10.0

    # Build log spaced strike grid
    grid_strikes = np.logspace(
        np.log10(max(S_min_safe, 1e-8)),
        np.log10(S_max_safe),
        _N_GRID,
    )

    # Evaluate local vol at each grid point
    grid_vols = np.empty(_N_GRID, dtype=float)
    for i, k in enumerate(grid_strikes):
        try:
            grid_vols[i] = local_vol_at(bridge, float(k), t)
        except Exception:
            grid_vols[i] = _atm_iv_fallback(bridge, t)

    # Handle any remaining nans in the grid
    nan_mask = np.isnan(grid_vols)
    if np.any(nan_mask):
        fallback = _atm_iv_fallback(bridge, t)
        grid_vols[nan_mask] = fallback

    # Clamp path strikes then interpolate
    s_clamped = np.clip(S, S_min_safe, S_max_safe)
    return np.interp(s_clamped, grid_strikes, grid_vols)



# Range accrual pricer
def price_range_accrual(
    bridge: SurfaceBridge,
    T: float,
    lower: float,
    upper: float,
    coupon_rate: float,
    n_paths: int = 10_000,
    n_steps_per_year: int = 252,
    rng_seed: int | None = None,
)-> RangeAccrualResult:
    """Price a range-accrual note via Monte Carlo under surface-implied local vol.

    Parameters
   ---------
    bridge:
        Surface bridge wrapping a fitted surface.
    T:
        Maturity in years.
    lower:
        Lower barrier (price level).
    upper:
        Upper barrier (price level).
    coupon_rate:
        Total coupon paid at maturity if every observation day is in-range.
        The accrual fraction (in-range days / total days) scales this
        proportionally.  (Not annualised.)  E.g. 0.05 = 5 % of notional
        at maturity for full accrual.
    n_paths:
        Number of Monte Carlo paths.  Default 10 000.
    n_steps_per_year:
        Number of time steps per year.  Default 252 (daily).
    rng_seed:
        Optional RNG seed for reproducibility.

    Returns
   ------
    RangeAccrualResult
        Present value per unit notional and simulation metadata.

    Notes
   ----
    The coupon is paid at maturity.  ``coupon_rate`` is the total payment
    when every observation day is in-range; the accrual fraction
    (in-range days / total days) scales this linearly.
    PV = coupon_rate × accrual_fraction × exp(-rT).
    This convention matches a fixed-coupon range accrual where the
    embedded note pays a capped maximum coupon if every day is in the
    corridor.

    Boundary tests
   -------------
    ``lower = 0, upper = +inf`` → full accrual every day,
      PV ≈ coupon_rate * exp(-r * T).
    ``lower`` extremely high above max path / ``upper`` extremely low
      below min path → no accrual, PV ≈ 0.
    """
    S0 = bridge.spot
    r = bridge.risk_free
    q = bridge.div_yield

    dt = 1.0 / n_steps_per_year
    n_steps = max(int(round(T * n_steps_per_year)), 1)

    rng = np.random.default_rng(rng_seed)

    # Initialise paths
    S = np.full((n_paths,), S0, dtype=float)

    # Running count of in range observations per path
    in_range_counts = np.zeros(n_paths, dtype=int)

    sqrt_dt = np.sqrt(dt)

    t = 0.0
    for _ in range(n_steps):
        # Local vol evaluated at the START of the step (time t, current spot S)
        sigma = _safe_local_vol(bridge, S, t, S0)

        # Generate random increments
        Z = rng.standard_normal(n_paths)

        # Euler-Maruyama step
        drift= (r-q) * S * dt
        diffusion= sigma * S * sqrt_dt * Z
        S = S + drift + diffusion
        S = np.maximum(S, 1e-12)  # floor at a tiny positive value
        t += dt  # advance time for the next step

        # Check range (on the new S after the step)
        in_range = (S >= lower) & (S <= upper)
        in_range_counts += in_range.astype(int)

    # Accrual fraction per path
    accrual_fraction = in_range_counts.astype(float) / n_steps

    accrual_mean = float(np.mean(accrual_fraction))
    accrual_std = float(np.std(accrual_fraction, ddof=1))

    # Discounted PV: coupon_rate * (accrual fraction) * exp(-r * T)
    discount = np.exp(-r * T)
    pv_per_path = coupon_rate * accrual_fraction * discount
    pv = float(np.mean(pv_per_path))

    return RangeAccrualResult(
        pv=pv,
        n_paths=n_paths,
        n_steps=n_steps,
        accrual_mean=accrual_mean,
        accrual_std=accrual_std,
    )
