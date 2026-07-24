"""Monte Carlo pricing of range-accrual structured notes under local vol.

Daily accrual: coupon accrues on each observation day where the spot
stays within [lower, upper].  Discounted to present and averaged across
paths.

SDE: dS = (r q) S dt + σ_loc(S, t) S dW   (Euler-Maruyama)

Performance note
----------------
The local-vol surface is pre-computed ONCE on a 2D (K, T) grid of
``_N_GRID`` strikes x ``n_steps`` time slices at the start of the
simulation, then per-step local vols are obtained via ``np.interp``.
This eliminates O(steps x _N_GRID) ``local_vol_at`` calls that were
previously made inside the loop.
"""

import math
import warnings
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
    pv_std:
        Standard error of the mean PV (=
        std(pv_per_path, ddof=1) / sqrt(n_paths)).
        Use 1.96 * pv_std for a 95% confidence half-width.
    """

    pv: float
    n_paths: int
    n_steps: int
    accrual_mean: float
    accrual_std: float
    pv_std: float


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

    Note
    ----
    Kept for compatibility; ``price_range_accrual`` now uses
    ``_precompute_lv_grid`` for performance (pre-computes the vol grid
    once for all time steps rather than per-step).
    """
    # Clamp t to the surface's minimum expiry so we don't
    # request a time slice before the earliest fitted slice.
    t = max(t, bridge.min_expiry)
    if t > bridge.max_expiry:
        warnings.warn(
            f"_safe_local_vol: t={t:.4f} exceeds surface max_expiry="
            f"{bridge.max_expiry:.4f}; clamping to max_expiry. "
            f"Local vol beyond the calibrated surface is unreliable -- "
            f"increase max_expiries in load_calibrated_surface() to cover the maturity.",
            RuntimeWarning,
            stacklevel=2,
        )
        t = min(t, bridge.max_expiry)

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


def _precompute_lv_grid(
    bridge: SurfaceBridge,
    grid_strikes_K: np.ndarray,
    times: np.ndarray,
    S0: float,
) -> np.ndarray:
    """Pre-compute local volatility on a 2D (time, strike) grid.

    Each element ``lv_grid[k, j]`` = local vol at ``(grid_strikes_K[j], times[k])``
    with the same fallback logic as ``_safe_local_vol`` (clamp time to
    ``min_expiry``, clamp strike to ``[S0/10, S0*10]``, ATM-IV fallback on
    exception, NaN fill).

    Returns
    -------
    np.ndarray
        Shape ``(len(times), len(grid_strikes_K))``.
    """
    n_times = len(times)
    n_strikes = len(grid_strikes_K)
    lv_grid = np.empty((n_times, n_strikes), dtype=float)

    S_min_safe = max(S0 * 0.1, 1e-4)
    S_max_safe = S0 * 10.0

    for k_idx, t in enumerate(times):
        t_clamped = max(t, bridge.min_expiry)
        if t > bridge.max_expiry:
            warnings.warn(
                f"_precompute_lv_grid: t={t:.4f} exceeds surface max_expiry="
                f"{bridge.max_expiry:.4f}; clamping to max_expiry. "
                f"Local vol beyond the calibrated surface is unreliable -- "
                f"increase max_expiries in load_calibrated_surface() to cover the maturity.",
                RuntimeWarning,
                stacklevel=2,
            )
            t_clamped = min(t_clamped, bridge.max_expiry)
        # Evaluate local vol at each grid point for this time slice
        row = np.empty(n_strikes, dtype=float)
        for j, K in enumerate(grid_strikes_K):
            if K < S_min_safe or K > S_max_safe:
                row[j] = _atm_iv_fallback(bridge, t_clamped)
            else:
                try:
                    row[j] = local_vol_at(bridge, float(K), t_clamped)
                except Exception:
                    row[j] = _atm_iv_fallback(bridge, t_clamped)

        # NaN fill
        nan_mask = np.isnan(row)
        if np.any(nan_mask):
            fallback = _atm_iv_fallback(bridge, t_clamped)
            row[nan_mask] = fallback

        lv_grid[k_idx] = row

    return lv_grid



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
    use_antithetic: bool = True,
)-> RangeAccrualResult:
    """Price a range-accrual note via Monte Carlo under surface-implied local vol.

    Parameters
    ----------
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
    use_antithetic:
        When True (default), generate ``n_paths // 2`` random normals
        and mirror them (Z and -Z) for the second half.  Total paths
        simulated = ``n_paths``; only ``n_paths // 2`` draws are made.
        This is a standard variance-reduction technique; on discontinuous
        payoffs (indicator-range) the reduction is weaker than the
        textbook 1/sqrt(2) but still positive.

    Returns
    -------
    RangeAccrualResult
        Present value per unit notional and simulation metadata.

    Notes
    -----
    The coupon is paid at maturity.  ``coupon_rate`` is the total payment
    when every observation day is in-range; the accrual fraction
    (in-range days / total days) scales this linearly.
    PV = coupon_rate × accrual_fraction × exp(-rT).
    This convention matches a fixed-coupon range accrual where the
    embedded note pays a capped maximum coupon if every day is in the
    corridor.

    Boundary tests
    --------------
    ``lower = 0, upper = +inf`` → full accrual every day,
      PV ≈ coupon_rate * exp(-r * T).
    ``lower`` extremely high above max path / ``upper`` extremely low
      below min path → no accrual, PV ≈ 0.
    """
    S0 = bridge.spot
    r = bridge.risk_free
    q = bridge.div_yield

    if use_antithetic and n_paths % 2 != 0:
        raise ValueError(
            f"n_paths must be even when use_antithetic=True; got {n_paths}"
        )

    dt = 1.0 / n_steps_per_year
    n_steps = max(int(round(T * n_steps_per_year)), 1)

    rng = np.random.default_rng(rng_seed)

    # Pre-compute local-vol grid on 2D (time, strike) for the step-start times
    S_min_safe = max(S0 * 0.1, 1e-4)
    S_max_safe = S0 * 10.0
    grid_strikes_K = np.logspace(
        np.log10(max(S_min_safe, 1e-8)),
        np.log10(S_max_safe),
        _N_GRID,
    )
    step_times = np.array([k * dt for k in range(n_steps)])  # 0, dt, 2dt, ...
    lv_grid = _precompute_lv_grid(bridge, grid_strikes_K, step_times, S0)

    # Initialise paths
    S = np.full((n_paths,), S0, dtype=float)

    # Running count of in range observations per path
    in_range_counts = np.zeros(n_paths, dtype=int)

    sqrt_dt = np.sqrt(dt)

    t = 0.0
    for k_idx in range(n_steps):
        # Local vol at the START of the step (time t, current spot S)
        # Interpolate from the pre-computed lv_grid for this time slice
        sigma = np.interp(
            np.clip(S, S_min_safe, S_max_safe),
            grid_strikes_K,
            lv_grid[k_idx],
        )

        # Generate random increments (antithetic if requested)
        if use_antithetic:
            half = n_paths // 2
            Z_half = rng.standard_normal(half)
            Z = np.concatenate([Z_half, -Z_half])
        else:
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
    # Standard error: for antithetic, average each pair first so the
    # pairing structure is reflected in the SE estimate.
    if use_antithetic:
        half = n_paths // 2
        pair_means = (pv_per_path[:half] + pv_per_path[half:]) * 0.5
        pv_std = float(np.std(pair_means, ddof=1) / math.sqrt(half))
    else:
        pv_std = float(np.std(pv_per_path, ddof=1) / math.sqrt(n_paths))

    return RangeAccrualResult(
        pv=pv,
        n_paths=n_paths,
        n_steps=n_steps,
        accrual_mean=accrual_mean,
        accrual_std=accrual_std,
        pv_std=pv_std,
    )
