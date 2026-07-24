"""Tests for pricing on a skewed (put-skew) SVI surface.

All existing tests run on a flat b=0, rho=0 surface where every exotic
collapses to its Black-Scholes closed form.  This file verifies that
the pricing engines produce genuinely surface-sensitive results when
the input has realistic curvature.

The skewed surface uses b=0.4, rho=-0.4 (put skew) so that the resulting
prices differ materially from the flat-surface baseline.
"""

import math

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.local_vol import local_vol_at
from voldrv.structured.range_accrual import price_range_accrual
from voldrv.swaps.variance_swap import fair_variance_strike


# ---------------------------------------------------------------------------
# Helpers (no conftest — each test file self-contains its helpers)
# ---------------------------------------------------------------------------
def _forward(T: float, S: float = 100.0, r: float = 0.05, q: float = 0.0) -> float:
    return S * math.exp((r - q) * T)


def _skewed_fs(
    sigma: float = 0.2,       # kept for signature symmetry; unused in the skewed body
    S: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T_low: float = 0.5,
    T_high: float = 2.0,
) -> FittedSurface:
    """Two-slice put-skew SVI surface (b=0.4, rho=-0.4) for surface-sensitivity tests.

    The ``sigma`` parameter is accepted for signature symmetry with
    ``_flat_fs`` but is **not** used in the SVI parameters — the slice
    parameters are hard-coded to produce a realistic put skew.
    """
    sl1 = FittedSlice(
        expiry_time=T_low,
        params=SVIParams(a=0.04 * T_low, b=0.4, rho=-0.4, m=0.0, sigma=0.15),
        rmse=0.0,
        forward_price=_forward(T_low, S, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    sl2 = FittedSlice(
        expiry_time=T_high,
        params=SVIParams(a=0.04 * T_high, b=0.4, rho=-0.4, m=0.0, sigma=0.15),
        rmse=0.0,
        forward_price=_forward(T_high, S, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )

    return FittedSurface(
        spot=S,
        risk_free=r,
        div_yield=q,
        forward_curve=(
            (T_low, _forward(T_low, S, r, q)),
            (T_high, _forward(T_high, S, r, q)),
        ),
        fitted_slices=(sl1, sl2),
    )


def _flat_fs(
    sigma: float = 0.2,
    S: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T_low: float = 0.5,
    T_high: float = 2.0,
) -> FittedSurface:
    """Two-slice flat (b=0) SVI surface — baseline for comparison."""
    sl1 = FittedSlice(
        expiry_time=T_low,
        params=SVIParams(a=sigma ** 2 * T_low, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=_forward(T_low, S, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    sl2 = FittedSlice(
        expiry_time=T_high,
        params=SVIParams(a=sigma ** 2 * T_high, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=_forward(T_high, S, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    return FittedSurface(
        spot=S,
        risk_free=r,
        div_yield=q,
        forward_curve=(
            (T_low, _forward(T_low, S, r, q)),
            (T_high, _forward(T_high, S, r, q)),
        ),
        fitted_slices=(sl1, sl2),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVarianceSwapSkewed:
    """Variance swap fair strike on a skewed surface."""

    def test_variance_swap_skewed_differs_from_flat(self) -> None:
        """The skewed-surface strike differs materially from the flat baseline."""
        flat_bridge = SurfaceBridge(_flat_fs())
        skew_bridge = SurfaceBridge(_skewed_fs())

        k_flat = fair_variance_strike(flat_bridge, T=1.0).strike_var
        k_skew = fair_variance_strike(skew_bridge, T=1.0).strike_var

        print(f"\n  K_var² flat = {k_flat:.8f}, K_var² skewed = {k_skew:.8f}, "
              f"diff = {abs(k_skew - k_flat):.6f}")

        # The gap must be non-trivial — the surface curvature genuinely
        # changes the integral of 1/K²-weighted option prices
        assert abs(k_skew - k_flat) > 0.002, (
            f"Skewed strike {k_skew:.6f} differs from flat {k_flat:.6f} by only "
            f"{abs(k_skew - k_flat):.6f}, expected > 0.002"
        )

    def test_variance_swap_skewed_is_positive_and_finite(self) -> None:
        """Sanity bounds on the skewed variance swap strike."""
        skew_bridge = SurfaceBridge(_skewed_fs())
        k_skew = fair_variance_strike(skew_bridge, T=1.0).strike_var
        # The b=0.4 put skew inflates OTM put prices at low strikes, so
        # K_var² can exceed ATM IV^2 (0.113) significantly.  Bound is generous.
        assert 0.0 < k_skew < 0.5, (
            f"K_var² skewed = {k_skew:.6f} outside expected (0, 0.5)"
        )

    def test_variance_swap_at_surface_expiry_boundaries(self) -> None:
        """Min and max expiry should not raise for fair_variance_strike."""
        skew_bridge = SurfaceBridge(_skewed_fs())
        # Min expiry
        result_min = fair_variance_strike(skew_bridge, T=skew_bridge.min_expiry)
        assert result_min.strike_var > 0.0
        # Max expiry
        result_max = fair_variance_strike(skew_bridge, T=skew_bridge.max_expiry)
        assert result_max.strike_var > 0.0


class TestLocalVolSkewed:
    """Dupire local vol on a skewed surface."""

    def test_local_vol_skewed_depends_on_strike(self) -> None:
        """Local vol differs between K=90 and K=110 (near-ATM, avoiding wing pathology)."""
        skew_bridge = SurfaceBridge(_skewed_fs())
        lv_low = local_vol_at(skew_bridge, K=90.0, T=1.0)
        lv_high = local_vol_at(skew_bridge, K=110.0, T=1.0)

        print(f"\n  LV(K=90, T=1) = {lv_low:.6f}, LV(K=110, T=1) = {lv_high:.6f}, "
              f"diff = {abs(lv_low - lv_high):.6f}")

        assert abs(lv_low - lv_high) > 1e-3, (
            f"Local vol at K=90 ({lv_low:.6f}) and K=110 ({lv_high:.6f}) differ by "
            f"only {abs(lv_low - lv_high):.6f}, expected > 1e-3"
        )

    def test_local_vol_at_surface_expiry_boundaries(self) -> None:
        """Min and max expiry should not raise for local_vol_at."""
        skew_bridge = SurfaceBridge(_skewed_fs())
        # Min expiry
        lv_min = local_vol_at(skew_bridge, K=100.0, T=skew_bridge.min_expiry)
        assert lv_min > 0.0
        # Max expiry
        lv_max = local_vol_at(skew_bridge, K=100.0, T=skew_bridge.max_expiry)
        assert lv_max > 0.0


class TestRangeAccrualSkewed:
    """Range accrual pricing on a skewed surface."""

    def test_range_accrual_skewed_differs_from_flat(self) -> None:
        """Range accrual PV on skewed surface differs from flat beyond MC noise.

        Uses a near-ATM corridor [S0*0.95, S0*1.05] where the local vol
        difference is material, and 20 000 paths for runtime efficiency.
        The corridor avoids wing pathology by staying close to ATM.

        If the comparison is too noisy at 20k paths, we document the chosen
        n_paths and rely on a generous absolute tolerance.
        """
        flat_bridge = SurfaceBridge(_flat_fs())
        skew_bridge = SurfaceBridge(_skewed_fs())

        # Near-ATM corridor, weekly steps, 20k paths
        result_flat = price_range_accrual(
            flat_bridge,
            T=1.0,
            lower=100.0 * 0.95,
            upper=100.0 * 1.05,
            coupon_rate=0.05,
            n_paths=20_000,
            n_steps_per_year=63,
            rng_seed=42,
        )
        result_skew = price_range_accrual(
            skew_bridge,
            T=1.0,
            lower=100.0 * 0.95,
            upper=100.0 * 1.05,
            coupon_rate=0.05,
            n_paths=20_000,
            n_steps_per_year=63,
            rng_seed=42,
        )

        pv_flat = result_flat.pv
        pv_skew = result_skew.pv

        print(f"\n  Range accrual PV flat = {pv_flat:.6f}, PV skewed = {pv_skew:.6f}, "
              f"diff = {abs(pv_skew - pv_flat):.6f}")

        # Use a generous absolute tolerance since the result dataclass does
        # not yet carry a pv_std field (Group B adds it).  The difference
        # should be material for a near-ATM corridor.
        assert abs(pv_skew - pv_flat) > 0.002, (
            f"Skewed PV {pv_skew:.6f} differs from flat PV {pv_flat:.6f} by only "
            f"{abs(pv_skew - pv_flat):.6f}, expected > 0.002"
        )
