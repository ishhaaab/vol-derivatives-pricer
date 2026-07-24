"""Tests for the range-accrual Monte Carlo pricer."""

import math

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.range_accrual import price_range_accrual


# ---------------------------------------------------------------------------
# Helpers (no conftest — follow upstream convention)
# ---------------------------------------------------------------------------
def _forward(T: float, S: float = 100.0, r: float = 0.05, q: float = 0.0) -> float:
    return S * math.exp((r - q) * T)


def _flat_fs(
    sigma: float = 0.2,
    S: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T_low: float = 0.5,
    T_high: float = 2.0,
) -> FittedSurface:
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
N_PATHS = 20_000
RNG_SEED = 42


class TestRangeAccrual:
    """Tests for the range-accrual Monte Carlo pricer.

    Uses a flat σ=0.2, S=100, r=0.05 surface.
    """

    def test_wide_range_full_accrual(self) -> None:
        """L=0, H=1e9: every day accrues → PV ≈ coupon_rate * exp(-rT).

        With coupon_rate=0.05, T=1.0, PV ≈ 0.05 * exp(-0.05) = 0.04756.
        """
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = price_range_accrual(
            bridge,
            T=1.0,
            lower=0.0,
            upper=1e9,
            coupon_rate=0.05,
            n_paths=N_PATHS,
            n_steps_per_year=252,
            rng_seed=RNG_SEED,
        )
        expected_pv = 0.05 * math.exp(-0.05)
        # Tolerance: 5 standard errors or rel=0.02
        assert result.pv == approx(expected_pv, rel=0.02), (
            f"PV={result.pv:.6f}, expected={expected_pv:.6f}"
        )
        assert result.accrual_mean == approx(1.0, abs=0.01)

    def test_tight_range_above_paths(self) -> None:
        """L=200, H=1e9: spot never reaches this high → PV ≈ 0."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = price_range_accrual(
            bridge,
            T=1.0,
            lower=200.0,
            upper=1e9,
            coupon_rate=0.05,
            n_paths=N_PATHS,
            n_steps_per_year=252,
            rng_seed=RNG_SEED,
        )
        assert result.pv == approx(0.0, abs=0.001)
        assert result.accrual_mean == approx(0.0, abs=0.001)

    def test_tight_range_below_paths(self) -> None:
        """L=0, H=10: spot never goes this low → PV ≈ 0."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = price_range_accrual(
            bridge,
            T=1.0,
            lower=0.0,
            upper=10.0,
            coupon_rate=0.05,
            n_paths=N_PATHS,
            n_steps_per_year=252,
            rng_seed=RNG_SEED,
        )
        assert result.pv == approx(0.0, abs=0.001)
        assert result.accrual_mean == approx(0.0, abs=0.001)

    def test_wide_range_T_two_years(self) -> None:
        """Wide range at T=2 confirms coupon_rate is the total maturity payment, not annualised.

        The existing test_wide_range_full_accrual uses T=1 where both
        conventions coincide.  At T=2 the discrepancy would be 2× if the
        convention were annualised.
        """
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = price_range_accrual(
            bridge,
            T=2.0,
            lower=0.0,
            upper=1e12,
            coupon_rate=0.05,
            n_paths=N_PATHS,
            n_steps_per_year=252,
            rng_seed=RNG_SEED,
        )
        # Code convention: PV = coupon_rate * accrual_fraction * exp(-rT)
        # With full accrual: PV = 0.05 * exp(-0.05 * 2)  (NOT multiplied by T)
        expected = 0.05 * math.exp(-0.05 * 2.0)
        assert result.pv == approx(expected, rel=0.05), (
            f"PV={result.pv:.6f}, expected={expected:.6f}"
        )
        assert result.accrual_mean == approx(1.0, abs=0.01)

    def test_result_fields_populated(self) -> None:
        """Returned object has correct fields."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = price_range_accrual(
            bridge,
            T=1.0,
            lower=0.0,
            upper=1e9,
            coupon_rate=0.05,
            n_paths=1000,
            n_steps_per_year=252,
            rng_seed=42,
        )
        assert result.n_paths == 1000
        assert result.n_steps == 252
        assert result.accrual_std >= 0.0
