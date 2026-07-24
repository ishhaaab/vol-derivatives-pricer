"""Tests for the variance swap fair-strike calculation."""

import math

import pytest
from pytest import approx

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.variance_swap import fair_variance_strike


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
    """Two-slice flat (b=0) SVI surface so iv_at(K, T) = sigma for any K."""
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
class TestFairVarianceStrike:
    """Tests for the Demeterfi static replication formula."""

    def test_flat_surface_atm(self) -> None:
        """For flat σ=0.2, T=1.0, q=0.0: K_var² ≈ 0.04 (σ²)."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.0)
        assert result.strike_var == approx(0.04, rel=1e-3)

    def test_with_dividends(self) -> None:
        """With dividends q=0.02, variance swap strike ≈ σ² = 0.04.

        Variance swaps are dividend-blind: the forward captures the
        dividend yield, and the replication weights by 1/K² neutralise
        the remaining impact.
        """
        sigma = 0.2
        q = 0.02
        S = 100.0
        r = 0.05
        fs = _flat_fs(sigma=sigma, S=S, r=r, q=q)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.0)
        assert result.strike_var == approx(sigma ** 2, rel=1e-3)

    def test_scales_with_sigma_sq(self) -> None:
        """K_var² scales linearly with σ²: σ=0.3 → K_var² ≈ 0.09."""
        fs = _flat_fs(sigma=0.3, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.0)
        assert result.strike_var == approx(0.09, rel=1e-3)

    def test_annualised_independent_of_T(self) -> None:
        """K_var² is the annualised variance rate, independent of T."""
        sigma = 0.2
        fs = _flat_fs(sigma=sigma, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.5)
        assert result.strike_var == approx(sigma ** 2, rel=1e-3)

    def test_return_fields(self) -> None:
        """Returned FairVarianceSwap object has sensible fields."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.0)
        assert result.forward == approx(100.0 * math.exp(0.05))
        assert result.expiry == approx(1.0)
        assert result.n_strikes >= 100
        assert result.pi_integral > 0.0
