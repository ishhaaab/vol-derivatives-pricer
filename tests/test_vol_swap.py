"""Tests for the vol swap convexity adjustment."""

import math

import pytest
from pytest import approx

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.vol_swap import (
    FairVolSwap,
    fair_vol_strike,
    fair_vol_strike_from_surface,
    nu_from_surface,
)


# ---------------------------------------------------------------------------
# Helpers (no conftest — each test file self-contains its helpers)
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


def _skewed_fs(
    sigma: float = 0.2,       # kept for signature symmetry; unused in the skewed body
    S: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T_low: float = 0.5,
    T_high: float = 2.0,
) -> FittedSurface:
    """Two-slice put-skew SVI surface (b=0.4, rho=-0.4) for surface-sensitivity tests."""
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


# ---------------------------------------------------------------------------
# Tests — FairVolStrike (Brockhaus-Long)
# ---------------------------------------------------------------------------
class TestFairVolStrike:
    """Tests for the Brockhaus-Long convexity adjustment."""

    def test_nu_zero_returns_sqrt_variance(self) -> None:
        """For nu=0, fair vol strike = sqrt(k_var_sq) = σ."""
        result = fair_vol_strike(k_var_sq=0.04, nu=0.0, T=1.0)
        assert result == approx(math.sqrt(0.04))  # 0.2

    def test_nu_nonzero_reduces_strike(self) -> None:
        """For nonzero nu, the adjustment reduces the fair vol strike."""
        result_zero = fair_vol_strike(k_var_sq=0.04, nu=0.0, T=1.0)
        result_pos = fair_vol_strike(k_var_sq=0.04, nu=0.5, T=1.0)
        assert result_pos < result_zero

    def test_nu_zero_independent_of_T(self) -> None:
        """At nu=0, fair vol strike = sqrt(k_var_sq) regardless of T."""
        assert fair_vol_strike(0.04, 0.0, 1.0) == approx(0.2)
        assert fair_vol_strike(0.04, 0.0, 2.0) == approx(0.2)
        assert fair_vol_strike(0.09, 0.0, 0.5) == approx(0.3)

    def test_scales_with_nu_sq(self) -> None:
        """Adjustment scales with nu²: doubling nu quadruples the reduction."""
        sq = math.sqrt(0.04)  # 0.2
        adj1 = sq - fair_vol_strike(0.04, 0.1, 1.0)
        adj2 = sq - fair_vol_strike(0.04, 0.2, 1.0)
        # adj2 / adj1 ≈ (0.2² / 0.1²) = 4
        assert adj2 == approx(4.0 * adj1, rel=1e-3)

    def test_numerical_example(self) -> None:
        """Manual numerical check: k_var_sq=0.04, nu=0.3, T=2.0."""
        # 0.04 → sqrt = 0.2
        # adj = 0.2 * (1 - 0.3² * 2 / 8) = 0.2 * (1 - 0.09*2/8) = 0.2*(1-0.0225) = 0.1955
        result = fair_vol_strike(0.04, 0.3, 2.0)
        expected = 0.2 * (1.0 - (0.09 * 2.0) / 8.0)
        assert result == approx(expected)


# ---------------------------------------------------------------------------
# Tests — nu_from_surface and end-to-end vol swap pricing
# ---------------------------------------------------------------------------
class TestNuFromSurface:
    """Tests for the surface-driven vol-of-vol estimator."""

    def test_nu_flat_surface_near_zero(self) -> None:
        """On a flat smile, nu ≈ 0 (no curvature)."""
        bridge = SurfaceBridge(_flat_fs(sigma=0.2))
        nu = nu_from_surface(bridge, T=1.0)
        print(f"\n  nu (flat) = {nu:.6f}")
        # Numerical noise in the quadratic fit should give nu < 0.01
        assert nu < 0.01, f"Expected nu ~ 0 on flat surface, got {nu:.6f}"

    def test_nu_skewed_surface_substantial(self) -> None:
        """On a put-skew smile, nu is materially positive."""
        bridge = SurfaceBridge(_skewed_fs())
        nu = nu_from_surface(bridge, T=1.0)
        print(f"\n  nu (skewed) = {nu:.6f}")
        assert nu > 0.1, f"Expected nu > 0.1 on skewed surface, got {nu:.6f}"

    def test_fair_vol_strike_from_surface_flat_equals_sigma(self) -> None:
        """On a flat 0.2 surface, end-to-end vol strike ≈ 0.2.

        Since nu ≈ 0, the Brockhaus-Long adjustment is negligible and
        K_vol ≈ σ = 0.2.
        """
        bridge = SurfaceBridge(_flat_fs(sigma=0.2))
        result = fair_vol_strike_from_surface(bridge, T=1.0)
        print(f"\n  K_vol (flat) = {result.strike_vol:.6f}, nu = {result.nu:.6f}")
        assert result.strike_vol == approx(0.2, rel=1e-2), (
            f"Fair vol strike on flat surface = {result.strike_vol:.6f}, expected ≈ 0.2"
        )
        assert isinstance(result, FairVolSwap)
        assert result.expiry == approx(1.0)

    def test_fair_vol_strike_from_surface_outside_range_raises(self) -> None:
        """T below min_expiry raises ValueError."""
        bridge = SurfaceBridge(_flat_fs(T_low=0.5, T_high=2.0))
        with pytest.raises(ValueError, match="outside surface range"):
            nu_from_surface(bridge, T=0.1)

    def test_nu_from_surface_returns_nonneg(self) -> None:
        """nu_from_surface always returns a non-negative value."""
        bridge_flat = SurfaceBridge(_flat_fs(sigma=0.2))
        bridge_skew = SurfaceBridge(_skewed_fs())
        assert nu_from_surface(bridge_flat, T=1.0) >= 0.0
        assert nu_from_surface(bridge_skew, T=1.0) >= 0.0
