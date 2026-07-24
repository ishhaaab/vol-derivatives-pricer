"""Tests for the vol swap convexity adjustment."""

import math

import pytest
from pytest import approx

from voldrv.swaps.vol_swap import fair_vol_strike


# ---------------------------------------------------------------------------
# Tests
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
