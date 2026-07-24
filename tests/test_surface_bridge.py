"""Tests for the surface bridge."""

import math

import pytest
from pytest import approx

from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge


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
class TestSurfaceBridge:
    def test_get_iv_flat_returns_constant_vol(self) -> None:
        fs = _flat_fs()
        b = SurfaceBridge(fs)
        assert b.get_iv(95.0, 1.0) == approx(0.2)
        assert b.get_iv(110.0, 1.0) == approx(0.2)

    def test_get_option_price_matches_bs(self) -> None:
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        b = SurfaceBridge(fs)
        for K in (90, 95, 100, 105, 110):
            expected_call = price_floats(100.0, K, 1.0, 0.05, 0.0, 0.2, True)
            expected_put = price_floats(100.0, K, 1.0, 0.05, 0.0, 0.2, False)
            assert b.get_option_price(K, 1.0, "C") == approx(expected_call)
            assert b.get_option_price(K, 1.0, "P") == approx(expected_put)

    def test_cp_case_insensitive(self) -> None:
        fs = _flat_fs()
        b = SurfaceBridge(fs)
        assert b.get_option_price(100, 1.0, "call") == approx(
            b.get_option_price(100, 1.0, "CALL")
        )
        assert b.get_option_price(100, 1.0, "put") == approx(
            b.get_option_price(100, 1.0, "P")
        )

    def test_invalid_cp_raises(self) -> None:
        fs = _flat_fs()
        b = SurfaceBridge(fs)
        with pytest.raises(ValueError):
            b.get_option_price(100, 1.0, "X")

    def test_iv_outside_expiry_range_raises(self) -> None:
        fs = _flat_fs(T_low=0.5, T_high=2.0)
        b = SurfaceBridge(fs)
        with pytest.raises(ValueError):
            b.get_iv(100, 0.1)
        with pytest.raises(ValueError):
            b.get_iv(100, 3.0)

    def test_forward_property(self) -> None:
        fs = _flat_fs(S=100.0, r=0.05, q=0.0)
        b = SurfaceBridge(fs)
        assert b.forward(1.0) == approx(100 * math.exp(0.05))

    def test_expiries_property(self) -> None:
        fs = _flat_fs(T_low=0.5, T_high=2.0)
        b = SurfaceBridge(fs)
        assert b.expiries == approx((0.5, 2.0))
        assert b.min_expiry == approx(0.5)
        assert b.max_expiry == approx(2.0)
