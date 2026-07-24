"""Tests for the local vol wrapper module."""

import math

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.pricing.local_vol import dupire_at
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.local_vol import LocalVolGrid, build_local_vol_grid, local_vol_at


# ---------------------------------------------------------------------------
# Helpers (no conftest — follow upstream convention)
# ---------------------------------------------------------------------------
def _forward(T: float, spot: float = 100.0, r: float = 0.05,
             q: float = 0.0) -> float:
    return spot * math.exp((r - q) * T)


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
class TestBuildLocalVolGrid:
    def test_grid_shape(self) -> None:
        """Grid dimensions match the input arrays."""
        fs = _flat_fs()
        bridge = SurfaceBridge(fs)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        maturities = np.array([0.5, 1.0, 1.5, 2.0])
        lvg = build_local_vol_grid(bridge, strikes, maturities)
        assert len(lvg.grid) == 4
        for row in lvg.grid:
            assert len(row) == 5

    def test_matches_upstream_dupire_at(self) -> None:
        """local_vol_at matches arbfree_vol's dupire_at numerically."""
        fs = _flat_fs()
        bridge = SurfaceBridge(fs)
        ref = dupire_at(fs, 100.0, 1.0)
        result = local_vol_at(bridge, 100.0, 1.0)
        assert result == approx(ref)

    def test_flat_vol_surface_dupire(self) -> None:
        """On a flat-vol surface, Dupire local vol ≈ 0.2 (tolerant due to FD jitter)."""
        fs = _flat_fs(sigma=0.2)
        bridge = SurfaceBridge(fs)
        vol = local_vol_at(bridge, 100.0, 1.0)
        assert vol == approx(0.2, rel=5e-3)

    def test_out_of_surface_raises(self) -> None:
        """T outside surface range propagates ValueError."""
        fs = _flat_fs(T_low=0.5, T_high=2.0)
        bridge = SurfaceBridge(fs)
        with pytest.raises(ValueError):
            local_vol_at(bridge, 100.0, 0.1)

    def test_local_vol_grid_return_type(self) -> None:
        """build_local_vol_grid returns a LocalVolGrid."""
        fs = _flat_fs()
        bridge = SurfaceBridge(fs)
        strikes = np.array([90.0, 100.0, 110.0])
        maturities = np.array([0.5, 2.0])
        lvg = build_local_vol_grid(bridge, strikes, maturities)
        assert isinstance(lvg, LocalVolGrid)
        assert len(lvg.strikes) == 3
        assert len(lvg.maturities) == 2
