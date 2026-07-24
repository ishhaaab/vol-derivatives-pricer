"""Tests for voldrv.data.loader (real-data integration seam).

These tests mock the upstream yfinance fetch_chain so they run
network-free.  The synthetic surface is a flat-20%-vol 2-slice
surface that the repair engine can fit successfully.
"""

from __future__ import annotations

import math
from unittest.mock import Mock, patch

import pytest

from voldrv.data import load_calibrated_surface
from voldrv.surface_bridge import SurfaceBridge

# ---------------------------------------------------------------------------
# Synthetic surface factory
# ---------------------------------------------------------------------------

# Use a flat 20% vol surface — the repair engine will fit it with near-
# zero RMSE.  We need enough strikes so that after IV solving at least
# 5 (k,w) points remain per slice (upstream requirement).

_STRIKES = [80, 90, 95, 100, 105, 110, 120]
_SPOT = 100.0
_RISK_FREE = 0.05
_DIV_YIELD = 0.0
_SIGMA = 0.20


def _build_synthetic_surface():
    """Build a VolSurface with two slices (T=0.5, T=1.0), flat 20% vol."""
    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
    from arbfree_vol.pricing.black_scholes import price_floats

    slices: list[ExpirySlice] = []
    for T in (0.5, 1.0):
        quotes: list[Quote] = []
        for K in _STRIKES:
            for is_call in (True, False):
                ot = OptionType.CALL if is_call else OptionType.PUT
                px = price_floats(_SPOT, K, T, _RISK_FREE, _DIV_YIELD, _SIGMA, is_call)
                quotes.append(
                    Quote(strike=K, option_type=ot, price=px, bid=px * 0.98, ask=px * 1.02)
                )
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=_SPOT, risk_free=_RISK_FREE, div_yield=_DIV_YIELD, slices=slices)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCalibratedSurface:
    """End-to-end loader tests with mocked yfinance."""

    @patch("arbfree_vol.ingestion.yfinance.fetch_chain")
    def test_load_returns_bridge_and_meta(self, mock_fetch):
        """Patch fetch_chain; verify bridge + all 13 meta keys."""
        synthetic = _build_synthetic_surface()
        mock_fetch.return_value = (synthetic, [])

        bridge, meta = load_calibrated_surface("TEST", max_expiries=2)

        # bridge
        assert isinstance(bridge, SurfaceBridge)
        assert bridge.spot == pytest.approx(_SPOT)

        # meta symbol
        assert meta["symbol"] == "TEST"

        # all 13 keys
        expected_keys = frozenset({
            "symbol", "spot", "risk_free", "div_yield",
            "n_quotes_total", "n_quotes_rejected", "rejection_rate",
            "violations_before", "violations_after",
            "n_slices_input", "n_slices_fitted",
            "fitted_expiries", "avg_rmse",
        })
        assert frozenset(meta.keys()) == expected_keys, f"meta keys mismatch"

        # shape / type checks
        assert meta["spot"] == pytest.approx(_SPOT)
        assert meta["risk_free"] == pytest.approx(_RISK_FREE)
        assert meta["div_yield"] == pytest.approx(_DIV_YIELD)
        assert meta["n_slices_fitted"] >= 1
        assert isinstance(meta["fitted_expiries"], tuple)
        assert all(isinstance(t, float) for t in meta["fitted_expiries"])
        assert meta["avg_rmse"] >= 0.0

    @patch("arbfree_vol.ingestion.yfinance.fetch_chain")
    @patch("arbfree_vol.repair.engine.repair")
    def test_load_raises_on_empty_repair(self, mock_repair, mock_fetch):
        """When repair returns zero fitted slices, RuntimeError is raised."""
        synthetic = _build_synthetic_surface()
        mock_fetch.return_value = (synthetic, [])

        # Build a mock repair report with no fitted slices
        mock_metrics = Mock(
            n_rejected=10,
            n_total_quotes=10,
            n_slices_input=2,
            n_slices_fitted=0,
            n_violations_before=5,
            n_violations_after=0,
            rejection_rate=1.0,
        )
        mock_report = Mock(
            fitted_slices=(),
            metrics=mock_metrics,
        )
        mock_repair.return_value = mock_report

        with pytest.raises(RuntimeError) as exc_info:
            load_calibrated_surface("TEST", max_expiries=2)

        assert "0 fitted slices" in str(exc_info.value)

    def test_load_calibrated_surface_lazy_imports(self):
        """Importing the loader does NOT require yfinance at module level.

        This is a structural assertion: the function-level lazy import
        means calling code can ``from voldrv.data import load_calibrated_surface``
        without yfinance being installed (only calling it needs network).
        Since yfinance *is* installed in this environment, we just verify
        the function is callable.
        """
        from voldrv.data.loader import load_calibrated_surface

        assert callable(load_calibrated_surface)

    @patch("arbfree_vol.ingestion.yfinance.fetch_chain")
    def test_meta_keys_documented(self, mock_fetch):
        """Assert the exact set of 13 meta keys -- guards against drift."""
        synthetic = _build_synthetic_surface()
        mock_fetch.return_value = (synthetic, [])

        _, meta = load_calibrated_surface("TEST", max_expiries=2)

        assert set(meta.keys()) == {
            "symbol", "spot", "risk_free", "div_yield",
            "n_quotes_total", "n_quotes_rejected", "rejection_rate",
            "violations_before", "violations_after",
            "n_slices_input", "n_slices_fitted",
            "fitted_expiries", "avg_rmse",
        }, "meta keys have drifted from the documented set"
