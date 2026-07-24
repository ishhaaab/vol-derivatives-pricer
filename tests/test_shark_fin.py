"""Tests for the shark-fin structured note pricer.

Covers:
  - Closed-form sanity (high barrier, low barrier, PV structure)
  - Flat-surface cross-check: MC matches BS to within 3 SE
  - Skewed-surface divergence: BS and MC differ beyond MC noise
  - Error handling (barrier below spot, invalid method)
  - Result dataclass field population
"""

import math
import time

import numpy as np
import pytest

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.shark_fin import SharkFinResult, price_shark_fin


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
    sigma: float = 0.2,       # kept for signature symmetry; unused in skewed body
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
# BS closed-form sanity checks
# ---------------------------------------------------------------------------
class TestSharkFinBS:
    """Closed-form sanity — fast, no MC."""

    def test_bs_no_touch_high_barrier_near_one(self) -> None:
        """B=1000 far above S0=100 → P_no_touch > 0.99."""
        bridge = SurfaceBridge(_flat_fs())
        result = price_shark_fin(bridge, T=1.0, B=1000.0, coupon_rate=0.05, method="bs")
        assert result.p_no_touch > 0.99, f"Expected > 0.99, got {result.p_no_touch}"

    def test_bs_no_touch_barrier_at_spot_is_zero(self) -> None:
        """B=100.01 just above S0=100 → P_no_touch < 0.05."""
        bridge = SurfaceBridge(_flat_fs())
        result = price_shark_fin(bridge, T=1.0, B=100.01, coupon_rate=0.05, method="bs")
        assert result.p_no_touch < 0.05, f"Expected < 0.05, got {result.p_no_touch}"

    def test_bs_pv_refund_zero_matches_p_no_touch_times_discount(self) -> None:
        """PV with refund=0 exactly equals discount * coupon * P_no_touch."""
        bridge = SurfaceBridge(_flat_fs())
        result = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0, method="bs",
        )
        expected = math.exp(-0.05) * 0.05 * result.p_no_touch
        assert abs(result.pv - expected) < 1e-12


# ---------------------------------------------------------------------------
# MC flat-surface cross-check
# ---------------------------------------------------------------------------
class TestSharkFinMCFlat:
    """MC vs BS agreement on a flat surface — both MUST match."""

    def test_mc_matches_bs_on_flat_surface_within_3se(self) -> None:
        """On flat σ=0.2: |MC PV − BS PV| < 3 × MC standard error."""
        bridge = SurfaceBridge(_flat_fs())
        bs = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0, method="bs",
        )
        t0 = time.perf_counter()
        mc = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0,
            method="mc", n_paths=50_000, n_steps_per_year=252, rng_seed=42,
        )
        elapsed = time.perf_counter() - t0
        diff = abs(mc.pv - bs.pv)
        n_se = diff / mc.pv_std if mc.pv_std and mc.pv_std > 0 else float("inf")
        print(
            f"\n  [Flat cross-check] BS PV = {bs.pv:.8f}, MC PV = {mc.pv:.8f}, "
            f"MC SE = {mc.pv_std:.8f}, |diff| = {diff:.8f}, "
            f"|diff|/SE = {n_se:.2f}, BS P_no_touch = {bs.p_no_touch:.6f}, "
            f"wall = {elapsed:.1f}s"
        )
        assert diff < 3.0 * mc.pv_std, (
            f"MC PV {mc.pv:.8f} differs from BS PV {bs.pv:.8f} by {diff:.8f} "
            f"({n_se:.1f} SE), expected < 3 SE"
        )

    def test_flat_bs_mc_matches_closed_form_within_3se(self) -> None:
        """On flat σ=0.2: |MC P_no_touch − BS P_no_touch| < 3 × Bernoulli SE."""
        bridge = SurfaceBridge(_flat_fs())
        bs = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0, method="bs",
        )
        mc = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0,
            method="mc", n_paths=50_000, n_steps_per_year=252, rng_seed=42,
        )
        bernoulli_se = math.sqrt(bs.p_no_touch * (1.0 - bs.p_no_touch) / 50_000)
        diff = abs(mc.p_no_touch - bs.p_no_touch)
        n_se = diff / bernoulli_se if bernoulli_se > 0 else float("inf")
        print(
            f"\n  [Flat P_no_touch] BS = {bs.p_no_touch:.6f}, MC = {mc.p_no_touch:.6f}, "
            f"Bernoulli SE = {bernoulli_se:.6f}, |diff| = {diff:.6f}, "
            f"|diff|/SE = {n_se:.2f}"
        )
        assert diff < 3.0 * bernoulli_se, (
            f"MC P_no_touch {mc.p_no_touch:.6f} vs BS {bs.p_no_touch:.6f} "
            f"diff {diff:.6f} ({n_se:.1f} Bernoulli SE)"
        )


# ---------------------------------------------------------------------------
# Smile-premium verification on a skewed surface
# ---------------------------------------------------------------------------
class TestSharkFinSkewed:
    """BS and MC MUST diverge on a put-skew surface — the smile premium."""

    def test_skewed_bs_and_mc_diverge_beyond_mc_noise(self) -> None:
        """On put-skew (b=0.4, rho=-0.4): |MC PV − BS PV| > 3 × MC SE."""
        bridge = SurfaceBridge(_skewed_fs())
        bs = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0, method="bs",
        )
        t0 = time.perf_counter()
        mc = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.0,
            method="mc", n_paths=50_000, n_steps_per_year=252, rng_seed=42,
        )
        elapsed = time.perf_counter() - t0
        diff = abs(mc.pv - bs.pv)
        n_se = diff / mc.pv_std if mc.pv_std and mc.pv_std > 0 else float("inf")
        print(
            f"\n  [Skewed divergence] BS PV = {bs.pv:.8f}, MC PV = {mc.pv:.8f}, "
            f"MC SE = {mc.pv_std:.8f}, |diff| = {diff:.8f}, "
            f"|diff|/SE = {n_se:.2f}, wall = {elapsed:.1f}s"
        )
        assert diff > 3.0 * mc.pv_std, (
            f"Skewed BS PV {bs.pv:.8f} and MC PV {mc.pv:.8f} differ by only "
            f"{diff:.8f} ({n_se:.1f} SE), expected > 3 SE divergence"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestSharkFinErrors:
    """Input validation raises as expected."""

    def test_barrier_below_spot_raises(self) -> None:
        bridge = SurfaceBridge(_flat_fs())
        with pytest.raises(ValueError):
            price_shark_fin(bridge, T=1.0, B=50.0, coupon_rate=0.05, method="bs")

    def test_invalid_method_raises(self) -> None:
        bridge = SurfaceBridge(_flat_fs())
        with pytest.raises(ValueError):
            price_shark_fin(bridge, T=1.0, B=120.0, coupon_rate=0.05, method="abc")


# ---------------------------------------------------------------------------
# Result dataclass smoke test
# ---------------------------------------------------------------------------
class TestSharkFinDataclass:
    """SharkFinResult fields are populated correctly for both methods."""

    def test_result_dataclass_fields(self) -> None:
        bridge = SurfaceBridge(_flat_fs())

        # BS
        bs = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05, refund=0.01, method="bs",
        )
        assert isinstance(bs.pv, float)
        assert isinstance(bs.p_no_touch, float)
        assert bs.method == "bs"
        assert bs.pv_std is None
        assert bs.n_paths is None
        assert bs.barrier == 120.0
        assert bs.expiry == 1.0
        assert bs.coupon_rate == 0.05
        assert bs.refund == 0.01

        # MC (small run for speed)
        mc = price_shark_fin(
            bridge, T=1.0, B=120.0, coupon_rate=0.05,
            method="mc", n_paths=1_000, n_steps_per_year=52, rng_seed=1,
        )
        assert isinstance(mc.pv, float)
        assert isinstance(mc.p_no_touch, float)
        assert mc.method == "mc"
        assert mc.pv_std is not None and mc.pv_std >= 0.0
        assert mc.n_paths == 1_000
        assert mc.barrier == 120.0
        assert mc.expiry == 1.0
        assert mc.coupon_rate == 0.05
        assert mc.refund == 0.0
