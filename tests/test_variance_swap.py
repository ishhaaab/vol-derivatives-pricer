"""Tests for the variance swap fair-strike calculation."""

import math
import time

import numpy as np
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


# ---------------------------------------------------------------------------
# Grid convergence tests
# ---------------------------------------------------------------------------
class TestGridConvergence:
    """Tests that the variance swap strike converges with grid resolution."""

    def test_converges_with_n_points(self) -> None:
        """Compute strike_var for increasing n_points; relative step below 5e-4 for n>=2000."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        ns = [500, 1000, 2000, 4000, 8000]
        values: list[float] = []

        for n in ns:
            result = fair_variance_strike(bridge, T=1.0, n_points=n)
            values.append(result.strike_var)

        # Print convergence table (captured by pytest; visible with -s)
        header = f"{'n_points':>8}  {'K_var²':>12}  {'rel step':>10}"
        print(f"\n{header}")
        print("-" * len(header))
        for i, n in enumerate(ns):
            rel_step = ""
            if i > 0:
                rel = abs(values[i] - values[i - 1]) / values[i]
                rel_step = f"{rel:>10.2e}"
            print(f"{n:>8}  {values[i]:>12.8f}  {rel_step}")
        print()

        # Check convergence criterion: relative change between successive
        # grids below 5e-4 for n >= 2000
        for i in range(2, len(ns)):  # start from i=2 (n=2000)
            rel = abs(values[i] - values[i // 2]) / values[i]
            assert rel < 5e-4, (
                f"n_points {ns[i]}: relative step vs {ns[i//2]} is {rel:.2e}, "
                f"expected < 5e-4"
            )

    def test_monotonic_no_oscillation(self) -> None:
        """On a flat surface, all values across [500..8000] are within 0.1% of each other."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        ns = [500, 1000, 2000, 4000, 8000]
        values: list[float] = []

        for n in ns:
            result = fair_variance_strike(bridge, T=1.0, n_points=n)
            values.append(result.strike_var)

        spread = max(values) - min(values)
        assert spread < 0.001 * values[-1], (
            f"Max-min spread {spread:.2e} exceeds 0.1% of final value {values[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Vectorised option-pricing tests
# ---------------------------------------------------------------------------
class TestVectorisedOptionPrices:
    """Tests for SurfaceBridge.get_option_prices and the fair_variance_strike refactor."""

    def test_vectorised_matches_loop(self) -> None:
        """get_option_prices matches per-element get_option_price."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        strikes = np.linspace(50.0, 200.0, 50)
        for cp in ("C", "P"):
            vec = bridge.get_option_prices(strikes, 1.0, cp)
            loop = np.array([bridge.get_option_price(float(k), 1.0, cp) for k in strikes])
            assert np.allclose(vec, loop, atol=1e-10), (
                f"Vectorised vs loop mismatch for cp={cp}"
            )

    def test_vectorised_invalid_cp_raises(self) -> None:
        """Invalid cp raises ValueError on get_option_prices."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        strikes = np.array([90.0, 100.0, 110.0])
        with pytest.raises(ValueError):
            bridge.get_option_prices(strikes, 1.0, "X")

    def test_fair_variance_strike_unchanged_after_vectorise(self) -> None:
        """The vectorised fair_variance_strike still gives sigma² ≈ 0.04."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        result = fair_variance_strike(bridge, T=1.0, n_points=4000)
        assert result.strike_var == approx(0.04, rel=1e-3)

    def test_perf_print(self) -> None:
        """Print wall times for fair_variance_strike at various n_points (info only)."""
        fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
        bridge = SurfaceBridge(fs)
        for n in (4000, 16000):
            t0 = time.perf_counter()
            _ = fair_variance_strike(bridge, T=1.0, n_points=n)
            elapsed = time.perf_counter() - t0
            print(f"  fair_variance_strike(n_points={n:>6}): {elapsed:.3f}s")
