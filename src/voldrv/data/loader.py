"""One-line ingestion: fetch live option chain, repair, wrap in a SurfaceBridge.

Wraps the upstream arbfree_vol pipeline (yfinance fetch -> cleaning ->
repair -> build_fitted_surface) into a single call so downstream
pricing code can do ``bridge, meta = load_calibrated_surface("SPY")``
with no knowledge of the surface-engine internals.

Single integration point for LIVE DATA (mirrors the role surface_bridge.py
plays for the surface object itself).
"""

from voldrv.surface_bridge import SurfaceBridge


def load_calibrated_surface(
    symbol: str,
    max_expiries: int = 12,
    use_ssvi: bool = False,
    use_sabr: bool = False,
    min_T_years: float = 7.0 / 365.0,
) -> tuple[SurfaceBridge, dict]:
    """Fetch a live option chain, calibrate an arb-free surface, wrap it.

    Pipeline:
      1. arbfree_vol.ingestion.yfinance.fetch_chain(symbol, ...)
      2. arbfree_vol.repair.engine.repair(surface, use_ssvi, use_sabr)
      3. (guard) raise if repair produced zero fitted slices
      4. arbfree_vol.surface.interpolate.build_fitted_surface(report)
      5. wrap in SurfaceBridge; build a metadata dict

    Parameters
    ----------
    symbol : str
        Equity ticker, e.g. "SPY".
    max_expiries : int
        Number of nearest expiries to pull (default 12 -- ~3 months of
        SPY weeklies+monthlies).
    use_ssvi / use_sabr : bool
        Pass through to repair(); default SVI smile model.
    min_T_years : float
        Drop expiries closer than this (default 7/365 = 1 week).

    Returns
    -------
    (SurfaceBridge, dict)
        bridge : ready to pass to fair_variance_strike / price_range_accrual
            / etc.
        meta keys : symbol, spot, risk_free, div_yield, n_quotes_total,
            n_quotes_rejected, rejection_rate, violations_before,
            violations_after, n_slices_input, n_slices_fitted,
            fitted_expiries (tuple of years), avg_rmse

    Raises
    ------
    RuntimeError
        if repair() returned a report with zero fitted slices (cannot
        build a usable surface -- e.g. all quotes rejected).
    """
    # Lazy imports so that importing voldrv.data never requires yfinance
    # at import time (keeps test imports network-free).
    from arbfree_vol.ingestion.yfinance import fetch_chain
    from arbfree_vol.repair.engine import repair
    from arbfree_vol.surface.interpolate import build_fitted_surface

    surface, _rejected = fetch_chain(
        symbol, max_expiries=max_expiries, min_T_years=min_T_years
    )
    report = repair(surface, use_ssvi=use_ssvi, use_sabr=use_sabr)

    fitted = report.fitted_slices
    if not fitted:
        raise RuntimeError(
            f"load_calibrated_surface({symbol!r}): repair() produced 0 "
            f"fitted slices (rejected {report.metrics.n_rejected} of "
            f"{report.metrics.n_total_quotes} quotes); cannot build a surface."
        )

    fs = build_fitted_surface(report)
    bridge = SurfaceBridge(fs)

    m = report.metrics
    avg_rmse = float(sum(s.rmse for s in fitted) / len(fitted)) if fitted else float("nan")
    meta = {
        "symbol": symbol,
        "spot": fs.spot,
        "risk_free": fs.risk_free,
        "div_yield": fs.div_yield,
        "n_quotes_total": m.n_total_quotes,
        "n_quotes_rejected": m.n_rejected,
        "rejection_rate": m.rejection_rate,
        "violations_before": m.n_violations_before,
        "violations_after": m.n_violations_after,
        "n_slices_input": m.n_slices_input,
        "n_slices_fitted": m.n_slices_fitted,
        "fitted_expiries": tuple(s.expiry_time for s in fitted),
        "avg_rmse": avg_rmse,
    }
    return bridge, meta
