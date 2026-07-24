# vol-derivatives-pricer

A Python engine for pricing volatility swaps, variance swaps, and exotic structured products (range accruals, shark fin notes) off a calibrated arbitrage-free implied volatility surface.

This project is a downstream consumer of [`arbfree_vol`](https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine) — the surface engine handles calibration and no-arbitrage repair; this repo handles pricing derivatives off that surface.

## Project Goal

Most student pricing projects price vanilla options off a flat Black-Scholes vol assumption. This project instead prices path-dependent and volatility-linked instruments using the shape of the *entire* calibrated smile/surface as the input — the same way a real vol desk would source pricing inputs.

Specifically:

- Static-replicate variance swaps from a strip of OTM option prices read off the surface
- Convexity-adjust into vol swap fair strikes
- Price exotic structured notes (range accrual, shark fin) via Monte Carlo simulation under the surface-implied local vol

## Dependency on the Surface Engine

This repo does not reimplement SVI calibration, IV solving, or arbitrage detection — it imports them.

```toml
# pyproject.toml
dependencies = [
    "arbfree_vol @ git+https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine.git"
]
```

`surface_bridge.py` is the only integration point: it wraps a calibrated `arbfree_vol` surface object and exposes a clean `get_iv(strike, expiry)` / `get_option_price(strike, expiry, cp)` interface. Nothing downstream talks to `arbfree_vol` directly.

## Current Scope

`v0.1` is live and testable. Four pricing modules are implemented against
a calibrated SVI surface:

- **Variance swap** — fair strike via log-contract static replication
  (Demeterfi et al.)
- **Vol swap** — Brockhaus-Long convexity-adjusted fair strike
- **Dupire local vol** — grid evaluation from the SVI surface
- **Range accrual** — Monte Carlo simulation under surface-implied local vol

27 tests pass on a flat synthetic SVI surface (`b=0, rho=0`) that collapses
each exotic to its Black-Scholes closed form.  Skewed-surface tests are
being added (see below).

## Features

### Surface Integration
- [x] `surface_bridge.py`: load a calibrated surface from `arbfree_vol`, expose `get_iv()` / `get_option_price()`

### Variance & Vol Swaps
- [x] Fair variance strike via log-contract static replication (Demeterfi et al.) — integrate `1/K²`-weighted OTM option prices across strikes pulled from the surface
- [x] Vol swap fair strike via convexity adjustment off the variance swap (Brockhaus-Long approximation)
- [x] Surface-driven vol-of-vol estimator (`nu_from_surface`) — quadratic fit in log-moneyness space

### Structured Products
- [x] Local vol construction from the SVI surface (Dupire), or direct strike/time IV interpolation as a simpler first pass
- [x] Range accrual note: Monte Carlo path simulation under surface-implied vol, coupon accrues per day spot is in-range, discounted to PV
- [ ] Shark fin note (stretch goal): decompose into knock-out barrier + digital, price both legs analytically and cross-check against Monte Carlo

## Examples

Two end-to-end demos on live SPY data (require network — they fetch via
`arbfree_vol`'s yfinance ingestion and calibrate an SVI surface before
pricing):

```
python examples/spy_variance_swap.py     # prints K_var^2, K_vol across expiries; saves PNG
python examples/spy_range_accrual.py     # prices a 1-year note; prints PV with 95% CI; saves PNG
```

The demos use `load_calibrated_surface("SPY", max_expiries=24)` — a
one-line wrapper over the upstream fetch -> repair -> FittedSurface
pipeline — so 1-year pricing lands on-surface (SPY monthlies extend ~2y).

![SPY variance swap fair strike](examples/spy_variance_swap.png)

## Quick Start

```bash
pip install -e .
pytest tests/ -q
python -c "from voldrv import SurfaceBridge, fair_variance_strike; print('OK')"
```

3. Price off live SPY data: see [Examples](#examples)

## Project Structure

```
vol-derivatives-pricer/
  pyproject.toml
  README.md
  examples/
    spy_variance_swap.py     # live SPY variance + vol swap fair strikes
    spy_range_accrual.py     # live SPY range-accrual MC pricer
    spy_variance_swap.png    # generated output plot
    spy_range_accrual.png    # generated output plot
  src/
    voldrv/
      __init__.py               # package-level re-exports
      surface_bridge.py         # wraps arbfree_vol, exposes iv(K,T) and option prices
      swaps/
        __init__.py             # subpackage re-exports
        variance_swap.py        # static replication -> fair variance strike
        vol_swap.py             # convexity-adjusted fair vol strike + nu estimator
      structured/
        __init__.py             # subpackage re-exports
        local_vol.py            # Dupire local vol from the SVI surface
        range_accrual.py        # MC pricing of range accrual notes
        shark_fin.py            # barrier + digital decomposition (stub — planned)
  tests/
    test_surface_bridge.py
    test_variance_swap.py
    test_vol_swap.py
    test_local_vol.py
    test_range_accrual.py
    test_skewed_surface.py
```

## Build Order

1. [x] `surface_bridge.py` — integration point with the surface engine
2. [x] Variance swap replication — correctness baseline, checkable against published examples
3. [x] Vol swap convexity adjustment + `nu_from_surface` estimator
4. [x] Local vol construction
5. [x] Range accrual Monte Carlo pricer
6. [ ] Shark fin note (stretch)

## Tech Stack

- Python
- NumPy, SciPy
- `arbfree_vol` (git dependency)
- Pytest

## References

- Demeterfi, Derman, Kamal, Zou — *More Than You Ever Wanted to Know About Volatility Swaps*
- Brockhaus & Long — convexity adjustment for volatility swaps from variance swaps
- Dupire — local volatility from the implied vol surface
