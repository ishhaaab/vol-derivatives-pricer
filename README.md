# vol-derivatives-pricer

A Python library for pricing variance swaps, vol swaps, and exotic structured products (range accruals, shark fin notes) off a calibrated, arbitrage-free implied volatility surface.

This is a downstream consumer of [`arbfree_vol`](https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine) — the upstream engine handles SVI calibration, IV solving, and arbitrage repair. This repo takes that surface as input and prices derivatives whose payoffs depend on realised volatility, barrier events, or the shape of the smile across strikes and maturities.

## The Problem

Vanilla options are priced with a single at-the-money implied volatility. But many derivatives on a volatility desk depend on the *entire* vol surface — the skew across strikes, the term structure across expiries, and the local volatility that drives path-dependent payoffs.

Specifically:

- A **variance swap** replicates realised variance with a log-contract portfolio of vanilla options across all strikes. Its fair strike is the integrated implied variance, weighted by 1/K².
- A **vol swap** pays realised volatility, not variance. Since volatility is a concave function of variance, there is a convexity gap that must be estimated from the surface's skew.
- A **range accrual** pays a coupon for every day the underlying stays within a corridor. Its price depends on the probability of staying in-range — which requires a full local-vol path simulation.
- A **shark fin note** has a barrier that extinguishes the payoff if breached. Pricing it requires both barrier probability and volatility smile awareness.

Most student pricing projects treat vol as a single number. This project prices these instruments the way a real vol desk would — using the shape of the calibrated SVI surface as the input.

## What It Does

Given a calibrated SVI vol surface, the library produces verifiable prices with documented error bounds:

**Variance swap** — Fair strike via Demeterfi static replication. Integrates 1/K²-weighted OTM option prices across a 4000-point log-spaced strike grid, read directly off the surface. On a flat SVI surface (b=0, ρ=0), the result converges to σ² within 5e-4 relative quadrature error by 4000 points.

**Vol swap** — Fair strike via Brockhaus-Long convexity correction. Estimates the vol-of-vol (ν) from the surface's skew slope using a quadratic fit in log-moneyness space, then adjusts the variance-swap strike to produce a vol-swap strike. On a skewed surface (b=0.4, ρ=-0.4), the estimated ν is 5.94, producing a measurable gap from the flat-vol approximation.

**Range accrual** — Monte Carlo under Dupire local vol with antithetic variates. Local vol is pre-computed on a 2D grid (strike × time) and interpolated during simulation, eliminating per-step Dupire calls. A 95% confidence interval is reported via pair-based standard error (textbook-correct for antithetic). On real SPY data (spot 738.25, r=3.79%, q=1.01%), a 1-year 95%-of-S0 corridor range accrual prices to PV=0.0301 with 95% CI [0.0299, 0.0303].

**Shark fin** — Two independent methods cross-checked against each other. The closed-form method uses the reflection principle to compute no-touch probability under flat vol. The MC method runs log-Euler with Brownian bridge barrier checking (Glasserman §6.2, corrects the 5-8 SE bias of raw discrete Euler). On a flat surface, the two methods agree within 1 SE (BS PV=0.02793, MC PV=0.02783, divergence = 1.02 SE). On a skewed surface, the MC prices 60% higher (BS PV=0.01993, MC PV=0.03198, divergence = 120.69 SE) — this is the smile premium, a real market effect, not noise.

## Architecture

The surface engine (`arbfree_vol`) is the single upstream dependency. `surface_bridge.py` is the only integration point: it wraps a calibrated `arbfree_vol` surface object and exposes a clean interface. Nothing downstream talks to `arbfree_vol` directly.

```
arbfree_vol (upstream)
  └── FittedSurface (calibrated SVI)
        └── surface_bridge.py    ← only integration point
              ├── get_iv(strike, expiry)
              ├── get_option_price(strike, expiry, cp)
              ├── get_option_prices(strikes, expiry, cp)   ← vectorised
              └── surface object (for direct access)
                    ├── swaps/variance_swap.py   ← Demeterfi replication
                    ├── swaps/vol_swap.py        ← Brockhaus-Long + ν estimator
                    ├── structured/range_accrual.py  ← MC + antithetic + local vol
                    ├── structured/shark_fin.py  ← BS formula + Brownian bridge MC
                    └── structured/local_vol.py  ← Dupire local vol grid
```

The upstream handles SVI calibration, IV solving, and arbitrage repair. This library takes the resulting surface and prices derivatives — the clean separation means you can swap in a different calibration method without changing any pricing code.

## Pricing Modules

### Variance Swap (`src/voldrv/swaps/variance_swap.py`)

Static replication of the log-contract payoff. Integrates OTM option prices across strikes weighted by 1/K² (puts for K ≤ F, calls for K > F). Uses 4000 log-spaced points by default. The integrand is vectorised via `scipy.special.erfc` — 4000 points evaluate in ~32ms.

### Vol Swap (`src/voldrv/swaps/vol_swap.py`)

Brockhaus-Long convexity adjustment. Estimates ν (vol-of-vol) from the surface's skew slope by fitting a quadratic to IV vs log-moneyness. Adjusts the variance-swap strike to produce a vol-swap strike. Also exposes `fair_vol_strike_from_surface()` as an end-to-end wrapper.

### Dupire Local Vol (`src/voldrv/structured/local_vol.py`)

Constructs a local vol grid from the SVI surface. Supports pre-computation with 2D interpolation (bilinear on time, spline on strike) — the grid is built once and reused across all MC paths, eliminating per-step Dupire calls.

### Range Accrual (`src/voldrv/structured/range_accrual.py`)

Monte Carlo simulation under surface-implied local vol. Antithetic variates reduce variance. Local vol is pre-computed on a 2D grid and interpolated during simulation. Reports PV with pair-based standard error (textbook-correct for antithetic). Expiry-time safety: if a path reaches an expiry beyond the surface's fitted range, the local vol is clamped to the nearest available expiry with a `RuntimeWarning`.

### Shark Fin (`src/voldrv/structured/shark_fin.py`)

Two methods. Closed-form: reflection principle for single upper barrier, no-touch probability under flat vol, exact formula. MC: log-Euler with Brownian bridge barrier checking (Glasserman §6.2, corrects discrete monitoring bias). On a flat surface the two agree within 1 SE. On a skewed surface the MC prices higher — the difference is the smile premium, a genuine market effect.

## Real-Data Results (SPY)

Run with live network data via `load_calibrated_surface("SPY")`:

| Expiry | K_var² | σ² | fwd | K_vol | ν |
|--------|--------|----|-----|-------|---|
| 5 days | 0.40 | 0.41 | 1.000 | 0.55 | — |
| 36 days | 0.14 | 0.12 | 1.001 | 0.30 | — |
| 65 days | 0.11 | 0.11 | 1.001 | 0.27 | — |
| 122 days | 0.09 | 0.09 | 1.002 | 0.24 | — |
| 357 days | 0.06 | 0.06 | 1.005 | 0.21 | 5.94 |

Range accrual (1-year, corridor [0.9·S₀, 1.1·S₀], 5% coupon): PV=0.0301, 95% CI [0.0299, 0.0303], accrual fraction=0.625.

Shark fin (flat-surface cross-check): BS PV=0.02793, MC PV=0.02783, divergence=1.02 SE. Skewed-surface divergence=120.69 SE (smile premium = 60% relative).

## Examples

Two end-to-end demos on live SPY data (require network — they fetch via `arbfree_vol`'s yfinance ingestion and calibrate an SVI surface before pricing):

```
python examples/spy_variance_swap.py     # prints K_var², K_vol across expiries; saves PNG
python examples/spy_range_accrual.py     # prices a 1-year note; prints PV with 95% CI; saves PNG
```

The demos use `load_calibrated_surface("SPY", max_expiries=24)` — a one-line wrapper over the upstream fetch → repair → FittedSurface pipeline — so 1-year pricing lands on-surface (SPY monthlies extend ~2y).

![SPY variance swap fair strike](examples/spy_variance_swap.png)

## Quick Start

```bash
# install (editable mode)
pip install -e .

# run tests
pytest tests/ -q

# verify top-level imports work without network
python -c "from voldrv import SurfaceBridge, fair_variance_strike; print('OK')"

# price off live SPY data (needs network)
python examples/spy_variance_swap.py
python examples/spy_range_accrual.py
```

## Continuous Integration

GitHub Actions runs `pytest tests/ -v` and `pyright src/` on every push and pull request to `main`, across Python 3.11 and 3.12 on Ubuntu and Windows (4 jobs). The `examples/` directory is intentionally not exercised in CI — those scripts need network access (yfinance fetch) and are run on demand.

See `.github/workflows/ci.yml` for the full matrix definition.

## Project Structure

```
vol-derivatives-pricer/
  pyproject.toml
  README.md
  examples/
    spy_variance_swap.py        # live SPY variance + vol swap fair strikes
    spy_range_accrual.py        # live SPY range-accrual MC pricer
    spy_variance_swap.png       # generated output plot
    spy_range_accrual.png       # generated output plot
  src/
    voldrv/
      __init__.py               # package-level re-exports
      surface_bridge.py         # wraps arbfree_vol, exposes iv(K,T) and option prices
      swaps/
        __init__.py             # subpackage re-exports
        variance_swap.py        # static replication → fair variance strike
        vol_swap.py             # convexity-adjusted fair vol strike + ν estimator
      structured/
        __init__.py             # subpackage re-exports
        local_vol.py            # Dupire local vol from the SVI surface
        range_accrual.py        # MC pricing of range accrual notes
        shark_fin.py            # BS formula + Brownian bridge MC
      data/
        __init__.py             # network-free package init
        loader.py               # load_calibrated_surface("SPY") one-line wrapper
  tests/
    test_surface_bridge.py
    test_variance_swap.py
    test_vol_swap.py
    test_local_vol.py
    test_range_accrual.py
    test_skewed_surface.py
    test_loader.py
    test_shark_fin.py
  bench/
    bench_variance_swap.py      # integrand vectorisation timing
    bench_range_accrual.py      # MC convergence + runtime timing
    README.md                   # benchmark methodology
```

## Tech Stack

- Python ≥ 3.11
- NumPy, SciPy (vectorised integrand via `scipy.special.erfc`)
- matplotlib (optional, for example plots)
- yfinance (optional, only in examples — fetched via `arbfree_vol`)
- [`arbfree_vol`](https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine) — upstream SVI calibration, IV solving, arbitrage repair
- Pytest (test suite)
- Pyright (static type checking, standard mode)

## References

- Demeterfi, Derman, Kamal, Zou — *More Than You Ever Wanted to Know About Volatility Swaps*
- Brockhaus & Long — convexity adjustment for volatility swaps from variance swaps
- Dupire — local volatility from the implied vol surface
- Glasserman — *Monte Carlo Methods in Financial Engineering* (§6.2, Brownian bridge barrier correction)
