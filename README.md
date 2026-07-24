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

Planning stage — no pricing modules implemented yet. Build order below.

## Planned Features

### Surface Integration
- [ ] `surface_bridge.py`: load a calibrated surface from `arbfree_vol`, expose `get_iv()` / `get_option_price()`

### Variance & Vol Swaps
- [ ] Fair variance strike via log-contract static replication (Demeterfi et al.) — integrate `1/K²`-weighted OTM option prices across strikes pulled from the surface
- [ ] Vol swap fair strike via convexity adjustment off the variance swap (Brockhaus-Long approximation)
- [ ] Validation: replicate published variance swap examples to check the replication math independently of the surface

### Structured Products
- [ ] Local vol construction from the SVI surface (Dupire), or direct strike/time IV interpolation as a simpler first pass
- [ ] Range accrual note: Monte Carlo path simulation under surface-implied vol, coupon accrues per day spot is in-range, discounted to PV
- [ ] Shark fin note (stretch goal): decompose into knock-out barrier + digital, price both legs analytically and cross-check against Monte Carlo

## Project Structure

```
vol-derivatives-pricer/
  pyproject.toml
  README.md
  src/
    voldrv/
      surface_bridge.py       # wraps arbfree_vol, exposes iv(K,T) and option prices
      swaps/
        variance_swap.py      # static replication -> fair variance strike
        vol_swap.py           # convexity-adjusted fair vol strike
      structured/
        local_vol.py          # Dupire local vol from the SVI surface
        range_accrual.py      # MC pricing of range accrual notes
        shark_fin.py          # barrier + digital decomposition
  tests/
    test_variance_swap.py
    test_vol_swap.py
    test_range_accrual.py
```

## Build Order

1. `surface_bridge.py` — integration point with the surface engine
2. Variance swap replication — correctness baseline, checkable against published examples
3. Vol swap convexity adjustment
4. Local vol construction
5. Range accrual Monte Carlo pricer
6. Shark fin note (stretch)

## Tech Stack

- Python
- NumPy, SciPy
- `arbfree_vol` (git dependency)
- Pytest

## References

- Demeterfi, Derman, Kamal, Zou — *More Than You Ever Wanted to Know About Volatility Swaps*
- Brockhaus & Long — convexity adjustment for volatility swaps from variance swaps
- Dupire — local volatility from the implied vol surface
