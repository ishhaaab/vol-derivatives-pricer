# Known Issues

## Issue 1: SVI constrained calibration fails with "maximum number of function evaluations is exceeded"

- **Severity:** Low (non-fatal, handled per slice) but can bias term-structure pricing
- **Origin:** upstream `arbfree-vol-surface` at `src/arbfree_vol/svi/calibration.py:87`, called from `repair/engine.py:112`
- **Status:** Observed on live SPY runs. Owned by upstream (SSOT). Not fixed in this repo.
- **First seen:** running `examples/spy_range_accrual.py`

### What the error means

To fit an SVI parametrisation to one expiry's option quotes, the surface engine runs a
constrained nonlinear least-squares optimisation (scipy `minimize`). The optimiser gave up
because it hit its iteration / function-evaluation budget (`maxfev` / `maxiter`) before it
converged. Put plainly: it could not find SVI parameters that fit that slice's implied vols
within the allowed number of tries.

### Why it happens

Almost always a data-quality problem for that specific expiry slice:

- too few valid (log-moneyness, total-variance) points after IV solving (the engine requires
  at least 5 and warns if there are fewer)
- sparse, stale, or illiquid quotes (a far-dated or low-volume expiry)
- wide bid/ask spreads producing noisy or non-monotonic implied vols that the smooth SVI form
  cannot match
- a poor optimiser starting guess for that slice

Because option chains refresh through the day (yfinance caching, new prints), the same symbol
can fail on one run and succeed on the next.

### How it is handled

Upstream `repair/engine.py:_fit_slice` wraps `calibrate_constrained` in a
`try / except RuntimeError` and returns `None` for that slice. It logs a warning and moves on,
so a single failed slice is skipped rather than aborting the whole fit.

Downstream `voldrv/data/loader.py` only raises if `repair()` returns **zero** fitted slices
(`if not fitted: raise RuntimeError`). With partial failures the surface is still built from the
surviving slices, which is why `examples/spy_range_accrual.py` completed and printed a price
despite the red warning.

### Impact on this library

- The affected expiry's vol information is dropped. If several slices fail, the fitted term
  structure is truncated.
- Pricing that needs expiries near a dropped region can fall back to the nearest fitted expiry.
  The range accrual's `_precompute_lv_grid` clamps with a `RuntimeWarning` in that case. That is
  honest, but it means the price uses off-surface vols and can be biased.
- In the observed SPY run, one slice failed but enough remained, so the 1-year range accrual
  still priced on-surface.

### Scope and action

This is upstream behaviour. Per project rules `arbfree-vol-surface` is the SSOT and must not be
modified from this repo, so there is no code change here.

Mitigations already in place or available on the downstream side:

- `load_calibrated_surface(..., max_expiries=24)` already widens the slice set so 1-year pricing
  stays on-surface (SPY monthlies run past 2 years).
- Re-running later often succeeds once fresher quotes are available.
- If a long-dated product keeps landing off-surface, increase `max_expiries` further or
  pre-filter the expiries passed to the loader.

### Reproduction

```
python examples/spy_range_accrual.py
```

Watch for the red `SVI constrained calibration failed for slice T=...; skipping` warnings.
The script still finishes; the warnings indicate dropped expiries.
