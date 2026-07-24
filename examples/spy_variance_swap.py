"""End-to-end SPY variance + vol swap fair strikes off a live surface.

Usage:  python examples/spy_variance_swap.py
Saves:  examples/spy_variance_swap.png
"""
import matplotlib
matplotlib.use("Agg")  # headless; safe even with no display
import matplotlib.pyplot as plt

from voldrv.data import load_calibrated_surface
from voldrv.swaps.variance_swap import fair_variance_strike
from voldrv.swaps.vol_swap import fair_vol_strike_from_surface


def main() -> None:
    symbol = "SPY"
    bridge, meta = load_calibrated_surface(symbol, max_expiries=12)
    print(f"\n{symbol} calibration summary")
    print(f"  spot        = {meta['spot']:.2f}")
    print(f"  risk-free r = {meta['risk_free']:.4f}")
    print(f"  div yield q = {meta['div_yield']:.4f}")
    print(f"  quotes kept = {meta['n_quotes_total'] - meta['n_quotes_rejected']:d}/{meta['n_quotes_total']:d}"
          f"  (rejected {meta['n_quotes_rejected']:d})")
    print(f"  violations  = {meta['violations_before']:d} -> {meta['violations_after']:d}")
    print(f"  fitted expiries (yrs): {meta['fitted_expiries']}")
    print(f"  avg RMSE     = {meta['avg_rmse']:.4e}")

    # pick up to 5 expiries within the fitted range, spaced out
    exps = list(meta["fitted_expiries"])
    Ts = sorted(set(exps))[:5]
    rows = []
    for T in Ts:
        vs = fair_variance_strike(bridge, T=float(T))
        vols = fair_vol_strike_from_surface(bridge, T=float(T))
        rows.append((T, vs.strike_var, vs.forward, vols.strike_vol, vols.nu))

    print(f"\n  {'T (yrs)':>8} {'K_var^2':>10} {'Fwd':>8} {'K_vol':>8} {'nu':>8}")
    for T, kv, fwd, kvol, nu in rows:
        print(f"  {T:>8.3f} {kv:>10.5f} {fwd:>8.2f} {kvol:>8.5f} {nu:>8.4f}")

    # plot K_var^2 vs T
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="K_var^2")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Fair variance strike (annualised)")
    ax.set_title(f"{symbol} variance swap fair strike from calibrated SVI surface")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "examples/spy_variance_swap.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
