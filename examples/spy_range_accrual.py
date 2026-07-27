"""End-to-end range-accrual note pricing off a live surface.

Usage:  python examples/spy_range_accrual.py [--symbol TICKER]
Saves:  examples/<TICKER>_range_accrual.png
"""
import argparse
import sys
from pathlib import Path

# pytest adds src/ to the path via pyproject, but a plain `python examples/foo.py`
# run does not. Point at the src layout so `import voldrv` works either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from voldrv.data import load_calibrated_surface
from voldrv.structured.range_accrual import price_range_accrual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price a range-accrual note off a live calibrated vol surface."
    )
    parser.add_argument(
        "--symbol", default="SPY",
        help="Ticker to fetch and price (default: SPY).",
    )
    args = parser.parse_args()
    symbol = args.symbol

    bridge, meta = load_calibrated_surface(symbol, max_expiries=24)
    S0 = meta["spot"]
    T = 1.0  # 1-year note
    lower = S0 * 0.90   # 90%-of-spot corridor floor
    upper = S0 * 1.10   # 110%-of-spot ceiling
    coupon_rate = 0.05  # 5% total paid at maturity if every day in range

    result = price_range_accrual(
        bridge, T=T, lower=lower, upper=upper, coupon_rate=coupon_rate,
        n_paths=20_000, n_steps_per_year=252, rng_seed=42,
    )

    ci_half = 1.96 * result.pv_std
    print(f"\n{symbol} range-accrual note (T={T:.1f}y, corridor [{lower:.2f}, {upper:.2f}])")
    print(f"  spot            = {S0:.2f}")
    print(f"  coupon_rate     = {coupon_rate:.4f}")
    print(f"  PV              = {result.pv:.6f}")
    print(f"  pv_std          = {result.pv_std:.6f}")
    print(f"  95% CI          = [{result.pv - ci_half:.6f}, {result.pv + ci_half:.6f}]")
    print(f"  accrual fraction= {result.accrual_mean:.4f}  (std {result.accrual_std:.4f})")
    print(f"  n_paths         = {result.n_paths}   n_steps = {result.n_steps}")

    # plot the IV smile at the nearest fitted expiry to T
    fs = bridge.fitted_surface
    # find the slice closest to T using fitted_slices (forward_curve is a
    # tuple of (expiry, forward_price) pairs, not slice objects)
    expiries = [sl.expiry_time for sl in fs.fitted_slices]
    T_near = min(expiries, key=lambda t: abs(t - T))
    Ks = np.linspace(S0 * 0.7, S0 * 1.3, 60)
    ivs = [bridge.get_iv(float(k), T_near) for k in Ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ks, ivs, "-")
    ax.axvline(lower, color="g", ls="--", label=f"lower {lower:.1f}")
    ax.axvline(upper, color="r", ls="--", label=f"upper {upper:.1f}")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Implied vol")
    ax.set_title(f"{symbol} IV smile @ T={T_near:.2f}y with range-accrual corridor")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f"examples/{symbol}_range_accrual.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
