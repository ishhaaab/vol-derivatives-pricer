import sys; sys.path.insert(0, "src")
import math
import time

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.structured.range_accrual import price_range_accrual


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


def main() -> None:
    fs = _flat_fs(sigma=0.2, S=100.0, r=0.05, q=0.0)
    bridge = SurfaceBridge(fs)

    configs = [
        (1000, 63),
        (10000, 252),
        (50000, 252),
    ]

    header = f"{'n_paths':>7}  {'n_steps':>7}  {'seconds':>7}  {'pv':>8}  {'pv_std':>9}  {'paths/s':>10}"
    print(header)
    print("-" * len(header))

    for n_paths, n_steps_per_year in configs:
        t0 = time.perf_counter()
        result = price_range_accrual(
            bridge,
            T=1.0,
            lower=95.0,
            upper=105.0,
            coupon_rate=0.05,
            n_paths=n_paths,
            n_steps_per_year=n_steps_per_year,
            rng_seed=42,
        )
        t = time.perf_counter() - t0

        paths_per_sec = n_paths / t if t > 0 else float("inf")
        print(
            f"{n_paths:>7}  {n_steps_per_year:>7}  {t:>7.3f}s  "
            f"{result.pv:>8.5f}  {result.pv_std:>9.6f}  "
            f"{paths_per_sec:>10.0f}"
        )


if __name__ == "__main__":
    main()
