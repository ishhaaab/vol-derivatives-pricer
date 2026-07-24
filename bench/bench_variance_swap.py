import sys; sys.path.insert(0, "src")
import math
import statistics
import time

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.svi.model import SVIParams

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.variance_swap import fair_variance_strike


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

    n_points_list = [1000, 4000, 16000]
    n_runs = 3

    header = f"{'n_points':>8}  {'seconds':>7}  {'us/point':>9}  {'speedup':>7}"
    print(header)
    print("-" * len(header))

    n1000_med = None
    for n in n_points_list:
        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = fair_variance_strike(bridge, T=1.0, n_points=n)
            times.append(time.perf_counter() - t0)

        med = statistics.median(times)
        if n1000_med is None:
            n1000_med = med
            speedup = 1.0
        else:
            speedup = n1000_med / med

        us_per_point = 1e6 * med / n
        print(f"{n:>8}  {med:>7.3f}s  {us_per_point:>9.2f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
