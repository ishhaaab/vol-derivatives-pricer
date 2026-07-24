"""Structured-product pricing modules."""

from voldrv.structured.local_vol import LocalVolGrid, build_local_vol_grid, local_vol_at
from voldrv.structured.range_accrual import RangeAccrualResult, price_range_accrual
from voldrv.structured.shark_fin import SharkFinResult, price_shark_fin

__all__ = [
    "LocalVolGrid",
    "build_local_vol_grid",
    "local_vol_at",
    "RangeAccrualResult",
    "price_range_accrual",
    "SharkFinResult",
    "price_shark_fin",
]
