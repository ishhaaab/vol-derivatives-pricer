"""Companion pricing library downstream of the arbfree_vol surface engine."""

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.variance_swap import FairVarianceSwap, fair_variance_strike
from voldrv.swaps.vol_swap import fair_vol_strike

__all__ = [
    "SurfaceBridge",
    "FairVarianceSwap",
    "fair_variance_strike",
    "fair_vol_strike",
]
