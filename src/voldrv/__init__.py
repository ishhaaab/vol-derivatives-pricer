"""Companion pricing library downstream of the arbfree_vol surface engine."""

from voldrv.surface_bridge import SurfaceBridge
from voldrv.swaps.variance_swap import FairVarianceSwap, fair_variance_strike
from voldrv.swaps.vol_swap import FairVolSwap, fair_vol_strike, fair_vol_strike_from_surface, nu_from_surface

__all__ = [
    "SurfaceBridge",
    "FairVarianceSwap",
    "fair_variance_strike",
    "FairVolSwap",
    "fair_vol_strike",
    "nu_from_surface",
    "fair_vol_strike_from_surface",
]
