"""Local volatility grid from the arbfree_vol SVI surface.

Thin wrapper that renames/reframes the upstream Dupire implementation
so structuredproduct pricing modules stay selfcontained within
``voldrv``.
"""

from dataclasses import dataclass

import numpy as np
from arbfree_vol.pricing.local_vol import dupire_at

from voldrv.surface_bridge import SurfaceBridge



# LocalVolGrid
@dataclass(frozen=True, slots=True)
class LocalVolGrid:
    """Sampled local volatility surface.

    Attributes
    
    strikes:
        Sorted tuple of absolute strikes.
    maturities:
        Sorted tuple of timetoexpiry values (years).
    grid:
        ``grid[i][j]``= local volatility at maturity *i*, strike *j*.
    """

    strikes: tuple[float, ...]
    maturities: tuple[float, ...]
    grid: tuple[tuple[float, ...], ...]


def build_local_vol_grid(
    bridge: SurfaceBridge,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> LocalVolGrid:
    """Build a localvol grid by calling ``arbfree_vol.dupire_at`` at each (K, T).

    Parameters
    
    bridge:
        Surface bridge wrapping a fitted surface.
    strikes:
        1D array of absolute strikes.
    maturities:
        1D array of maturities in years.

    Returns
    
    LocalVolGrid
        Frozen container with the sampled grid.
    """
    grid= np.empty((len(maturities), len(strikes)), dtype=float)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            grid[i, j]= local_vol_at(bridge, float(K), float(T))

    return LocalVolGrid(
        strikes=tuple(float(k) for k in strikes),
        maturities=tuple(float(t) for t in maturities),
        grid=tuple(tuple(float(v) for v in row) for row in grid),
    )


def local_vol_at(bridge: SurfaceBridge, K: float, T: float) -> float:
    """Singlepoint local volatility query.

    Delegates to ``arbfree_vol.pricing.local_vol.dupire_at``.
    """
    return dupire_at(bridge.fitted_surface, K, T)
