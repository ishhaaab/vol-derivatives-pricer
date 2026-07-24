"""Bridge to the calibrated arbfree_vol surface.

Single integration point wrapping a FittedSurface so that downstream
pricing modules never touch arbfree_vol directly.
"""

from dataclasses import dataclass
from math import exp

from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.surface.interpolate import FittedSurface, iv_at


# 
# SurfaceBridge

@dataclass(frozen=True, slots=True)
class SurfaceBridge:
    """Thin wrapper over arbfree_vol.surface.FittedSurface.

    Exposes ``get_iv(K, T)`` and ``get_option_price(K, T, cp)``.  All
    downstream modules receive a ``SurfaceBridge`` instance; none import
    ``arbfree_vol`` directly.
    """

    fitted_surface: FittedSurface

    #  convenience properties

    @property
    def spot(self) -> float:
        return self.fitted_surface.spot

    @property
    def risk_free(self) -> float:
        return self.fitted_surface.risk_free

    @property
    def div_yield(self) -> float:
        return self.fitted_surface.div_yield

    @property
    def expiries(self) -> tuple[float, ...]:
        return tuple(sl.expiry_time for sl in self.fitted_surface.fitted_slices)

    @property
    def min_expiry(self) -> float:
        return min(sl.expiry_time for sl in self.fitted_surface.fitted_slices)

    @property
    def max_expiry(self) -> float:
        return max(sl.expiry_time for sl in self.fitted_surface.fitted_slices)

    #  forward price 

    def forward(self, T: float) -> float:
        """Forward price at expiry *T* = S · exp((r - q) · T)."""
        return self.spot * exp((self.risk_free - self.div_yield) * T)

    #  surface queries 

    def get_iv(self, strike: float, expiry: float) -> float:
        """Implied volatility at (strike, expiry).

        Delegates to ``arbfree_vol.surface.interpolate.iv_at``.
        """
        return iv_at(self.fitted_surface, strike, expiry)

    def get_option_price(self, strike: float, expiry: float, cp: str) -> float:
        """Black-Scholes price of a call or put at (strike, expiry).

        Parameters
        ----------
        cp:
            ``"C"`` / ``"CALL"`` / ``"c"`` for a call,
            ``"P"`` / ``"PUT"`` / ``"p"`` for a put (case-insensitive).

        Returns
        -------
        float
            Option premium in the same units as ``spot``.
        """
        u = cp.upper()
        is_call = u.startswith("C")
        if not is_call and not u.startswith("P"):
            raise ValueError(f"cp must indicate call or put; got {cp!r}")
        sigma = self.get_iv(strike, expiry)
        return price_floats(
            S=self.spot,
            K=strike,
            T=expiry,
            r=self.risk_free,
            q=self.div_yield,
            sigma=sigma,
            is_call=is_call,
        )
