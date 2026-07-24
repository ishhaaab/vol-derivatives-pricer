"""Bridge to the calibrated arbfree_vol surface.

Single integration point wrapping a FittedSurface so that downstream
pricing modules never touch arbfree_vol directly.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erfc  # vectorised norm_cdf via erfc; upstream uses scalar math.erf

from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.surface.interpolate import FittedSurface, iv_at


# ---------------------------------------------------------------------------
# Vectorised Black-Scholes helpers
# ---------------------------------------------------------------------------
# The upstream price_floats uses scalar math.erf.  We provide a vectorised
# equivalent via scipy.special.erfc so that SurfaceBridge.get_option_prices
# can price an array of strikes in a single numpy call.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised standard normal CDF via erfc."""
    return 0.5 * erfc(-x * _INV_SQRT2)


def _bs_prices_vec(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    sigma: np.ndarray,
    is_call: bool,
) -> np.ndarray:
    """Vectorised Black-Scholes price for an array of (K, sigma) pairs.

    Mirrors the upstream ``price_floats`` signature but accepts numpy arrays
    for *K* and *sigma*.  The CDF is evaluated via ``_norm_cdf_vec`` which
    uses ``scipy.special.erfc`` (vectorised) rather than ``math.erf`` (scalar).
    """
    sqrt_T = math.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if is_call:
        return S * df_q * _norm_cdf_vec(d1) - K * df_r * _norm_cdf_vec(d2)
    return K * df_r * _norm_cdf_vec(-d2) - S * df_q * _norm_cdf_vec(-d1)


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
        return self.spot * math.exp((self.risk_free - self.div_yield) * T)

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

    def get_option_prices(
        self, strikes: np.ndarray, expiry: float, cp: str
    ) -> np.ndarray:
        """Black-Scholes prices for an array of strikes at one expiry.

        Vectorised via ``scipy.special.erfc``; the per-element IV lookup
        still iterates (arbfree_vol's ``iv_at`` is scalar) but the BS
        pricing itself is fully vectorised.  *cp* is case-insensitive
        like :meth:`get_option_price`.

        Parameters
        ----------
        strikes:
            1-D array of absolute strike prices.
        expiry:
            Time to expiry in years.
        cp:
            ``"C"`` / ``"CALL"`` / ``"c"`` for calls,
            ``"P"`` / ``"PUT"`` / ``"p"`` for puts (case-insensitive).

        Returns
        -------
        np.ndarray
            Option premiums, one per strike.
        """
        u = cp.upper()
        is_call = u.startswith("C")
        if not is_call and not u.startswith("P"):
            raise ValueError(f"cp must indicate call or put; got {cp!r}")
        sigmas = np.array([self.get_iv(float(k), expiry) for k in strikes])
        return _bs_prices_vec(
            S=self.spot, K=strikes, T=expiry, r=self.risk_free,
            q=self.div_yield, sigma=sigmas, is_call=is_call,
        )
