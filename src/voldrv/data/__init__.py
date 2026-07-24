"""Real-data ingestion seam: yfinance -> repair -> SurfaceBridge."""

from voldrv.data.loader import load_calibrated_surface

__all__ = ["load_calibrated_surface"]
