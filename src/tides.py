"""Phase 3: FES2022 tidal and beach-slope correction.

Applies tidal correction to extracted shorelines using FES2022 (via pyTMD/pyfes)
and beach slopes estimated with CoastSat.slope (Lomb-Scargle). Converts tidal
elevation to a horizontal shoreline shift via X = (Z_tide - Z_MSL) / tan(beta).
"""
