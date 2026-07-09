"""Phase 2: sub-pixel shoreline extraction.

Extracts satellite-derived shorelines using a supervised sand/water/whitewater
classifier combined with marching-squares sub-pixel contouring on the MNDWI
interface (CoastSat/CoastSeg family). Deliberately avoids reduceToVectors and
simple pixel thresholding.
"""
