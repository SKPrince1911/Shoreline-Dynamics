"""Phase 1: image retrieval and quality/cloud masking.

Retrieves Sentinel-2 SR Harmonized and Landsat 5/7/8/9 Collection 2 L2 imagery
from Google Earth Engine and applies quality/cloud masking (Cloud Score+ for
Sentinel-2 at cs ~0.55; QA_PIXEL/CFMask for Landsat), producing the dry-season
(Nov–Mar) annual median composites and single-scene stacks that feed later
phases.
"""
