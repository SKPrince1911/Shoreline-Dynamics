"""Project configuration: single source of truth.

Defines the area of interest (AOI), coordinate reference systems, study period,
Google Earth Engine collection IDs, masking thresholds, seasonal windows, and
notable cyclone events used across every phase of the pipeline.

Pure standard library only — this module deliberately does NOT ``import ee`` at
module level so it imports cleanly in any environment (including this sandbox,
where GEE is unavailable). GEE collection IDs are stored as plain strings and
resolved to ``ee.ImageCollection`` objects by the phase modules that need them.

CRS convention: EPSG:4326 for storage; EPSG:32646 (UTM 46N) for all metric
operations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Area of interest (AOI)
# ---------------------------------------------------------------------------
# Path to the AOI polygon, relative to the repository root. To study a
# different coast, replace data/aoi.geojson (a single-Polygon FeatureCollection
# in EPSG:4326 / CRS84).  <-- TUNABLE
AOI_PATH: str = "data/aoi.geojson"

# Repository root = parent of the directory containing this module (src/).
# Used so AOI loading works regardless of the current working directory.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def load_aoi() -> dict:
    """Read the AOI GeoJSON file and return it as a parsed dictionary.

    Returns:
        The full GeoJSON ``FeatureCollection`` as a Python dict.

    Raises:
        FileNotFoundError: If the AOI file does not exist at ``AOI_PATH``.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    aoi_file: Path = _PROJECT_ROOT / AOI_PATH
    with open(aoi_file, "r", encoding="utf-8") as f:
        aoi: Dict[str, Any] = json.load(f)
    return aoi


def aoi_coordinates() -> list:
    """Return the AOI polygon's exterior coordinate ring.

    Extracts the exterior ring of the first feature's Polygon geometry, i.e. a
    list of ``[longitude, latitude]`` pairs in EPSG:4326. This is the form GEE
    expects when constructing an ``ee.Geometry.Polygon``.

    Returns:
        The exterior linear ring as a list of ``[lon, lat]`` coordinate pairs.
    """
    aoi: Dict[str, Any] = load_aoi()
    geometry: Dict[str, Any] = aoi["features"][0]["geometry"]
    # Polygon coordinates are [ring0, ring1, ...]; ring0 is the exterior ring.
    ring: List[List[float]] = geometry["coordinates"][0]
    return ring


# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
STORAGE_CRS: str = "EPSG:4326"   # Geographic; used for storing vector data.
METRIC_CRS: str = "EPSG:32646"   # UTM zone 46N; used for all metric operations.

# ---------------------------------------------------------------------------
# Study period (inclusive), ISO-8601 date strings
# ---------------------------------------------------------------------------
STUDY_START: str = "1988-01-01"  # <-- TUNABLE
STUDY_END: str = "2025-12-31"    # <-- TUNABLE

# ---------------------------------------------------------------------------
# Google Earth Engine collection IDs
# ---------------------------------------------------------------------------
# Sentinel-2 Surface Reflectance, harmonized.
S2_SR_HARMONIZED: str = "COPERNICUS/S2_SR_HARMONIZED"
# Cloud Score+ quality bands aligned to the harmonized S2 archive.
CLOUD_SCORE_PLUS: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

# Landsat Collection 2, Level 2 (surface reflectance) by sensor.
LANDSAT5_C2_L2: str = "LANDSAT/LT05/C02/T1_L2"
LANDSAT7_C2_L2: str = "LANDSAT/LE07/C02/T1_L2"
LANDSAT8_C2_L2: str = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_C2_L2: str = "LANDSAT/LC09/C02/T1_L2"

# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------
# Cloud Score+ 'cs' band cutoff: pixels with cs below this are masked.
# Lower keeps more (possibly hazier) pixels; higher is stricter.  <-- TUNABLE
CS_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# Scene selection thresholds (per dry-season-year)
# ---------------------------------------------------------------------------
# A candidate is eligible for the annual shoreline only if its AOI cloud cover
# is at or below this percentage.  <-- TUNABLE
CLOUD_THRESHOLD_PCT: float = 10.0
# ...and it covers at least this percentage of the AOI (valid data pixels).
# <-- TUNABLE
COVERAGE_THRESHOLD_PCT: float = 95.0
# Landsat 7 lost its Scan Line Corrector after this date; later ETM+ scenes
# have ~22% striping gaps and are used only to fill otherwise-empty years.
SLC_OFF_DATE: str = "2003-05-31"

# ---------------------------------------------------------------------------
# Seasonal windows (calendar month numbers, 1 = January)
# ---------------------------------------------------------------------------
# Dry season used for the annual median trend composites (spans the new year).
DRY_SEASON_MONTHS: List[int] = [11, 12, 1, 2, 3]   # <-- TUNABLE
# Monsoon season used for pre/post-monsoon event windows.
MONSOON_MONTHS: List[int] = [6, 7, 8, 9]           # <-- TUNABLE

# ---------------------------------------------------------------------------
# Cyclone events (approximate landfall dates near the AOI, ISO-8601)
# ---------------------------------------------------------------------------
# Add or adjust entries as needed for event-based pre/post-cyclone analysis.
# <-- TUNABLE
CYCLONE_EVENTS: Dict[str, str] = {
    "Roanu": "2016-05-21",
    "Mora": "2017-05-30",
    "Mocha": "2023-05-14",
}

# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
# Directory (relative to the repository root) for generated artifacts.
OUTPUT_DIR: str = "outputs"
