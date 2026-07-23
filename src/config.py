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


# Path to reference tidal-channel lines (LineString FeatureCollection in
# EPSG:4326) used as a review overlay.  <-- TUNABLE
TIDAL_CHANNELS_PATH: str = "data/tidal_channels.geojson"


def load_tidal_channels() -> dict:
    """Read the tidal-channel GeoJSON file and return it as a parsed dictionary.

    Returns:
        The GeoJSON ``FeatureCollection`` (LineString features) as a Python dict.

    Raises:
        FileNotFoundError: If the file does not exist at ``TIDAL_CHANNELS_PATH``.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    channels_file: Path = _PROJECT_ROOT / TIDAL_CHANNELS_PATH
    with open(channels_file, "r", encoding="utf-8") as f:
        channels: Dict[str, Any] = json.load(f)
    return channels


# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
STORAGE_CRS: str = "EPSG:4326"   # Geographic; used for storing vector data.
METRIC_CRS: str = "EPSG:32646"   # UTM zone 46N; used for all metric operations.

# ---------------------------------------------------------------------------
# Study period (inclusive), ISO-8601 date strings
# ---------------------------------------------------------------------------
# TM (Landsat 4/5) is the first SWIR-capable 30 m sensor (Aug 1982), but
# Landsat 5 only became operational in March 1984, so 1985 is the first
# dry-season-year (1984-1985) with a realistic chance of complete coverage
# using identical MNDWI/sub-pixel logic. Earlier years are expected to be
# sparse. Dry-season-years then run 1985 through 2025.
STUDY_START: str = "1985-01-01"  # <-- TUNABLE
STUDY_END: str = "2025-12-31"    # <-- TUNABLE

# ---------------------------------------------------------------------------
# Google Earth Engine collection IDs
# ---------------------------------------------------------------------------
# Sentinel-2 Surface Reflectance, harmonized.
S2_SR_HARMONIZED: str = "COPERNICUS/S2_SR_HARMONIZED"
# Cloud Score+ quality bands aligned to the harmonized S2 archive.
CLOUD_SCORE_PLUS: str = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

# Landsat Collection 2, Level 2 (surface reflectance) by sensor.
# Landsat 4 TM shares Landsat 5's band layout, QA, and scaling exactly; it
# extends the record back to 1982 (though useful data is sparse before 1985).
LANDSAT4_C2_L2: str = "LANDSAT/LT04/C02/T1_L2"
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
# Relaxed AOI coverage floor: used only when no scene meets the strict
# threshold, so a slightly-clipped but clear image can still be selected
# (flagged relaxed_coverage=True).  <-- TUNABLE
COVERAGE_THRESHOLD_RELAXED_PCT: float = 90.0
# A single scene at or above this AOI coverage counts as full-coast, so no
# gap-fill composite is needed.  <-- TUNABLE
COVERAGE_COMPLETE_PCT: float = 99.5

# Scales (m) for AOI coverage/cloud reductions: a coarse grid for fast
# screening and a finer grid for final coverage verification.  <-- TUNABLE
COVERAGE_SCALE: int = 100
COVERAGE_SCALE_FINE: int = 30

# Greedy gap-fill composite (minimal set-cover) tolerances, in percent.
# Ties within a tolerance are broken by lowest cloud. A candidate date whose
# marginal coverage gain is below the minimum is not added (avoids redundant
# dates that would each add another acquisition time/tide).  <-- TUNABLE
SEED_COVERAGE_TOLERANCE_PCT: float = 1.0
MARGINAL_GAIN_TOLERANCE_PCT: float = 0.5
MIN_MARGINAL_GAIN_PCT: float = 0.25
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

# ---------------------------------------------------------------------------
# Phase 2 — sub-pixel shoreline extraction (see PHASE2_SPEC.md §1)
# ---------------------------------------------------------------------------
# Surface reflectance is retained (D3): Landsat C2 L2 + S2_SR_HARMONIZED. The
# CoastSat-shipped classifiers are TOA-trained and invalid here, so a LOCAL
# classifier is trained instead.
REFLECTANCE_LEVEL: str = "SR"  # locked: Landsat C2 L2 + S2_SR_HARMONIZED

# SR band aliases per sensor. Keys are the canonical names used everywhere
# downstream; values are the native GEE band names. ``swir2`` is carried on
# every scene because AWEInsh (a live water-index candidate, D4) needs it, and
# retrofitting a band after fetch_scene exists means re-downloading every scene.
#   Landsat TM/ETM+ (L4/L5/L7):  SR_B7 = SWIR2 (~2.2 um)
#   Landsat OLI    (L8/L9):      SR_B7 = SWIR2 (~2.2 um)
#   Sentinel-2 MSI (S2):         B12   = SWIR2 (~2.19 um); B11 = SWIR1
BAND_MAP: Dict[str, Dict[str, str]] = {
    "L4": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"},
    "L5": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"},
    "L7": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"},
    "L8": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
    "L9": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
    "S2": {"blue": "B2",    "green": "B3",    "red": "B4",    "nir": "B8",    "swir1": "B11",   "swir2": "B12"},
}

# Sensor -> reflectance-scale family. Used to look up SR_SCALE and the
# per-family georef-RMSE fallback without hardcoding sensor lists downstream.
SR_SCALE_FAMILY: Dict[str, str] = {
    "L4": "LANDSAT", "L5": "LANDSAT", "L7": "LANDSAT",
    "L8": "LANDSAT", "L9": "LANDSAT", "S2": "S2",
}

# Reflectance scaling to physical [0, 1], as (gain, offset): value*gain + offset.
# Landsat C2 L2: DN*2.75e-5 - 0.2. S2_SR_HARMONIZED: DN/10000 (the harmonized
# collection already removes the post-2022-01-25 baseline-04.00 +1000 offset —
# this is why the HARMONIZED collection is used).
SR_SCALE: Dict[str, tuple] = {
    "LANDSAT": (2.75e-5, -0.2),
    "S2": (1e-4, 0.0),
}

# Native analysis grid (m) per sensor. Pansharpening is OFF by default: the
# Landsat pan band (B8) exists only in the L1 TOA collections, so pansharpening
# SR multispectral requires a cross-product fetch. Evaluated as an option in the
# benchmark (PHASE2_SPEC.md §6), not assumed.  <-- TUNABLE
PIXEL_SIZE_M: Dict[str, int] = {
    "L4": 30, "L5": 30, "L7": 30, "L8": 30, "L9": 30, "S2": 10,
}
PANSHARPEN: bool = False  # <-- TUNABLE

# Fallback georeferencing RMSE (m) when per-scene metadata is absent (D5).
# CoastSat uses 12 m as the Landsat-collection average.  <-- TUNABLE
GEOREF_RMSE_DEFAULT_M: Dict[str, float] = {"LANDSAT": 12.0, "S2": 11.0}

# Pixel classes (classifier output codes, D3).
CLASS_OTHER: int = 0
CLASS_SAND: int = 1
CLASS_WHITEWATER: int = 2
CLASS_WATER: int = 3

# Apply a 3x3 majority filter to the classifier LABEL MAP (only) after predict,
# before it reaches the interface threshold. Speckled misclassification poisons
# the sand-union-water histogram and drifts the Otsu threshold. This filters the
# discrete label raster, NOT the water index or the sub-pixel contour (those are
# never smoothed — that would reintroduce the pixel quantisation this design
# avoids).  <-- TUNABLE
LABEL_MAJORITY_FILTER: bool = True

# Water indices available to the benchmark (D4).  <-- TUNABLE
WATER_INDICES: List[str] = ["mndwi", "ndwi", "aweinsh", "scowi"]
WATER_INDEX_DEFAULT: str = "mndwi"
THRESHOLD_METHODS: List[str] = ["otsu", "weighted_peaks"]
THRESHOLD_METHOD_DEFAULT: str = "otsu"

# Contour filtering: minimum length of a MERGED shoreline segment (applied AFTER
# linemerge, so fragmented contours from cloud edges or L7 SLC-off striping are
# stitched before this floor is applied). This coast is cut by 13 tidal channels,
# so legitimate inter-channel shoreline segments are well under 2 km — the floor
# only removes short spurious specks, not real inter-channel reaches.  <-- TUNABLE
MIN_SHORELINE_LENGTH_M: float = 300.0  # <-- TUNABLE
# QGIS-digitised search zone constraining extraction to the coast (D1 filtering;
# replaces CoastSat's scalar max_dist_ref). See PHASE2_SPEC.md §4.
SEARCH_ZONE_PATH: str = "data/shoreline_search_zone.geojson"

# A dry-season composite whose contributing acquisitions span more than this many
# days mixes shoreline positions from different parts of the season (e.g. 1995 =
# 1994-11-20 + 1995-03-19, 119 days apart). Per-scene extraction (D1) fixes the
# tide problem for these but not the temporal one. The scene list, shoreline
# records, and merged annual file carry ``composite_date_spread_days``; years
# above this threshold are flagged so Phase 4 can add an E_temporal term to the
# RSS budget (~ local change rate x spread/2).  <-- TUNABLE
COMPOSITE_SPREAD_FLAG_DAYS: float = 60.0

# Series B (dense, all-season) query envelope (D2).  <-- TUNABLE
# End at 2026-04-30 (not 2025-12-31) so the COMPLETE 2025-2026 dry season
# (Nov 2025 - Mar 2026) is captured for slope density. A Series B dry_year spans
# Nov(Y-1)-Oct(Y), so the boundary years 1999 (missing its Nov-Dec 1998 head) and
# 2026 (missing its 2026 monsoon tail) are all-season-incomplete; they are marked
# season_complete=False so no per-dry_year seasonal statistic uses a partial year.
DENSE_START: str = "1999-01-01"
DENSE_END: str = "2026-04-30"
DENSE_SENSORS: List[str] = ["L7", "L8", "L9", "S2"]
DENSE_CLOUD_MAX_PCT: float = 30.0     # relaxed vs the 10% annual rule
DENSE_COVERAGE_MIN_PCT: float = 50.0  # partial scenes are still useful for slope

# Landsat CFMask misflags bright beach/whitewater as cloud (CoastSat exposes
# ``cloud_mask_issue`` for exactly this). When True, pixels flagged cloud but
# classified sand/whitewater with high confidence are NOT masked.  <-- TUNABLE
LANDSAT_CLOUD_MASK_ISSUE: bool = True
