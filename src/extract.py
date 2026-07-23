"""Phase 2: sub-pixel shoreline extraction (CoastSat/CoastSeg family).

Implements the locked Phase 2 design (see ``PHASE2_SPEC.md`` at the repo root
and the D1–D7 block in ``CLAUDE.md``):

* **D1** — the extraction unit is the SCENE (one acquisition / one tide), never
  the dry-season-year. Same-overpass granules (adjacent Landsat rows, adjacent
  S2 tiles) are mosaicked into one scene; multi-*date* products are split so
  every output segment keeps its own ``image_id`` and ``acq_datetime_utc``.
* **D2** — two scene series: A (38 annual dry-season products, the trend layer)
  and B (dense, all-season 1999–2025, for CoastSat.slope).
* **D3** — surface reflectance is retained and a LOCAL sand/water/whitewater/
  other MLP classifier is trained per sensor group (TM = L4/L5/L7, OLI = L8/L9,
  MSI = S2). CoastSat's shipped TOA classifiers are invalid on SR imagery.
* **D4** — the water index is a parameter (mndwi/ndwi/aweinsh/scowi ×
  otsu/weighted_peaks); the Otsu/peaks threshold is computed on the index values
  of **sand ∪ water pixels only** (the sub-pixel step), not the whole scene.
* **D5** — per-scene georeferencing RMSE is read from image metadata.
* **D6** — inter-sensor bias is quantified (:func:`intersensor_bias`).
* **D7** — study period 1988–2025; 1985–87 archive gap; 1991 ``partial``.

Execution model: heavy logic lives here; the Colab notebook is a thin driver.
Google Earth Engine is imported LAZILY inside the functions that need it so the
pure-NumPy science (indices, thresholding, contouring, the classifier) can be
imported and exercised without GEE authentication. ``config`` stays ``ee``-free.

CRS convention: EPSG:4326 for stored geometries; EPSG:32646 (UTM 46N) on a fixed
pixel grid for all metric work and every raster fetch (so scenes are
pixel-aligned — required for the inter-sensor bias test and the benchmark).
"""

from __future__ import annotations

import csv
import logging
import math
import os
import urllib.request

# EE's downloaded GeoTIFF tiles carry a band count that doesn't match a colour
# interpretation, so GDAL/rasterio logs a harmless "Sum of Photometric ... and
# ExtraSamples doesn't match SamplesPerPixel" warning per tile. Quiet that noise
# without hiding real errors.
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd
import pyproj
import rasterio
from shapely.geometry import (
    LineString, MultiLineString, Polygon, box, mapping,
)
from shapely.ops import linemerge, transform as shapely_transform, unary_union
from rasterio import features as rio_features
from rasterio.io import MemoryFile
from rasterio.transform import Affine, xy as raster_xy
from skimage import filters, measure, morphology
from scipy import ndimage, signal
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

from . import config, data

# ---------------------------------------------------------------------------
# Module constants  (<-- TUNABLE where marked)
# ---------------------------------------------------------------------------
# Sensor -> classifier sensor group (D3). TM and ETM+ share the band layout, so
# L4/L5/L7 train one classifier; OLI (L8/L9) and MSI (S2) each train their own.
SENSOR_GROUP: Dict[str, str] = {
    "L4": "TM", "L5": "TM", "L7": "TM", "L8": "OLI", "L9": "OLI", "S2": "MSI",
}

# Canonical band order used everywhere downstream (keys of config.BAND_MAP).
CANONICAL_BANDS: List[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]

# Base layers whose value AND 3x3 local std become classifier features. Ten
# layers x {value, texture} = the 20-feature CoastSat-style vector (D3).
_FEATURE_BASE: List[str] = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "mndwi", "ndwi", "ndvi", "mndwi2",
]
FEATURE_NAMES: List[str] = (
    _FEATURE_BASE + [f"{name}_std3" for name in _FEATURE_BASE]
)  # length 20

# Landsat Collection 2 L2 IDs by sensor (reuse the Phase 1 map).
_LANDSAT_COLLECTIONS: Dict[str, str] = data._LANDSAT_COLLECTIONS

# Fetch tiling: the AOI is ~90 km long, so one getDownloadURL request exceeds
# GEE's payload limit. Each tile is fetched separately and mosaicked client-side.
DEFAULT_TILE_PX: int = 768  # <-- TUNABLE (px per fetch tile; keep tile*tile*nbands*4 < ~32 MB)
DOWNLOAD_RETRIES: int = 4   # <-- TUNABLE

# Approximate open-coast alongshore length (m), used to report the fraction of
# the coast a scene's shoreline spans. ~92 km Cox's Bazar–Teknaf coast.
AOI_ALONGSHORE_M: float = 92000.0  # <-- TUNABLE

# AOI-specific geometry: this coast faces the Bay of Bengal to the WEST, so land
# lies to the EAST (larger UTM easting). Used to decide which side of a tidal-
# channel closure line is "landward" when trimming up-channel contour intrusions.
LAND_IS_EAST: bool = True             # <-- TUNABLE
CHANNEL_BUFFER_M: float = 250.0       # <-- TUNABLE (half-width searched around each closure line)

# Classifier / model persistence.
DEFAULT_CLASSIFIER_VERSION: str = "v1"
MODELS_DIR: str = "models"

# Output locations (D-locked schema, PHASE2_SPEC.md §3).
SHORELINE_DIR: str = os.path.join(config.OUTPUT_DIR, "shorelines")
SDS_SCENES_PATH: str = os.path.join(SHORELINE_DIR, "sds_scenes.geojson")
SDS_ANNUAL_PATH: str = os.path.join(SHORELINE_DIR, "sds_annual_merged.geojson")
EXTRACTION_LOG_PATH: str = os.path.join(config.OUTPUT_DIR, "extraction_log.csv")
SCENE_LIST_DENSE_PATH: str = os.path.join(config.OUTPUT_DIR, "scene_list_dense.csv")

# Transects (DSAS/QSCAT convention) — cast shore-normal from an operator-digitised
# baseline for the benchmark and inter-sensor bias. Kept in extract.py for now;
# Phase 4 (change.py) can adopt them.
BASELINE_PATH: str = "data/baseline.geojson"
TRANSECTS_PATH: str = os.path.join(config.OUTPUT_DIR, "transects.geojson")
TRANSECT_SPACING_M: float = 50.0     # <-- TUNABLE (alongshore spacing between transects)
TRANSECT_LENGTH_M: float = 1500.0    # <-- TUNABLE (seaward reach from the baseline)

# The locked one-LineString-per-scene attribute schema (geometry stored aside).
OUTPUT_SCHEMA: List[str] = [
    "image_id", "sensor", "sensor_group", "acq_datetime_utc", "dry_year",
    "season_label", "series", "pixel_size_m", "georef_rmse_m", "aoi_cloud_pct",
    "aoi_coverage_pct", "slc_off", "composite_date_spread_days", "season_complete",
    "water_index", "threshold_method", "threshold_value", "classifier_version",
    "length_m", "n_vertices", "pct_aoi_alongshore_covered", "flags",
]

# Reusable CRS transformers (EPSG:4326 <-> EPSG:32646, always lon/lat order).
_TF_WGS_TO_UTM = pyproj.Transformer.from_crs(
    config.STORAGE_CRS, config.METRIC_CRS, always_xy=True
)
_TF_UTM_TO_WGS = pyproj.Transformer.from_crs(
    config.METRIC_CRS, config.STORAGE_CRS, always_xy=True
)


def _to_utm(geom):
    """Reproject a shapely geometry EPSG:4326 -> EPSG:32646 (metres)."""
    return shapely_transform(_TF_WGS_TO_UTM.transform, geom)


def _to_wgs(geom):
    """Reproject a shapely geometry EPSG:32646 -> EPSG:4326 (for storage)."""
    return shapely_transform(_TF_UTM_TO_WGS.transform, geom)


# ---------------------------------------------------------------------------
# Small identity/parse helpers
# ---------------------------------------------------------------------------
def sensor_group(sensor: str) -> str:
    """Return the classifier sensor group (``TM``/``OLI``/``MSI``) for a sensor."""
    try:
        return SENSOR_GROUP[sensor]
    except KeyError as exc:
        raise ValueError(f"unknown sensor {sensor!r}") from exc


def sensor_from_image_id(image_id: str) -> str:
    """Infer the sensor label from a scene ``system:index``.

    Landsat IDs start with a mission code (``LT04``/``LT05``/``LE07``/``LC08``/
    ``LC09``); Sentinel-2 IDs start with a ``YYYYMMDDT`` sensing timestamp.
    """
    head = image_id.split(",")[0].strip()
    prefix = head[:4]
    landsat = {"LT04": "L4", "LT05": "L5", "LE07": "L7", "LC08": "L8", "LC09": "L9"}
    if prefix in landsat:
        return landsat[prefix]
    if len(head) >= 9 and head[:8].isdigit() and head[8] == "T":
        return "S2"
    raise ValueError(f"cannot infer sensor from image_id {image_id!r}")


def date_from_image_id(image_id: str) -> str:
    """Parse the acquisition date (``YYYY-MM-DD``) from a single granule ID."""
    head = image_id.strip()
    sensor = sensor_from_image_id(head)
    if sensor == "S2":
        ymd = head[:8]
    else:
        # LXNN_PPPRRR_YYYYMMDD_...  -> third underscore field is the date.
        ymd = head.split("_")[2]
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def assign_dry_year(when: datetime) -> int:
    """Map any acquisition datetime to a dry-season-year (canonical key).

    Uses the locked labelling extended to all months so every Series B scene
    gets an integer key: November/December roll into the next calendar year,
    January–October stay in the same calendar year. For dry-season (Nov–Mar)
    acquisitions this equals the Phase 1 ``dry_year`` exactly.
    """
    return when.year + 1 if when.month >= 11 else when.year


def _season_complete(
    dry_year: int, query_start: pd.Timestamp, query_end: pd.Timestamp
) -> bool:
    """Whether a dry-season-year's full all-season window fits the query range.

    A Series B ``dry_year`` spans Nov (Y−1) → Oct (Y). It is 'complete' only if
    that whole window lies within ``[query_start, query_end]``. Boundary years are
    incomplete: 1999 (its Nov–Dec 1998 head predates ``DENSE_START``) and the
    trailing year (its monsoon tail postdates ``DENSE_END``). Incomplete years
    must be excluded from any per-dry_year seasonal statistic — the flag makes
    that impossible to miss.
    """
    win_start = pd.Timestamp(f"{dry_year - 1}-11-01")
    win_end = pd.Timestamp(f"{dry_year}-10-31")
    return bool(win_start >= query_start and win_end <= query_end)


def _class_code(name: str) -> int:
    """Map a training-polygon class name to its integer code (D3)."""
    table = {
        "other": config.CLASS_OTHER,
        "sand": config.CLASS_SAND,
        "whitewater": config.CLASS_WHITEWATER,
        "water": config.CLASS_WATER,
    }
    key = str(name).strip().lower()
    if key not in table:
        raise ValueError(
            f"training class {name!r} not in {sorted(table)}; check "
            "data/training_polygons.geojson 'class' field"
        )
    return table[key]


# ===========================================================================
# 2.1  Scene lists
# ===========================================================================
def build_scene_list_annual(
    inventory_csv: str = "outputs/image_inventory.csv",
) -> pd.DataFrame:
    """Explode the approved annual inventory into one row per SCENE (D1).

    Each approved year-product in ``outputs/image_inventory.csv`` is split by
    acquisition *date* (the tide-consistent unit): same-date granules — adjacent
    Landsat rows or Sentinel-2 tiles from one overpass — are grouped into a
    single scene whose ``image_id`` is the comma-joined granule list, while
    different dates in a multi-date composite become separate scenes so the
    scene→tide mapping survives for Phase 3. ``product_type='none'`` gap years
    (1985–87) contribute no rows.

    Args:
        inventory_csv: Path to the Phase 1 inventory CSV (repo-relative).

    Returns:
        A DataFrame with the shared scene-list schema (``image_id``, ``sensor``,
        ``acq_datetime_utc`` [tz-aware UTC], ``dry_year``, ``season_label``,
        ``series='A'``, ``aoi_cloud_pct``, ``aoi_coverage_pct``, ``slc_off``,
        ``georef_rmse_m`` [NaN until read from metadata in :func:`fetch_scene`])
        plus ``product_type`` and ``review_status`` for provenance. Expect ~60
        scenes across the 38 non-gap dry-season-years.
    """
    inv = pd.read_csv(inventory_csv)
    rows: List[dict] = []
    for _, r in inv.iterrows():
        image_ids = str(r.get("image_ids", "") or "").strip()
        if str(r.get("product_type")) == "none" or not image_ids:
            continue
        granules = [g.strip() for g in image_ids.split(",") if g.strip()]
        dates = [d.strip() for d in str(r["dates"]).split(";") if d.strip()]
        times = [t.strip() for t in str(r["timestamps_utc"]).split(";") if t.strip()]
        # Map each contributing date to its acquisition timestamp.
        date_to_time = dict(zip(dates, times))
        # Product-level temporal spread: days between the earliest and latest
        # contributing acquisitions (0 for a single-date product). Every scene of
        # a multi-date composite carries the SAME value, so the merged annual
        # shoreline knows its segments come from up to this many days apart.
        parsed_dates = sorted(pd.Timestamp(d) for d in dates)
        spread_days = (
            int((parsed_dates[-1] - parsed_dates[0]).days)
            if len(parsed_dates) > 1 else 0
        )
        # Group granules by their own parsed date.
        by_date: Dict[str, List[str]] = {}
        for g in granules:
            by_date.setdefault(date_from_image_id(g), []).append(g)
        dry_year = int(r["dry_year"])
        for date in dates:
            grp = by_date.get(date)
            if not grp:
                continue
            sensor = sensor_from_image_id(grp[0])
            ts = date_to_time.get(date, f"{date}T00:00:00")
            rows.append(
                {
                    "image_id": ",".join(grp),
                    "sensor": sensor,
                    "acq_datetime_utc": _parse_utc(ts),
                    "dry_year": dry_year,
                    "season_label": data.season_label(dry_year),
                    "series": "A",
                    "aoi_cloud_pct": _to_float(r.get("cloud_pct")),
                    "aoi_coverage_pct": _to_float(r.get("achieved_coverage_pct")),
                    "slc_off": bool(data.slc_off(sensor, date)),
                    "composite_date_spread_days": spread_days,
                    # Series A products are deliberate per-year dry-season products,
                    # each temporally its own year -> always season-complete.
                    "season_complete": True,
                    "georef_rmse_m": np.nan,
                    "product_type": r.get("product_type"),
                    "review_status": r.get("review_status"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dry_year", "acq_datetime_utc"]).reset_index(drop=True)
    return df


def build_scene_list_dense(
    start: str = config.DENSE_START,
    end: str = config.DENSE_END,
    sensors: Sequence[str] = tuple(config.DENSE_SENSORS),
    cloud_max_pct: float = config.DENSE_CLOUD_MAX_PCT,
    coverage_min_pct: float = config.DENSE_COVERAGE_MIN_PCT,
    chunk: int = data._MATERIALIZE_BATCH_SIZE,
    write_csv: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the dense, all-season Series B scene list via GEE (D2).

    Queries every month (monsoon included) over ``start``–``end`` for the dense
    sensors and groups scenes by ``(sensor, date)`` — the same tide-consistent
    unit as Series A. Reuses the Phase 1 AOI-reduced masking/reducers in
    ``src.data`` (``_prepare_landsat``/``_prepare_s2`` with ``all_season=True``,
    plus ``_aoi_pixel_count``/``_num_or_zero``) — it does NOT use scene-wide
    ``CLOUDY_PIXEL_PERCENTAGE``/``CLOUD_COVER``.

    The query runs **one calendar year at a time**: a single all-season query
    across all ~27 years builds one enormous merged collection whose
    ``aggregate_array(...).getInfo()`` trips GEE's "User memory limit exceeded".
    A ``(sensor, date)`` group lies wholly within one calendar year, so the union
    over years is identical to one query, but every request stays bounded.
    Per-scene AOI cloud % and coverage % are still evaluated in key chunks
    (``chunk`` per request) to stay under the concurrent-aggregation (HTTP 429)
    limit.

    Scenes are kept when ``aoi_cloud_pct <= cloud_max_pct`` and
    ``aoi_coverage_pct >= coverage_min_pct`` (partial-coast scenes are retained —
    they still constrain the slope estimate at the transects they cover).

    With ``verbose`` (default ``True``) a progress line is printed per year
    (``"YYYY: k kept of n evaluated | running total …"``) so the multi-minute
    build can be watched advancing.

    Returns:
        A DataFrame in the shared scene-list schema with ``series='B'`` (and a
        ``dry_year`` assigned by :func:`assign_dry_year`). Also written to
        ``outputs/scene_list_dense.csv`` when ``write_csv`` (push it with the
        existing ``save_outputs()``).
    """
    import ee  # lazy: GEE only needed for the dense query.

    aoi = ee.Geometry.Polygon(config.aoi_coordinates())
    total_px = ee.Number(data._aoi_pixel_count(aoi, config.COVERAGE_SCALE))
    slc_millis = ee.Date(config.SLC_OFF_DATE).millis()

    def _prepared_for_window(win_start: str, win_end: str) -> Optional["ee.ImageCollection"]:
        prepared: List["ee.ImageCollection"] = []
        for sensor in sensors:
            if sensor == "S2":
                prepared.append(data._prepare_s2(aoi, win_start, win_end, all_season=True))
            else:
                prepared.append(
                    data._prepare_landsat(
                        _LANDSAT_COLLECTIONS[sensor], sensor, aoi, win_start, win_end,
                        all_season=True,
                    )
                )
        if not prepared:
            return None
        merged = prepared[0]
        for col in prepared[1:]:
            merged = merged.merge(col)
        return merged

    def _feature_for_key_fn(merged: "ee.ImageCollection"):
        def _feature_for_key(key: "ee.String") -> "ee.Feature":
            key = ee.String(key)
            group = merged.filter(ee.Filter.eq("sensor_date", key))
            mosaic = group.mosaic()
            stats = mosaic.select(["cloudy", "valid"]).reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.count(), sharedInputs=True
                ),
                geometry=aoi,
                scale=config.COVERAGE_SCALE,
                crs=config.METRIC_CRS,
                maxPixels=int(1e10),
                bestEffort=False,
            )
            cloud = data._num_or_zero(stats.get("cloudy_mean")).multiply(100)
            coverage = (
                data._num_or_zero(stats.get("valid_count"))
                .divide(total_px)
                .multiply(100)
                .min(100.0)
            )
            parts = key.split("_")
            sensor = ee.String(parts.get(0))
            date = ee.String(parts.get(1))
            image_ids = (
                group.aggregate_array("system:index")
                .map(lambda idx: data.clean_scene_id(ee.String(idx)))
                .join(",")
            )
            is_slc_off = sensor.compareTo("L7").eq(0).And(
                ee.Date(date).millis().gt(slc_millis)
            )
            return ee.Feature(
                None,
                {
                    "sensor": sensor,
                    "date": date,
                    "timestamp_utc": ee.Date(
                        group.aggregate_min("system:time_start")
                    ).format(),
                    "image_ids": image_ids,
                    "n_scenes": group.size(),
                    "aoi_cloud_pct": cloud,
                    "aoi_coverage_pct": coverage,
                    "slc_off": is_slc_off,
                },
            )
        return _feature_for_key

    # Per-year windows whose union is exactly [start, end). Filtering and the
    # ``verbose`` progress line ("YYYY: k kept of n | running total …") happen
    # per year so a long dense build (tens of minutes to a couple of hours) can
    # be watched advancing rather than sitting silent.
    q_start, q_end = pd.Timestamp(start), pd.Timestamp(end)
    start_year, end_year = int(start[:4]), int(end[:4])
    rows: List[dict] = []
    for year in range(start_year, end_year + 1):
        win_start = start if year == start_year else f"{year}-01-01"
        win_end = end if year == end_year else f"{year + 1}-01-01"
        merged = _prepared_for_window(win_start, win_end)
        keys: List[str] = (
            merged.aggregate_array("sensor_date").distinct().getInfo()
            if merged is not None else []
        )
        year_records: List[dict] = []
        if keys:
            feature_for_key = _feature_for_key_fn(merged)
            for i in range(0, len(keys), chunk):
                subset = ee.List(keys[i:i + chunk])
                fc = ee.FeatureCollection(subset.map(feature_for_key))
                year_records.extend(f["properties"] for f in fc.getInfo()["features"])

        kept = 0
        for rec in year_records:
            if rec["aoi_cloud_pct"] > cloud_max_pct:
                continue
            if rec["aoi_coverage_pct"] < coverage_min_pct:
                continue
            ts = _parse_utc(rec["timestamp_utc"])
            dry_year = assign_dry_year(ts)
            rows.append(
                {
                    "image_id": rec["image_ids"],
                    "sensor": rec["sensor"],
                    "acq_datetime_utc": ts,
                    "dry_year": dry_year,
                    "season_label": data.season_label(dry_year),
                    "series": "B",
                    "aoi_cloud_pct": float(rec["aoi_cloud_pct"]),
                    "aoi_coverage_pct": float(rec["aoi_coverage_pct"]),
                    "slc_off": bool(rec["slc_off"]),
                    # Each dense scene is a single acquisition -> no composite spread.
                    "composite_date_spread_days": 0,
                    "season_complete": _season_complete(dry_year, q_start, q_end),
                    "georef_rmse_m": np.nan,
                }
            )
            kept += 1
        if verbose:
            print(f"{year}: {kept} kept of {len(year_records)} evaluated "
                  f"| running total {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["acq_datetime_utc", "sensor"]).reset_index(drop=True)
    if write_csv:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        df.to_csv(SCENE_LIST_DENSE_PATH, index=False)
    return df


def _to_float(value) -> float:
    """Coerce a possibly-empty inventory cell to float (NaN if blank)."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_utc(text: str) -> pd.Timestamp:
    """Parse an ISO timestamp to a tz-aware UTC ``pandas.Timestamp``."""
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# ===========================================================================
# 2.2  Imagery
# ===========================================================================
@dataclass
class Scene:
    """One tide-consistent acquisition, fetched onto the fixed EPSG:32646 grid.

    Attributes:
        image_id: Comma-joined ``system:index`` of the mosaicked granule(s).
        sensor: Sensor label (``L4``..``L9``/``S2``).
        acq_datetime_utc: Acquisition instant (tz-aware UTC) from
            ``system:time_start`` — the tide reference for Phase 3.
        bands: Canonical scaled reflectance arrays in [0, 1]; keys
            ``blue, green, red, nir, swir1, swir2`` (``swir2`` carried for
            AWEInsh, D4). Invalid pixels are ``np.nan``.
        valid: Boolean array — True where the pixel is usable (not fill, not
            saturated, not cloud/shadow).
        transform: The ``rasterio``/``affine`` transform of the grid (EPSG:32646).
        pixel_size_m: Native analysis grid size (10 m S2, 30 m Landsat).
        georef_rmse_m: Per-scene georeferencing RMSE (D5), from metadata when
            available else the ``config.GEOREF_RMSE_DEFAULT_M`` fallback.

    Scene-list attributes (dry_year, series, cloud/coverage, slc_off) are stashed
    on the instance by :func:`_attach_row_metadata` before extraction so the
    output record can carry them; they are not core imagery fields.
    """

    image_id: str
    sensor: str
    acq_datetime_utc: datetime
    bands: Dict[str, np.ndarray]
    valid: np.ndarray
    transform: Affine
    pixel_size_m: int
    georef_rmse_m: float


def _ee_asset_id(granule: str, sensor: str) -> str:
    """Full GEE asset ID for a granule ``system:index``."""
    if sensor == "S2":
        return f"{config.S2_SR_HARMONIZED}/{granule}"
    return f"{_LANDSAT_COLLECTIONS[sensor]}/{granule}"


def _fixed_grid(
    minx: float, miny: float, maxx: float, maxy: float, scale: int
) -> Tuple[Affine, int, int]:
    """Snap a UTM bounding box to a fixed ``scale``-m grid (north-up).

    Anchoring the origin to a multiple of ``scale`` makes every scene, at a given
    sensor resolution, share one grid — so scenes are pixel-aligned (needed for
    D6 and the benchmark). Returns ``(transform, width, height)``.
    """
    x0 = np.floor(minx / scale) * scale
    y1 = np.ceil(maxy / scale) * scale
    x1 = np.ceil(maxx / scale) * scale
    y0 = np.floor(miny / scale) * scale
    width = int(round((x1 - x0) / scale))
    height = int(round((y1 - y0) / scale))
    transform = Affine(scale, 0.0, x0, 0.0, -scale, y1)
    return transform, width, height


def extraction_region_utm() -> Polygon:
    """Return the UTM polygon the fetch is clipped to (search zone else AOI).

    Fetching only the digitised search-zone envelope (a narrow coastal band)
    instead of the full ~800 km² AOI keeps per-scene arrays small enough for
    Colab RAM. Falls back to the AOI polygon if the search zone is absent.
    """
    zone = load_search_zone()
    if zone is not None:
        return zone
    return _to_utm(Polygon(config.aoi_coordinates()))


def _download_tiled(
    image: "ee.Image",
    band_names: List[str],
    transform: Affine,
    width: int,
    height: int,
    scale: int,
    tile_px: int = DEFAULT_TILE_PX,
    keep_geom: Optional[Polygon] = None,
) -> Dict[str, np.ndarray]:
    """Fetch an ``ee.Image`` onto the fixed grid, tile by tile, as NumPy arrays.

    The AOI is ~90 km long, so a single ``getDownloadURL`` exceeds GEE's payload
    limit; the grid is fetched in ``tile_px`` blocks and stitched. Any bilinear
    resampling of reflectance must be applied to the SOURCE images (which carry a
    native projection) BEFORE they are mosaicked — never here: ``.resample()`` on
    a ``mosaic()`` (whose default projection is EPSG:4326 at 1°) makes GEE compute
    at ~1° and upsample, collapsing the whole scene to a near-constant blur. Tiles
    whose footprint does not intersect ``keep_geom`` (the search-zone/extraction
    polygon, EPSG:32646) are skipped and left NaN — the fetch grid is the bbox of
    an irregular coastal band, so this avoids downloading (and holding in RAM) the
    empty corner tiles.

    Returns a dict of ``band_name -> (height, width) float32`` arrays.
    """
    x0, y1 = transform.c, transform.f
    out = {b: np.full((height, width), np.nan, dtype=np.float32) for b in band_names}
    for r0 in range(0, height, tile_px):
        for c0 in range(0, width, tile_px):
            th = min(tile_px, height - r0)
            tw = min(tile_px, width - c0)
            if keep_geom is not None:
                tile_box = box(x0 + c0 * scale, y1 - (r0 + th) * scale,
                               x0 + (c0 + tw) * scale, y1 - r0 * scale)
                if not keep_geom.intersects(tile_box):
                    continue
            tile_transform = [scale, 0.0, x0 + c0 * scale, 0.0, -scale, y1 - r0 * scale]
            params = {
                "bands": band_names,
                "crs": config.METRIC_CRS,
                "crs_transform": tile_transform,
                "dimensions": f"{tw}x{th}",
                "format": "GEO_TIFF",
            }
            arr = _download_geotiff(image, params, len(band_names), th, tw)
            for k, b in enumerate(band_names):
                out[b][r0:r0 + th, c0:c0 + tw] = arr[k]
    return out


def _download_geotiff(
    image: "ee.Image", params: dict, n_bands: int, th: int, tw: int
) -> np.ndarray:
    """Download one GeoTIFF tile with retries; return a ``(n_bands, th, tw)`` array."""
    last_exc: Optional[Exception] = None
    for _ in range(DOWNLOAD_RETRIES):
        try:
            url = image.getDownloadURL(params)
            with urllib.request.urlopen(url, timeout=300) as resp:
                blob = resp.read()
            with MemoryFile(blob) as mem, mem.open() as ds:
                return ds.read().astype(np.float32)
        except Exception as exc:  # network / EE transient
            last_exc = exc
    raise RuntimeError(f"tile download failed after {DOWNLOAD_RETRIES} tries: {last_exc}")


def fetch_scene(
    row: pd.Series,
    tile_px: int = DEFAULT_TILE_PX,
    region_utm: Optional[Polygon] = None,
) -> Scene:
    """Fetch one scene onto the fixed EPSG:32646 grid as a :class:`Scene` (D1/D5).

    Steps:
      1. Resolve the granule ID(s) in ``row['image_id']`` to GEE assets and
         mosaic them (one overpass -> one scene).
      2. Reflectance: select ``config.BAND_MAP`` bands, download (bilinear) over
         the fixed grid, and scale to [0, 1] with ``config.SR_SCALE`` client-side
         (S2 20 m SWIR is bilinearly resampled to the 10 m grid on the way).
      3. Data presence (nearest): an explicit ``_obs`` band (1 where the SR
         mosaic has data, 0 outside the footprint) marks observed pixels — NOT
         QA bit 0, which reads 0 (= not fill) in the getDownloadURL no-data fill;
         a belt-and-braces all-bands-raw-0 test backs it up. Cloud/shadow: S2 ->
         Cloud Score+ ``cs < CS_THRESHOLD``; Landsat -> ``QA_PIXEL`` bits 1/2/3/4
         (dilated/cirrus/cloud/shadow) + ``QA_RADSAT`` saturation. When
         ``config.LANDSAT_CLOUD_MASK_ISSUE`` a morphological opening drops
         isolated cloud flags over bright sand/surf (CoastSat's ``cloud_mask_issue``).
      4. ``georef_rmse_m`` from ``GEOMETRIC_RMSE_MODEL`` (Landsat) else the
         ``config.GEOREF_RMSE_DEFAULT_M`` fallback; ``acq_datetime_utc`` from
         ``system:time_start``.

    Args:
        row: A scene-list row (from :func:`build_scene_list_annual`/``_dense``).
        tile_px: Fetch tile size in pixels.
        region_utm: Optional EPSG:32646 polygon to fetch instead of the default
            extraction region (search zone else full AOI). Pass a small window
            for cheap visual QC before the search zone has been digitised, or to
            keep Sentinel-2 arrays inside Colab RAM.

    Returns:
        The populated :class:`Scene`.
    """
    import ee

    sensor = str(row["sensor"])
    family = config.SR_SCALE_FAMILY[sensor]
    scale = config.PIXEL_SIZE_M[sensor]
    granules = [g.strip() for g in str(row["image_id"]).split(",") if g.strip()]

    # Fixed grid over the requested region (default: search zone else AOI). The
    # region polygon also drives per-tile skipping: only tiles that intersect it
    # are downloaded (the grid is the bbox of an irregular coastal band).
    region = region_utm if region_utm is not None else extraction_region_utm()
    minx, miny, maxx, maxy = region.bounds
    transform, width, height = _fixed_grid(minx, miny, maxx, maxy, scale)

    band_src = [config.BAND_MAP[sensor][b] for b in CANONICAL_BANDS]
    # Resample each SOURCE image (native projection intact) so S2 20 m SWIR is
    # bilinearly brought to the 10 m grid and reflectance interpolates cleanly.
    # Resampling here — not on the mosaic — avoids the 1° collapse bug.
    refl_imgs = [
        ee.Image(_ee_asset_id(g, sensor)).select(band_src, CANONICAL_BANDS)
        .resample("bilinear")
        for g in granules
    ]
    refl_image = ee.ImageCollection(refl_imgs).mosaic()
    refl_raw = _download_tiled(
        refl_image, CANONICAL_BANDS, transform, width, height, scale,
        tile_px=tile_px, keep_geom=region,
    )
    gain, offset = config.SR_SCALE[family]
    bands: Dict[str, np.ndarray] = {
        b: refl_raw[b].astype(np.float32) * gain + offset for b in CANONICAL_BANDS
    }

    # Explicit data-presence band (D1 partial-coverage safety). getDownloadURL
    # fills EVERY band — reflectance AND QA — with 0 outside a granule's
    # footprint, so QA bit 0 (fill) reads as "not fill" and a no-data area would
    # be marked valid with reflectance -0.2 (Landsat) or 0 (S2) -> MNDWI == 0
    # exactly along the footprint edge -> a false straight shoreline that survives
    # the min-length filter. This corrupts every partial-coverage product (the 19
    # composites, 1991). So attach an explicit, sensor-agnostic presence band: 1
    # where the SR mosaic has data, 0 outside. Built from the un-resampled
    # reflectance mosaic's mask and downloaded nearest, so it is crisp 0/1.
    presence = (
        ee.ImageCollection(
            [ee.Image(_ee_asset_id(g, sensor)).select([band_src[0]]) for g in granules]
        )
        .mosaic()
        .mask()
        .rename("_obs")
    )

    # Mask source bands (nearest-neighbour download), with _obs alongside.
    if sensor == "S2":
        # Cloud Score+ 'cs' via the Phase 1 linkCollection join (robust to any
        # index-format quirk), bounded to the acquisition day for speed.
        day = ee.Date(date_from_image_id(granules[0]))
        cs_mosaic = (
            ee.ImageCollection(config.S2_SR_HARMONIZED)
            .filterDate(day, day.advance(1, "day"))
            .filter(ee.Filter.inList("system:index", ee.List(granules)))
            .linkCollection(ee.ImageCollection(config.CLOUD_SCORE_PLUS), ["cs"])
            .select("cs")
            .mosaic()
        )
        dl = _download_tiled(
            ee.Image.cat([presence, cs_mosaic]), ["_obs", "cs"],
            transform, width, height, scale, tile_px=tile_px, keep_geom=region,
        )
        # cs is filled with 0 outside the footprint; the _obs gate (below) handles
        # that, so cloud is judged purely on cs where observed.
        cloud = np.nan_to_num(dl["cs"], nan=0.0) < config.CS_THRESHOLD
        shadow = np.zeros(cloud.shape, dtype=bool)
        saturated = np.zeros(cloud.shape, dtype=bool)
    else:
        qa_imgs = [ee.Image(_ee_asset_id(g, sensor)).select(["QA_PIXEL", "QA_RADSAT"])
                   for g in granules]
        dl = _download_tiled(
            ee.Image.cat([presence, ee.ImageCollection(qa_imgs).mosaic()]),
            ["_obs", "QA_PIXEL", "QA_RADSAT"],
            transform, width, height, scale, tile_px=tile_px, keep_geom=region,
        )
        qap = np.nan_to_num(dl["QA_PIXEL"], nan=0.0).astype(np.uint16)
        radsat = np.nan_to_num(dl["QA_RADSAT"], nan=0.0).astype(np.uint16)
        dilated = (qap & (1 << 1)) != 0
        cirrus = (qap & (1 << 2)) != 0
        cloud_bit = (qap & (1 << 3)) != 0
        shadow = (qap & (1 << 4)) != 0
        cloud = dilated | cirrus | cloud_bit
        saturated = radsat != 0
        if config.LANDSAT_CLOUD_MASK_ISSUE:
            # Drop small isolated cloud flags (bright beach/whitewater misflags).
            cloud = morphology.binary_opening(cloud, np.ones((3, 3), bool))

    # Observed = explicit presence band (symmetric across sensors). Belt-and-
    # braces: a pixel whose RAW reflectance is exactly 0 in ALL bands is no-data
    # too (catches any residual fill the presence band missed).
    observed = np.nan_to_num(dl["_obs"], nan=0.0) >= 0.5
    all_zero = np.logical_and.reduce([refl_raw[b] == 0 for b in CANONICAL_BANDS])
    observed = observed & ~all_zero

    valid = observed & ~cloud & ~shadow & ~saturated
    # Reflectance is only meaningful where valid.
    for b in CANONICAL_BANDS:
        arr = bands[b]
        arr[~valid] = np.nan
        bands[b] = arr

    meta_img = ee.Image(_ee_asset_id(granules[0], sensor))
    georef = _read_georef_rmse(meta_img, family)
    acq = _read_time_start(meta_img, row)

    return Scene(
        image_id=str(row["image_id"]),
        sensor=sensor,
        acq_datetime_utc=acq,
        bands=bands,
        valid=valid,
        transform=transform,
        pixel_size_m=scale,
        georef_rmse_m=georef,
    )


def _read_georef_rmse(meta_img: "ee.Image", family: str) -> float:
    """Per-scene georeferencing RMSE (m) from metadata, else the fallback (D5).

    Landsat Collection 2 publishes a per-scene numeric ``GEOMETRIC_RMSE_MODEL``
    (metres), read here. Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) exposes NO
    per-scene numeric geolocation RMSE — its only geometric-quality property is a
    categorical ``GEOMETRIC_QUALITY`` PASSED/FAILED flag, and the ~11 m S2
    geolocation accuracy is characterised globally, not per scene. So E_georef is
    a per-scene MEASUREMENT for Landsat but a fixed ASSUMPTION for Sentinel-2
    (``config.GEOREF_RMSE_DEFAULT_M['S2']``); Phase 4 must report it as such.
    """
    import ee

    fallback = config.GEOREF_RMSE_DEFAULT_M[family]
    if family != "LANDSAT":
        # Sentinel-2: no per-scene geolocation RMSE field exists in GEE -> constant.
        return float(fallback)
    try:
        value = ee.Image(meta_img).get("GEOMETRIC_RMSE_MODEL").getInfo()
    except Exception:
        value = None
    return float(value) if value is not None else float(fallback)


def _read_time_start(meta_img: "ee.Image", row: pd.Series) -> pd.Timestamp:
    """Authoritative acquisition instant from ``system:time_start`` (UTC)."""
    import ee

    try:
        millis = ee.Image(meta_img).get("system:time_start").getInfo()
        if millis is not None:
            return pd.Timestamp(int(millis), unit="ms", tz="UTC")
    except Exception:
        pass
    return _parse_utc(str(row.get("acq_datetime_utc")))


def export_scene_geotiff(
    row: pd.Series,
    path: str,
    bands: Sequence[str] = ("red", "green", "blue", "swir1"),
    region_utm: Optional[Polygon] = None,
    tile_px: int = DEFAULT_TILE_PX,
) -> str:
    """Fetch a scene and write it as a georeferenced multi-band GeoTIFF (UTM 46N).

    Writes the scaled reflectance ``bands`` on the exact fixed EPSG:32646 grid the
    pipeline extracts from, so training polygons and reference shorelines digitised
    over this file in QGIS line up pixel-for-pixel with what the classifier and
    contour see. Invalid pixels are written as NaN nodata (transparent in QGIS).

    Args:
        row: A scene-list row (as for :func:`fetch_scene`).
        path: Output ``.tif`` path (e.g. on the mounted Drive).
        bands: Canonical band names to write, in order (default R, G, B, SWIR1 —
            SWIR1 makes the wet/dry line and water easy to see).
        region_utm: Optional EPSG:32646 fetch window (see :func:`fetch_scene`).
        tile_px: Fetch tile size.

    Returns:
        ``path`` (the file written).
    """
    scene = fetch_scene(row, tile_px=tile_px, region_utm=region_utm)
    stack = np.stack([scene.bands[b] for b in bands], axis=0).astype(np.float32)
    height, width = scene.valid.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(bands),
        "dtype": "float32",
        "crs": config.METRIC_CRS,
        "transform": scene.transform,
        "nodata": float("nan"),
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(bands, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(
            image_id=scene.image_id,
            sensor=scene.sensor,
            acq_datetime_utc=str(scene.acq_datetime_utc),
            pixel_size_m=str(scene.pixel_size_m),
        )
    return path


# ===========================================================================
# 2.3  Classifier (D3)
# ===========================================================================
def feats_dim() -> int:
    """Number of classifier features (len of ``FEATURE_NAMES``)."""
    return len(FEATURE_NAMES)


def _local_std(a: np.ndarray, size: int = 3) -> np.ndarray:
    """3x3 local standard deviation (texture) via uniform filters.

    ``std = sqrt(E[x^2] - E[x]^2)``; NaNs are treated as 0 so the filter does not
    propagate them (the caller masks invalid pixels separately).
    """
    a = np.nan_to_num(a.astype(np.float64), nan=0.0)
    mean = ndimage.uniform_filter(a, size=size, mode="reflect")
    sqr = ndimage.uniform_filter(a * a, size=size, mode="reflect")
    var = np.clip(sqr - mean * mean, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference ``(a - b) / (a + b)`` with 0/0 -> NaN."""
    num = a - b
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, num / den)
    return out.astype(np.float32)


def _base_layers(scene: Scene) -> Dict[str, np.ndarray]:
    """Reflectances + water/veg indices used as classifier feature bases."""
    b = scene.bands
    layers = {name: b[name] for name in CANONICAL_BANDS}
    layers["mndwi"] = _norm_diff(b["green"], b["swir1"])
    layers["ndwi"] = _norm_diff(b["green"], b["nir"])
    layers["ndvi"] = _norm_diff(b["nir"], b["red"])
    layers["mndwi2"] = _norm_diff(b["green"], b["swir2"])
    return layers


def _iter_feature_layers(scene: Scene):
    """Yield the 20 feature layers one at a time (order = ``FEATURE_NAMES``).

    Ten base layers (6 reflectances + MNDWI/NDWI/NDVI/MNDWI2), then their 3x3
    local std (texture separates bright whitewater from dry sand). Yielding one
    (H, W) layer at a time — rather than stacking all 20 — keeps peak memory to a
    couple of full arrays instead of ~20x the grid (a 10 m search-zone bbox is
    tens of millions of pixels; a 20-deep float32 stack would be gigabytes).
    """
    layers = _base_layers(scene)
    for name in _FEATURE_BASE:
        yield layers[name]
    for name in _FEATURE_BASE:
        yield _local_std(layers[name])


def _feature_matrix(scene: Scene, mask: np.ndarray) -> np.ndarray:
    """Build the ``(N, 20)`` feature matrix for the ``mask`` pixels only.

    Extracts each feature layer's values at the ``N = mask.sum()`` selected
    pixels without ever materialising the full ``(H, W, 20)`` cube — the layer is
    freed before the next is built.
    """
    flat = mask.reshape(-1)
    cols = [layer.reshape(-1)[flat] for layer in _iter_feature_layers(scene)]
    return np.column_stack(cols).astype(np.float32) if cols else np.empty((0, feats_dim()))


def _search_zone_mask(scene: Scene) -> np.ndarray:
    """Rasterise the search zone onto the scene grid (all-True if none exists).

    Restricting classification to this mask keeps inland aquaculture ponds and
    the Bakkhali/Naf estuaries out of the feature stack and the Otsu histogram,
    and bounds the number of pixels ``clf.predict`` runs on.
    """
    zone = load_search_zone()
    if zone is None:
        return np.ones(scene.valid.shape, dtype=bool)
    return rio_features.rasterize(
        [(mapping(zone), 1)],
        out_shape=scene.valid.shape,
        transform=scene.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def build_training_set(
    scenes: List[Scene], labels_path: str = "data/training_polygons.geojson",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample classifier features under the labelled training polygons (D3).

    For every scene, each polygon in ``labels_path`` (a ``class`` field in
    {other, sand, whitewater, water}) is rasterised onto the scene grid and the
    20-feature vectors of the valid pixels it covers are collected. Pass scenes
    of ONE sensor group (their band layouts must match).

    Args:
        scenes: Training scenes (same sensor group), already fetched.
        labels_path: Path to ``data/training_polygons.geojson``.

    Returns:
        ``(X, y)`` with ``X`` shape ``(N, 20)`` and integer ``y`` in {0,1,2,3}.
    """
    gdf = gpd.read_file(labels_path).to_crs(config.METRIC_CRS)
    if "class" not in gdf.columns:
        raise ValueError(f"{labels_path} has no 'class' column")
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for scene in scenes:
        # Burn every labelled polygon into one class-code raster (last polygon
        # wins on overlap), then extract features once for all labelled pixels —
        # no full-scene feature cube is built.
        class_map = np.full(scene.valid.shape, -1, dtype=np.int16)
        for _, poly in gdf.iterrows():
            if poly.geometry is None or poly.geometry.is_empty:
                continue
            code = _class_code(poly["class"])
            burned = rio_features.rasterize(
                [(mapping(poly.geometry), 1)],
                out_shape=scene.valid.shape,
                transform=scene.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            class_map[burned & scene.valid] = code
        sample = class_map >= 0
        if not sample.any():
            continue
        xs.append(_feature_matrix(scene, sample))
        ys.append(class_map[sample].astype(int))
    if not xs:
        raise ValueError(
            "no training pixels sampled — check that training polygons overlap "
            "the provided scenes and fall inside their valid data"
        )
    return np.vstack(xs), np.concatenate(ys)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sensor_group: str,
    version: str = DEFAULT_CLASSIFIER_VERSION,
) -> Pipeline:
    """Train and persist the local MLP classifier for a sensor group (D3).

    A ``StandardScaler`` + ``MLPClassifier(hidden=(100, 50), max_iter=500)``
    pipeline (fixed ``random_state`` for reproducibility) is fitted and dumped to
    ``models/clf_{sensor_group}_{version}.joblib``.

    Returns:
        The fitted scikit-learn pipeline.
    """
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                                  random_state=0)),
        ]
    )
    clf.fit(X, y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"clf_{sensor_group}_{version}.joblib")
    joblib.dump(clf, path)
    return clf


def load_classifier(
    sensor_group: str, version: str = DEFAULT_CLASSIFIER_VERSION
) -> Pipeline:
    """Load a persisted classifier for a sensor group."""
    path = os.path.join(MODELS_DIR, f"clf_{sensor_group}_{version}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"classifier {path} not found — train it with train_classifier() first"
        )
    return joblib.load(path)


def evaluate_classifier(
    clf: Pipeline, X: np.ndarray, y: np.ndarray, write_report: bool = False,
    sensor_group: str = "", version: str = DEFAULT_CLASSIFIER_VERSION,
) -> str:
    """Per-class precision/recall/F1 report on held-out samples (D3 deliverable).

    Optionally appends the report to ``models/classifier_report.md`` (a paper
    table). ``X``/``y`` should be from a scene NOT used in training.
    """
    labels = [config.CLASS_OTHER, config.CLASS_SAND, config.CLASS_WHITEWATER,
              config.CLASS_WATER]
    names = ["other", "sand", "whitewater", "water"]
    report = classification_report(
        y, clf.predict(np.nan_to_num(X)), labels=labels, target_names=names,
        zero_division=0,
    )
    if write_report:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "classifier_report.md"), "a",
                  encoding="utf-8") as fh:
            stamp = datetime.now(timezone.utc).isoformat()
            fh.write(f"\n### {sensor_group} classifier {version} — {stamp}\n\n")
            fh.write("```\n" + report + "\n```\n")
    return report


def _majority_filter_labels(labels: np.ndarray, size: int = 3) -> np.ndarray:
    """3x3 majority (modal) filter of a discrete class-label raster.

    Vectorised over the four classes: the count of each class in the ``size`` x
    ``size`` window (including the centre) is computed with a box filter, and each
    pixel takes the most common class. Operates on the LABEL MAP only — never the
    water index or the contour. Ties break toward the lower class code.
    """
    classes = [config.CLASS_OTHER, config.CLASS_SAND, config.CLASS_WHITEWATER,
               config.CLASS_WATER]
    # uniform_filter gives the window mean of the 0/1 membership; the argmax is
    # unchanged by the constant 1/window-size factor, so no need to rescale.
    counts = np.stack(
        [ndimage.uniform_filter((labels == c).astype(np.float32), size=size,
                                 mode="nearest")
         for c in classes],
        axis=0,
    )
    return np.asarray(classes)[np.argmax(counts, axis=0)].astype(labels.dtype)


def classify_scene(
    scene: Scene, clf: Pipeline, zone_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Classify a scene into {other, sand, whitewater, water} codes (D3).

    Only pixels inside the search zone AND valid are classified; everything else
    is ``CLASS_OTHER``. This bounds ``clf.predict`` to the coastal band (a
    whole-grid 20-feature predict on tens of millions of 10 m pixels would
    OOM/crawl in Colab) and keeps inland ponds/estuaries out of the interface
    used for the Otsu threshold. When ``config.LABEL_MAJORITY_FILTER`` a 3x3
    majority filter cleans label speckle (which would otherwise poison the
    sand∪water histogram) — applied to the LABEL MAP only, never the index or
    contour.

    Args:
        zone_mask: Optional precomputed search-zone boolean mask (same shape as
            ``scene.valid``); computed from the digitised zone when omitted.

    Returns:
        An ``(H, W)`` int array in {0, 1, 2, 3}.
    """
    height, width = scene.valid.shape
    if zone_mask is None:
        zone_mask = _search_zone_mask(scene)
    mask = scene.valid & zone_mask
    labels = np.full(height * width, config.CLASS_OTHER, dtype=int)
    flat = mask.reshape(-1)
    if flat.any():
        X = np.nan_to_num(_feature_matrix(scene, mask), nan=0.0, posinf=0.0, neginf=0.0)
        labels[flat] = clf.predict(X)
    labels = labels.reshape(height, width)
    if config.LABEL_MAJORITY_FILTER and flat.any():
        # Filter the label raster, then re-force out-of-zone pixels back to OTHER
        # so the filter cannot leak classes past the search-zone boundary.
        labels = np.where(mask, _majority_filter_labels(labels), config.CLASS_OTHER)
    return labels


# ===========================================================================
# 2.4  Index, threshold, sub-pixel contour (D4)
# ===========================================================================
# scowi (Bergsma et al. 2024) is feature-flagged until its published definition
# is verified against the paper; the default index is mndwi, so this never
# blocks the pipeline.  <-- TUNABLE
SCOWI_VERIFIED: bool = False


def water_index(scene: Scene, name: str = config.WATER_INDEX_DEFAULT) -> np.ndarray:
    """Compute a water index over the scene (D4).

    * ``mndwi``  = (green − swir1) / (green + swir1)         (Xu 2006)
    * ``ndwi``   = (green − nir)  / (green + nir)            (McFeeters 1996)
    * ``aweinsh``= 4·(green − swir1) − (0.25·nir + 2.75·swir2)  (Feyisa 2014)
    * ``scowi``  = Bergsma et al. 2024 (feature-flagged, see ``SCOWI_VERIFIED``)

    Higher values indicate water. Invalid pixels are ``np.nan``.
    """
    b = scene.bands
    name = name.lower()
    if name == "mndwi":
        return _norm_diff(b["green"], b["swir1"])
    if name == "ndwi":
        return _norm_diff(b["green"], b["nir"])
    if name == "aweinsh":
        return (4.0 * (b["green"] - b["swir1"])
                - (0.25 * b["nir"] + 2.75 * b["swir2"])).astype(np.float32)
    if name == "scowi":
        if not SCOWI_VERIFIED:
            raise NotImplementedError(
                "scowi is feature-flagged pending verification of the Bergsma "
                "et al. (2024) definition; set extract.SCOWI_VERIFIED=True once "
                "implemented. Default index is 'mndwi'."
            )
        raise NotImplementedError("scowi definition not yet implemented")
    raise ValueError(f"unknown water index {name!r}; choose from {config.WATER_INDICES}")


def interface_threshold(
    index: np.ndarray,
    labels: np.ndarray,
    method: str = config.THRESHOLD_METHOD_DEFAULT,
) -> float:
    """Threshold the index on the SAND ∪ WATER interface only (the sub-pixel step).

    The threshold is derived from the index values of pixels the classifier
    called sand or water — not the full-scene histogram. Restricting to the
    land/water interface is what removes the whitewater / wet-sand seaward bias
    of plain full-scene MNDWI+Otsu (D4).

    * ``otsu``           — Otsu's between-class variance threshold.
    * ``weighted_peaks`` — the density valley between the sand and water modes
      (a two-peak variant after Doherty et al. 2022); sits higher in the swash.

    Args:
        index: The water-index array.
        labels: Classifier output (same shape); sand/water taken from it.
        method: ``'otsu'`` or ``'weighted_peaks'``.

    Returns:
        The scalar index threshold separating sand from water.
    """
    interface = np.isin(labels, [config.CLASS_SAND, config.CLASS_WATER])
    values = index[interface & np.isfinite(index)]
    if values.size < 2:
        raise ValueError(
            "too few sand/water interface pixels to threshold — check the "
            "classifier output and the search zone"
        )
    method = method.lower()
    if method == "otsu":
        return float(filters.threshold_otsu(values))
    if method == "weighted_peaks":
        return _weighted_peaks_threshold(values)
    raise ValueError(
        f"unknown threshold method {method!r}; choose from {config.THRESHOLD_METHODS}"
    )


def _weighted_peaks_threshold(values: np.ndarray) -> float:
    """Two-mode density valley of the sand∪water index (Doherty-style).

    Smooths the histogram, finds the two most prominent peaks (sand mode low,
    water mode high), and returns the density minimum between them. Falls back to
    Otsu when a clean bimodal shape is not found.
    """
    hist, edges = np.histogram(values, bins=200)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = ndimage.gaussian_filter1d(hist.astype(float), 2.0)
    if smooth.max() <= 0:
        return float(filters.threshold_otsu(values))
    peaks, props = signal.find_peaks(smooth, prominence=smooth.max() * 0.05)
    if peaks.size < 2:
        return float(filters.threshold_otsu(values))
    top2 = np.sort(peaks[np.argsort(props["prominences"])[::-1][:2]])
    lo, hi = int(top2[0]), int(top2[1])
    valley = lo + int(np.argmin(smooth[lo:hi + 1]))
    return float(centers[valley])


def extract_contour(
    index: np.ndarray,
    threshold: float,
    valid: np.ndarray,
    transform: Affine,
) -> List[LineString]:
    """Marching-squares sub-pixel contour of the index at ``threshold`` (D4).

    ``skimage.measure.find_contours`` returns vertices at sub-pixel (fractional
    row/col) precision; these are mapped to EPSG:32646 pixel *centres* via the
    affine transform. Contours are broken (not bridged) across invalid/masked
    pixels: any vertex whose 3x3 neighbourhood touches invalid data is dropped,
    splitting the polyline there.

    Returns:
        A list of ``LineString`` geometries in EPSG:32646.
    """
    arr = index.astype(np.float64)
    good = np.isfinite(arr) & valid.astype(bool)
    if good.sum() < 2:
        return []
    fill = float(np.median(arr[good]))
    arr = np.where(good, arr, fill)
    invalid_dil = morphology.binary_dilation(~good, np.ones((3, 3), bool))
    height, width = arr.shape

    lines: List[LineString] = []
    for contour in measure.find_contours(arr, level=float(threshold)):
        rows = contour[:, 0]
        cols = contour[:, 1]
        rr = np.clip(np.round(rows).astype(int), 0, height - 1)
        cc = np.clip(np.round(cols).astype(int), 0, width - 1)
        ok = ~invalid_dil[rr, cc]
        for run in _true_runs(ok):
            if run.stop - run.start < 2:
                continue
            xs, ys = raster_xy(
                transform, rows[run].tolist(), cols[run].tolist(), offset="center"
            )
            xy = list(zip(np.atleast_1d(xs), np.atleast_1d(ys)))
            if len(xy) >= 2:
                lines.append(LineString(xy))
    return lines


def _true_runs(mask: np.ndarray) -> List[slice]:
    """Return slices of maximal runs of ``True`` in a 1-D boolean array."""
    runs: List[slice] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append(slice(start, i))
            start = None
    if start is not None:
        runs.append(slice(start, len(mask)))
    return runs


# ===========================================================================
# 2.5  Filtering
# ===========================================================================
def load_search_zone() -> Optional[Polygon]:
    """Load the digitised search zone as a UTM polygon, or ``None`` if absent.

    ``data/shoreline_search_zone.geojson`` (operator-digitised, PHASE2_SPEC.md
    §4) constrains extraction to the coast — CoastSat's ``max_dist_ref`` made
    human-controlled and auditable.
    """
    path = config.SEARCH_ZONE_PATH
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path).to_crs(config.METRIC_CRS)
    if gdf.empty:
        return None
    return unary_union(list(gdf.geometry))


def load_channel_lines_utm() -> List[LineString]:
    """Load the 13 tidal-channel closure lines as UTM ``LineString``s."""
    geojson = config.load_tidal_channels()
    lines: List[LineString] = []
    for feature in geojson["features"]:
        geom = feature.get("geometry")
        if geom and geom.get("type") == "LineString":
            lines.append(_to_utm(LineString(geom["coordinates"])))
    return lines


def _as_line_list(geom) -> List[LineString]:
    """Flatten a geometry to a list of non-empty ``LineString``s."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    # GeometryCollection or mixed: keep only line parts.
    out: List[LineString] = []
    for g in getattr(geom, "geoms", []):
        out.extend(_as_line_list(g))
    return out


def filter_contours(
    contours: List[LineString],
    search_zone: Optional[Polygon],
    channel_lines: List[LineString],
    min_length_m: float = config.MIN_SHORELINE_LENGTH_M,
):
    """Reduce raw contours to a single ordered shoreline geometry.

    Order matters: the min-length filter is applied AFTER the merge, otherwise
    fragmented contours (cloud edges, and especially L7 SLC-off striping gaps in
    2002/2003/2013) would be deleted before they could stitch into one shoreline.

    1. Clip to the search zone (removes inland/offshore false detections).
    2. Handle tidal-channel mouths: within a buffer of each closure line, drop
       contour parts that run landward (up the channel) and splice in the closure
       line — only where a contour actually reaches the mouth — so the shoreline
       follows the mouth-closing segment (D1). Land is east (``LAND_IS_EAST``);
       this coast faces west.
    3. ``unary_union`` + ``linemerge`` to stitch the fragments into continuous
       reaches.
    4. THEN drop merged reaches shorter than ``min_length_m`` (spurious specks
       only — real inter-channel reaches survive).
    5. Order alongshore (north -> south).

    Returns:
        A ``LineString``/``MultiLineString`` (EPSG:32646), or ``None`` if nothing
        survives.
    """
    lines = [ln for ln in contours if ln is not None and ln.length > 0]
    if not lines:
        return None
    geom = unary_union(lines)
    if search_zone is not None:
        geom = geom.intersection(search_zone)
    parts = _as_line_list(geom)
    if not parts:
        return None

    # Splice channel closures BEFORE merging so mouth-bridging segments join the
    # adjacent reaches; only THEN drop short *merged* reaches.
    parts = _apply_channel_closures(parts, channel_lines)
    merged = linemerge(unary_union(parts))
    ordered = [ln for ln in _as_line_list(merged) if ln.length >= min_length_m]
    if not ordered:
        return None
    ordered.sort(key=lambda ln: ln.centroid.y, reverse=True)  # north (higher y) first
    if len(ordered) == 1:
        return ordered[0]
    return MultiLineString(ordered)


def _apply_channel_closures(
    lines: List[LineString], channel_lines: List[LineString]
) -> List[LineString]:
    """Trim up-channel contour intrusions and bridge reached mouths with closures.

    For each closure line, contour material within ``CHANNEL_BUFFER_M`` that lies
    landward of the mouth (east when ``LAND_IS_EAST``) is discarded. The closure
    line is spliced in **only when a contour actually reaches that mouth** — an
    unconditional splice would add all 13 closure lines to every scene (and, with
    the corrected order, leave stray closure stubs where no shoreline reaches).
    """
    if not channel_lines:
        return lines
    out = list(lines)
    for closure in channel_lines:
        buf = closure.buffer(CHANNEL_BUFFER_M)
        xs = [c[0] for c in closure.coords]
        mouth_x = max(xs) if LAND_IS_EAST else min(xs)
        touched = False
        kept: List[LineString] = []
        for ln in out:
            if not ln.intersects(buf):
                kept.append(ln)
                continue
            touched = True
            kept.extend(_as_line_list(ln.difference(buf)))  # outside the mouth: keep
            for part in _as_line_list(ln.intersection(buf)):
                seaward = (part.centroid.x <= mouth_x if LAND_IS_EAST
                           else part.centroid.x >= mouth_x)
                if seaward:
                    kept.append(part)  # near the mouth but seaward: keep
                # landward (up-channel): dropped
        out = kept
        if touched:
            out.append(closure)  # bridge only mouths the shoreline actually reaches
    return out


# ===========================================================================
# 2.6  Drivers
# ===========================================================================
def default_settings(
    water_index_name: str = config.WATER_INDEX_DEFAULT,
    threshold_method: str = config.THRESHOLD_METHOD_DEFAULT,
    classifier_version: str = DEFAULT_CLASSIFIER_VERSION,
    min_length_m: float = config.MIN_SHORELINE_LENGTH_M,
) -> dict:
    """Assemble an extraction ``settings`` dict (loads search zone + closures once)."""
    return {
        "water_index": water_index_name,
        "threshold_method": threshold_method,
        "classifier_version": classifier_version,
        "min_length_m": min_length_m,
        "search_zone": load_search_zone(),
        "channel_lines": load_channel_lines_utm(),
    }


def _attach_row_metadata(scene: Scene, row: pd.Series) -> Scene:
    """Stash scene-list attributes on the Scene for :func:`extract_shoreline`."""
    scene._dry_year = int(row["dry_year"]) if pd.notna(row.get("dry_year")) else 0
    scene._season_label = row.get("season_label")
    scene._series = row.get("series")
    scene._aoi_cloud_pct = _to_float(row.get("aoi_cloud_pct"))
    scene._aoi_coverage_pct = _to_float(row.get("aoi_coverage_pct"))
    scene._slc_off = bool(row.get("slc_off", False))
    scene._composite_date_spread_days = _to_float(
        row.get("composite_date_spread_days", 0)
    )
    scene._season_complete = bool(row.get("season_complete", True))
    return scene


def extract_shoreline(scene: Scene, clf: Pipeline, settings: dict) -> dict:
    """Extract one scene's shoreline and return a schema record (D1/§3).

    Classifies the scene, computes the chosen water index, thresholds it on the
    sand∪water interface, marching-squares contours it, filters to the search
    zone + channel closures, and packages the locked output attributes. The
    geometry is reprojected to EPSG:4326 for storage under the ``geometry`` key.
    Scene-list attributes are read from those stashed by
    :func:`_attach_row_metadata` (falling back to defaults when absent).
    """
    labels = classify_scene(scene, clf)
    index = water_index(scene, settings["water_index"])
    threshold = interface_threshold(index, labels, settings["threshold_method"])
    contours = extract_contour(index, threshold, scene.valid, scene.transform)
    line_utm = filter_contours(
        contours, settings.get("search_zone"), settings.get("channel_lines", []),
        settings.get("min_length_m", config.MIN_SHORELINE_LENGTH_M),
    )

    length_m = float(line_utm.length) if line_utm is not None else 0.0
    n_vertices = _count_vertices(line_utm)
    geom_wgs = _to_wgs(line_utm) if line_utm is not None else None
    dry_year = int(getattr(scene, "_dry_year", 0))
    spread_days = float(getattr(scene, "_composite_date_spread_days", 0.0) or 0.0)

    flags: List[str] = []
    if bool(getattr(scene, "_slc_off", False)):
        flags.append("slc_off")
    if dry_year == 1991:
        flags.append("partial_1991_inflated_uncertainty")
    if scene.sensor in ("L4", "L5", "L7") and config.LANDSAT_CLOUD_MASK_ISSUE:
        flags.append("landsat_cloud_mask_issue")
    if spread_days > config.COMPOSITE_SPREAD_FLAG_DAYS:
        flags.append(f"composite_spread_gt_{int(config.COMPOSITE_SPREAD_FLAG_DAYS)}d")
    if line_utm is None:
        flags.append("no_shoreline")

    return {
        "image_id": scene.image_id,
        "sensor": scene.sensor,
        "sensor_group": sensor_group(scene.sensor),
        "acq_datetime_utc": pd.Timestamp(scene.acq_datetime_utc).isoformat(),
        "dry_year": dry_year or None,
        "season_label": getattr(scene, "_season_label", None),
        "series": getattr(scene, "_series", None),
        "pixel_size_m": scene.pixel_size_m,
        "georef_rmse_m": scene.georef_rmse_m,
        "aoi_cloud_pct": getattr(scene, "_aoi_cloud_pct", None),
        "aoi_coverage_pct": getattr(scene, "_aoi_coverage_pct", None),
        "slc_off": bool(getattr(scene, "_slc_off", False)),
        "composite_date_spread_days": spread_days,
        "season_complete": bool(getattr(scene, "_season_complete", True)),
        "water_index": settings["water_index"],
        "threshold_method": settings["threshold_method"],
        "threshold_value": float(threshold),
        "classifier_version": settings["classifier_version"],
        "length_m": length_m,
        "n_vertices": n_vertices,
        "pct_aoi_alongshore_covered": min(100.0, 100.0 * length_m / AOI_ALONGSHORE_M),
        "flags": ",".join(flags),
        "geometry": geom_wgs,
    }


def _count_vertices(geom) -> int:
    """Total vertex count across a (Multi)LineString (0 for ``None``)."""
    if geom is None:
        return 0
    return sum(len(ln.coords) for ln in _as_line_list(geom))


def extract_all(
    scene_list: pd.DataFrame,
    settings: dict,
    classifiers: Optional[Dict[str, Pipeline]] = None,
    checkpoint_path: str = SDS_SCENES_PATH,
    log_path: str = EXTRACTION_LOG_PATH,
    resume: bool = True,
    tile_px: int = DEFAULT_TILE_PX,
) -> "gpd.GeoDataFrame":
    """Extract every scene in a list, checkpointing after each one (D1/§3).

    Series-agnostic: works on either scene list. For each row it fetches the
    scene, picks the sensor-group classifier, extracts the shoreline, and appends
    the record — rewriting ``checkpoint_path`` (``sds_scenes.geojson``) after
    EVERY scene so a Colab disconnect over the ~600–900 dense scenes never loses
    completed work. Failures are logged to ``extraction_log.csv`` (one row per
    scene attempted, with the reason) and skipped. With ``resume`` the already
    completed ``image_id``s in an existing checkpoint are not recomputed.

    Args:
        scene_list: Output of ``build_scene_list_annual``/``_dense``.
        settings: From :func:`default_settings`.
        classifiers: ``{sensor_group: pipeline}``; loaded from ``models/`` when
            omitted.
        checkpoint_path / log_path: Output paths.
        resume: Skip scenes already present in the checkpoint.

    Returns:
        A ``GeoDataFrame`` (EPSG:4326) of all extracted shorelines.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    version = settings.get("classifier_version", DEFAULT_CLASSIFIER_VERSION)

    records: List[dict] = []
    done: set = set()
    if resume and os.path.exists(checkpoint_path):
        prev = gpd.read_file(checkpoint_path)
        for _, r in prev.iterrows():
            rec = r.drop(labels="geometry").to_dict()
            rec["geometry"] = r.geometry
            records.append(rec)
            done.add(rec.get("image_id"))

    clf_cache: Dict[str, Pipeline] = dict(classifiers or {})
    for _, row in scene_list.iterrows():
        image_id = str(row["image_id"])
        if image_id in done:
            continue
        group = sensor_group(str(row["sensor"]))
        try:
            if group not in clf_cache:
                clf_cache[group] = load_classifier(group, version)
            scene = fetch_scene(row, tile_px=tile_px)
            scene = _attach_row_metadata(scene, row)
            record = extract_shoreline(scene, clf_cache[group], settings)
            records.append(record)
            _write_checkpoint(records, checkpoint_path)
            _append_log(log_path, image_id, "ok",
                        f"length_m={record['length_m']:.1f}")
        except Exception as exc:  # keep going; a bad scene must not abort the run
            _append_log(log_path, image_id, "error", f"{type(exc).__name__}: {exc}")
    return _records_to_gdf(records)


def _records_to_gdf(records: List[dict]) -> "gpd.GeoDataFrame":
    """Build a EPSG:4326 GeoDataFrame with the locked column order."""
    if not records:
        return gpd.GeoDataFrame(columns=OUTPUT_SCHEMA + ["geometry"],
                                geometry="geometry", crs=config.STORAGE_CRS)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=config.STORAGE_CRS)
    cols = [c for c in OUTPUT_SCHEMA if c in gdf.columns] + ["geometry"]
    return gdf[cols]


def _write_checkpoint(records: List[dict], path: str) -> None:
    """Persist all records so far to a GeoJSON checkpoint (called every scene)."""
    gdf = _records_to_gdf([r for r in records if r.get("geometry") is not None])
    if len(gdf) == 0:
        return
    gdf.to_file(path, driver="GeoJSON")


def _append_log(path: str, image_id: str, status: str, reason: str) -> None:
    """Append one row to the extraction log (created with a header if new)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["image_id", "status", "reason", "logged_at_utc"])
        writer.writerow([image_id, status, reason,
                         datetime.now(timezone.utc).isoformat()])


def merge_annual(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Assemble the Series A merged file Phase 3 consumes (D1/§3).

    Keeps one feature per source scene (so each segment retains
    ``source_image_id`` and ``source_acq_datetime_utc`` for per-patch tidal
    correction) while carrying the shared ``dry_year``/``season_label``. Only
    Series A rows are included.

    Returns:
        A ``GeoDataFrame`` (EPSG:4326) written by the caller to
        ``sds_annual_merged.geojson``.
    """
    series_a = gdf[gdf["series"] == "A"].copy()
    series_a = series_a.rename(
        columns={"image_id": "source_image_id",
                 "acq_datetime_utc": "source_acq_datetime_utc"}
    )
    keep = [
        "dry_year", "season_label", "source_image_id", "source_acq_datetime_utc",
        "sensor", "sensor_group", "series", "composite_date_spread_days",
        "season_complete", "water_index", "threshold_method", "threshold_value",
        "georef_rmse_m", "length_m", "flags", "geometry",
    ]
    cols = [c for c in keep if c in series_a.columns]
    out = series_a[cols].sort_values(["dry_year", "source_acq_datetime_utc"])
    return out.reset_index(drop=True)


# ===========================================================================
# 6.  Benchmark + validation (Phase 2 deliverables)
# ===========================================================================
def build_transects(
    baseline_path: str = BASELINE_PATH,
    spacing_m: float = TRANSECT_SPACING_M,
    length_m: float = TRANSECT_LENGTH_M,
    write: bool = True,
) -> "gpd.GeoDataFrame":
    """Cast shore-normal transects from a digitised baseline (DSAS/QSCAT).

    Reads ``data/baseline.geojson`` — a single LineString digitised landward of
    and roughly parallel to the coast — and casts transects every ``spacing_m``
    along it, each ``length_m`` long and normal to the local baseline tangent,
    pointing seaward (west, since ``LAND_IS_EAST``). Each transect is a
    ``LineString`` ordered land->sea (vertex 0 on the baseline) with a stable
    integer ``transect_id`` increasing alongshore — the convention
    :func:`benchmark_extraction` and :func:`intersensor_bias` expect.

    Args:
        baseline_path: Path to the digitised baseline GeoJSON (EPSG:4326).
        spacing_m: Alongshore spacing between transects.
        length_m: Seaward reach of each transect from the baseline.
        write: Write ``outputs/transects.geojson`` (EPSG:4326) when True.

    Returns:
        A ``GeoDataFrame`` of transects in EPSG:4326 with ``transect_id``.
    """
    gdf = gpd.read_file(baseline_path).to_crs(config.METRIC_CRS)
    base = linemerge(unary_union(list(gdf.geometry)))
    baseline_parts = _as_line_list(base)
    if not baseline_parts:
        raise ValueError(f"{baseline_path} has no LineString baseline")

    records: List[dict] = []
    tid = 0
    for line in baseline_parts:
        n = int(np.floor(line.length / spacing_m))
        for i in range(n + 1):
            d = i * spacing_m
            p = line.interpolate(d)
            eps = max(1.0, min(spacing_m, line.length) * 0.5)
            a = line.interpolate(max(0.0, d - eps))
            b = line.interpolate(min(line.length, d + eps))
            tx, ty = (b.x - a.x), (b.y - a.y)
            norm = math.hypot(tx, ty) or 1.0
            tx, ty = tx / norm, ty / norm
            nx, ny = -ty, tx  # left normal of the tangent
            # Orient seaward: land is east, so seaward normals have nx < 0.
            if (nx > 0) == LAND_IS_EAST:
                nx, ny = -nx, -ny
            end = (p.x + nx * length_m, p.y + ny * length_m)
            records.append({
                "transect_id": tid,
                "geometry": LineString([(p.x, p.y), end]),
            })
            tid += 1

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=config.METRIC_CRS)
    out = out.to_crs(config.STORAGE_CRS)
    if write:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        out.to_file(TRANSECTS_PATH, driver="GeoJSON")
    return out


def load_transects(path: str = TRANSECTS_PATH) -> "gpd.GeoDataFrame":
    """Load ``outputs/transects.geojson`` reprojected to EPSG:32646 (metric).

    The benchmark and inter-sensor bias measure distances along transects, so
    they need them in the metric CRS. Build them first with
    :func:`build_transects`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run build_transects() (needs data/baseline.geojson)"
        )
    return gpd.read_file(path).to_crs(config.METRIC_CRS)


def _transect_crossings(line, transect, origin) -> List[float]:
    """Land->sea distances from ``origin`` to every crossing of ``line``.

    Returns a sorted list of the distances at which ``line`` crosses ``transect``
    (empty if it never does). A shore-normal transect can cross a complex
    shoreline more than once (spits, channel mouths, double sandbars near
    Bakkhali and the Naf), so ALL crossings are returned — letting the caller
    detect and exclude multi-crossing transects rather than silently collapsing
    them to one point. A rare collinear (LineString) overlap contributes its
    representative point.
    """
    if line is None:
        return []
    inter = line.intersection(transect)
    if inter.is_empty:
        return []
    dists: List[float] = []
    for g in getattr(inter, "geoms", [inter]):
        if g.geom_type == "Point":
            dists.append(float(origin.distance(g)))
        else:  # collinear LineString/MultiLineString overlap (degenerate)
            dists.append(float(origin.distance(g.representative_point())))
    return sorted(dists)


def _transect_offsets(
    line_a, line_b, transects: "gpd.GeoDataFrame"
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed cross-shore offsets (m) between two shorelines along transects.

    For each transect (a cross-shore ``LineString`` ordered land->sea) the offset
    is ``d(line_a) − d(line_b)`` (positive = A seaward of B), where each ``d`` is
    the land->sea distance from the transect origin to the shoreline crossing.

    A transect contributes an offset ONLY when each shoreline crosses it exactly
    once. If either line crosses more than once, the transect is flagged
    multi-crossing and its offset is ``NaN`` — excluded rather than resolved by an
    arbitrary nearest-crossing pick, which would inject a systematic landward bias
    exactly at the complex mouths (and inflate the benchmark RMSE). Transects that
    miss either line are also ``NaN`` (but not flagged multi). All geometries must
    be in a metric CRS (EPSG:32646).

    Returns:
        ``(offsets, multi)`` — a float array of offsets (``NaN`` where excluded or
        missing) and a bool array marking transects excluded because either line
        crossed more than once.
    """
    offsets: List[float] = []
    multi: List[bool] = []
    for _, t in transects.iterrows():
        tr = t.geometry
        origin = tr.interpolate(0.0)
        ca = _transect_crossings(line_a, tr, origin)
        cb = _transect_crossings(line_b, tr, origin)
        multi.append(len(ca) > 1 or len(cb) > 1)
        offsets.append(ca[0] - cb[0] if (len(ca) == 1 and len(cb) == 1) else np.nan)
    return np.asarray(offsets, dtype=float), np.asarray(multi, dtype=bool)


def _labels_all_interface(scene: Scene) -> np.ndarray:
    """Fallback 'no classifier' labelling: every valid pixel is interface.

    Used by the benchmark's ``classifier='none'`` arm so the threshold is taken
    on the full valid-scene index (the old-pipeline behaviour) for comparison.
    """
    labels = np.full(scene.valid.shape, config.CLASS_OTHER, dtype=int)
    labels[scene.valid] = config.CLASS_SAND  # sand∪water == all valid
    return labels


def benchmark_extraction(
    scenes: List[Scene],
    reference_shorelines: "gpd.GeoDataFrame",
    transects: Optional["gpd.GeoDataFrame"] = None,
    classifiers: Optional[Dict[str, Pipeline]] = None,
    indices: Sequence[str] = tuple(config.WATER_INDICES),
    threshold_methods: Sequence[str] = tuple(config.THRESHOLD_METHODS),
    use_classifier_options: Sequence[bool] = (True,),
) -> pd.DataFrame:
    """Grid-search the extraction configuration against manual references (§6).

    Evaluates ``index × threshold × classifier`` on the digitised reference
    scenes and reports, per sensor group, the transect-normal error (RMSE, mean
    bias, MAE) so the operational configuration can be selected and a paper
    figure produced. Reference shorelines are matched to a scene by ``image_id``
    (an ``image_id`` column on the reference layer). ``transects`` must be in
    EPSG:32646; when omitted they are loaded from ``outputs/transects.geojson``
    (build them once with :func:`build_transects`).

    Returns:
        A tidy DataFrame: one row per ``(sensor_group, water_index,
        threshold_method, classifier)`` with ``rmse_m``, ``bias_m``, ``mae_m``,
        ``n`` (transects used) and ``n_multi_excluded`` (transects dropped because
        a shoreline crossed them more than once — reported for honesty).
    """
    if transects is None:
        transects = load_transects()
    ref_by_id = {
        str(r["image_id"]): r.geometry for _, r in reference_shorelines.iterrows()
    }
    search_zone = load_search_zone()
    channel_lines = load_channel_lines_utm()
    rows: List[dict] = []
    for use_clf in use_classifier_options:
        for name in indices:
            for method in threshold_methods:
                acc: Dict[str, Dict[str, object]] = {}
                for scene in scenes:
                    ref = ref_by_id.get(scene.image_id)
                    if ref is None:
                        continue
                    group = sensor_group(scene.sensor)
                    clf = (classifiers or {}).get(group)
                    if use_clf and clf is None:
                        clf = load_classifier(group)
                    try:
                        labels = (classify_scene(scene, clf) if use_clf
                                  else _labels_all_interface(scene))
                        index = water_index(scene, name)
                        thr = interface_threshold(index, labels, method)
                        contours = extract_contour(index, thr, scene.valid,
                                                   scene.transform)
                        line = filter_contours(contours, search_zone, channel_lines)
                        line_utm = _to_utm(line) if line is not None else None
                    except Exception:
                        continue
                    off, multi = _transect_offsets(line_utm, _to_utm(ref), transects)
                    bucket = acc.setdefault(group, {"off": [], "n_multi": 0})
                    bucket["off"].extend(off[np.isfinite(off)].tolist())
                    bucket["n_multi"] += int(multi.sum())
                for group, bucket in acc.items():
                    arr = np.asarray(bucket["off"], dtype=float)
                    if arr.size == 0:
                        continue
                    rows.append(
                        {
                            "sensor_group": group,
                            "water_index": name,
                            "threshold_method": method,
                            "classifier": "local" if use_clf else "none",
                            "rmse_m": float(np.sqrt(np.mean(arr ** 2))),
                            "bias_m": float(np.mean(arr)),
                            "mae_m": float(np.mean(np.abs(arr))),
                            "n": int(arr.size),
                            "n_multi_excluded": int(bucket["n_multi"]),
                        }
                    )
    return pd.DataFrame(rows)


def intersensor_bias(
    gdf: "gpd.GeoDataFrame",
    transects: Optional["gpd.GeoDataFrame"] = None,
    max_days: int = 2,
) -> pd.DataFrame:
    """Quantify cross-sensor offset from near-coincident scene pairs (D6/§6).

    Finds scene pairs from different sensors acquired within ``max_days`` and
    measures the mean transect-normal offset between their shorelines (e.g. the
    L5+L7 2001/2005 overlaps and the 2016–2021 L8/L9×S2 overlap). A significant,
    consistent offset is the most obvious confound for a 40-year multi-sensor
    trend; report it (and correct it) before computing rates. ``transects`` must
    be in EPSG:32646; when omitted they are loaded from
    ``outputs/transects.geojson`` (build them with :func:`build_transects`).

    Returns:
        One row per sensor pair with ``mean_offset_m``, ``std_offset_m``,
        ``median_offset_m``, ``n_pairs``, ``n_transect_samples``, and
        ``n_multi_excluded`` (transects dropped for a >1-crossing shoreline).
    """
    if transects is None:
        transects = load_transects()
    g = gdf.copy()
    g["_t"] = pd.to_datetime(g["acq_datetime_utc"], utc=True)
    pairs: Dict[Tuple[str, str], List[float]] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}
    multi_counts: Dict[Tuple[str, str], int] = {}
    rows = list(g.iterrows())
    for i in range(len(rows)):
        _, a = rows[i]
        for j in range(i + 1, len(rows)):
            _, b = rows[j]
            if a["sensor"] == b["sensor"]:
                continue
            if abs((a["_t"] - b["_t"]).total_seconds()) > max_days * 86400:
                continue
            key = tuple(sorted((str(a["sensor"]), str(b["sensor"]))))
            off, multi = _transect_offsets(_to_utm(a.geometry), _to_utm(b.geometry),
                                           transects)
            finite = off[np.isfinite(off)]
            multi_counts[key] = multi_counts.get(key, 0) + int(multi.sum())
            if finite.size == 0:
                continue
            # Orient the offset consistently by the sorted-pair order.
            sign = 1.0 if (str(a["sensor"]), str(b["sensor"])) == key else -1.0
            pairs.setdefault(key, []).extend((sign * finite).tolist())
            pair_counts[key] = pair_counts.get(key, 0) + 1
    out: List[dict] = []
    for key, vals in pairs.items():
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "sensor_a": key[0],
                "sensor_b": key[1],
                "mean_offset_m": float(np.mean(arr)),
                "std_offset_m": float(np.std(arr)),
                "median_offset_m": float(np.median(arr)),
                "n_pairs": pair_counts[key],
                "n_transect_samples": int(arr.size),
                "n_multi_excluded": multi_counts.get(key, 0),
            }
        )
    return pd.DataFrame(out)
