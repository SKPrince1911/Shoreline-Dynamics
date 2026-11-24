"""Phase 3a — image co-registration of every shoreline scene to one Sentinel-2 reference.

Why: satellite georeferencing errors shift a whole image (and therefore its
shoreline) by metres to tens of metres. Vos et al. (2023, Commun. Earth Environ.
4:345) found systematic Landsat/Sentinel-2 offsets of several metres and
recommend co-registration; Mikkelsen et al. (2026, Sci. Rep.) reduced S2 shoreline
RMSE ~26% with it. Uncorrected, these offsets leak into rates and inter-sensor bias.

Method (translation-only, per scene):
  1. A fixed set of square CHIPS (``CHIP_SIZE_M``) is laid along the coast on the
     LAND side of ``data/baseline.geojson`` (two rows, ``CHIP_OFFSETS_M`` inland),
     away from the moving beach/water. Chip corners are multiples of 30 m so the
     10 m and 30 m grids nest exactly.
  2. One reference scene = the clearest dry-season Sentinel-2 acquisition
     (chosen automatically, :func:`select_reference`).
  3. For each target scene and chip: NIR is fetched on the sensor's grid; pixels
     that are invalid (no data / cloud / shadow / saturated) or water (MNDWI > 0)
     in EITHER image are excluded; both chips are z-scored and Hann-windowed; the
     sub-pixel translation is found by upsampled cross-correlation
     (Guizar-Sicairos et al. 2008, Opt. Lett. 33:156; scikit-image
     ``phase_cross_correlation``), refined once. A chip is kept only if the
     shift is ≤ ``MAX_SHIFT_M`` and the post-shift Pearson r ≥ ``MIN_CHIP_R``.
  4. Per scene: robust median of chip shifts after 3·MAD outlier rejection;
     standard error and spread are stored for the Phase 4 uncertainty budget; a
     linear along-coast trend of the shifts is stored as a QC flag for
     non-translational (rotation/scale) error.
  5. :func:`apply_shifts` translates each scene's shoreline VECTOR by its shift
     (no re-extraction needed; the image and its waterline move together).

Sign convention: ``dx_m``/``dy_m`` is the translation to ADD to the target
scene's map coordinates (easting/northing, EPSG:32646) to align it to the
reference.

Google Earth Engine is imported lazily (this module imports without GEE).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import Affine
from scipy import ndimage
from shapely.affinity import translate
from shapely.geometry import Point, box
from skimage.filters import window
from skimage.registration import phase_cross_correlation

from . import config, extract

# ---------------------------------------------------------------------------
# Tunables (record any change in CLAUDE.md decision log)
# ---------------------------------------------------------------------------
CHIP_SIZE_M: float = 1920.0            # 64 Landsat px / 192 S2 px
CHIP_SPACING_M: float = 3000.0         # along the baseline
CHIP_OFFSETS_M: Tuple[float, ...] = (1500.0, 3500.0)  # chip centre, inland of baseline
GRID_ANCHOR_M: int = 30                # chip corners on multiples of 30 m
MAX_SHIFT_M: float = 90.0              # reject chip shifts > 3 Landsat px
MIN_CHIP_VALID_FRAC: float = 0.80      # usable (valid & non-water in both) fraction
MIN_CHIP_R: float = 0.50               # post-shift Pearson r (reliability gate)
MIN_CHIPS_OK: int = 6                  # inlier chips needed for status "ok"
MAD_K: float = 3.0                     # outlier rejection threshold (× 1.4826·MAD)
UPSAMPLE: int = 100                    # 1/100 px precision (0.3 m on Landsat)
N_WORKERS: int = 8                     # concurrent chip downloads
WATER_MNDWI: float = 0.0               # MNDWI > this -> water (excluded)

COREG_DIR: str = os.path.join(config.OUTPUT_DIR, "coreg")
DRIVE_COREG_DIR: str = "/content/drive/MyDrive/shoreline_dynamics/coreg"
SHIFTS_CSV: str = "coreg_shifts.csv"
REFERENCE_JSON: str = "coreg_reference.json"
CHIPS_GEOJSON: str = "coreg_chips.geojson"

_COLUMNS = [
    "image_id", "sensor", "acq_datetime_utc", "series", "ref_image_id",
    "n_chips", "n_chips_measured", "n_chips_used", "dx_m", "dy_m",
    "se_dx_m", "se_dy_m", "spread_m", "median_r",
    "trend_dx_m_per_10km", "trend_dy_m_per_10km", "status", "note",
    "run_utc",
]


# ===========================================================================
# Chips
# ===========================================================================
def _landward_unit_normal(baseline, s: float, seaward_pt: Optional[Point]) -> Tuple[float, float]:
    """Unit normal to the baseline at arc-length ``s``, pointing LANDWARD."""
    ds = 200.0
    a = baseline.interpolate(max(s - ds, 0.0))
    b = baseline.interpolate(min(s + ds, baseline.length))
    tx, ty = b.x - a.x, b.y - a.y
    n = math.hypot(tx, ty)
    nx, ny = -ty / n, tx / n                      # left-hand normal
    p = baseline.interpolate(s)
    if seaward_pt is not None:
        # Landward = the side AWAY from the offshore reference point.
        to_sea = (seaward_pt.x - p.x, seaward_pt.y - p.y)
        if nx * to_sea[0] + ny * to_sea[1] > 0:
            nx, ny = -nx, -ny
    elif extract.LAND_IS_EAST and nx < 0:
        nx, ny = -nx, -ny
    return nx, ny


def build_chips(write: bool = True) -> gpd.GeoDataFrame:
    """Lay the co-registration chips on the land side of the baseline.

    Returns a GeoDataFrame (EPSG:32646) with ``chip_id``, ``along_km``,
    ``offset_m`` and a square polygon whose corners are multiples of
    ``GRID_ANCHOR_M``.
    """
    parts = extract._baseline_parts(
        gpd.read_file(extract.BASELINE_PATH).to_crs(config.METRIC_CRS).union_all()
    )
    if not parts:
        raise RuntimeError("data/baseline.geojson is empty or invalid")
    baseline = max(parts, key=lambda ln: ln.length)
    seaward = extract._seaward_reference_utm()
    half = CHIP_SIZE_M / 2.0
    rows = []
    s = CHIP_SPACING_M / 2.0
    while s < baseline.length:
        p = baseline.interpolate(s)
        nx, ny = _landward_unit_normal(baseline, s, seaward)
        for off in CHIP_OFFSETS_M:
            cx = round((p.x + nx * off) / GRID_ANCHOR_M) * GRID_ANCHOR_M
            cy = round((p.y + ny * off) / GRID_ANCHOR_M) * GRID_ANCHOR_M
            rows.append({
                "chip_id": f"c{len(rows):03d}",
                "along_km": s / 1000.0,
                "offset_m": off,
                "geometry": box(cx - half, cy - half, cx + half, cy + half),
            })
        s += CHIP_SPACING_M
    gdf = gpd.GeoDataFrame(rows, crs=config.METRIC_CRS)
    if write:
        for path in _out_paths(CHIPS_GEOJSON):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    return gdf


# ===========================================================================
# Fetching
# ===========================================================================
def _chip_ee_image(image_id: str, sensor: str) -> "ee.Image":
    """One float ee.Image with green, nir, swir1 (bilinear) + presence + QA bands."""
    import ee

    granules = [g.strip() for g in str(image_id).split(",") if g.strip()]
    bm = config.BAND_MAP[sensor]
    src = [bm["green"], bm["nir"], bm["swir1"]]
    refl = ee.ImageCollection([
        ee.Image(extract._ee_asset_id(g, sensor)).select(src, ["green", "nir", "swir1"])
        .resample("bilinear")
        for g in granules
    ]).mosaic()
    presence = ee.ImageCollection(
        [ee.Image(extract._ee_asset_id(g, sensor)).select([src[0]]) for g in granules]
    ).mosaic().mask().rename("obs_mask")
    if sensor == "S2":
        day = ee.Date(extract.date_from_image_id(granules[0]))
        qa = (
            ee.ImageCollection(config.S2_SR_HARMONIZED)
            .filterDate(day, day.advance(1, "day"))
            .filter(ee.Filter.inList("system:index", ee.List(granules)))
            .linkCollection(ee.ImageCollection(config.CLOUD_SCORE_PLUS), ["cs"])
            .select("cs").mosaic()
        )
    else:
        qa = ee.ImageCollection([
            ee.Image(extract._ee_asset_id(g, sensor)).select(["QA_PIXEL", "QA_RADSAT"])
            for g in granules
        ]).mosaic()
    return ee.Image.cat([refl, presence, qa]).toFloat()


def _decode_chip(arr: np.ndarray, sensor: str) -> Dict[str, np.ndarray]:
    """Raw downloaded bands -> nir, water, valid (same rules as extract.fetch_scene)."""
    family = config.SR_SCALE_FAMILY[sensor]
    gain, offset = config.SR_SCALE[family]
    raw_green, raw_nir, raw_swir = arr[0], arr[1], arr[2]
    green = raw_green * gain + offset
    nir = raw_nir * gain + offset
    swir = raw_swir * gain + offset
    observed = np.nan_to_num(arr[3], nan=0.0) >= 0.5
    observed &= ~((raw_green == 0) & (raw_nir == 0) & (raw_swir == 0))
    if sensor == "S2":
        bad = np.nan_to_num(arr[4], nan=0.0) < config.CS_THRESHOLD
    else:
        qap = np.nan_to_num(arr[4], nan=0.0).astype(np.uint16)
        radsat = np.nan_to_num(arr[5], nan=0.0).astype(np.uint16)
        bad = ((qap & 0b11110) != 0) | (radsat != 0)   # bits 1-4: dilated/cirrus/cloud/shadow
    valid = observed & ~bad & np.isfinite(nir)
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (green - swir) / (green + swir)
    water = np.nan_to_num(mndwi, nan=0.0) > WATER_MNDWI
    return {"nir": nir.astype(np.float32), "water": water, "valid": valid}


def _chip_grid(poly, res: int) -> Tuple[List[float], int, int]:
    minx, miny, maxx, maxy = poly.bounds
    w = int(round((maxx - minx) / res))
    h = int(round((maxy - miny) / res))
    return [res, 0.0, minx, 0.0, -res, maxy], w, h


def fetch_chips(
    image_id: str, sensor: str, chips: gpd.GeoDataFrame, res: Optional[int] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Download every chip of one scene (concurrently). Failed chips are omitted."""
    res = int(res or config.PIXEL_SIZE_M[sensor])
    img = _chip_ee_image(image_id, sensor)
    nb = 5 if sensor == "S2" else 6

    def _one(row):
        tr, w, h = _chip_grid(row.geometry, res)
        params = {"crs": config.METRIC_CRS, "crs_transform": tr,
                  "dimensions": f"{w}x{h}", "format": "GEO_TIFF"}
        try:
            arr = extract._download_geotiff(img, params, nb, h, w)
            return row.chip_id, _decode_chip(arr, sensor)
        except Exception:
            return row.chip_id, None

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        out = dict(pool.map(_one, [r for r in chips.itertuples()]))
    return {k: v for k, v in out.items() if v is not None}


def _block_reduce_ref(ch: Dict[str, np.ndarray], factor: int) -> Dict[str, np.ndarray]:
    """Aggregate a 10 m reference chip to a coarser nested grid (nan-aware mean)."""
    if factor == 1:
        return ch
    h, w = ch["nir"].shape
    h2, w2 = h // factor, w // factor

    def _blocks(a):
        return a[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor)

    nir = np.where(ch["valid"], ch["nir"], np.nan)
    with np.errstate(invalid="ignore"):
        nir_c = np.nanmean(_blocks(nir), axis=(1, 3))
    valid_c = _blocks(ch["valid"]).all(axis=(1, 3))
    water_c = _blocks(ch["water"]).mean(axis=(1, 3)) > 0.5
    return {"nir": nir_c.astype(np.float32), "water": water_c, "valid": valid_c & np.isfinite(nir_c)}


# ===========================================================================
# Shift estimation
# ===========================================================================
def _prep(a: np.ndarray, m: np.ndarray, w: np.ndarray) -> np.ndarray:
    v = a[m]
    z = np.zeros_like(a, dtype=np.float64)
    z[m] = (v - v.mean()) / (v.std() + 1e-12)
    return z * w


def estimate_chip_shift(
    ref: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray], px: float
) -> Optional[Dict[str, float]]:
    """Sub-pixel translation (m) that aligns ``tgt`` onto ``ref`` for one chip.

    Returns ``{"dx_m","dy_m","r"}`` or ``None`` if the chip fails a quality gate.
    """
    if ref["nir"].shape != tgt["nir"].shape:
        return None
    m = ref["valid"] & tgt["valid"] & ~ref["water"] & ~tgt["water"]
    if m.mean() < MIN_CHIP_VALID_FRAC:
        return None
    w = window("hann", m.shape)
    r0 = _prep(np.nan_to_num(ref["nir"]), m, w)
    t0 = _prep(np.nan_to_num(tgt["nir"]), m, w)
    if not (np.any(r0) and np.any(t0)):
        return None
    s1, _, _ = phase_cross_correlation(r0, t0, upsample_factor=UPSAMPLE, normalization=None)
    t_raw = np.nan_to_num(tgt["nir"])
    t1 = ndimage.shift(t_raw, s1, order=3, mode="nearest")
    m1 = m & (ndimage.shift(m.astype(float), s1, order=0, mode="constant", cval=0) > 0.5)
    if m1.mean() < MIN_CHIP_VALID_FRAC * 0.8:
        return None
    s2, _, _ = phase_cross_correlation(r0, _prep(t1, m1, w), upsample_factor=UPSAMPLE,
                                       normalization=None)
    s = np.asarray(s1) + np.asarray(s2)
    dx_m, dy_m = float(s[1]) * px, float(-s[0]) * px
    if math.hypot(dx_m, dy_m) > MAX_SHIFT_M:
        return None
    t_fin = ndimage.shift(t_raw, s, order=3, mode="nearest")
    mf = m1
    a, b = np.nan_to_num(ref["nir"])[mf], t_fin[mf]
    if a.size < 50 or a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r) or r < MIN_CHIP_R:
        return None
    return {"dx_m": dx_m, "dy_m": dy_m, "r": r}


def aggregate_shifts(per_chip: pd.DataFrame) -> Dict[str, object]:
    """Robust per-scene translation from chip shifts (median after 3·MAD rejection)."""
    out: Dict[str, object] = {
        "n_chips_measured": int(len(per_chip)), "n_chips_used": 0,
        "dx_m": np.nan, "dy_m": np.nan, "se_dx_m": np.nan, "se_dy_m": np.nan,
        "spread_m": np.nan, "median_r": np.nan,
        "trend_dx_m_per_10km": np.nan, "trend_dy_m_per_10km": np.nan,
    }
    if len(per_chip) == 0:
        return out
    keep = np.ones(len(per_chip), bool)
    for c in ("dx_m", "dy_m"):
        v = per_chip[c].to_numpy()
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        mad = max(mad, 0.25 * 10.0)          # floor: never tighter than ~2.5 m
        keep &= np.abs(v - med) <= MAD_K * mad
    inl = per_chip[keep]
    n = len(inl)
    out["n_chips_used"] = int(n)
    if n == 0:
        return out
    dx, dy = inl["dx_m"].to_numpy(), inl["dy_m"].to_numpy()
    out["dx_m"], out["dy_m"] = float(np.median(dx)), float(np.median(dy))
    if n > 1:
        out["se_dx_m"] = float(1.2533 * dx.std(ddof=1) / math.sqrt(n))
        out["se_dy_m"] = float(1.2533 * dy.std(ddof=1) / math.sqrt(n))
        out["spread_m"] = float(np.sqrt(np.mean((dx - out["dx_m"]) ** 2 + (dy - out["dy_m"]) ** 2)))
    out["median_r"] = float(inl["r"].median())
    if n >= 4 and inl["along_km"].nunique() > 2:
        out["trend_dx_m_per_10km"] = float(np.polyfit(inl["along_km"], dx, 1)[0] * 10)
        out["trend_dy_m_per_10km"] = float(np.polyfit(inl["along_km"], dy, 1)[0] * 10)
    return out


# ===========================================================================
# Reference
# ===========================================================================
def select_reference(
    chips: gpd.GeoDataFrame,
    scene_list: Optional[pd.DataFrame] = None,
    years: Tuple[int, int] = (2022, 2026),
    n_try: int = 8,
    min_chip_frac: float = 0.85,
) -> Dict[str, object]:
    """Pick the clearest dry-season Sentinel-2 scene as the reference (automatic).

    Candidates: S2 rows of the Series B scene list with ``dry_year`` in ``years``
    and acquisition month in Nov–Mar, ranked by AOI coverage (desc) then AOI cloud
    (asc). The first candidate whose chips are ≥ ``min_chip_frac`` usable
    (valid & non-water ≥ ``MIN_CHIP_VALID_FRAC``) is chosen.

    Returns ``{"image_id", "sensor", "acq_datetime_utc", "chips_10m", "usable_frac",
    "candidates_tried"}``.
    """
    sl = scene_list if scene_list is not None else extract.get_scene_list_dense()
    s2 = sl[sl["sensor"] == "S2"].copy()
    t = extract._parse_utc_series(s2["acq_datetime_utc"])
    s2 = s2[(t.dt.month.isin([11, 12, 1, 2, 3])).to_numpy()
            & s2["dry_year"].between(*years).to_numpy()]
    s2 = s2.sort_values(["aoi_coverage_pct", "aoi_cloud_pct"], ascending=[False, True])
    tried = []
    for _, row in s2.head(n_try).iterrows():
        ch = fetch_chips(row["image_id"], "S2", chips, res=10)
        usable = [
            (v["valid"] & ~v["water"]).mean() >= MIN_CHIP_VALID_FRAC for v in ch.values()
        ]
        frac = (sum(usable) / len(chips)) if len(chips) else 0.0
        tried.append({"image_id": row["image_id"], "usable_frac": round(frac, 3),
                      "aoi_cloud_pct": row["aoi_cloud_pct"]})
        print(f"  ref candidate {row['image_id'][:60]}  usable chips {frac:.0%}", flush=True)
        if frac >= min_chip_frac:
            return {"image_id": row["image_id"], "sensor": "S2",
                    "acq_datetime_utc": str(row["acq_datetime_utc"]),
                    "chips_10m": ch, "usable_frac": frac, "candidates_tried": tried}
    raise RuntimeError(f"no S2 reference reached {min_chip_frac:.0%} usable chips: {tried}")


def load_or_select_reference(chips: gpd.GeoDataFrame, **kwargs) -> Dict[str, object]:
    """Reuse the reference recorded in ``coreg_reference.json`` if present (so a
    resumed run uses the SAME reference), else select one and record it."""
    import json

    paths = _out_paths(REFERENCE_JSON)
    for p in reversed(paths):
        if os.path.exists(p):
            meta = json.load(open(p))
            print("reusing recorded reference:", meta["image_id"], flush=True)
            ch = fetch_chips(meta["image_id"], "S2", chips, res=10)
            return {**meta, "chips_10m": ch}
    ref = select_reference(chips, **kwargs)
    meta = {k: v for k, v in ref.items() if k != "chips_10m"}
    meta["selected_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["params"] = {"CHIP_SIZE_M": CHIP_SIZE_M, "CHIP_SPACING_M": CHIP_SPACING_M,
                      "CHIP_OFFSETS_M": list(CHIP_OFFSETS_M), "MAX_SHIFT_M": MAX_SHIFT_M,
                      "MIN_CHIP_VALID_FRAC": MIN_CHIP_VALID_FRAC, "MIN_CHIP_R": MIN_CHIP_R,
                      "MIN_CHIPS_OK": MIN_CHIPS_OK, "UPSAMPLE": UPSAMPLE}
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
    return ref


def reference_on_grid(reference: Dict[str, object], res: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Reference chips on a target sensor's grid (10 m native, else block mean)."""
    factor = int(round(res / 10))
    return {k: _block_reduce_ref(v, factor) for k, v in reference["chips_10m"].items()}


# ===========================================================================
# Per-scene + batch
# ===========================================================================
def coregister_scene(
    image_id: str,
    sensor: str,
    chips: gpd.GeoDataFrame,
    reference: Dict[str, object],
    _ref_cache: Optional[Dict[int, dict]] = None,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Co-register one scene. Returns (summary record, per-chip DataFrame)."""
    res = int(config.PIXEL_SIZE_M[sensor])
    if _ref_cache is not None and res in _ref_cache:
        ref = _ref_cache[res]
    else:
        ref = reference_on_grid(reference, res)
        if _ref_cache is not None:
            _ref_cache[res] = ref
    tgt = fetch_chips(image_id, sensor, chips, res=res)
    along = dict(zip(chips["chip_id"], chips["along_km"]))
    rows = []
    for cid, t in tgt.items():
        if cid not in ref:
            continue
        est = estimate_chip_shift(ref[cid], t, float(res))
        if est is not None:
            rows.append({"chip_id": cid, "along_km": along[cid], **est})
    per_chip = pd.DataFrame(rows, columns=["chip_id", "along_km", "dx_m", "dy_m", "r"])
    agg = aggregate_shifts(per_chip)
    status = "ok" if agg["n_chips_used"] >= MIN_CHIPS_OK else (
        "few_tiepoints" if agg["n_chips_used"] > 0 else "failed")
    rec = {"image_id": image_id, "sensor": sensor,
           "ref_image_id": reference["image_id"], "n_chips": int(len(chips)),
           **agg, "status": status, "note": "",
           "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return rec, per_chip


def scenes_to_coregister() -> pd.DataFrame:
    """Unique scenes of Series A ∪ Series B (image_id, sensor, acq time, series)."""
    a = gpd.read_file(extract.SDS_SCENES_PATH, ignore_geometry=True)
    b = gpd.read_file(os.path.join(extract.SHORELINE_DIR, "sds_scenes_B.geojson"),
                      ignore_geometry=True)
    cols = ["image_id", "sensor", "acq_datetime_utc"]
    a = a[cols].assign(series="A")
    b = b[cols].assign(series="B")
    both = pd.concat([a, b], ignore_index=True)
    s = both.groupby("image_id")["series"].apply(lambda x: "".join(sorted(set(x))))
    out = both.drop_duplicates("image_id").drop(columns="series").merge(
        s.rename("series"), left_on="image_id", right_index=True)
    return out.sort_values("acq_datetime_utc").reset_index(drop=True)


def _out_paths(filename: str, base: Optional[str] = None) -> List[str]:
    if base is not None:
        return [os.path.join(base, filename)]
    paths = [os.path.join(COREG_DIR, filename)]
    if os.path.isdir(extract.DRIVE_MOUNT):
        paths.append(os.path.join(DRIVE_COREG_DIR, filename))
    return paths


def _read_existing(paths: List[str]) -> pd.DataFrame:
    for p in reversed(paths):  # prefer Drive (durable) copy
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame(columns=_COLUMNS)


def run_coregistration(
    scenes: pd.DataFrame,
    chips: gpd.GeoDataFrame,
    reference: Dict[str, object],
    out_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Co-register every scene, resumably (already-done image_ids are skipped).

    ``out_dir`` = None writes to ``outputs/coreg`` (+ Drive mirror). Pass a scratch
    directory (e.g. ``/content/smoke``) for smoke tests, per project practice.
    """
    paths = _out_paths(SHIFTS_CSV, out_dir)
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    done = _read_existing(paths)
    done_ids = set(done["image_id"])
    todo = scenes[~scenes["image_id"].isin(done_ids)]
    if limit is not None:
        todo = todo.head(limit)
    print(f"{len(done_ids)} already done, {len(todo)} to run", flush=True)
    recs = done.to_dict("records")
    cache: Dict[int, dict] = {}
    t0 = datetime.now()
    for i, row in enumerate(todo.itertuples(), 1):
        try:
            rec, _ = coregister_scene(row.image_id, row.sensor, chips, reference, cache)
        except Exception as exc:
            rec = {"image_id": row.image_id, "sensor": row.sensor, "status": "error",
                   "ref_image_id": reference["image_id"], "note": str(exc)[:200],
                   "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec["acq_datetime_utc"] = row.acq_datetime_utc
        rec["series"] = row.series
        recs.append(rec)
        el = (datetime.now() - t0).total_seconds()
        print(f"[{i}/{len(todo)}] {row.sensor} {str(row.acq_datetime_utc)[:10]} "
              f"{rec['status']:<13} dx={rec.get('dx_m', np.nan):6.1f} "
              f"dy={rec.get('dy_m', np.nan):6.1f} m  n={rec.get('n_chips_used', 0)}  "
              f"({el / i:.0f}s/scene)", flush=True)
        if i % checkpoint_every == 0 or i == len(todo):
            df = pd.DataFrame(recs).reindex(columns=_COLUMNS)
            for p in paths:
                df.to_csv(p, index=False)
    df = pd.DataFrame(recs).reindex(columns=_COLUMNS)
    for p in paths:
        df.to_csv(p, index=False)
    return df


# ===========================================================================
# Apply to shoreline vectors
# ===========================================================================
def apply_shifts(
    gdf: gpd.GeoDataFrame, shifts: pd.DataFrame, ok_status: Tuple[str, ...] = ("ok",)
) -> gpd.GeoDataFrame:
    """Translate each scene's shoreline by its co-registration shift.

    Scenes whose status is not in ``ok_status`` keep their geometry and get
    ``coreg_status`` = that status (they are NOT silently dropped; Phase 4 can
    exclude or down-weight them). Geometry is translated in EPSG:32646 and
    returned in the input CRS.
    """
    s = shifts.set_index("image_id")
    src_crs = gdf.crs
    utm = gdf.to_crs(config.METRIC_CRS).copy()
    dx = utm["image_id"].map(s["dx_m"])
    dy = utm["image_id"].map(s["dy_m"])
    st = utm["image_id"].map(s["status"]).fillna("missing")
    ok = st.isin(ok_status) & dx.notna() & dy.notna()
    utm.loc[ok, "geometry"] = [
        translate(g, xoff=x, yoff=y) if g is not None else None
        for g, x, y in zip(utm.loc[ok, "geometry"], dx[ok], dy[ok])
    ]
    utm["coreg_dx_m"] = dx.where(ok, 0.0)
    utm["coreg_dy_m"] = dy.where(ok, 0.0)
    utm["coreg_se_m"] = np.hypot(utm["image_id"].map(s["se_dx_m"]),
                                 utm["image_id"].map(s["se_dy_m"])).where(ok)
    utm["coreg_status"] = st
    utm["coreg_ref"] = utm["image_id"].map(s["ref_image_id"])
    return utm.to_crs(src_crs)


# ===========================================================================
# Offline self-test (no GEE): known synthetic shifts must be recovered
# ===========================================================================
def selftest(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Recover known synthetic translations; returns per-case errors (m, 30 m px)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        big = ndimage.gaussian_filter(rng.random((200, 200)), rng.uniform(1.0, 3.0))
        big = (big - big.min()) / (big.max() - big.min()) * 0.4
        d = rng.uniform(-2.5, 2.5, 2)                      # content shift (rows, cols)
        sl = (slice(68, 132), slice(68, 132))
        ref = big[sl]
        tgt = ndimage.shift(big, d, order=3)[sl] + rng.normal(0, 0.005, (64, 64))
        valid = np.ones_like(ref, bool)
        water = np.zeros_like(ref, bool)
        est = estimate_chip_shift({"nir": ref, "valid": valid, "water": water},
                                  {"nir": tgt.astype(np.float32), "valid": valid,
                                   "water": water}, 30.0)
        true_dx, true_dy = -d[1] * 30.0, d[0] * 30.0       # correction to apply
        if est is None:
            out.append({"true_dx": true_dx, "true_dy": true_dy, "err_m": np.nan})
        else:
            out.append({"true_dx": true_dx, "true_dy": true_dy,
                        "err_m": math.hypot(est["dx_m"] - true_dx, est["dy_m"] - true_dy)})
    return pd.DataFrame(out)
