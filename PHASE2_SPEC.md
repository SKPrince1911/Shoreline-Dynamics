# Phase 2 — Sub-Pixel Shoreline Extraction: Implementation Brief

Repo: `SKPrince1911/Shoreline-Dynamics` · Target module: `src/extract.py`
Status: design locked, ready to author. Execution: Google Colab + GEE.

---

## 0. Decisions locked in this phase (amend `CLAUDE.md`)

| # | Decision | Rationale (citable) |
|---|---|---|
| D1 | **The extraction unit is the SCENE, never the dry-season-year.** Shorelines are extracted per scene and merged as *vectors*, each segment carrying its own `image_id` and `acq_datetime_utc`. | 19 of 38 year-products in `outputs/image_inventory.csv` are multi-date set-cover mosaics. Tidal correction is X = (Z_tide − Z_MSL)/tanβ, evaluated at the acquisition instant; a merged raster destroys the scene→tide mapping and would make those 19 years uncorrectable in Phase 3. |
| D2 | **Two shoreline series are produced.** Series A = 38 annual dry-season products (trend layer). Series B = dense, all-season, 1999–2025, L7/L8/L9/S2 (slope + event layer). | CoastSat.slope isolates the tidal signal by frequency-domain analysis and requires sub-monthly sampling (reference config: `n_days = 8`, date range 1999–2020, chosen so two Landsats are simultaneously in orbit → 8-day combined revisit). A 365-day annual series aliases the tidal band completely. Vos et al. (2020), *GRL*, doi:10.1029/2020GL088365. |
| D3 | **Surface reflectance (SR) is retained; a LOCAL sand/water/whitewater/other classifier is trained on this coast.** CoastSat's shipped classifiers are invalid on SR inputs (they were trained on Landsat Tier-1 TOA and Sentinel-2 L1C). | Locally trained classifiers reduced CoastSat RMSE from 9–21 m to 7–12 m — the largest single gain of any tuning step tested. Billet et al. (2026), *Coasts*. Their explicit guideline: use local classifiers on heterogeneous beaches. Cox's Bazar–Teknaf qualifies (turbid GBM water, dark wet sand, Himchari cliff section, dissipative surf, Marine Drive hard structures). |
| D4 | **The water index is a parameter, not a constant.** Default MNDWI (Xu 2006), but MNDWI / NDWI / AWEInsh / SCoWI × {Otsu, weighted-peaks} are benchmarked against manually digitised reference shorelines. | No generic global index exists: on a **mesotidal dissipative** beach AWEInsh+erosion filter won (RMSE 5.96 m) with MNDWI+Otsu close behind (Cabezas-Rabadán et al. 2024, *Remote Sensing* 16, doi:10.3390/rs16142594). In **Bangladesh's own GBM sediment-laden setting** (Sandwip Is.) NDWI beat MNDWI, AWEIsh/nsh and MBWI (Langat et al. 2024, *Remote Sensing*). Lake-derived indices are contaminated by wave breaking, motivating coastal-specific indices (Bergsma et al. 2024, *Remote Sensing* 16(16):2795). |
| D5 | **Per-scene georeferencing accuracy is recorded, not assumed constant.** | CoastSat defaults to a 12 m Landsat collection RMSE; USGS publishes per-scene `GEOMETRIC_RMSE_MODEL` in Landsat metadata precisely so products can be filtered for time-series suitability. This becomes E_georef in the Phase 4 RSS budget. |
| D6 | **Inter-sensor bias is a Phase 2 deliverable.** | The record is 21 yr of 30 m TM → 15/30 m ETM+/OLI → 10 m MSI. SDS error is demonstrably sensor-dependent (CoastSat at macrotidal Truc Vert: 38.3 m S2 vs 27.9 m Landsat; Vos et al. 2023, *Comm. Earth Environ.* 4:345). An uncorrected sensor offset masquerades as a long-term trend. |
| D7 | **Reported study period becomes 1988–2025 (38 dry-season-years).** 1985–87 are a Landsat archive gap over the AOI and are reported as such in the data-availability section. 1991 is retained as `partial` with an inflated uncertainty flag (92.6% coverage, 23.9% AOI cloud — a review override). | |

---

## 1. `src/config.py` — additions

```python
# ---------------------------------------------------------------------------
# Phase 2 — extraction
# ---------------------------------------------------------------------------
REFLECTANCE_LEVEL: str = "SR"  # locked: Landsat C2 L2 + S2_SR_HARMONIZED

# SR band aliases per sensor. Keys are the canonical names used everywhere
# downstream; values are the native GEE band names.
BAND_MAP: Dict[str, Dict[str, str]] = {
    "L4": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5"},
    "L5": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5"},
    "L7": {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5"},
    "L8": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6"},
    "L9": {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6"},
    "S2": {"blue": "B2",    "green": "B3",    "red": "B4",    "nir": "B8",    "swir1": "B11"},
}

# Reflectance scaling to physical [0,1]. Landsat C2 L2: DN*2.75e-5 - 0.2.
# S2_SR_HARMONIZED: DN/10000 (the harmonized collection already removes the
# post-2022-01-25 baseline-04.00 +1000 offset — this is why HARMONIZED is used).
SR_SCALE: Dict[str, tuple] = {
    "LANDSAT": (2.75e-5, -0.2),
    "S2": (1e-4, 0.0),
}

# Native analysis grid (m) per sensor. Pansharpening is OFF by default: the
# Landsat pan band (B8) exists only in the L1 TOA collections, so pansharpening
# SR multispectral requires a cross-product fetch. Evaluated as an option in the
# benchmark (Section 6), not assumed.  <-- TUNABLE
PIXEL_SIZE_M: Dict[str, int] = {"L4": 30, "L5": 30, "L7": 30, "L8": 30, "L9": 30, "S2": 10}
PANSHARPEN: bool = False  # <-- TUNABLE

# Fallback georeferencing RMSE (m) when per-scene metadata is absent.
# CoastSat uses 12 m as the Landsat-collection average.  <-- TUNABLE
GEOREF_RMSE_DEFAULT_M: Dict[str, float] = {"LANDSAT": 12.0, "S2": 11.0}

# Pixel classes (classifier output codes)
CLASS_OTHER, CLASS_SAND, CLASS_WHITEWATER, CLASS_WATER = 0, 1, 2, 3

# Water indices available to the benchmark.  <-- TUNABLE
WATER_INDICES: List[str] = ["mndwi", "ndwi", "aweinsh", "scowi"]
WATER_INDEX_DEFAULT: str = "mndwi"
THRESHOLD_METHODS: List[str] = ["otsu", "weighted_peaks"]
THRESHOLD_METHOD_DEFAULT: str = "otsu"

# Contour filtering.  <-- TUNABLE
MIN_SHORELINE_LENGTH_M: float = 2000.0   # CoastSat default is 500 m; this coast is ~90 km
SEARCH_ZONE_PATH: str = "data/shoreline_search_zone.geojson"  # QGIS-digitised (Section 4)

# Series B (dense) query envelope.  <-- TUNABLE
DENSE_START: str = "1999-01-01"
DENSE_END: str = "2025-12-31"
DENSE_SENSORS: List[str] = ["L7", "L8", "L9", "S2"]
DENSE_CLOUD_MAX_PCT: float = 30.0      # relaxed vs the 10% annual rule
DENSE_COVERAGE_MIN_PCT: float = 50.0   # partial scenes are useful for slope

# Landsat CFMask misflags bright beach/whitewater as cloud (CoastSat exposes
# `cloud_mask_issue` for exactly this). When True, pixels flagged cloud but
# classified sand/whitewater with high confidence are NOT masked.  <-- TUNABLE
LANDSAT_CLOUD_MASK_ISSUE: bool = True
```

---

## 2. `src/extract.py` — function contracts

All functions typed + docstringed. GEE is imported lazily inside functions (config stays `ee`-free).

### 2.1 Scene lists

```python
def build_scene_list_annual(inventory_csv: str = "outputs/image_inventory.csv") -> pd.DataFrame
```
Explodes the semicolon-delimited `image_ids` / `dates` / `timestamps_utc` columns into **one row per scene**. Preserves `dry_year`, `season_label`, `product_type`, `review_status`. Adds `series="A"`. Expect ~62 scenes from 38 year-products.

```python
def build_scene_list_dense(start=DENSE_START, end=DENSE_END, sensors=DENSE_SENSORS,
                           cloud_max_pct=DENSE_CLOUD_MAX_PCT,
                           coverage_min_pct=DENSE_COVERAGE_MIN_PCT) -> pd.DataFrame
```
GEE query over **all months** (monsoon included). Reuses the AOI-reduced cloud/coverage reducers already in `src/data.py` — do not re-implement, and do **not** use scene-wide `CLOUDY_PIXEL_PERCENTAGE`/`CLOUD_COVER`. Adds `series="B"`. Writes `outputs/scene_list_dense.csv` and pushes via the existing `save_outputs()`. Expect several hundred scenes.

Both return the same schema, so `extract_all()` is series-agnostic:
`image_id, sensor, acq_datetime_utc, dry_year, season_label, series, aoi_cloud_pct, aoi_coverage_pct, slc_off, georef_rmse_m`

### 2.2 Imagery

```python
@dataclass
class Scene:
    image_id: str; sensor: str; acq_datetime_utc: datetime
    bands: dict[str, np.ndarray]   # blue, green, red, nir, swir1 — scaled reflectance
    valid: np.ndarray              # bool: not cloud, not shadow, not nodata
    transform: Affine              # EPSG:32646
    pixel_size_m: int; georef_rmse_m: float

def fetch_scene(row: pd.Series) -> Scene
```
- Reprojects to **EPSG:32646** on a fixed grid so all scenes are pixel-aligned (essential for D6 and for the benchmark).
- Applies `SR_SCALE`. S2 SWIR (B11, 20 m) bilinearly resampled to 10 m.
- Mask: S2 → Cloud Score+ `cs < CS_THRESHOLD` (0.55). Landsat → `QA_PIXEL` bits 1 (dilated), 2 (cirrus), 3 (cloud), 4 (shadow) + `QA_RADSAT`; honour `LANDSAT_CLOUD_MASK_ISSUE`.
- Reads `GEOMETRIC_RMSE_MODEL` (Landsat) / geometric-quality flag (S2); falls back to `GEOREF_RMSE_DEFAULT_M`.
- Transport: `geemap.ee_to_numpy` or `ee.Image.getDownloadURL` on the AOI bounding box. **Tile the fetch** — the AOI is ~90 km long; a single request will exceed the GEE download limit.

### 2.3 Classifier (D3)

```python
def build_training_set(scenes: list[Scene], labels_path: str) -> tuple[np.ndarray, np.ndarray]
def train_classifier(X, y, sensor_group: str, version: str) -> MLPClassifier
def classify_scene(scene: Scene, clf) -> np.ndarray   # int array in {0,1,2,3}
```
- **Feature vector per pixel (CoastSat convention):** the 5 scaled reflectances + MNDWI + NDWI + NDVI + a 3×3 local standard deviation of each of the 5 bands (texture separates whitewater from sand). 20 features.
- **Model:** `sklearn.neural_network.MLPClassifier`, hidden `(100, 50)`, `max_iter=500`. Persist with joblib to `models/clf_{sensor_group}_{version}.joblib`.
- **Sensor groups:** `TM` (L4/L5/L7), `OLI` (L8/L9), `MSI` (S2) — three classifiers. TM and ETM+ share band layout; OLI and MSI do not.
- **Training labels:** digitise polygons in QGIS on ~6 scenes per sensor group (spread across dry/monsoon and across the AOI: Bakkhali mouth, Kolatoli/Marine Drive, Himchari cliffs, Inani, Teknaf, Shah Porir Dwip) → `data/training_polygons.geojson` with a `class` field in {other, sand, whitewater, water}. Aim ≥5,000 labelled pixels per class per sensor group.
- Report per-class precision/recall on a held-out scene. This table goes in the paper.

### 2.4 Index, threshold, sub-pixel contour

```python
def water_index(scene: Scene, name: str) -> np.ndarray
```
- `mndwi  = (green − swir1) / (green + swir1)`  (Xu 2006)
- `ndwi   = (green − nir) / (green + nir)`  (McFeeters 1996)
- `aweinsh = 4·(green − swir1) − (0.25·nir + 2.75·swir2)` — **note:** SWIR2 is not in `BAND_MAP`; either add `swir2` (SR_B7 / B12) or use the no-SWIR2 variant and say so. Add it — it is one extra band and AWEInsh is a live candidate.
- `scowi` — per Bergsma et al. (2024), doi:10.3390/rs16162795. Implement from the paper's definition; keep it behind a feature flag until verified.

```python
def interface_threshold(index: np.ndarray, labels: np.ndarray,
                        method: str = "otsu") -> float
```
**This is the sub-pixel step that separates you from the old pipeline.** The threshold is computed on the index values of **sand ∪ water pixels only** (from `classify_scene`), not on the full scene histogram. That is what suppresses the whitewater / wet-sand seaward bias of plain full-scene MNDWI+Otsu. `weighted_peaks` (Doherty et al. 2022) is the alternative and sits higher in the swash — carry both through the benchmark; recent lidar validation found Otsu waterlines track the lidar waterline better while weighted-peaks track a runup bulk statistic (Brown et al. 2025, *Coastal Engineering*).

```python
def extract_contour(index, threshold, valid, transform) -> list[LineString]
```
`skimage.measure.find_contours(index, threshold)` → **sub-pixel** vertex coordinates in pixel space → apply the affine transform to EPSG:32646. Contours touching invalid/masked pixels are broken, not bridged.

### 2.5 Filtering

```python
def filter_contours(contours, search_zone, channel_lines,
                    min_length_m=MIN_SHORELINE_LENGTH_M) -> LineString | MultiLineString
```
1. Clip to the **search zone** (Section 4) — replaces CoastSat's `max_dist_ref` scalar.
2. Cut at the 13 `data/tidal_channels.geojson` closure lines; discard segments landward of them.
3. Drop segments shorter than `min_length_m`.
4. Order vertices alongshore (north→south) and return a single geometry.

### 2.6 Drivers

```python
def extract_shoreline(scene: Scene, clf, settings: dict) -> dict     # one record
def extract_all(scene_list: pd.DataFrame, settings: dict) -> gpd.GeoDataFrame
```
Checkpoint after every scene (append to a Parquet/GeoJSON on Drive) — a Colab disconnect mid-run over 600 dense scenes must not cost the whole run.

---

## 3. Output schema (locked)

`outputs/shorelines/sds_scenes.geojson` — **one LineString per scene**:

```
image_id, sensor, sensor_group, acq_datetime_utc, dry_year, season_label, series,
pixel_size_m, georef_rmse_m, aoi_cloud_pct, aoi_coverage_pct, slc_off,
water_index, threshold_method, threshold_value, classifier_version,
length_m, n_vertices, pct_aoi_alongshore_covered, flags
```

`outputs/shorelines/sds_annual_merged.geojson` — Series A merged per `dry_year`. Segments **retain `source_image_id` and `source_acq_datetime_utc`** (D1). This is the file Phase 3 consumes.

`outputs/extraction_log.csv` — one row per scene attempted, incl. failures and reason.
`models/clf_{TM,OLI,MSI}_{version}.joblib` + `models/classifier_report.md`.

---

## 4. Manual QGIS artifacts required before the first run

Upload via the GitHub web interface (per the repo's static-data convention):

1. **`data/shoreline_search_zone.geojson`** — a polygon enclosing every plausible shoreline position 1988–2025. Roughly ±300 m about the present shoreline along the open Marine Drive coast, widening to ±800–1000 m within ~3 km of the Bakkhali and Naf mouths, where deltaic change is large. This replaces a scalar `max_dist_ref` and keeps the decision human-controlled and auditable.
2. **`data/training_polygons.geojson`** — classifier labels (Section 2.3).
3. **`data/reference_shorelines/`** — 10 manually digitised validation shorelines (Section 6). Digitise at native resolution on the scene itself, zoomed to ~1:2,000, along the wet/dry line.

---

## 5. Verification in Colab (run in this order)

1. `build_scene_list_annual()` → assert 38 dry-years present, ~62 scenes, no NaT timestamps.
2. `build_scene_list_dense()` → print scenes/year; sanity-check the 2003+ L7 SLC-off share and the S2 ramp from 2016.
3. `fetch_scene()` on one L5, one L8, one S2 → plot RGB + mask; confirm reflectance in [0,1] and that the AOI is fully tiled.
4. Train `clf_MSI` → per-class precision/recall on a held-out S2 scene. **Gate: ≥0.90 for water and sand.** If not, add training polygons before proceeding.
5. `extract_shoreline()` on that S2 scene → overlay on RGB. Visually: no contour on the Marine Drive road edge, none inside tidal channels, none along the cloud mask boundary.
6. Only then run `extract_all()` on Series A, then Series B.

---

## 6. Benchmark + validation (a Phase 2 deliverable, not an afterthought)

```python
def benchmark_extraction(scenes, reference_shorelines, transects) -> pd.DataFrame
```
Grid: `index ∈ {mndwi, ndwi, aweinsh, scowi}` × `threshold ∈ {otsu, weighted_peaks}` × `classifier ∈ {local, none}` × `pansharpen ∈ {on, off}` (Landsat only), on the 10 manually digitised scenes. Metric: transect-normal offset → **RMSE, mean bias, MAE**, reported per sensor group. Selects the operational configuration and produces a paper figure.

```python
def intersensor_bias(gdf, max_days=2) -> pd.DataFrame
```
Near-coincident pairs already present in your data: **L5+L7 in 2001 (2000-11-11 / 2000-11-12) and 2005**, plus the 2016–2021 L8/L9 × S2 overlap. Compute the mean transect-normal offset per sensor pair. If significant, apply a constant correction and report it — this closes the most obvious attack on a 40-year multi-sensor trend (D6).

**Decision gate (from your own methodology doc):** if post-extraction RMSE against the manual references stays **> 20 m**, the limiting factor is tides, not extraction — stop tuning and go straight to Phase 3.

---

## 7. `requirements.txt` — additions

```
scikit-learn
joblib
scipy
pyproj
```
(`scikit-image`, `geopandas`, `shapely`, `rasterio`, `geemap` already present.)

---

## References

- Vos, K., Splinter, K.D., Harley, M.D., Simmons, J.A., Turner, I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. *Environmental Modelling & Software* 122:104528. doi:10.1016/j.envsoft.2019.104528
- Vos, K., Harley, M.D., Splinter, K.D., Walker, A., Turner, I.L. (2020). Beach slopes from satellite-derived shorelines. *Geophysical Research Letters* 47:e2020GL088365. doi:10.1029/2020GL088365
- Vos, K. et al. (2023). Benchmarking satellite-derived shoreline mapping algorithms. *Communications Earth & Environment* 4:345. doi:10.1038/s43247-023-01001-2
- Billet, C. et al. (2026). Regional validation of satellite-derived beach width and slope in microtidal environments: the role of water level forcing and classifier training. *Coasts*.
- Cabezas-Rabadán, C. et al. (2024). Assessing satellite-derived shoreline detection on a mesotidal dissipative beach. *Remote Sensing* 16.
- Langat, P. et al. (2024). Mapping coastal dynamics induced land use change in Sandwip Island, Bangladesh. *Remote Sensing* 16.
- Bergsma, E. et al. (2024). Shoreliner: a sub-pixel coastal waterline extraction pipeline for multi-spectral satellite optical imagery. *Remote Sensing* 16(16):2795. doi:10.3390/rs16162795
- Brown, S. et al. (2025). Assessing shorelines extracted from satellite imagery using coincident terrestrial lidar linescans. *Coastal Engineering*.
- Doherty, Y. et al. (2022). CoastSat.PlanetScope. *Environmental Modelling & Software* 157:105512.
- Xu, H. (2006). Modification of normalised difference water index (NDWI). *Int. J. Remote Sensing* 27(14):3025–3033.
- Himmelstoss, E.A. et al. (2018/2021). DSAS v5.1 user guide. USGS Open-File Report 2018-1179.
