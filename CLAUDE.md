# Project: Shoreline Dynamics — Cox's Bazar–Teknaf Marine Drive Coast, Bangladesh

## Goal
Q1-publishable, fully reproducible pipeline for satellite-derived shoreline
(SDS) change and driver attribution along the ~92 km sandy/deltaic Cox's
Bazar–Teknaf coast (Bay of Bengal). Meso-tidal (~3 m spring range), monsoon-
dominated, cyclone-affected. Target period: 1985–2025 (dry-season-years
1984-1985 … 2024-2025).

## Execution model (IMPORTANT)
- Claude Code (web) only WRITES/edits code in this repo.
- ALL code EXECUTES in Google Colab, where Google Earth Engine (GEE) is
  authenticated. This sandbox cannot run GEE or authenticate to it.
- Heavy logic lives in src/ modules (.py). The root notebook
  Shoreline_Dynamics.ipynb is a THIN driver: installs deps, authenticates GEE,
  imports src/, runs each phase.
- Colab sync: the notebook setup cell is CLONE-OR-PULL (git pull -q if the repo
  folder exists, else clone) so a session always runs the latest committed
  code, and the imports cell calls importlib.reload(config); reload(data) so a
  running kernel picks up the freshly pulled modules without a restart.
- Pin dependencies in requirements.txt for reproducibility.

## Repo structure
- src/config.py  : AOI, CRS, date ranges, constants (single source of truth)
- src/data.py    : image retrieval + quality/cloud masking      (Phase 1)
- src/extract.py : sub-pixel shoreline extraction               (Phase 2)
- src/tides.py   : FES2022 tidal + beach-slope correction       (Phase 3)
- src/change.py  : transects, NSM/EPR/LRR/WLR/LMS, uncertainty  (Phase 4)
- src/drivers.py : wave/cyclone/SLR/sediment attribution        (Phase 5)

## Locked methodology (supersedes any old code)
- Imagery: Sentinel-2 SR Harmonized + Landsat 4/5/7/8/9 Collection 2 L2.
  Landsat 4 (TM) shares Landsat 5's exact band layout, QA_PIXEL masking, and
  DN×0.0000275−0.2 scaling, and sits alongside L5 in the sensor tie-break.
  Rationale: TM (Landsat 4/5) is the first SWIR-capable 30 m sensor (Aug 1982),
  but Landsat 5 only became operational in March 1984, so 1985 is the first
  dry-season-year with a realistic chance of complete coverage using identical
  MNDWI/sub-pixel logic; earlier years are expected to be sparse.
- Masking: Cloud Score+ for S2 (cs ~0.55); QA_PIXEL/CFMask for Landsat.
- Composites: dry-season (Nov–Mar) annual medians for the trend layer; single
  tidally-corrected images for cyclone/monsoon event analysis.
- Extraction: per-SCENE (D1) supervised LOCAL sand/water/whitewater/other
  classifier (D3) + marching-squares sub-pixel contour on the water-index
  interface (CoastSat/CoastSeg family); the water index is a benchmarked
  PARAMETER, default MNDWI (D4). NOT reduceToVectors / pixel thresholding.
  See "## Phase 2 — sub-pixel shoreline extraction (locked: D1–D7)" below,
  which supersedes this line where they differ.
- Tidal correction: FES2022 via pyTMD/pyfes; slope via CoastSat.slope
  (Lomb-Scargle). Horizontal shift X = (Z_tide - Z_MSL) / tan(beta).
- Uncertainty: root-sum-of-squares per-shoreline budget (pixel, georef, tidal,
  seasonal, digitizing) -> EPR_unc = sqrt(U1^2 + U2^2)/T; WLR weights = 1/U^2.
- Rates: NSM, EPR, LRR, WLR, LMS + Mann-Kendall/Sen + BEAST breakpoints.
- Attribution: ERA5 wave power/direction, IBTrACS cyclones, altimetry/gauge
  relative SLR, GBM sediment flux; Kamphuis longshore transport; Random Forest
  importance, GAM, wavelet coherence. Pre/post-monsoon & pre/post-cyclone windows.

## Phase 2 — sub-pixel shoreline extraction (locked: D1–D7)
These decisions are LOCKED (argued with citations in PHASE2_SPEC.md, the
authoritative Phase 2 brief at the repo root) and supersede any older Phase-2
statement above. They do not alter the verified
Phase 1 inventory (18 single / 19 composite / 1 partial / 3 gap).

- **D1 — the extraction unit is the SCENE, never the dry-season-year.**
  Shorelines are extracted per scene and merged as VECTORS; every output
  segment retains its own `image_id` and `acq_datetime_utc`. Rationale: 19 of
  the 38 year-products are multi-date set-cover mosaics — extracting from a
  merged raster would destroy the scene→tide mapping and make those years
  uncorrectable in Phase 3 (FES2022 tidal correction is per acquisition time).
- **D2 — two shoreline series.**
  - Series A = the 38 annual dry-season products (the trend layer).
  - Series B = a dense, all-season series, 1999–2025, from L7/L8/L9/S2. It is
    required for CoastSat.slope, which needs sub-monthly sampling: a 365-day
    annual series aliases the tidal band entirely, so beach-slope estimation is
    impossible from Series A alone.
- **D3 — surface reflectance + a LOCAL classifier.**
  Keep SR imagery (`S2_SR_HARMONIZED` + Landsat C2 L2). Train a local
  sand/water/whitewater/other classifier in THREE sensor groups by band layout:
  TM (L4/L5/L7 — ETM+ shares the TM band set), OLI (L8/L9), MSI (S2).
  CoastSat's shipped classifiers are TOA-trained and are INVALID on SR imagery.
- **D4 — the water index is a PARAMETER, not hardcoded.**
  Choice is `{mndwi, ndwi, aweinsh, scowi} × {otsu, weighted_peaks}`,
  benchmarked against manually digitised reference shorelines. Default `mndwi`
  pending the benchmark. (AWEInsh needs SWIR2, so `swir2` is carried on every
  fetched scene — see config BAND_MAP.)
- **D5 — per-scene georeferencing RMSE is read from image metadata**, not
  assumed constant, and feeds the per-shoreline uncertainty budget.
- **D6 — inter-sensor bias quantification is a Phase 2 deliverable** (e.g. at
  sensor-overlap dates on pixel-aligned grids), so cross-sensor offsets are
  characterised before rates are computed.
- **D7 — reported study period is 1988–2025 (38 dry-season-years).**
  1985–87 is a Landsat archive gap (no usable imagery over the AOI). 1991
  remains `partial` and carries an INFLATED uncertainty flag.
- **Composite temporal spread (E_temporal precursor).** A dry-season composite
  whose contributing acquisitions span a wide window mixes shoreline positions
  from different parts of the season (e.g. 1995 = 1994-11-20 + 1995-03-19, 119
  days; also 2005, 2010, 2011, 2014 > 60 days). Per-scene extraction (D1) fixes
  the tide problem but not this temporal one. The scene list, per-scene shoreline
  records, and `sds_annual_merged.geojson` carry `composite_date_spread_days`
  (max−min of the product's contributing dates; 0 for single-date products);
  years above `config.COMPOSITE_SPREAD_FLAG_DAYS` (60) are flagged
  (`composite_spread_gt_60d`). Phase 4 turns this into an `E_temporal` RSS term
  (≈ local change rate × spread/2); Phase 2 only records the field and flag.

### Phase 2 inputs — operator-digitised in QGIS (manual-artifact pathway)
Uploaded by the operator via the GitHub web interface (NOT written by
`save_outputs()`). Extraction code reads them; exact schemas are defined with
the code that consumes them:
- `data/shoreline_search_zone.geojson` — buffer/search zone constraining
  extraction to the coast (suppresses false detections inland and offshore).
- `data/training_polygons.geojson` — labelled polygons, `class ∈ {other, sand,
  whitewater, water}`, per D3's three sensor groups.
- `data/reference_shorelines/` — 10 manually digitised validation shorelines
  for the D4 water-index/threshold benchmark.
- `data/baseline.geojson` — a single smooth LineString landward of and roughly
  parallel to the coast; `extract.build_transects()` casts shore-normal DSAS
  transects from it (default 50 m spacing, 1500 m long) to
  `outputs/transects.geojson` for the benchmark and inter-sensor bias (D6).

## Temporal design (locked)
- Annual shoreline is derived from the DRY SEASON ONLY (Nov–Mar); monsoon
  months are excluded from the trend layer. This annual layer is Series A (D2);
  Phase 2 additionally builds Series B — a dense, all-season 1999–2025 series
  needed for tidal-slope estimation (see D2 below).
- One BEST (clearest) image per dry-season-year — a single scene, not a
  composite — is selected for the annual shoreline. (This is the Phase 1
  *product selection*. For Phase 2 *extraction*, the unit is the SCENE, not the
  year: each contributing scene of a multi-date product is extracted separately
  and the results merged as vectors — see D1 below.)
- Cloud threshold: ≤10% cloud cover computed OVER THE AOI (per-pixel cloud
  mask reduced over the AOI polygon), NOT scene-wide metadata such as
  CLOUDY_PIXEL_PERCENTAGE / CLOUD_COVER.
- Dry-season-year labelling: a scene acquired in Nov or Dec belongs to the NEXT
  calendar year; a scene acquired in Jan–Mar belongs to the SAME calendar year.
  (Dry-season-year Y spans (Y-1)-11-01 to Y-03-31.)
- All candidate scenes are logged (sensor, date, AOI cloud %); dry-season-years
  with no ≤10% image are flagged as GAPS.
- Both Sentinel-2 (10 m) and Landsat (30 m) are used; the sensor of the chosen
  image is tracked per year to feed the per-sensor uncertainty budget.
- Landsat 7 SLC-off (after 2003-05) scenes are flagged and used ONLY to fill
  gaps (years with no other ≤10% image).

## Naming convention (locked)
- `dry_year` (int) is the CANONICAL key for every sort, join, and later rate
  computation; it never changes form.
- `season_label` (str) is the human-readable DISPLAY field:
  `f"{dry_year-1}-{dry_year}"` (e.g. 1995 -> "1994-1995"), reflecting that the
  dry season spans Nov of the prior calendar year through Mar of the labelled
  year.
- Both `dry_year` and `season_label` are present on EVERY feature, dict, and
  DataFrame that carries a dry-season-year (candidates, ranked candidates,
  selection, product recommendation, approvals CSV, final inventory).

## Product hierarchy (locked)
- Per dry-season-year the recommended product is chosen in order:
  1. `single` — the top coverage-first candidate that is full-coast
     (`is_complete`, coverage ≥ COVERAGE_COMPLETE_PCT).
  2. `composite` — otherwise the greedy minimal gap-fill composite (fewest
     dates via set-cover) if it reaches COVERAGE_COMPLETE_PCT.
  3. `partial` — the same composite when it cannot reach full coverage (the
     highest-coverage product achieved is kept).
  4. `none` — no usable imagery at all (achieved_coverage_pct = 0).
- Every year is HUMAN-REVIEWED and the decision is RECORDED: approvals are
  appended to a persistent CSV (Drive when mounted, else outputs/) written on
  every call, with `review_status` in
  {approved, approved_override, rejected, auto}. `approve_single` and
  `force_composite` are recorded overrides; `reject` marks a year `none`.
- The final inventory (build_inventory) uses the approved row per year when one
  exists, else the automatic `year_product`, spanning all of 1985–2025.

## Output persistence (locked)
- Colab-generated outputs (image_inventory.csv, the approvals CSV) are saved to
  Google Drive AND pushed to GitHub automatically via `save_outputs()`, which
  authenticates with a `GITHUB_TOKEN` stored in Colab Secrets (never printed or
  written to a file). No manual upload step for generated outputs.
- Only hand-made artifacts (e.g. QGIS files) are uploaded to the repo manually.

## Conventions
- CRS: EPSG:4326 for storage; EPSG:32646 (UTM 46N) for all metric operations.
- Every function: docstring + type hints. Validate units at stage boundaries.
- Work ONE phase at a time. After each change, tell the user exactly how to
  verify it in Colab before moving on. Explain what changed and why.
- Never delete verified working code or break reproducibility.
