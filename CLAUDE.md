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
- Extraction: supervised sand/water/whitewater classifier + marching-squares
  sub-pixel contour on the MNDWI interface (CoastSat/CoastSeg family).
  NOT reduceToVectors / pixel thresholding.
- Tidal correction: FES2022 via pyTMD/pyfes; slope via CoastSat.slope
  (Lomb-Scargle). Horizontal shift X = (Z_tide - Z_MSL) / tan(beta).
- Uncertainty: root-sum-of-squares per-shoreline budget (pixel, georef, tidal,
  seasonal, digitizing) -> EPR_unc = sqrt(U1^2 + U2^2)/T; WLR weights = 1/U^2.
- Rates: NSM, EPR, LRR, WLR, LMS + Mann-Kendall/Sen + BEAST breakpoints.
- Attribution: ERA5 wave power/direction, IBTrACS cyclones, altimetry/gauge
  relative SLR, GBM sediment flux; Kamphuis longshore transport; Random Forest
  importance, GAM, wavelet coherence. Pre/post-monsoon & pre/post-cyclone windows.

## Temporal design (locked)
- Annual shoreline is derived from the DRY SEASON ONLY (Nov–Mar); monsoon
  months are excluded from the trend layer.
- One BEST (clearest) image per dry-season-year — a single scene, not a
  composite — is selected for the annual shoreline.
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

## Conventions
- CRS: EPSG:4326 for storage; EPSG:32646 (UTM 46N) for all metric operations.
- Every function: docstring + type hints. Validate units at stage boundaries.
- Work ONE phase at a time. After each change, tell the user exactly how to
  verify it in Colab before moving on. Explain what changed and why.
- Never delete verified working code or break reproducibility.
