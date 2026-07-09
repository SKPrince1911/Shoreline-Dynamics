# Project: Shoreline Dynamics — Cox's Bazar–Teknaf Marine Drive Coast, Bangladesh

## Goal
Q1-publishable, fully reproducible pipeline for satellite-derived shoreline
(SDS) change and driver attribution along the ~92 km sandy/deltaic Cox's
Bazar–Teknaf coast (Bay of Bengal). Meso-tidal (~3 m spring range), monsoon-
dominated, cyclone-affected. Target period: 1988–2025.

## Execution model (IMPORTANT)
- Claude Code (web) only WRITES/edits code in this repo.
- ALL code EXECUTES in Google Colab, where Google Earth Engine (GEE) is
  authenticated. This sandbox cannot run GEE or authenticate to it.
- Heavy logic lives in src/ modules (.py). The root notebook
  Shoreline_Dynamics.ipynb is a THIN driver: installs deps, authenticates GEE,
  imports src/, runs each phase.
- Pin dependencies in requirements.txt for reproducibility.

## Repo structure
- src/config.py  : AOI, CRS, date ranges, constants (single source of truth)
- src/data.py    : image retrieval + quality/cloud masking      (Phase 1)
- src/extract.py : sub-pixel shoreline extraction               (Phase 2)
- src/tides.py   : FES2022 tidal + beach-slope correction       (Phase 3)
- src/change.py  : transects, NSM/EPR/LRR/WLR/LMS, uncertainty  (Phase 4)
- src/drivers.py : wave/cyclone/SLR/sediment attribution        (Phase 5)

## Locked methodology (supersedes any old code)
- Imagery: Sentinel-2 SR Harmonized + Landsat 5/7/8/9 Collection 2 L2.
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

## Conventions
- CRS: EPSG:4326 for storage; EPSG:32646 (UTM 46N) for all metric operations.
- Every function: docstring + type hints. Validate units at stage boundaries.
- Work ONE phase at a time. After each change, tell the user exactly how to
  verify it in Colab before moving on. Explain what changed and why.
- Never delete verified working code or break reproducibility.
