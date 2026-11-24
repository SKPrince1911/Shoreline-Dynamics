# SCOPE — locked 2026-11-25

Two papers. Paper 1 first; nothing from Paper 2 is started until Paper 1 is submitted.

## Paper 1

Working title: *A water-level-corrected, 38-year satellite shoreline record for a
monsoon-dominated meso-macrotidal coast: Cox's Bazar–Teknaf, Bangladesh*

Targets: Coastal Engineering or Geomorphology (fallback: Estuarine, Coastal and
Shelf Science; Regional Studies in Marine Science).

- **RQ1** How accurately can sub-pixel, tide- and setup-corrected shorelines be
  extracted on this coast (TM / OLI / MSI), against independent data?
- **RQ2** What are the long-term rates, monsoon seasonal amplitude and trend
  breaks (1988–2025), and where are they detectable above uncertainty?
- **RQ3** How are these changes associated with waves, cyclones and coastal works
  (Marine Drive, geobags)?

Figures: 1 study area · 2 workflow · 3 tide-model validation · 4 beach-slope map ·
5 accuracy / ablation · 6 rates + uncertainty map · 7 seasonality & BEAST breaks ·
8 driver associations

**NOT in Paper 1:** forecasting, LULC/exposure, household survey, MoST hotspot
figures, super-resolution, deep learning, MSS imagery.

### Validation plan (no RTK/DGPS/drone available)
- BIWTA Cox's Bazar hourly tide-gauge data → tide-model validation.
- Field: GNSS-tracked waterline walks timed to Sentinel-2 / Landsat overpasses
  (dry season 2026-27), plus levelled beach profiles if equipment can be borrowed.
- Benchmark RMSE reported separately for training vs independent reference scenes;
  add independent OLI/MSI reference scenes and L5 2009 training labels.

### Milestones (review every Sunday)
| Week | Date (approx.) | Milestone |
|---|---|---|
| 1 | 2026-12-01 | Scope locked, housekeeping, BIWTA data request sent |
| 3 | 2026-12-15 | Co-registration done |
| 7 | 2027-01-12 | Tide model validated; slope + MSL correction done |
| 9 | 2027-01-26 | Validation (field + independent scenes) done |
| 12 | 2027-02-16 | Change statistics + uncertainty done |
| 15 | 2027-03-09 | Driver analysis done |
| 20 | 2027-04-13 | Full draft |
| 22 | 2027-04-27 | Submit |

## Paper 2 (after Paper 1 submission)
Hindcast-tested shoreline forecasts (equilibrium + ensemble Kalman, ML, ensemble)
to 2040/2050 and LULC exposure of coastal assets. Target: Ocean & Coastal
Management. Delivers the MoST 2026–27 commitments.

## Side-task rule
Accept a side task (MoST, BIMRAD, viva, magazine) only if it REUSES Paper 1
outputs. Never start new analysis for it.
