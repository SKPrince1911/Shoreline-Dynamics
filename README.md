# Shoreline Dynamics — Cox's Bazar–Teknaf Marine Drive Coast, Bangladesh

A reproducible pipeline for satellite-derived shoreline (SDS) change along the
~92 km sandy/deltaic Cox's Bazar–Teknaf coast on the Bay of Bengal — a
meso-macrotidal (~3.6 m spring range at the Cox's Bazar gauge),
monsoon-dominated, cyclone-affected shoreline, studied over 1988–2025.

All code runs in Google Colab with Google Earth Engine; heavy logic lives in
`src/`, and the phase notebooks are in `notebooks/`.

## Status (2026-11-25)

| Phase | Module / notebook | Status |
|---|---|---|
| 1 Image inventory | `src/data.py`, `notebooks/02_image_inventory.ipynb` | Done |
| 2 Sub-pixel extraction (Series A + B), benchmark | `src/extract.py`, `notebooks/03a_*`, `03b_*` | Done |
| 3a Co-registration | `src/coreg.py`, `notebooks/05_coregistration.ipynb` | Code ready, run pending |
| 3b Tide / beach-slope / setup correction | `src/tides.py` | Not started (stub) |
| 4 Change rates + uncertainty | `src/change.py` | Not started (stub) |
| 5 Driver attribution | `src/drivers.py` | Not started (stub) |

`notebooks/03_shoreline_extraction.ipynb` (superseded by 03a/03b) and
`notebooks/04_coastsat_baseline.ipynb` (abandoned) are kept for history.
The root `Shoreline_Dynamics.ipynb` is an empty placeholder.

Authoritative outputs are under `outputs/shorelines/` and
`outputs/benchmark/`; `outputs/archive/` holds old duplicates only.

See [CLAUDE.md](CLAUDE.md) for locked methods and the decision log, and
[SCOPE.md](SCOPE.md) for the paper scope and milestones.
