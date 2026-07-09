# Shoreline Dynamics — Cox's Bazar–Teknaf Marine Drive Coast, Bangladesh

A Q1-publishable, fully reproducible pipeline for satellite-derived shoreline
(SDS) change detection and driver attribution along the ~92 km sandy/deltaic
Cox's Bazar–Teknaf coast on the Bay of Bengal — a meso-tidal (~3 m spring
range), monsoon-dominated, cyclone-affected shoreline studied over 1988–2025.
Heavy logic lives in the `src/` modules (image retrieval and cloud masking,
sub-pixel shoreline extraction, FES2022 tidal and beach-slope correction,
transect-based change rates with an explicit uncertainty budget, and
wave/cyclone/sea-level/sediment attribution), while the root
`Shoreline_Dynamics.ipynb` is a thin driver that installs dependencies,
authenticates Google Earth Engine, and runs each phase in Google Colab. See
[CLAUDE.md](CLAUDE.md) for the full goal, execution model, locked methodology,
and conventions.
