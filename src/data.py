"""Phase 1: image retrieval and quality/cloud masking (image inventory).

Builds the per-year candidate-scene inventory for the dry-season shoreline
pipeline. For each dry-season-year it retrieves Sentinel-2 SR Harmonized and
Landsat 5/7/8/9 Collection 2 L2 imagery, computes an AOI-based cloud percentage
(Cloud Score+ for Sentinel-2, QA_PIXEL bit flags for Landsat) rather than
relying on scene-wide metadata, and returns a tidy table of candidates from
which the single clearest image per year is later chosen.

Execution model: this module uses the Google Earth Engine Python API. It assumes
``ee.Initialize(...)`` has ALREADY been called by the caller (e.g. the Colab
driver notebook); it deliberately does NOT authenticate or initialize here.
"""

import ee

from functools import reduce
from typing import Dict, List, Tuple

from . import config

# ---------------------------------------------------------------------------
# Sensor operational windows, expressed as inclusive dry-season-year ranges.
# A sensor is a candidate for dry-season-year Y only if first <= Y <= last.
# Extend the ``last`` values as new acquisitions become available.  <-- TUNABLE
# ---------------------------------------------------------------------------
_PRESENT_YEAR: int = 9999  # Sentinel for "still acquiring".

_SENSOR_OPERATIONAL: Dict[str, Tuple[int, int]] = {
    "L5": (1984, 2012),          # Landsat 5 TM.
    "L7": (1999, 2022),          # Landsat 7 ETM+ (SLC-off after 2003-05).
    "L8": (2013, _PRESENT_YEAR),  # Landsat 8 OLI/TIRS.
    "L9": (2022, _PRESENT_YEAR),  # Landsat 9 OLI-2/TIRS-2 (first light 2021-10).
    "S2": (2017, _PRESENT_YEAR),  # Sentinel-2 SR Harmonized.
}

# Map each Landsat sensor label to its Collection 2 L2 collection ID.
_LANDSAT_COLLECTIONS: Dict[str, str] = {
    "L5": config.LANDSAT5_C2_L2,
    "L7": config.LANDSAT7_C2_L2,
    "L8": config.LANDSAT8_C2_L2,
    "L9": config.LANDSAT9_C2_L2,
}

# Landsat Collection 2 QA_PIXEL bit positions used for cloud masking.
_QA_DILATED_CLOUD_BIT: int = 1  # Dilated cloud.
_QA_CLOUD_BIT: int = 3          # Cloud.
_QA_CLOUD_SHADOW_BIT: int = 4   # Cloud shadow.


def dry_season_month_filter() -> ee.Filter:
    """Return a month filter selecting the dry season (Nov–Mar).

    The dry season straddles the new year, so it is expressed as the union of
    two calendar-month ranges: November–December and January–March.

    Returns:
        An ``ee.Filter`` matching images whose acquisition month is 11, 12, 1,
        2, or 3.
    """
    return ee.Filter.Or(
        ee.Filter.calendarRange(11, 12, "month"),
        ee.Filter.calendarRange(1, 3, "month"),
    )


def dry_season_window(year: int) -> Tuple[str, str]:
    """Return the date window for a dry-season-year.

    Dry-season-year ``year`` spans from 1 November of the previous calendar year
    through 31 March of ``year`` (so its Nov/Dec scenes are labelled with the
    next calendar year, and its Jan–Mar scenes with the same calendar year).

    Args:
        year: The dry-season-year label (e.g. 2010).

    Returns:
        A ``(start, end)`` tuple of ISO-8601 date strings,
        ``"(year-1)-11-01"`` to ``"(year)-03-31"``.
    """
    start: str = f"{year - 1}-11-01"
    end: str = f"{year}-03-31"
    return start, end


def s2_with_aoi_cloud(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Load Sentinel-2 SR Harmonized with an AOI-based cloud percentage.

    Filters the collection to the AOI, date window, and dry-season months, links
    the Cloud Score+ ``cs`` band, and attaches an ``aoi_cloud_pct`` property to
    each image: the mean of the cloudy mask (``cs`` below ``CS_THRESHOLD``)
    reduced over the AOI polygon and expressed as a percentage.

    Args:
        aoi: Area of interest as an ``ee.Geometry``.
        start: Inclusive start date, ISO-8601 (``YYYY-MM-DD``).
        end: Exclusive end date, ISO-8601 (``YYYY-MM-DD``).

    Returns:
        An ``ee.ImageCollection`` of Sentinel-2 images, each carrying an
        ``aoi_cloud_pct`` property.
    """
    collection: ee.ImageCollection = (
        ee.ImageCollection(config.S2_SR_HARMONIZED)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(dry_season_month_filter())
        .linkCollection(ee.ImageCollection(config.CLOUD_SCORE_PLUS), ["cs"])
    )

    def _attach_cloud(image: ee.Image) -> ee.Image:
        # Cloud Score+ 'cs': higher is clearer, so cloudy = cs below threshold.
        cloudy: ee.Image = image.select("cs").lt(config.CS_THRESHOLD)
        cloud_fraction = cloudy.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e9,
            bestEffort=True,
        ).get("cs")
        aoi_cloud_pct = ee.Number(cloud_fraction).multiply(100)
        return image.set("aoi_cloud_pct", aoi_cloud_pct)

    return collection.map(_attach_cloud)


def landsat_with_aoi_cloud(
    collection_id: str,
    sensor_label: str,
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> ee.ImageCollection:
    """Load a Landsat C2 L2 collection with an AOI-based cloud percentage.

    Filters the collection to the AOI, date window, and dry-season months, then
    attaches an ``aoi_cloud_pct`` property to each image from the QA_PIXEL band:
    a pixel is cloudy if the dilated-cloud (bit 1), cloud (bit 3), or
    cloud-shadow (bit 4) flag is set. The mean of that cloudy mask over the AOI
    is expressed as a percentage. The ``sensor`` property is also set.

    Args:
        collection_id: GEE Landsat Collection 2 L2 collection ID.
        sensor_label: Short sensor tag (e.g. ``"L5"``, ``"L7"``, ``"L8"``,
            ``"L9"``).
        aoi: Area of interest as an ``ee.Geometry``.
        start: Inclusive start date, ISO-8601 (``YYYY-MM-DD``).
        end: Exclusive end date, ISO-8601 (``YYYY-MM-DD``).

    Returns:
        An ``ee.ImageCollection`` of Landsat images, each carrying
        ``aoi_cloud_pct`` and ``sensor`` properties.
    """
    collection: ee.ImageCollection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(dry_season_month_filter())
    )

    def _attach_cloud(image: ee.Image) -> ee.Image:
        qa: ee.Image = image.select("QA_PIXEL")
        dilated: ee.Image = qa.bitwiseAnd(1 << _QA_DILATED_CLOUD_BIT).neq(0)
        cloud: ee.Image = qa.bitwiseAnd(1 << _QA_CLOUD_BIT).neq(0)
        shadow: ee.Image = qa.bitwiseAnd(1 << _QA_CLOUD_SHADOW_BIT).neq(0)
        cloudy: ee.Image = dilated.Or(cloud).Or(shadow).rename("cloudy")
        cloud_fraction = cloudy.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9,
            bestEffort=True,
        ).get("cloudy")
        aoi_cloud_pct = ee.Number(cloud_fraction).multiply(100)
        return image.set("aoi_cloud_pct", aoi_cloud_pct).set("sensor", sensor_label)

    return collection.map(_attach_cloud)


def list_scenes(year: int, aoi: ee.Geometry) -> ee.FeatureCollection:
    """Assemble all candidate scenes for a dry-season-year across sensors.

    Includes every sensor operational in ``year`` (Landsat 5/7/8/9 per their
    operational ranges, plus Sentinel-2 for years >= 2017), each with an
    AOI-based cloud percentage. Every candidate image is reduced to a
    null-geometry feature carrying only the inventory properties.

    Args:
        year: The dry-season-year label (e.g. 2010).
        aoi: Area of interest as an ``ee.Geometry``.

    Returns:
        An ``ee.FeatureCollection`` of null-geometry features, one per candidate
        image, with properties: ``date`` (``YYYY-MM-DD``), ``timestamp_utc``
        (ISO-8601 from ``system:time_start``), ``sensor`` (L5/L7/L8/L9/S2),
        ``image_id`` (``system:index``), ``dry_year``, and ``aoi_cloud_pct``.
    """
    start, end = dry_season_window(year)
    collections: List[ee.ImageCollection] = []

    # Landsat sensors operational for this dry-season-year.
    for label, collection_id in _LANDSAT_COLLECTIONS.items():
        first, last = _SENSOR_OPERATIONAL[label]
        if first <= year <= last:
            collections.append(
                landsat_with_aoi_cloud(collection_id, label, aoi, start, end)
            )

    # Sentinel-2 (tagged with its sensor label) once operational.
    s2_first, s2_last = _SENSOR_OPERATIONAL["S2"]
    if s2_first <= year <= s2_last:
        collections.append(
            s2_with_aoi_cloud(aoi, start, end).map(
                lambda image: image.set("sensor", "S2")
            )
        )

    # No operational sensors (outside the archive) -> empty inventory.
    if not collections:
        return ee.FeatureCollection([])

    merged: ee.ImageCollection = reduce(
        lambda acc, col: acc.merge(col), collections
    )

    def _to_feature(element: ee.ComputedObject) -> ee.Feature:
        image: ee.Image = ee.Image(element)
        date: ee.Date = image.date()
        properties = {
            "date": date.format("YYYY-MM-dd"),
            "timestamp_utc": date.format(),  # ISO-8601, UTC.
            "sensor": image.get("sensor"),
            "image_id": image.get("system:index"),
            "dry_year": year,
            "aoi_cloud_pct": image.get("aoi_cloud_pct"),
        }
        return ee.Feature(None, properties)

    # ImageCollection.map requires an Image return, so map over a server-side
    # list of images to build null-geometry features instead.
    images: ee.List = merged.toList(merged.size())
    return ee.FeatureCollection(images.map(_to_feature))
