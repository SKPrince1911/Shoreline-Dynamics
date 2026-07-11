"""Phase 1: image retrieval and quality/cloud masking (image inventory).

Builds the per-year candidate-scene inventory for the dry-season shoreline
pipeline. For each dry-season-year it retrieves Sentinel-2 SR Harmonized and
Landsat 5/7/8/9 Collection 2 L2 imagery, computes an AOI-based cloud percentage
(Cloud Score+ for Sentinel-2, QA_PIXEL bit flags for Landsat) rather than
relying on scene-wide metadata, mosaics same-day scenes to measure AOI
coverage, and selects the single clearest full-coverage image per year (flagging
gap years when none qualifies).

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
_QA_FILL_BIT: int = 0           # Fill / no-data (incl. Landsat 7 SLC-off gaps).

# Common AOI analysis scale (m) for the per-date coverage/cloud reductions.
# Uses the coarser Landsat pixel so metrics are comparable across sensors.
# <-- TUNABLE
_AOI_ANALYSIS_SCALE: int = 30

# How many sensor-date features to materialize per Earth Engine request.
# Evaluating a whole aggregation-heavy FeatureCollection at once trips the "Too
# many concurrent aggregations" limit (HTTP 429), so results are fetched in
# chunks of this many keys — each request builds a FeatureCollection of only
# this many features. Lower this if 429s persist.  <-- TUNABLE
_MATERIALIZE_BATCH_SIZE: int = 8

# Sensor selection preference (lower rank = preferred) used to break ties
# between equally-clean candidates: finer/better sensors first.
_RESOLUTION_RANK: Dict[str, int] = {
    "S2": 0,   # 10 m Sentinel-2.
    "L8": 1,   # 30 m Landsat 8 OLI.
    "L9": 1,   # 30 m Landsat 9 OLI-2.
    "L5": 2,   # 30 m Landsat 5 TM.
    "L7": 3,   # 30 m Landsat 7 ETM+ (SLC-off risk).
}


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


def clean_scene_id(system_index: ee.String) -> ee.String:
    """Strip merge-index prefixes from a ``system:index`` value.

    ``ee.ImageCollection.merge`` prepends running numeric prefixes (e.g.
    ``"1_1_2_"``) to each image's ``system:index`` to keep IDs unique. This
    removes those leading ``<digits>_`` groups, returning the original scene ID.
    Genuine scene IDs never begin with digits immediately followed by an
    underscore (Landsat IDs start with letters; Sentinel-2 IDs start with a
    date followed by ``T``), so the strip is unambiguous.

    Args:
        system_index: The (possibly prefixed) ``system:index`` string.

    Returns:
        The clean scene ID as an ``ee.String``.
    """
    return ee.String(system_index).replace("^([0-9]+_)+", "")


def slc_off(sensor: str, date: str) -> bool:
    """Return whether a scene is a Landsat 7 SLC-off acquisition.

    Landsat 7 lost its Scan Line Corrector on 2003-05-31; later ETM+ scenes
    carry ~22% striping gaps. Such scenes are usable only to fill otherwise
    empty dry-season-years.

    Args:
        sensor: Sensor label (e.g. ``"L7"``).
        date: Acquisition date as an ISO-8601 ``YYYY-MM-DD`` string. ISO dates
            sort lexicographically, so a plain string comparison is correct.

    Returns:
        ``True`` if ``sensor`` is Landsat 7 and ``date`` is after
        ``config.SLC_OFF_DATE``, else ``False``.
    """
    return sensor == "L7" and date > config.SLC_OFF_DATE


def _prepare_landsat(
    collection_id: str,
    sensor_label: str,
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> ee.ImageCollection:
    """Prepare a Landsat collection for per-date mosaicking.

    Each returned image carries a ``cloudy`` band (1 where dilated-cloud,
    cloud, or cloud-shadow is flagged, masked to valid data) and a ``valid``
    band (1 over non-fill data, masked elsewhere — so Landsat 7 SLC-off gaps
    reduce coverage), plus the ``sensor``, ``date``, and ``sensor_date``
    grouping properties.
    """
    collection: ee.ImageCollection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(dry_season_month_filter())
    )

    def _prepare(image: ee.Image) -> ee.Image:
        qa: ee.Image = image.select("QA_PIXEL")
        data_mask: ee.Image = qa.bitwiseAnd(1 << _QA_FILL_BIT).eq(0)  # not fill
        dilated: ee.Image = qa.bitwiseAnd(1 << _QA_DILATED_CLOUD_BIT).neq(0)
        cloud: ee.Image = qa.bitwiseAnd(1 << _QA_CLOUD_BIT).neq(0)
        shadow: ee.Image = qa.bitwiseAnd(1 << _QA_CLOUD_SHADOW_BIT).neq(0)
        cloudy: ee.Image = (
            dilated.Or(cloud).Or(shadow).updateMask(data_mask).rename("cloudy")
        )
        valid: ee.Image = data_mask.selfMask().rename("valid")
        return _tag_prepared(ee.Image.cat([cloudy, valid]), image, sensor_label)

    return collection.map(_prepare)


def _prepare_s2(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Prepare Sentinel-2 SR Harmonized for per-date mosaicking.

    Each returned image carries a ``cloudy`` band (Cloud Score+ ``cs`` below
    ``CS_THRESHOLD``, masked to valid data) and a ``valid`` band (1 over data,
    masked elsewhere), plus the ``sensor``, ``date``, and ``sensor_date``
    grouping properties.
    """
    collection: ee.ImageCollection = (
        ee.ImageCollection(config.S2_SR_HARMONIZED)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(dry_season_month_filter())
        .linkCollection(ee.ImageCollection(config.CLOUD_SCORE_PLUS), ["cs"])
    )

    def _prepare(image: ee.Image) -> ee.Image:
        # Data footprint from a reflectance band (masked over no-data).
        data_mask: ee.Image = image.select("B8").mask()
        cloudy: ee.Image = (
            image.select("cs")
            .lt(config.CS_THRESHOLD)
            .updateMask(data_mask)
            .rename("cloudy")
        )
        valid: ee.Image = data_mask.selfMask().rename("valid")
        return _tag_prepared(ee.Image.cat([cloudy, valid]), image, "S2")

    return collection.map(_prepare)


def _tag_prepared(
    prepared: ee.Image, source: ee.Image, sensor_label: str
) -> ee.Image:
    """Attach grouping properties to a prepared (cloudy+valid) image.

    Carries the source ``system:time_start`` and ``system:index`` (so the
    original scene ID survives merging) and sets ``sensor``, ``date``
    (``YYYY-MM-DD``), and a ``sensor_date`` key used to group same-day scenes.
    """
    date: ee.Date = source.date()
    date_str: ee.String = date.format("YYYY-MM-dd")
    return prepared.set(
        {
            "sensor": sensor_label,
            "date": date_str,
            "sensor_date": ee.String(sensor_label).cat("_").cat(date_str),
            "system:time_start": source.get("system:time_start"),
            "system:index": source.get("system:index"),
        }
    )


def candidates_by_date(year: int, aoi: ee.Geometry) -> ee.FeatureCollection:
    """Group candidate scenes by sensor-date and mosaic each group.

    For a dry-season-year, every operational sensor's scenes are grouped by
    ``(sensor, acquisition-date)`` and mosaicked. For each mosaic the AOI cloud
    percentage (mean of the cloudy mask over observed pixels) and AOI coverage
    percentage (valid AOI pixels / total AOI pixels) are computed.

    Args:
        year: The dry-season-year label (e.g. 2010).
        aoi: Area of interest as an ``ee.Geometry``.

    Returns:
        An ``ee.FeatureCollection`` of null-geometry features, one per
        sensor-date, with properties: ``dry_year``, ``date`` (``YYYY-MM-DD``),
        ``timestamp_utc`` (ISO-8601, earliest scene in the group), ``sensor``,
        ``n_scenes``, ``slc_off``, ``image_ids`` (comma-joined clean IDs),
        ``aoi_cloud_pct``, and ``aoi_coverage_pct``.
    """
    merged, keys, total_aoi_px, slc_off_millis = _candidate_components(year, aoi)
    if merged is None:
        return ee.FeatureCollection([])
    return _candidates_fc(keys, merged, aoi, year, total_aoi_px, slc_off_millis)


def _candidate_components(year: int, aoi: ee.Geometry):
    """Build the merged prepared collection and the shared reduction inputs.

    Returns:
        A ``(merged, keys, total_aoi_px, slc_off_millis)`` tuple, where
        ``merged`` is the prepared+merged ``ee.ImageCollection`` (``None`` if no
        sensor is operational this year), ``keys`` is the ``ee.List`` of distinct
        ``sensor_date`` strings, ``total_aoi_px`` is the AOI pixel count at the
        analysis scale (coverage denominator), and ``slc_off_millis`` is the
        SLC-off cutoff in epoch milliseconds.
    """
    start, end = dry_season_window(year)
    prepared: List[ee.ImageCollection] = []

    for label, collection_id in _LANDSAT_COLLECTIONS.items():
        first, last = _SENSOR_OPERATIONAL[label]
        if first <= year <= last:
            prepared.append(_prepare_landsat(collection_id, label, aoi, start, end))

    s2_first, s2_last = _SENSOR_OPERATIONAL["S2"]
    if s2_first <= year <= s2_last:
        prepared.append(_prepare_s2(aoi, start, end))

    if not prepared:
        return None, None, None, None

    merged: ee.ImageCollection = reduce(
        lambda acc, col: acc.merge(col), prepared
    )
    # Distinct sensor-date groups present this year.
    keys: ee.List = merged.aggregate_array("sensor_date").distinct()
    slc_off_millis: ee.Number = ee.Date(config.SLC_OFF_DATE).millis()
    # Total AOI pixel count at the analysis scale, computed once and reused as
    # the coverage denominator (keeps per-feature work to a single reduceRegion).
    total_aoi_px: ee.Number = ee.Number(
        ee.Image.constant(1)
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=_AOI_ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("constant")
    )
    return merged, keys, total_aoi_px, slc_off_millis


def _candidates_fc(
    keys: ee.List,
    merged: ee.ImageCollection,
    aoi: ee.Geometry,
    year: int,
    total_aoi_px: ee.Number,
    slc_off_millis: ee.Number,
) -> ee.FeatureCollection:
    """Build the null-geometry candidate features for a set of sensor-date keys.

    The number of features (and thus concurrent reductions) equals ``len(keys)``,
    so passing a small key subset bounds the aggregations evaluated per request.
    """

    def _feature_for_key(key: ee.String) -> ee.Feature:
        key = ee.String(key)
        # sensor_date == "<sensor>_<YYYY-MM-DD>"; neither part contains an
        # underscore, so split gives [sensor, date] without any aggregation.
        parts: ee.List = key.split("_")
        sensor: ee.String = ee.String(parts.get(0))
        date: ee.String = ee.String(parts.get(1))

        group: ee.ImageCollection = merged.filter(
            ee.Filter.eq("sensor_date", key)
        )
        mosaic: ee.Image = group.mosaic()

        # One reduceRegion for both metrics: mean(cloudy) over observed pixels
        # and count(valid) = number of data pixels in the AOI.
        stats: ee.Dictionary = mosaic.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.count(), sharedInputs=True
            ),
            geometry=aoi,
            scale=_AOI_ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )
        aoi_cloud_pct = ee.Number(stats.get("cloudy_mean")).multiply(100)
        # Clamp to 100: Sentinel-2 mosaics can report ~100.5 because the coarse
        # screening grid overcounts valid pixels near the AOI edge.
        aoi_coverage_pct = (
            ee.Number(stats.get("valid_count"))
            .divide(total_aoi_px)
            .multiply(100)
            .min(100.0)
        )

        timestamp_utc: ee.String = ee.Date(
            group.aggregate_min("system:time_start")
        ).format()
        # Clean the merge-prefixed system:index of every scene in the group.
        image_ids: ee.String = (
            group.aggregate_array("system:index")
            .map(lambda idx: clean_scene_id(ee.String(idx)))
            .join(",")
        )

        # Server-side mirror of slc_off(): L7 acquired after the SLC-off date.
        # Use compareTo(...).eq(0) so the result is a typed ee.Number (which
        # has .And); ee.String.equals returns an untyped ComputedObject.
        is_l7 = sensor.compareTo("L7").eq(0)
        is_after_slc = ee.Date(date).millis().gt(slc_off_millis)
        is_slc_off = is_l7.And(is_after_slc)

        properties = {
            "dry_year": year,
            "date": date,
            "timestamp_utc": timestamp_utc,
            "sensor": sensor,
            "n_scenes": group.size(),
            "slc_off": is_slc_off,
            "image_ids": image_ids,
            "aoi_cloud_pct": aoi_cloud_pct,
            "aoi_coverage_pct": aoi_coverage_pct,
        }
        return ee.Feature(None, properties)

    return ee.FeatureCollection(ee.List(keys).map(_feature_for_key))


def _materialize_candidates(
    year: int, aoi: ee.Geometry, chunk: int = _MATERIALIZE_BATCH_SIZE
) -> List[dict]:
    """Fetch per-date candidate properties by paging over sensor-date keys.

    Evaluating every sensor-date in one request runs all their reductions
    concurrently and trips Earth Engine's HTTP 429 "Too many concurrent
    aggregations" limit on high-sensor years (e.g. 2022: L7+L8+L9+S2). Instead,
    the distinct keys are listed once (cheap), then a *small* FeatureCollection
    of at most ``chunk`` features is built and evaluated per request, so each
    ``getInfo`` runs only ~``chunk`` reductions.

    Args:
        year: The dry-season-year label.
        aoi: Area of interest as an ``ee.Geometry``.
        chunk: Number of sensor-dates to evaluate per request.

    Returns:
        A list of per-sensor-date property dictionaries.
    """
    merged, keys, total_aoi_px, slc_off_millis = _candidate_components(year, aoi)
    if merged is None:
        return []
    key_list: List[str] = keys.getInfo()  # cheap: list of "sensor_date" strings
    records: List[dict] = []
    for start in range(0, len(key_list), chunk):
        subset = key_list[start:start + chunk]
        fc = _candidates_fc(
            ee.List(subset), merged, aoi, year, total_aoi_px, slc_off_millis
        )
        records.extend(
            feature["properties"] for feature in fc.getInfo()["features"]
        )
    return records


def candidates_dataframe(
    year: int, aoi: ee.Geometry, chunk: int = _MATERIALIZE_BATCH_SIZE
):
    """Return the per-date candidate inventory as a sorted pandas DataFrame.

    Convenience wrapper over :func:`candidates_by_date` that retrieves results
    in key chunks (see :func:`_materialize_candidates`) to avoid the "Too many
    concurrent aggregations" (HTTP 429) limit, then sorts by AOI cloud %.

    Args:
        year: The dry-season-year label.
        aoi: Area of interest as an ``ee.Geometry``.
        chunk: Number of sensor-dates to evaluate per request.

    Returns:
        A ``pandas.DataFrame`` of candidates sorted by ``aoi_cloud_pct``
        (ascending); empty if the year has no candidate scenes.
    """
    import pandas as pd

    rows = _materialize_candidates(year, aoi, chunk)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("aoi_cloud_pct").reset_index(drop=True)
    return df


def _rank_key(row: dict) -> tuple:
    """Sort key for a qualifying candidate: clearest, then most reliable sensor.

    Lowest AOI cloud first; ties broken by preferring non-SLC-off scenes, then
    finer/preferred sensors (S2 > L8/L9 > L5 > L7).
    """
    return (
        row["aoi_cloud_pct"],
        slc_off(row["sensor"], row["date"]),
        _RESOLUTION_RANK.get(row["sensor"], 99),
    )


def _cloud_ok(row: dict) -> bool:
    """Whether a candidate's AOI cloud cover is within the threshold."""
    return row["aoi_cloud_pct"] <= config.CLOUD_THRESHOLD_PCT


def _classify_candidates(rows: List[dict]):
    """Partition candidates into the two qualifying tiers and the remainder.

    Returns:
        A ``(tier1, tier2, rest)`` tuple of disjoint lists, each already sorted:
        ``tier1`` (coverage >= strict threshold and cloud OK) and ``tier2``
        (relaxed <= coverage < strict, cloud OK) by :func:`_rank_key`; ``rest``
        (everything else) by coverage descending then cloud ascending.
    """
    tier1 = sorted(
        [
            row
            for row in rows
            if row["aoi_coverage_pct"] >= config.COVERAGE_THRESHOLD_PCT
            and _cloud_ok(row)
        ],
        key=_rank_key,
    )
    tier2 = sorted(
        [
            row
            for row in rows
            if config.COVERAGE_THRESHOLD_RELAXED_PCT
            <= row["aoi_coverage_pct"]
            < config.COVERAGE_THRESHOLD_PCT
            and _cloud_ok(row)
        ],
        key=_rank_key,
    )
    used = {(row["sensor"], row["date"]) for row in tier1 + tier2}
    rest = sorted(
        [row for row in rows if (row["sensor"], row["date"]) not in used],
        key=lambda row: (-row["aoi_coverage_pct"], row["aoi_cloud_pct"]),
    )
    return tier1, tier2, rest


def select_best(year: int, aoi: ee.Geometry) -> ee.Feature:
    """Select the single best candidate scene for a dry-season-year.

    Two-tier adaptive rule:

    1. If any candidate has coverage >= ``COVERAGE_THRESHOLD_PCT`` and cloud
       <= ``CLOUD_THRESHOLD_PCT``, pick the lowest-cloud one (tie-break:
       non-SLC-off preferred, then S2 > L8/L9 > L5 > L7) with
       ``relaxed_coverage=False``.
    2. Otherwise, if any candidate has coverage >= ``COVERAGE_THRESHOLD_RELAXED_PCT``
       and cloud <= ``CLOUD_THRESHOLD_PCT``, pick it the same way with
       ``relaxed_coverage=True``.
    3. Otherwise return a gap feature (``selected=False``) with a ``gap_reason``,
       keeping the best-available row: highest coverage among cloud-OK
       candidates, or the lowest-cloud candidate if none are cloud-OK.

    Every returned feature carries ``relaxed_coverage``. The candidate table is
    materialized client-side (paged over sensor-date keys, via
    :func:`_materialize_candidates`) so the tiering and tie-breaks stay
    unambiguous.

    Args:
        year: The dry-season-year label (e.g. 2010).
        aoi: Area of interest as an ``ee.Geometry``.

    Returns:
        An ``ee.Feature`` (null geometry) with the chosen (or best-available)
        candidate's properties plus ``selected`` and ``relaxed_coverage`` (and,
        when unselected, ``gap_reason``).
    """
    rows: List[dict] = _materialize_candidates(year, aoi)
    tier1, tier2, _rest = _classify_candidates(rows)

    if tier1:
        return ee.Feature(
            None, {**tier1[0], "selected": True, "relaxed_coverage": False}
        )
    if tier2:
        return ee.Feature(
            None, {**tier2[0], "selected": True, "relaxed_coverage": True}
        )

    # Gap year: no qualifying scene at either coverage tier.
    if not rows:
        return ee.Feature(
            None,
            {
                "dry_year": year,
                "selected": False,
                "relaxed_coverage": False,
                "gap_reason": "no candidate scenes",
            },
        )

    cloud_ok = [row for row in rows if _cloud_ok(row)]
    if cloud_ok:
        best_available = max(cloud_ok, key=lambda row: row["aoi_coverage_pct"])
        gap_reason = "no >=90% coverage"
    else:
        best_available = min(rows, key=lambda row: row["aoi_cloud_pct"])
        gap_reason = "no <=10% cloud"

    return ee.Feature(
        None,
        {
            **best_available,
            "selected": False,
            "relaxed_coverage": False,
            "gap_reason": gap_reason,
        },
    )


def _ranked_rows(year: int, aoi: ee.Geometry) -> List[dict]:
    """Return all candidate rows for the year, ranked (rank 0 = select_best pick).

    Order: tier-1 qualifying (strict coverage) then tier-2 relaxed coverage —
    both cloud-OK, by :func:`_rank_key` — then the remainder by coverage
    descending, cloud ascending. Each row gains ``rank`` (0-based) and the
    booleans ``qualifies_95`` and ``qualifies_90``.
    """
    rows = _materialize_candidates(year, aoi)
    tier1, tier2, rest = _classify_candidates(rows)
    ranked: List[dict] = []
    for index, row in enumerate(tier1 + tier2 + rest):
        ranked.append(
            {
                **row,
                "rank": index,
                "qualifies_95": bool(
                    row["aoi_coverage_pct"] >= config.COVERAGE_THRESHOLD_PCT
                    and _cloud_ok(row)
                ),
                "qualifies_90": bool(
                    row["aoi_coverage_pct"] >= config.COVERAGE_THRESHOLD_RELAXED_PCT
                    and _cloud_ok(row)
                ),
            }
        )
    return ranked


def year_candidates_ranked(year: int, aoi: ee.Geometry) -> ee.FeatureCollection:
    """Return all candidate sensor-dates for the year, ranked and flagged.

    Rank 0 is the :func:`select_best` pick. Each null-geometry feature carries
    the candidate properties plus ``rank`` (0-based), ``qualifies_95``, and
    ``qualifies_90``. Features are built from already-materialized scalars, so
    fetching this collection is cheap (no per-feature reductions).

    Args:
        year: The dry-season-year label.
        aoi: Area of interest as an ``ee.Geometry``.

    Returns:
        An ``ee.FeatureCollection`` ordered by ``rank`` ascending.
    """
    ranked = _ranked_rows(year, aoi)
    return ee.FeatureCollection([ee.Feature(None, row) for row in ranked])


def candidate_mosaic_truecolor(
    year: int, aoi: ee.Geometry, rank: int = 0
) -> ee.Image:
    """Rebuild a true-color RGB mosaic for the rank-th ranked candidate.

    Reloads the raw imagery for that candidate's sensor and acquisition date,
    mosaics it, and returns a visualized RGB image clipped to the AOI, with
    the correct bands and scaling per sensor:

    * Sentinel-2: ``B4, B3, B2`` (display min 0, max 3000).
    * Landsat 8/9 C2 L2: ``SR_B4, SR_B3, SR_B2``; Landsat 5/7 C2 L2:
      ``SR_B3, SR_B2, SR_B1`` — scaled to reflectance (DN * 0.0000275 - 0.2,
      display min 0.0, max 0.3).

    Args:
        year: The dry-season-year label.
        aoi: Area of interest as an ``ee.Geometry``.
        rank: 0-based rank from :func:`year_candidates_ranked` (0 = best).

    Returns:
        A visualized 3-band RGB ``ee.Image`` clipped to the AOI.
    """
    ranked = _ranked_rows(year, aoi)
    row = ranked[rank]
    sensor: str = row["sensor"]
    day_start: ee.Date = ee.Date(row["date"])
    day_end: ee.Date = day_start.advance(1, "day")

    if sensor == "S2":
        mosaic = (
            ee.ImageCollection(config.S2_SR_HARMONIZED)
            .filterBounds(aoi)
            .filterDate(day_start, day_end)
            .mosaic()
        )
        rgb = mosaic.select(["B4", "B3", "B2"]).visualize(min=0, max=3000)
    else:
        bands = (
            ["SR_B4", "SR_B3", "SR_B2"]
            if sensor in ("L8", "L9")
            else ["SR_B3", "SR_B2", "SR_B1"]
        )
        mosaic = (
            ee.ImageCollection(_LANDSAT_COLLECTIONS[sensor])
            .filterBounds(aoi)
            .filterDate(day_start, day_end)
            .mosaic()
        )
        # Collection 2 Level 2 surface-reflectance scaling.
        reflectance = mosaic.select(bands).multiply(0.0000275).add(-0.2)
        rgb = reflectance.visualize(min=0.0, max=0.3)

    return rgb.clip(aoi)


def tidal_channels_fc() -> ee.FeatureCollection:
    """Load the tidal-channel reference lines as an ``ee.FeatureCollection``.

    Reads ``data/tidal_channels.geojson`` (LineString features) via
    :func:`config.load_tidal_channels` and builds the collection directly with
    ``ee.Geometry`` / ``ee.Feature`` — no dependency on ``geemap`` GeoJSON
    helpers, whose names differ across versions.

    Returns:
        An ``ee.FeatureCollection`` of the tidal-channel lines (EPSG:4326).
    """
    geojson = config.load_tidal_channels()
    features = [
        ee.Feature(
            ee.Geometry(feature["geometry"]), feature.get("properties") or {}
        )
        for feature in geojson["features"]
    ]
    return ee.FeatureCollection(features)
