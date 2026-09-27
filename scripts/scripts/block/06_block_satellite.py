"""
SIH26 — Block-level satellite/environment collector
====================================================

Purpose
-------
Collect real, block-level satellite/environmental predictors that complement
the coarse atmospheric forecast:

    Sentinel-2 : NDVI, EVI, SAVI
    Sentinel-1 : VV, VH
    Landsat    : daytime land-surface temperature (LST)

The collector performs polygon zonal statistics over the BLOCK boundary.
It does not use a centroid-only point sample.

Scientific rules
----------------
- Real Google Earth Engine data only.
- No synthetic/sample fallback.
- No temporal interpolation or fabricated replacement values.
- Cloud screening is applied to Sentinel-2.
- Every observation retains acquisition date/time and source provenance.
- Existing observations are preserved; only genuinely new acquisition dates
  are appended.
- The script is universal: State/District/Block are selected at runtime.
- Static terrain variables are intentionally NOT extracted here; those belong
  to the GP-level terrain module where fine-scale spatial variability matters.

Important forecasting/leakage note
----------------------------------
Satellite variables are observations of land-surface state. For a forecasting
experiment, only observations that were actually available before the
forecast issuance/target should enter the feature vector. The downstream
training/merge stage must therefore apply the appropriate availability cutoff.
This collector archives the observations without pretending they are future
information.

GEE datasets
------------
Sentinel-2:
    COPERNICUS/S2_SR_HARMONIZED
Sentinel-1:
    COPERNICUS/S1_GRD
Landsat 8/9 LST:
    LANDSAT/LC08/C02/T1_L2
    LANDSAT/LC09/C02/T1_L2

Outputs
-------
block/<State>/<District>/<Block>/
    raw/satellite/
        sentinel2/
        sentinel1/
        landsat_lst/
    processed/satellite/
        sentinel2_block.csv
        sentinel1_block.csv
        landsat_lst_block.csv
        block_satellite_collection_metadata.json
"""


from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================


def find_project_root(start: Path) -> Path:
    start = start.resolve()

    for p in [start, *start.parents]:
        if all((p / x).exists() for x in [
            "block",
            "gp",
            "datasets",
            "scripts",
        ]):
            return p

    raise RuntimeError(
        "SIH26 project root not found. Expected block/, gp/, "
        "datasets/, scripts/."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)


# ============================================================
# CONFIGURATION
# ============================================================


SCRIPT_VERSION = "SIH26-06-block-satellite-v2"

DEFAULT_START_DATE = "2021-08-15"
BATCH_DAYS = 90

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"

LANDSAT_COLLECTIONS = [
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]

S2_CLOUD_LIMIT = 80

S2_SCALE = 10
S1_SCALE = 10
LST_SCALE = 30

GEE_TILE_SCALE = 4


# ============================================================
# LOCATION
# ============================================================


def clean_name(value: str) -> str:
    value = str(value).strip()

    if not value:
        raise ValueError("Location name cannot be empty.")

    return re.sub(r'[<>:"/\\|?*]', "_", value)


def get_location():
    print("\nEnter the target block location.")

    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    if not state or not district or not block:
        raise ValueError(
            "State, District and Block are required."
        )

    location_dir = (
        PROJECT_ROOT
        / "block"
        / clean_name(state)
        / clean_name(district)
        / clean_name(block)
    )

    metadata_path = (
        location_dir
        / "metadata"
        / "block_metadata.json"
    )

    boundary_path = (
        location_dir
        / "processed"
        / "block_boundary.geojson"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Block metadata not found:\n{metadata_path}\n"
            "Run 01_block_boundary.py first."
        )

    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Block boundary not found:\n{boundary_path}\n"
            "Run 01_block_boundary.py first."
        )

    raw_dir = location_dir / "raw" / "satellite"
    processed_dir = location_dir / "processed" / "satellite"

    for p in [
        raw_dir / "sentinel2",
        raw_dir / "sentinel1",
        raw_dir / "landsat_lst",
        processed_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return (
        state,
        district,
        block,
        location_dir,
        metadata_path,
        boundary_path,
        raw_dir,
        processed_dir,
    )


# ============================================================
# GEE INITIALIZATION
# ============================================================


def initialize_gee():
    """
    Use the user's existing GEE authentication/project configuration.

    No service-account key is hard-coded into this SIH26 script.
    """
    try:
        ee.Initialize()
        print("GEE initialized.")
        return

    except Exception:
        print("Normal GEE initialization failed.")
        print("Attempting interactive GEE authentication...")

        try:
            ee.Authenticate()
            ee.Initialize()
            print("GEE authenticated and initialized.")

        except Exception as exc:
            raise RuntimeError(
                "Could not initialize Google Earth Engine. "
                "Authenticate GEE first and rerun."
            ) from exc


# ============================================================
# BLOCK GEOMETRY
# ============================================================


def load_block_geometry(boundary_path: Path):
    """
    Read the local GeoJSON boundary and convert it to an EE geometry.
    """
    with boundary_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])

        if not features:
            raise RuntimeError("Block GeoJSON contains no features.")

        geometries = [
            f["geometry"]
            for f in features
            if f.get("geometry")
        ]

        if not geometries:
            raise RuntimeError(
                "Block GeoJSON contains no valid geometries."
            )

        if len(geometries) == 1:
            geometry_json = geometries[0]
        else:
            geometry_json = {
                "type": "MultiPolygon",
                "coordinates": [],
            }

            for geom in geometries:
                if geom["type"] == "Polygon":
                    geometry_json["coordinates"].append(
                        geom["coordinates"]
                    )
                elif geom["type"] == "MultiPolygon":
                    geometry_json["coordinates"].extend(
                        geom["coordinates"]
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported geometry type: {geom['type']}"
                    )

    elif geojson.get("type") in [
        "Polygon",
        "MultiPolygon",
    ]:
        geometry_json = geojson

    else:
        raise RuntimeError(
            "Unsupported block GeoJSON structure."
        )

    return ee.Geometry(geometry_json)


# ============================================================
# DATE RANGE
# ============================================================


def existing_latest_date(path: Path):
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        if df.empty or "acquisition_date" not in df.columns:
            return None

        dates = pd.to_datetime(
            df["acquisition_date"],
            errors="coerce",
        ).dropna()

        if dates.empty:
            return None

        return dates.max().date()

    except Exception:
        return None


def choose_start_date(path: Path):
    latest = existing_latest_date(path)

    if latest is None:
        return DEFAULT_START_DATE

    return latest.strftime("%Y-%m-%d")


def choose_end_date():
    # GEE filterDate end is exclusive. Use tomorrow so today's acquisitions
    # are included where the source has already published them.
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    return (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# ============================================================
# SENTINEL-2
# ============================================================


def mask_s2_clouds(image):
    """
    Mask Sentinel-2 clouds/cirrus using QA60 and SCL.

    QA60:
      bit 10 = opaque cloud
      bit 11 = cirrus

    SCL:
      3  = cloud shadow
      8  = medium probability cloud
      9  = high probability cloud
      10 = cirrus
      11 = snow/ice
    """
    qa = image.select("QA60")

    qa_mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )

    scl = image.select("SCL")

    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(
        qa_mask.And(scl_mask)
    )


def add_s2_indices(image):
    """
    Sentinel-2 SR reflectance is scaled by 10000.
    Scale to reflectance before computing indices.
    """
    image = mask_s2_clouds(image)

    optical = image.select(
        ["B2", "B3", "B4", "B8"]
    ).multiply(0.0001)

    blue = optical.select("B2")
    red = optical.select("B4")
    nir = optical.select("B8")

    ndvi = nir.subtract(red).divide(
        nir.add(red)
    ).rename("NDVI")

    evi = optical.expression(
        "2.5 * ((nir - red) / "
        "(nir + 6 * red - 7.5 * blue + 1))",
        {
            "nir": nir,
            "red": red,
            "blue": blue,
        },
    ).rename("EVI")

    savi = optical.expression(
        "1.5 * ((nir - red) / (nir + red + 0.5))",
        {
            "nir": nir,
            "red": red,
        },
    ).rename("SAVI")

    return image.addBands(
        [ndvi, evi, savi]
    )


def s2_feature(image, block_geometry):
    stats = image.select(
        ["NDVI", "EVI", "SAVI"]
    ).reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True,
            )
            .combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True,
            ),
        geometry=block_geometry,
        scale=S2_SCALE,
        bestEffort=True,
        tileScale=GEE_TILE_SCALE,
        maxPixels=1e9,
    )

    return ee.Feature(
        None,
        stats,
    ).set({
        "acquisition_datetime": image.date().format(
            "YYYY-MM-dd HH:mm:ss"
        ),
        "acquisition_date": image.date().format(
            "YYYY-MM-dd"
        ),
        "system_index": image.get("system:index"),
        "cloud_percentage": image.get(
            "CLOUDY_PIXEL_PERCENTAGE"
        ),
        "source_dataset": S2_COLLECTION,
        "sensor": "Sentinel-2",
    })


def collect_sentinel2(
    block_geometry,
    start_date,
    end_date,
):
    """
    Collect Sentinel-2 statistics in date batches.

    A single getInfo() over hundreds of mapped images can exceed the
    Earth Engine computation timeout. We therefore process 90-day windows.
    Each window is an independent server-side computation and the results
    are concatenated locally.

    No observations are fabricated if a batch fails.
    """
    print("\n[1/3] Sentinel-2")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    batch_days = BATCH_DAYS
    all_parts = []

    batch_start = start

    while batch_start < end:
        batch_end = min(
            batch_start + pd.Timedelta(days=batch_days),
            end,
        )

        batch_start_str = batch_start.strftime("%Y-%m-%d")
        batch_end_str = batch_end.strftime("%Y-%m-%d")

        print(
            f"  Batch: {batch_start_str} -> "
            f"{batch_end_str}"
        )

        collection = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(block_geometry)
            .filterDate(
                batch_start_str,
                batch_end_str,
            )
            .filter(
                ee.Filter.lte(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    S2_CLOUD_LIMIT,
                )
            )
            .map(add_s2_indices)
        )

        count = collection.size().getInfo()

        print(
            f"    Candidate scenes: {count}"
        )

        if count > 0:
            features = collection.map(
                lambda img: s2_feature(
                    img,
                    block_geometry,
                )
            )

            try:
                info = features.getInfo()
            except Exception as exc:
                raise RuntimeError(
                    "Sentinel-2 batch computation timed out or failed "
                    f"for {batch_start_str} -> {batch_end_str}. "
                    "The script intentionally stops instead of "
                    "creating fallback values."
                ) from exc

            rows = [
                feature.get("properties", {})
                for feature in info.get(
                    "features",
                    [],
                )
            ]

            if rows:
                all_parts.append(
                    pd.DataFrame(rows)
                )

        batch_start = batch_end

    if not all_parts:
        print("No Sentinel-2 observations available.")
        return pd.DataFrame()

    df = pd.concat(
        all_parts,
        ignore_index=True,
    )

    return normalize_satellite_df(
        df,
        "Sentinel-2",
    )


# ============================================================
# SENTINEL-1
# ============================================================


def s1_feature(image, block_geometry):
    stats = image.select(
        ["VV", "VH"]
    ).reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True,
            )
            .combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True,
            ),
        geometry=block_geometry,
        scale=S1_SCALE,
        bestEffort=True,
        tileScale=GEE_TILE_SCALE,
        maxPixels=1e9,
    )

    return ee.Feature(
        None,
        stats,
    ).set({
        "acquisition_datetime": image.date().format(
            "YYYY-MM-dd HH:mm:ss"
        ),
        "acquisition_date": image.date().format(
            "YYYY-MM-dd"
        ),
        "system_index": image.get("system:index"),
        "orbit_pass": image.get("orbitProperties_pass"),
        "relative_orbit": image.get(
            "relativeOrbitNumber_start"
        ),
        "source_dataset": S1_COLLECTION,
        "sensor": "Sentinel-1",
    })


def collect_sentinel1(
    block_geometry,
    start_date,
    end_date,
):
    """
    Collect Sentinel-1 statistics in 90-day Earth Engine batches to avoid
    large server-side getInfo() computations.
    """
    print("\n[2/3] Sentinel-1")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    batch_days = BATCH_DAYS
    all_parts = []
    batch_start = start

    while batch_start < end:
        batch_end = min(
            batch_start + pd.Timedelta(days=batch_days),
            end,
        )

        batch_start_str = batch_start.strftime("%Y-%m-%d")
        batch_end_str = batch_end.strftime("%Y-%m-%d")

        print(
            f"  Batch: {batch_start_str} -> "
            f"{batch_end_str}"
        )

        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterBounds(block_geometry)
            .filterDate(
                batch_start_str,
                batch_end_str,
            )
            .filter(
                ee.Filter.eq(
                    "instrumentMode",
                    "IW",
                )
            )
            .filter(
                ee.Filter.listContains(
                    "transmitterReceiverPolarisation",
                    "VV",
                )
            )
            .filter(
                ee.Filter.listContains(
                    "transmitterReceiverPolarisation",
                    "VH",
                )
            )
            .select(["VV", "VH"])
        )

        count = collection.size().getInfo()

        print(
            f"    Candidate scenes: {count}"
        )

        if count > 0:
            features = collection.map(
                lambda img: s1_feature(
                    img,
                    block_geometry,
                )
            )

            try:
                info = features.getInfo()
            except Exception as exc:
                raise RuntimeError(
                    "Sentinel-1 batch computation timed out or failed "
                    f"for {batch_start_str} -> {batch_end_str}."
                ) from exc

            rows = [
                feature.get("properties", {})
                for feature in info.get(
                    "features",
                    [],
                )
            ]

            if rows:
                all_parts.append(
                    pd.DataFrame(rows)
                )

        batch_start = batch_end

    if not all_parts:
        print("No Sentinel-1 observations available.")
        return pd.DataFrame()

    df = pd.concat(
        all_parts,
        ignore_index=True,
    )

    return normalize_satellite_df(
        df,
        "Sentinel-1",
    )


# ============================================================
# LANDSAT LST
# ============================================================


def mask_landsat(image):
    qa = image.select("QA_PIXEL")

    # QA_PIXEL:
    # bit 1 = dilated cloud
    # bit 2 = cirrus
    # bit 3 = cloud
    # bit 4 = cloud shadow
    # bit 5 = snow
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )

    # Landsat Collection 2 Level-2 ST_B10:
    # Kelvin = DN * 0.00341802 + 149.0
    lst_kelvin = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .rename("LST_K")
    )

    lst_c = (
        lst_kelvin
        .subtract(273.15)
        .rename("LST_C")
    )

    return image.addBands(
        [lst_kelvin, lst_c]
    ).updateMask(mask)


def landsat_feature(image, block_geometry):
    stats = image.select(
        ["LST_C"]
    ).reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True,
            )
            .combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True,
            ),
        geometry=block_geometry,
        scale=LST_SCALE,
        bestEffort=True,
        tileScale=GEE_TILE_SCALE,
        maxPixels=1e9,
    )

    return ee.Feature(
        None,
        stats,
    ).set({
        "acquisition_datetime": image.date().format(
            "YYYY-MM-dd HH:mm:ss"
        ),
        "acquisition_date": image.date().format(
            "YYYY-MM-dd"
        ),
        "system_index": image.get("system:index"),
        "source_dataset": image.get(
            "SPACECRAFT_ID"
        ),
        "sensor": "Landsat-8/9",
    })


def collect_landsat_lst(
    block_geometry,
    start_date,
    end_date,
):
    """
    Collect Landsat LST in 90-day Earth Engine batches.
    """
    print("\n[3/3] Landsat LST")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    batch_days = BATCH_DAYS
    all_parts = []
    batch_start = start

    while batch_start < end:
        batch_end = min(
            batch_start + pd.Timedelta(days=batch_days),
            end,
        )

        batch_start_str = batch_start.strftime("%Y-%m-%d")
        batch_end_str = batch_end.strftime("%Y-%m-%d")

        print(
            f"  Batch: {batch_start_str} -> "
            f"{batch_end_str}"
        )

        collections = [
            (
                ee.ImageCollection(collection_id)
                .filterBounds(block_geometry)
                .filterDate(
                    batch_start_str,
                    batch_end_str,
                )
                .filter(
                    ee.Filter.eq(
                        "PROCESSING_LEVEL",
                        "L2SP",
                    )
                )
                .map(mask_landsat)
            )
            for collection_id in LANDSAT_COLLECTIONS
        ]

        collection = collections[0].merge(
            collections[1]
        )

        count = collection.size().getInfo()

        print(
            f"    Candidate scenes: {count}"
        )

        if count > 0:
            features = collection.map(
                lambda img: landsat_feature(
                    img,
                    block_geometry,
                )
            )

            try:
                info = features.getInfo()
            except Exception as exc:
                raise RuntimeError(
                    "Landsat LST batch computation timed out or failed "
                    f"for {batch_start_str} -> {batch_end_str}."
                ) from exc

            rows = [
                feature.get("properties", {})
                for feature in info.get(
                    "features",
                    [],
                )
            ]

            if rows:
                all_parts.append(
                    pd.DataFrame(rows)
                )

        batch_start = batch_end

    if not all_parts:
        print("No Landsat LST observations available.")
        return pd.DataFrame()

    df = pd.concat(
        all_parts,
        ignore_index=True,
    )

    return normalize_satellite_df(
        df,
        "Landsat-8/9",
    )


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_satellite_df(df, sensor):
    df = df.copy()

    if "acquisition_datetime" in df.columns:
        df["acquisition_datetime"] = pd.to_datetime(
            df["acquisition_datetime"],
            errors="coerce",
        )

    if "acquisition_date" in df.columns:
        df["acquisition_date"] = pd.to_datetime(
            df["acquisition_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    # Convert numeric environmental columns.
    exclude = {
        "acquisition_datetime",
        "acquisition_date",
        "system_index",
        "source_dataset",
        "sensor",
        "orbit_pass",
    }

    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["sensor"] = sensor

    # Remove completely empty statistics rows. We do NOT impute them.
    stat_cols = [
        c for c in df.columns
        if c not in exclude
    ]

    if stat_cols:
        df = df.dropna(
            subset=stat_cols,
            how="all",
        )

    # Keep acquisition-level records. Same date may legitimately contain
    # multiple Sentinel acquisitions/orbits.
    return df.reset_index(drop=True)


# ============================================================
# ARCHIVE
# ============================================================


def append_archive(
    path: Path,
    new_df: pd.DataFrame,
    sensor: str,
):
    if path.exists():
        existing = pd.read_csv(path)
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing.copy()
    else:
        combined = pd.concat(
            [existing, new_df],
            ignore_index=True,
        )

    if combined.empty:
        return combined

    # Prefer exact acquisition identity.
    keys = [
        "sensor",
        "system_index",
    ]

    available_keys = [
        k for k in keys
        if k in combined.columns
    ]

    if available_keys:
        combined = combined.drop_duplicates(
            subset=available_keys,
            keep="last",
        )
    else:
        combined = combined.drop_duplicates()

    if "acquisition_datetime" in combined.columns:
        combined["acquisition_datetime"] = pd.to_datetime(
            combined["acquisition_datetime"],
            errors="coerce",
        )

        combined = combined.sort_values(
            "acquisition_datetime"
        )

        combined["acquisition_datetime"] = (
            combined["acquisition_datetime"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

    combined.to_csv(
        path,
        index=False,
    )

    return combined.reset_index(drop=True)


# ============================================================
# RAW CYCLE SNAPSHOT
# ============================================================


def save_raw_snapshot(
    df: pd.DataFrame,
    raw_dir: Path,
    sensor_name: str,
):
    if df.empty:
        return None

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    path = raw_dir / f"{sensor_name}_{stamp}.csv"

    df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# QUALITY AUDIT
# ============================================================


def audit_dataset(
    df: pd.DataFrame,
    sensor: str,
):
    print("\n" + "-" * 65)
    print(f"{sensor} AUDIT")
    print("-" * 65)

    if df.empty:
        print("No records.")
        return

    print(f"Rows: {len(df)}")

    if "acquisition_date" in df.columns:
        dates = pd.to_datetime(
            df["acquisition_date"],
            errors="coerce",
        ).dropna()

        if not dates.empty:
            print(
                f"Date range: {dates.min().date()} -> "
                f"{dates.max().date()}"
            )

    if "system_index" in df.columns:
        duplicate_count = df.duplicated(
            subset=["system_index"]
        ).sum()

        print(
            f"Duplicate acquisition IDs: "
            f"{duplicate_count}"
        )

        if duplicate_count:
            raise RuntimeError(
                f"{sensor}: duplicate acquisition IDs found."
            )

    print("Synthetic data: DISABLED")
    print("Interpolation: DISABLED")
    print("Audit: PASS")


# ============================================================
# METADATA
# ============================================================


def save_metadata(
    path: Path,
    state: str,
    district: str,
    block: str,
    start_date: str,
    end_date: str,
):
    metadata = {
        "state": state,
        "district": district,
        "block": block,
        "collection_date_range": {
            "requested_start": start_date,
            "requested_end_exclusive": end_date,
        },
        "datasets": {
            "sentinel2": {
                "collection": S2_COLLECTION,
                "variables": [
                    "NDVI",
                    "EVI",
                    "SAVI",
                ],
                "scale_m": S2_SCALE,
                "cloud_limit_percent": S2_CLOUD_LIMIT,
                "cloud_mask": "QA60 + SCL",
            },
            "sentinel1": {
                "collection": S1_COLLECTION,
                "variables": [
                    "VV",
                    "VH",
                ],
                "scale_m": S1_SCALE,
            },
            "landsat_lst": {
                "collections": LANDSAT_COLLECTIONS,
                "variable": "LST_C",
                "scale_m": LST_SCALE,
            },
        },
        "spatial_method": (
            "polygon zonal statistics over block boundary; "
            "not centroid-only extraction"
        ),
        "synthetic": False,
        "fallback_used": False,
        "temporal_interpolation": False,
        "future_information_used": False,
        "availability_cutoff_note": (
            "Downstream forecasting dataset must restrict satellite "
            "observations to those available before forecast issuance."
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================


def main():
    (
        state,
        district,
        block,
        location_dir,
        metadata_path,
        boundary_path,
        raw_dir,
        processed_dir,
    ) = get_location()

    initialize_gee()

    block_geometry = load_block_geometry(
        boundary_path
    )

    s2_path = (
        processed_dir
        / "sentinel2_block.csv"
    )

    s1_path = (
        processed_dir
        / "sentinel1_block.csv"
    )

    lst_path = (
        processed_dir
        / "landsat_lst_block.csv"
    )

    # Each sensor gets its own incremental start date.
    s2_start = choose_start_date(s2_path)
    s1_start = choose_start_date(s1_path)
    lst_start = choose_start_date(lst_path)

    end_date = choose_end_date()

    print("\n" + "=" * 70)
    print("SIH26 — BLOCK SATELLITE COLLECTION")
    print("=" * 70)
    print(f"State   : {state}")
    print(f"District: {district}")
    print(f"Block   : {block}")
    print(f"Boundary: {boundary_path}")
    print()
    print("Sentinel-2 start:", s2_start)
    print("Sentinel-1 start:", s1_start)
    print("Landsat LST start:", lst_start)
    print("End date (exclusive):", end_date)
    print()
    print("Polygon zonal statistics: ENABLED")
    print("Synthetic/fallback data: DISABLED")
    print("Interpolation: DISABLED")

    # --------------------------------------------------------
    # Sentinel-2
    # --------------------------------------------------------

    s2_new = collect_sentinel2(
        block_geometry,
        s2_start,
        end_date,
    )

    s2_raw = save_raw_snapshot(
        s2_new,
        raw_dir / "sentinel2",
        "sentinel2",
    )

    s2_archive = append_archive(
        s2_path,
        s2_new,
        "Sentinel-2",
    )

    audit_dataset(
        s2_archive,
        "Sentinel-2",
    )

    # --------------------------------------------------------
    # Sentinel-1
    # --------------------------------------------------------

    s1_new = collect_sentinel1(
        block_geometry,
        s1_start,
        end_date,
    )

    s1_raw = save_raw_snapshot(
        s1_new,
        raw_dir / "sentinel1",
        "sentinel1",
    )

    s1_archive = append_archive(
        s1_path,
        s1_new,
        "Sentinel-1",
    )

    audit_dataset(
        s1_archive,
        "Sentinel-1",
    )

    # --------------------------------------------------------
    # Landsat LST
    # --------------------------------------------------------

    lst_new = collect_landsat_lst(
        block_geometry,
        lst_start,
        end_date,
    )

    lst_raw = save_raw_snapshot(
        lst_new,
        raw_dir / "landsat_lst",
        "landsat_lst",
    )

    lst_archive = append_archive(
        lst_path,
        lst_new,
        "Landsat-8/9",
    )

    audit_dataset(
        lst_archive,
        "Landsat LST",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_out = (
        processed_dir
        / "block_satellite_collection_metadata.json"
    )

    save_metadata(
        metadata_out,
        state,
        district,
        block,
        min(
            s2_start,
            s1_start,
            lst_start,
        ),
        end_date,
    )

    print("\n" + "=" * 70)
    print("BLOCK SATELLITE COLLECTION COMPLETE")
    print("=" * 70)

    print("\nProcessed:")
    print("  Sentinel-2:", s2_path)
    print("  Sentinel-1:", s1_path)
    print("  Landsat LST:", lst_path)
    print("  Metadata:", metadata_out)

    print("\nRaw snapshots:")
    print("  Sentinel-2:", s2_raw)
    print("  Sentinel-1:", s1_raw)
    print("  Landsat LST:", lst_raw)

    print("\nNo synthetic or interpolated values were generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
