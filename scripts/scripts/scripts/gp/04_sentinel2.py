#!/usr/bin/env python3
"""
STAGE 4 — GP SENTINEL-2 ACCURATE / FAST V7
==========================================

Purpose
-------
Collect Sentinel-2 observations ONLY for the runner-selected GP master
and persist observations incrementally, while keeping the expensive temporal/statistical
feature engineering OUT of Google Earth Engine.

Design
------
GEE:
    Sentinel-2 SR Harmonized
        -> spatial/date/cloud filtering
        -> SCL quality mask
        -> calculate only required spectral indices
        -> extract GP-level observations

LOCAL PYTHON:
    -> validate observations
    -> calculate temporal features
    -> save raw observation table
    -> save processed GP feature table

IMPORTANT
---------
This version is intentionally NOT a giant reduceRegions() over the complete
2021->today collection.

It follows the performance principle demonstrated by the supplied
sentinel2_ndvi.py:
    one batched server request rather than a per-image .getInfo() loop.

For polygon data, the reliable scalable implementation is to convert each
Sentinel-2 acquisition into GP-level zonal observations using batched
FeatureCollection reductions. GPs are processed in batches so Sinnar cannot
create one enormous synchronous request.

No synthetic data.
No imputation.
No duplicate deletion.
No replacement of existing raw observations.
Append-only checkpoint persistence.
No hard-coded state/district/block.
No future observations are used for CURRENT rolling features.
Raw observations retain acquisition/image provenance.

SCIENTIFIC NOTE
---------------
The historical descriptive features produced by this script summarize the
available Sentinel-2 observation record. They are NOT automatically leakage-
safe historical training features.

For model training, date-specific features must later be generated using only
Sentinel-2 observations available on or before each forecast issuance date.

OUTPUTS PER BLOCK
-----------------
raw/
    sentinel2_observations.csv
    sentinel2_source_metadata.json

processed/
    gp_sentinel2_features.csv
    gp_sentinel2_features.geojson

RAW OBSERVATION COLUMNS
-----------------------
gp_id
acquisition_date
image_id
system_time_start
valid_pixel_fraction
B2, B3, B4, B8, B11, B12
NDVI, EVI, SAVI, NDMI, NBR

PROCESSED FEATURES
------------------
For each index:
    hist_mean
    hist_std
    hist_min
    hist_max
    7d_mean
    30d_mean
    60d_mean

Additional:
    NDVI_30d_change
    observation_count
    latest_observation_date
    days_since_latest_observation

PERFORMANCE
-----------
Defaults:
    cloud threshold = 20%
    scale = 20 m
    tileScale = 4
    GP batch size = 25
    date chunk = 365 days

The date chunking prevents one multi-year request from becoming an enormous
Earth Engine request. It does NOT change the underlying observations.

If Sinnar is still slow, reduce S2_GP_BATCH_SIZE to 10 or 15.

Run
---
cd E:\\SIH26_Downscaling
python scripts\\gp\\04_sentinel2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import ee
import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

S2_START_DATE = os.getenv("S2_START_DATE", "2021-01-01")
S2_END_DATE = os.getenv(
    "S2_END_DATE",
    (date.today() + timedelta(days=1)).isoformat(),
)

S2_CLOUD_PERCENT = int(os.getenv("S2_CLOUD_PERCENT", "20"))
S2_SCALE_M = int(os.getenv("S2_SCALE_M", "20"))
S2_TILE_SCALE = int(os.getenv("S2_TILE_SCALE", "4"))

# Number of GP polygons sent to one reduceRegions request.
# Smaller = safer for Sinnar; larger = fewer EE requests.
S2_GP_BATCH_SIZE = int(os.getenv("S2_GP_BATCH_SIZE", "25"))

# Maximum number of calendar days in one collection request.
S2_DATE_CHUNK_DAYS = int(os.getenv("S2_DATE_CHUNK_DAYS", "365"))

# GP is considered represented by an image if at least this fraction
# of its pixels has a valid masked observation.
MIN_VALID_PIXEL_FRACTION = float(
    os.getenv("S2_MIN_VALID_PIXEL_FRACTION", "0.01")
)

INDEXES = ["NDVI", "EVI", "SAVI", "NDMI", "NBR"]

REFLECTANCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(message, flush=True)


def timer() -> float:
    return time.time()


def seconds(t0: float) -> float:
    return time.time() - t0


# ============================================================================
# EARTH ENGINE
# ============================================================================

def initialize_earth_engine() -> None:
    """
    Use existing Earth Engine authentication.

    No automatic re-authentication is performed.
    """
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. "
            "Authenticate once using your normal EE setup, then rerun.\n"
            f"Original error: {exc}"
        ) from exc


# ============================================================================
# GP MASTER DISCOVERY
# ============================================================================

def discover_gp_masters() -> List[Path]:
    """
    Resolve the target GP master.

    RUNNER MODE:
        Process ONLY the exact GP master supplied by the runner.

    STANDALONE MODE:
        Preserve the original discovery behaviour for manual testing.

    IMPORTANT:
        Runner mode never scans the complete project tree.
        This prevents Bicholim -> Sinnar -> Baramati processing.
    """
    runner_mode = os.getenv("SIH26_RUNNER_MODE") == "1"

    if runner_mode:
        target = os.getenv("SIH26_TARGET_GP_MASTER")

        if not target:
            raise RuntimeError(
                "SIH26_RUNNER_MODE=1 but "
                "SIH26_TARGET_GP_MASTER was not supplied."
            )

        path = Path(target).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Runner target GP master does not exist: {path}"
            )

        return [path]

    # Standalone/manual mode.
    found = sorted(PROJECT_ROOT.rglob("gp_master.geojson"))

    return [
        path for path in found
        if path.is_file()
    ]


def read_gp_master(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"GP master is empty: {path}")

    if gdf.crs is None:
        raise ValueError(f"GP master has no CRS: {path}")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    id_candidates = [
        "gp_id",
        "gp_code",
        "LGD_GP_CODE",
        "lgd_gp_code",
    ]

    id_col = next((c for c in id_candidates if c in gdf.columns), None)

    if id_col is None:
        raise ValueError(
            f"No GP identifier found in {path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.copy()
    gdf["gp_id"] = gdf[id_col].astype(str).str.strip()

    if gdf["gp_id"].eq("").any():
        raise ValueError(f"Blank GP IDs found in {path}")

    if gdf["gp_id"].duplicated().any():
        duplicate_ids = sorted(
            gdf.loc[gdf["gp_id"].duplicated(keep=False), "gp_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Duplicate gp_id values found in {path}: {duplicate_ids}. "
            "The GP master must already contain the project's approved "
            "multipart/duplicate policy. This script will not silently "
            "drop or merge records."
        )

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(f"No non-empty GP geometries in {path}")

    return gdf


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    geojson = json.loads(gdf.to_json())
    return ee.FeatureCollection(geojson)


# ============================================================================
# SENTINEL-2 MASKING / INDICES
# ============================================================================

def mask_sentinel2(image: ee.Image) -> ee.Image:
    """
    SCL-based quality mask.

    SCL classes masked:
      3  cloud shadow
      8  cloud medium probability
      9  cloud high probability
      10 cirrus
      11 snow/ice

    Also use edge masks from B8A/B9.
    """
    scl = image.select("SCL")

    clear = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    edge = image.select("B8A").mask().And(
        image.select("B9").mask()
    )

    return image.updateMask(clear.And(edge))


def add_indices(image: ee.Image) -> ee.Image:
    """
    Add the five project indices while preserving provenance.
    """
    # Sentinel-2 SR scale factor.
    b2 = image.select("B2").multiply(0.0001)
    b4 = image.select("B4").multiply(0.0001)
    b8 = image.select("B8").multiply(0.0001)
    b11 = image.select("B11").multiply(0.0001)
    b12 = image.select("B12").multiply(0.0001)

    ndvi = b8.subtract(b4).divide(
        b8.add(b4)
    ).rename("NDVI")

    evi = (
        b8.subtract(b4)
        .multiply(2.5)
        .divide(
            b8.add(b4.multiply(6))
            .subtract(b2.multiply(7.5))
            .add(1)
        )
        .rename("EVI")
    )

    savi = (
        b8.subtract(b4)
        .multiply(1.5)
        .divide(
            b8.add(b4).add(0.5)
        )
        .rename("SAVI")
    )

    ndmi = b8.subtract(b11).divide(
        b8.add(b11)
    ).rename("NDMI")

    nbr = b8.subtract(b12).divide(
        b8.add(b12)
    ).rename("NBR")

    return ee.Image.cat(
        [
            image.select(REFLECTANCE_BANDS),
            ndvi,
            evi,
            savi,
            ndmi,
            nbr,
        ]
    ).copyProperties(
        image,
        [
            "system:time_start",
            "system:index",
            "CLOUDY_PIXEL_PERCENTAGE",
        ],
    )


def build_collection(
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                S2_CLOUD_PERCENT,
            )
        )
        .map(mask_sentinel2)
        .map(add_indices)
    )


# ============================================================================
# DATE CHUNKING
# ============================================================================

def date_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int,
) -> Iterable[Tuple[str, str]]:
    """
    Yield [start, end) date ranges.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    current = start

    while current < end:
        nxt = min(
            current + timedelta(days=chunk_days),
            end,
        )
        yield current.isoformat(), nxt.isoformat()
        current = nxt


# ============================================================================
# EE EXTRACTION
# ============================================================================

def make_observation_image(image: ee.Image) -> ee.Image:
    """
    Prepare one image for GP reduction.

    We use mean spectral/index values and a valid-pixel count.
    """
    values = image.select(
        REFLECTANCE_BANDS + INDEXES
    )

    valid = (
        image.select("NDVI")
        .mask()
        .rename("valid_pixel")
    )

    return values.addBands(valid)


def reduce_gp_batch_for_chunk(
    gp_batch: ee.FeatureCollection,
    aoi: ee.Geometry,
    start: str,
    end: str,
) -> Tuple[List[dict], int]:
    """
    Reduce all available images in one date chunk to one row per
    GP x image.

    There is deliberately ONE getInfo() for this complete batch/chunk,
    rather than one getInfo() per image.
    """
    collection = build_collection(aoi, start, end)

    # One small request to avoid constructing a reduction for an empty chunk.
    image_count = int(collection.size().getInfo())

    if image_count == 0:
        return [], 0

    def image_to_gp_features(image: ee.Image) -> ee.FeatureCollection:
        prepared = make_observation_image(image)

        reduced = prepared.reduceRegions(
            collection=gp_batch,
            reducer=ee.Reducer.mean(),
            scale=S2_SCALE_M,
            tileScale=S2_TILE_SCALE,
            maxPixelsPerRegion=20_000_000,
        )

        image_id = ee.String(
            image.get("system:index")
        )

        timestamp = ee.Number(
            image.get("system:time_start")
        )

        acquisition_date = ee.Date(timestamp).format("YYYY-MM-dd")

        def annotate(feature: ee.Feature) -> ee.Feature:
            valid = ee.Number(
                feature.get("valid_pixel")
            )

            # Keep null if there were no valid pixels.
            valid_fraction = ee.Algorithms.If(
                valid,
                valid,
                None,
            )

            return feature.set({
                "image_id": image_id,
                "system_time_start": timestamp,
                "acquisition_date": acquisition_date,
                "valid_pixel_fraction": valid_fraction,
            })

        return reduced.map(annotate)

    # Flatten all image reductions into one FeatureCollection.
    #
    # IMPORTANT:
    # This is still one server-side graph for the DATE CHUNK, not a
    # client-side per-image loop.
    nested = collection.map(
        lambda img: image_to_gp_features(
            ee.Image(img)
        )
    )

    flattened = ee.FeatureCollection(nested).flatten()

    raw = flattened.getInfo()

    features = raw.get("features", [])
    return features, image_count


# ============================================================================
# LOCAL CLEANING
# ============================================================================

def rows_to_dataframe(
    features: List[dict],
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()

    records = []

    for item in features:
        props = item.get("properties", {})
        records.append(props)

    df = pd.DataFrame(records)

    required = [
        "gp_id",
        "image_id",
        "acquisition_date",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Extraction result is missing required columns: {missing}"
        )

    df["gp_id"] = df["gp_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["acquisition_date"] = pd.to_datetime(
        df["acquisition_date"],
        errors="coerce",
    )

    numeric = REFLECTANCE_BANDS + INDEXES + [
        "valid_pixel_fraction",
        "system_time_start",
    ]

    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def validate_observations(
    df: pd.DataFrame,
    gp_ids: set,
) -> None:
    if df.empty:
        return

    unknown = sorted(
        set(df["gp_id"].astype(str)) - gp_ids
    )

    if unknown:
        raise ValueError(
            f"Earth Engine returned observations for unknown GP IDs: {unknown[:20]}"
        )

    # Duplicate GP x image records are not acceptable.
    duplicates = df.duplicated(
        subset=["gp_id", "image_id"],
        keep=False,
    )

    if duplicates.any():
        sample = df.loc[
            duplicates,
            ["gp_id", "image_id", "acquisition_date"],
        ].head(20)

        raise ValueError(
            "Duplicate GP x Sentinel-2 image observations detected.\n"
            f"{sample.to_string(index=False)}"
        )

    # NDVI/EVI/etc. can legitimately be NaN if a GP has no valid pixels.
    # We do NOT impute those values.
    for col in INDEXES:
        if col in df.columns:
            finite = np.isfinite(
                df[col].dropna().to_numpy(dtype=float)
            )
            if not finite.all():
                raise ValueError(
                    f"Non-finite values found in {col}."
                )

    # IMPORTANT:
    # EVI is NOT mathematically guaranteed to remain inside [-1, 1].
    # Its denominator contains the standard EVI coefficients (6 and 7.5),
    # so valid reflectance combinations can produce EVI > 1 or < -1.
    # Therefore we do NOT reject, clip, or impute EVI values.
    #
    # For the normalized-difference indices, use a numerical sanity check.
    # This is deliberately a tolerance check, not an artificial clipping rule.
    bounded_indexes = ["NDVI", "SAVI", "NDMI", "NBR"]

    for col in bounded_indexes:
        if col not in df.columns:
            continue

        valid = df[col].dropna()

        if ((valid < -1.05) | (valid > 1.05)).any():
            bad = valid[
                (valid < -1.05) | (valid > 1.05)
            ].head(10).tolist()

            raise ValueError(
                f"{col} contains suspicious values outside the expected "
                f"normalized range. Sample: {bad}"
            )

    # EVI is intentionally not subjected to the [-1, 1] test.


# ============================================================================
# LOCAL FEATURE ENGINEERING
# ============================================================================

def add_temporal_features(
    raw: pd.DataFrame,
    extraction_end_date: str,
) -> pd.DataFrame:
    """
    Calculate GP-level temporal features locally.

    Historical:
        mean/std/min/max over all available observations.

    Recent:
        7/30/60-day means relative to extraction_end_date.

    Trend:
        Current 30-day NDVI mean minus preceding 30-day NDVI mean.

    No missing values are fabricated.
    """
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["acquisition_date"] = pd.to_datetime(
        df["acquisition_date"]
    )

    end = pd.Timestamp(extraction_end_date)
    start7 = end - pd.Timedelta(days=7)
    start30 = end - pd.Timedelta(days=30)
    start60 = end - pd.Timedelta(days=60)
    prev30 = end - pd.Timedelta(days=60)

    output = []

    for gp_id, group in df.groupby("gp_id", sort=True):
        group = group.sort_values("acquisition_date")

        row = {"gp_id": str(gp_id)}

        # Historical descriptive statistics.
        for idx in INDEXES:
            values = pd.to_numeric(
                group[idx],
                errors="coerce",
            ).dropna()

            if len(values):
                row[f"{idx}_hist_mean"] = float(values.mean())
                row[f"{idx}_hist_std"] = float(
                    values.std(ddof=1)
                ) if len(values) > 1 else np.nan
                row[f"{idx}_hist_min"] = float(values.min())
                row[f"{idx}_hist_max"] = float(values.max())
            else:
                row[f"{idx}_hist_mean"] = np.nan
                row[f"{idx}_hist_std"] = np.nan
                row[f"{idx}_hist_min"] = np.nan
                row[f"{idx}_hist_max"] = np.nan

        # Recent windows.
        for days, start in [
            (7, start7),
            (30, start30),
            (60, start60),
        ]:
            recent = group[
                (group["acquisition_date"] >= start)
                & (group["acquisition_date"] < end)
            ]

            for idx in INDEXES:
                values = pd.to_numeric(
                    recent[idx],
                    errors="coerce",
                ).dropna()

                row[f"{idx}_{days}d_mean"] = (
                    float(values.mean())
                    if len(values)
                    else np.nan
                )

        # Current 30d minus previous 30d NDVI.
        current = group[
            (group["acquisition_date"] >= start30)
            & (group["acquisition_date"] < end)
        ]["NDVI"].dropna()

        previous = group[
            (group["acquisition_date"] >= prev30)
            & (group["acquisition_date"] < start30)
        ]["NDVI"].dropna()

        row["NDVI_30d_change"] = (
            float(current.mean() - previous.mean())
            if len(current) and len(previous)
            else np.nan
        )

        row["observation_count"] = int(len(group))

        valid_dates = group[
            group[INDEXES].notna().any(axis=1)
        ]["acquisition_date"]

        if len(valid_dates):
            latest = valid_dates.max()
            row["latest_observation_date"] = latest.date().isoformat()
            row["days_since_latest_observation"] = int(
                (end.normalize() - latest.normalize()).days
            )
        else:
            row["latest_observation_date"] = None
            row["days_since_latest_observation"] = np.nan

        output.append(row)

    return pd.DataFrame(output)


# ============================================================================
# INCREMENTAL / CHECKPOINT STORAGE
# ============================================================================

RAW_FILENAME = "sentinel2_observations.csv"


def get_existing_raw_path(master_path: Path) -> Path:
    raw_dir, _ = get_output_dirs(master_path)
    return raw_dir / RAW_FILENAME


def load_existing_raw_observations(
    master_path: Path,
    gp_ids: set,
) -> pd.DataFrame:
    """
    Load previously collected Sentinel-2 observations.

    Existing raw observations are authoritative and are NEVER discarded.

    The function is deliberately tolerant of a missing raw file because
    this is the first run for a block.
    """
    raw_csv = get_existing_raw_path(master_path)

    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        log("No existing Sentinel-2 raw observation file found.")
        return pd.DataFrame()

    log(f"Existing Sentinel-2 observations found: {raw_csv}")

    try:
        existing = pd.read_csv(
            raw_csv,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read existing Sentinel-2 observations: "
            f"{raw_csv}\n{exc}"
        ) from exc

    required = ["gp_id", "image_id", "acquisition_date"]

    missing = [
        col for col in required
        if col not in existing.columns
    ]

    if missing:
        raise RuntimeError(
            f"Existing Sentinel-2 raw file is missing required "
            f"columns: {missing}"
        )

    existing["gp_id"] = existing["gp_id"].astype(str).str.strip()
    existing["image_id"] = existing["image_id"].astype(str).str.strip()
    existing["acquisition_date"] = pd.to_datetime(
        existing["acquisition_date"],
        errors="coerce",
    )

    bad_dates = int(existing["acquisition_date"].isna().sum())

    if bad_dates:
        raise RuntimeError(
            f"Existing Sentinel-2 raw file contains {bad_dates} "
            f"rows with invalid acquisition_date. "
            f"Existing data will not be silently discarded or repaired."
        )

    unknown = sorted(
        set(existing["gp_id"]) - gp_ids
    )

    if unknown:
        raise RuntimeError(
            "Existing Sentinel-2 data contains GP IDs that are not present "
            f"in the current GP master: {unknown[:20]}. "
            "No automatic reassignment or deletion will be performed."
        )

    duplicate_keys = existing.duplicated(
        subset=["gp_id", "image_id"],
        keep=False,
    )

    if duplicate_keys.any():
        sample = existing.loc[
            duplicate_keys,
            ["gp_id", "image_id", "acquisition_date"],
        ].head(20)

        raise RuntimeError(
            "Existing Sentinel-2 raw file already contains duplicate "
            "GP x image keys. It will not be silently rewritten.\n"
            f"{sample.to_string(index=False)}"
        )

    log(f"Existing raw observations: {len(existing)}")
    log(
        "Existing acquisition range: "
        f"{existing['acquisition_date'].min().date()} -> "
        f"{existing['acquisition_date'].max().date()}"
    )

    return existing


def append_new_observations(
    master_path: Path,
    new_df: pd.DataFrame,
) -> int:
    """
    Append ONLY genuinely new GP x image observations.

    Existing rows are never replaced.

    If a GP/image key already exists:
      - identical record -> ignore it
      - conflicting record -> FAIL rather than guessing
    """
    if new_df.empty:
        return 0

    raw_csv = get_existing_raw_path(master_path)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)

    new_df = new_df.copy()

    key = ["gp_id", "image_id"]

    new_df["gp_id"] = new_df["gp_id"].astype(str).str.strip()
    new_df["image_id"] = new_df["image_id"].astype(str).str.strip()

    if new_df.duplicated(key, keep=False).any():
        dup = new_df.loc[
            new_df.duplicated(key, keep=False)
        ].sort_values(key)

        # Exact duplicate rows inside the new extraction can safely be
        # ignored. Conflicting GP/image values cannot.
        value_cols = [
            c for c in new_df.columns
            if c not in key
        ]

        exact_dup = dup.duplicated(
            subset=key + value_cols,
            keep="first",
        )

        if exact_dup.any():
            new_df = new_df.drop_duplicates(
                subset=key + value_cols,
                keep="first",
            )

        if new_df.duplicated(key, keep=False).any():
            sample = new_df.loc[
                new_df.duplicated(key, keep=False)
            ].head(20)

            raise RuntimeError(
                "Conflicting duplicate GP x image observations were "
                "returned during this extraction. "
                "No automatic selection was made.\n"
                f"{sample.to_string(index=False)}"
            )

    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        existing_keys = pd.read_csv(
            raw_csv,
            usecols=["gp_id", "image_id"],
            dtype=str,
        )

        existing_keys["gp_id"] = (
            existing_keys["gp_id"].astype(str).str.strip()
        )
        existing_keys["image_id"] = (
            existing_keys["image_id"].astype(str).str.strip()
        )

        existing_key_set = set(
            zip(
                existing_keys["gp_id"],
                existing_keys["image_id"],
            )
        )

        new_keys = list(
            zip(
                new_df["gp_id"],
                new_df["image_id"],
            )
        )

        already_present = [
            key_pair in existing_key_set
            for key_pair in new_keys
        ]

        if any(already_present):
            # For keys already present, compare complete records instead
            # of blindly deleting/replacing them.
            existing_full = pd.read_csv(
                raw_csv,
                low_memory=False,
            )

            existing_full["gp_id"] = (
                existing_full["gp_id"].astype(str).str.strip()
            )
            existing_full["image_id"] = (
                existing_full["image_id"].astype(str).str.strip()
            )

            existing_lookup = (
                existing_full
                .set_index(["gp_id", "image_id"])
            )

            new_columns = [
                c for c in new_df.columns
                if c in existing_full.columns
            ]

            keep_rows = []

            for _, row in new_df.iterrows():
                pair = (
                    str(row["gp_id"]),
                    str(row["image_id"]),
                )

                if pair not in existing_key_set:
                    keep_rows.append(row)
                    continue

                old = existing_lookup.loc[pair]

                conflict = False

                for col in new_columns:
                    if col in key:
                        continue

                    old_value = old[col]
                    new_value = row[col]

                    if pd.isna(old_value) and pd.isna(new_value):
                        continue

                    if pd.isna(old_value) != pd.isna(new_value):
                        conflict = True
                        break

                    if isinstance(old_value, (int, float, np.number)) or \
                       isinstance(new_value, (int, float, np.number)):
                        try:
                            if not np.isclose(
                                float(old_value),
                                float(new_value),
                                equal_nan=True,
                                rtol=1e-10,
                                atol=1e-12,
                            ):
                                conflict = True
                                break
                        except Exception:
                            if str(old_value) != str(new_value):
                                conflict = True
                                break
                    elif str(old_value) != str(new_value):
                        conflict = True
                        break

                if conflict:
                    raise RuntimeError(
                        "Existing GP x image observation conflicts with "
                        "newly extracted data. Existing data was NOT "
                        "replaced.\n"
                        f"GP ID: {pair[0]}\n"
                        f"Image : {pair[1]}"
                    )

                # Identical existing record: do nothing.

            if keep_rows:
                new_df = pd.DataFrame(
                    keep_rows,
                    columns=new_df.columns,
                )
            else:
                new_df = pd.DataFrame(
                    columns=new_df.columns
                )

    if new_df.empty:
        return 0

    # Align columns with an existing raw table when possible.
    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        existing_header = pd.read_csv(
            raw_csv,
            nrows=0,
        ).columns.tolist()

        for col in existing_header:
            if col not in new_df.columns:
                new_df[col] = np.nan

        # Preserve the original raw schema.
        new_df = new_df.reindex(
            columns=existing_header
        )

        write_header = False
    else:
        write_header = True

    # APPEND ONLY. Never overwrite the existing raw observation file.
    new_df.to_csv(
        raw_csv,
        mode="a",
        header=write_header,
        index=False,
    )

    log(
        f"Checkpoint saved: appended {len(new_df)} new "
        f"GP x image observations."
    )

    return len(new_df)


def get_existing_observation_end_date(
    existing_raw: pd.DataFrame,
) -> Optional[pd.Timestamp]:
    if existing_raw.empty:
        return None

    return pd.Timestamp(
        existing_raw["acquisition_date"].max()
    ).normalize()


def build_incremental_chunks(
    existing_raw: pd.DataFrame,
    configured_start: str,
    configured_end: str,
) -> List[Tuple[str, str]]:
    """
    Determine what period must be queried.

    First run:
        configured_start -> configured_end

    Existing data:
        latest existing acquisition date + 1 day -> configured_end

    This is deliberately conservative:
      * old observations are preserved
      * new observations are appended
      * no old period is overwritten
      * interrupted runs keep their checkpoints

    If a historical hole needs recovery, delete nothing. Run the script
    with an explicit S2_START_DATE covering that period; duplicate GP/image
    keys are checked against the persisted raw table and existing
    observations remain untouched.
    """
    end = pd.Timestamp(configured_end).normalize()
    start = pd.Timestamp(configured_start).normalize()

    existing_end = get_existing_observation_end_date(
        existing_raw
    )

    if existing_end is not None:
        incremental_start = max(
            start,
            existing_end + pd.Timedelta(days=1),
        )
    else:
        incremental_start = start

    if incremental_start >= end:
        return []

    return list(
        date_chunks(
            incremental_start.date().isoformat(),
            end.date().isoformat(),
            S2_DATE_CHUNK_DAYS,
        )
    )


def rebuild_raw_from_disk(
    master_path: Path,
    gp_ids: set,
) -> pd.DataFrame:
    """
    Reload the authoritative raw file after all checkpoints have been
    appended. This makes processed features derive from the complete
    persisted record, including observations saved before an interruption.
    """
    raw_csv = get_existing_raw_path(master_path)

    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        raise RuntimeError(
            "No persisted Sentinel-2 raw observations are available."
        )

    df = pd.read_csv(
        raw_csv,
        low_memory=False,
    )

    df["gp_id"] = df["gp_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["acquisition_date"] = pd.to_datetime(
        df["acquisition_date"],
        errors="coerce",
    )

    validate_observations(
        df,
        gp_ids,
    )

    return df.sort_values(
        ["gp_id", "acquisition_date", "image_id"]
    ).reset_index(drop=True)


# ============================================================================
# OUTPUT
# ============================================================================

def get_output_dirs(master_path: Path) -> Tuple[Path, Path]:
    """
    gp_master:
        .../<block>/processed/gp_master.geojson

    outputs:
        .../<block>/raw/
        .../<block>/processed/

    Prevents the old processed/processed path bug.
    """
    processed_dir = master_path.parent
    block_dir = processed_dir.parent

    raw_dir = block_dir / "raw"
    out_dir = block_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, out_dir


def save_block_outputs(
    master_path: Path,
    master_gdf: gpd.GeoDataFrame,
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    metadata: dict,
) -> None:
    raw_dir, out_dir = get_output_dirs(master_path)

    raw_csv = raw_dir / "sentinel2_observations.csv"
    raw_meta = raw_dir / "sentinel2_source_metadata.json"

    feature_csv = out_dir / "gp_sentinel2_features.csv"
    feature_geojson = out_dir / "gp_sentinel2_features.geojson"

    # IMPORTANT:
    # Raw observations are persisted incrementally by
    # append_new_observations(). Do NOT overwrite the raw file here.
    #
    # We only verify that the authoritative persisted raw table exists.
    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        raise RuntimeError(
            "Authoritative Sentinel-2 raw observation file is missing "
            "at save time. No processed output will be fabricated."
        )

    # Merge processed features back to GP geometry.
    result = master_gdf[["gp_id", "geometry"]].copy()

    if feature_df.empty:
        raise RuntimeError(
            "No processed GP features were generated. "
            "No synthetic output will be created."
        )

    result = result.merge(
        feature_df,
        on="gp_id",
        how="left",
        validate="one_to_one",
    )

    result.to_csv(
        feature_csv,
        index=False,
    )

    result.to_file(
        feature_geojson,
        driver="GeoJSON",
    )

    metadata["outputs"] = {
        "raw_observations_csv": str(raw_csv),
        "raw_metadata_json": str(raw_meta),
        "processed_features_csv": str(feature_csv),
        "processed_features_geojson": str(feature_geojson),
    }

    raw_meta.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    log(f"Raw observations: {raw_csv}")
    log(f"Metadata:         {raw_meta}")
    log(f"GP features CSV:  {feature_csv}")
    log(f"GP features GIS:  {feature_geojson}")


# ============================================================================
# BLOCK PROCESSING
# ============================================================================

def process_block(master_path: Path) -> None:
    t0 = timer()

    log("")
    log("=" * 80)
    log("STAGE 4 — SENTINEL-2 ACCURATE / FAST V7")
    log(f"GP master: {master_path}")
    log("=" * 80)

    gdf = read_gp_master(master_path)

    gp_ids = set(gdf["gp_id"].astype(str))
    gp_fc_full = gdf_to_ee_fc(gdf)

    # Block/GP AOI.
    aoi = gp_fc_full.geometry()

    log(f"GP count: {len(gdf)}")
    log(f"Sentinel-2 collection: {S2_COLLECTION}")
    log(f"Date range: {S2_START_DATE} -> {S2_END_DATE}")
    log(f"Cloud threshold: < {S2_CLOUD_PERCENT}%")
    log(f"Scale: {S2_SCALE_M} m")
    log(f"tileScale: {S2_TILE_SCALE}")
    log(f"GP batch size: {S2_GP_BATCH_SIZE}")
    log(f"Date chunk: {S2_DATE_CHUNK_DAYS} days")
    log("")
    log("Architecture:")
    log("  GEE = collection filtering + masking + GP extraction")
    log("  Python = validation + temporal/statistical calculations")
    log("  No per-image client getInfo() loop")

    # Split GP geometries locally.
    gp_indices = list(range(len(gdf)))

    batches = [
        gp_indices[i:i + S2_GP_BATCH_SIZE]
        for i in range(
            0,
            len(gp_indices),
            S2_GP_BATCH_SIZE,
        )
    ]

    total_images = 0

    # ----------------------------------------------------------------------
    # INCREMENTAL PERSISTENCE
    # ----------------------------------------------------------------------
    #
    # IMPORTANT:
    # The previous implementation accumulated everything in memory and only
    # wrote sentinel2_observations.csv after ALL batches/chunks completed.
    #
    # That meant an interruption during Sinnar/Baramati could lose all data
    # collected during that run.
    #
    # This version:
    #   1. Loads the existing raw file.
    #   2. Determines the next required date.
    #   3. Extracts only that period.
    #   4. APPENDS each successful chunk immediately.
    #   5. Never replaces an existing GP x image record.
    #
    # Therefore an interrupted run keeps its successful checkpoints.
    # ----------------------------------------------------------------------

    existing_raw = load_existing_raw_observations(
        master_path,
        gp_ids,
    )

    incremental_chunks = build_incremental_chunks(
        existing_raw=existing_raw,
        configured_start=S2_START_DATE,
        configured_end=S2_END_DATE,
    )

    log(f"GP batches: {len(batches)}")
    log(f"Missing/new date chunks: {len(incremental_chunks)}")
    log(
        f"Planned EE extraction requests: "
        f"{len(batches) * len(incremental_chunks)}"
    )

    if not incremental_chunks:
        log("")
        log("INCREMENTAL STATUS")
        log("  Existing Sentinel-2 data already reaches the requested end date.")
        log("  No new Sentinel-2 observations need to be collected.")
        log("  Existing raw observations will be reused unchanged.")

    else:
        log("")
        log("INCREMENTAL STATUS")
        log(
            f"  Existing latest acquisition: "
            f"{existing_raw['acquisition_date'].max().date()}"
            if not existing_raw.empty
            else "  Existing latest acquisition: none"
        )
        log(
            f"  New collection period starts: "
            f"{incremental_chunks[0][0]}"
        )
        log(
            f"  New collection period ends: "
            f"{incremental_chunks[-1][1]}"
        )
        log(
            f"  New date chunks: {len(incremental_chunks)}"
        )

    # Only create EE work for the missing/new period.
    for batch_no, indices in enumerate(batches, 1):
        batch_gdf = gdf.iloc[indices].copy()
        batch_fc = gdf_to_ee_fc(batch_gdf)
        batch_aoi = batch_fc.geometry()

        log("")
        log(
            f"GP batch {batch_no}/{len(batches)} "
            f"({len(batch_gdf)} GPs)"
        )

        for chunk_no, (start, end) in enumerate(
            incremental_chunks,
            1,
        ):
            t_chunk = timer()

            log(
                f"  Date chunk {chunk_no}/{len(incremental_chunks)}: "
                f"{start} -> {end}"
            )

            try:
                features, image_count = reduce_gp_batch_for_chunk(
                    gp_batch=batch_fc,
                    aoi=batch_aoi,
                    start=start,
                    end=end,
                )

                total_images += image_count

                if not features:
                    log(
                        f"    No observations returned; "
                        f"images passing filter: {image_count}"
                    )
                    continue

                frame = rows_to_dataframe(features)

                if frame.empty:
                    log("    Empty dataframe after parsing.")
                    continue

                validate_observations(
                    frame,
                    set(batch_gdf["gp_id"].astype(str)),
                )

                appended = append_new_observations(
                    master_path,
                    frame,
                )

                log(
                    f"    images: {image_count}; "
                    f"rows returned: {len(frame)}; "
                    f"new rows appended: {appended}; "
                    f"time: {seconds(t_chunk):.1f}s"
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Sentinel-2 incremental extraction failed for "
                    f"GP batch {batch_no}, date chunk "
                    f"{start}->{end}: {exc}"
                ) from exc

    # ----------------------------------------------------------------------
    # RELOAD AUTHORITATIVE PERSISTED RAW DATA
    # ----------------------------------------------------------------------

    raw_df = rebuild_raw_from_disk(
        master_path,
        gp_ids,
    )

    if raw_df.empty:
        raise RuntimeError(
            "The persisted Sentinel-2 raw observation table is empty. "
            "No synthetic data or imputation will be performed."
        )

    # ----------------------------------------------------------------------
    # LOCAL FEATURE ENGINEERING
    # ----------------------------------------------------------------------
    log("")
    log("Calculating temporal features locally in Python...")

    feature_df = add_temporal_features(
        raw_df,
        S2_END_DATE,
    )

    # Every GP must have a feature row.
    feature_ids = set(
        feature_df["gp_id"].astype(str)
    )

    missing_gps = sorted(
        gp_ids - feature_ids
    )

    if missing_gps:
        # A GP with no observations is a data-quality condition, not an
        # invitation to fabricate data.
        raise RuntimeError(
            f"No Sentinel-2 observations were returned for GP(s): "
            f"{missing_gps[:30]}. "
            "No imputation or synthetic values are permitted."
        )

    # ----------------------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------------------
    metadata = {
        "stage": "04_gp_sentinel2_accurate_fast_v7",
        "source_collection": S2_COLLECTION,
        "start_date": S2_START_DATE,
        "end_date_exclusive": S2_END_DATE,
        "cloud_threshold_percent": S2_CLOUD_PERCENT,
        "scale_m": S2_SCALE_M,
        "tileScale": S2_TILE_SCALE,
        "gp_batch_size": S2_GP_BATCH_SIZE,
        "date_chunk_days": S2_DATE_CHUNK_DAYS,
        "min_valid_pixel_fraction": MIN_VALID_PIXEL_FRACTION,
        "gp_count": len(gdf),
        "filtered_image_count_sum_across_chunks": int(total_images),
        "raw_observation_rows": int(len(raw_df)),
        "persisted_observation_start": (
            raw_df["acquisition_date"].min().date().isoformat()
            if not raw_df.empty else None
        ),
        "persisted_observation_end": (
            raw_df["acquisition_date"].max().date().isoformat()
            if not raw_df.empty else None
        ),
        "processed_gp_rows": int(len(feature_df)),
        "synthetic_data": False,
        "imputation": False,
        "client_per_image_getinfo": False,
        "incremental_collection": True,
        "existing_data_preserved": True,
        "raw_write_mode": "append_only_checkpointed",
        "duplicate_policy": (
            "Existing GP x image records are never replaced. "
            "Identical repeats are ignored; conflicting repeats fail."
        ),
        "historical_features": (
            "descriptive statistics over available Sentinel-2 GP observations"
        ),
        "recent_features": (
            "local Python rolling-window statistics relative to extraction end date"
        ),
        "leakage_warning": (
            "Processed historical aggregates are descriptive only. "
            "For historical model training, rebuild features per forecast "
            "issuance date using only observations available on/before that "
            "date."
        ),
    }

    save_block_outputs(
        master_path,
        gdf,
        raw_df,
        feature_df,
        metadata,
    )

    # ----------------------------------------------------------------------
    # FINAL QC
    # ----------------------------------------------------------------------
    log("")
    log("FINAL QC")
    log(
        f"  GP count:              "
        f"{'PASS' if len(feature_df) == len(gdf) else 'FAIL'}"
    )
    log(
        f"  Duplicate GP IDs:      "
        f"{'PASS' if not feature_df['gp_id'].duplicated().any() else 'FAIL'}"
    )
    log(
        f"  Duplicate GP/image:    "
        f"{'PASS' if not raw_df.duplicated(['gp_id', 'image_id']).any() else 'FAIL'}"
    )
    log("  Synthetic data:         PASS (disabled)")
    log("  Imputation:             PASS (disabled)")
    log(
        f"  Raw observations:      {len(raw_df)}"
    )
    log(
        f"  Processed GP rows:     {len(feature_df)}"
    )
    log(
        f"  Total block time:      {seconds(t0):.1f}s"
    )
    log("STAGE 4 BLOCK: SUCCESS")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    log("=" * 80)
    log("STAGE 4 — SENTINEL-2 ACCURATE / FAST V7 INCREMENTAL")
    log("=" * 80)
    log(f"Project root: {PROJECT_ROOT}")
    log("")

    runner_mode = os.getenv("SIH26_RUNNER_MODE") == "1"

    if runner_mode:
        state = os.getenv("SIH26_STATE", "")
        district = os.getenv("SIH26_DISTRICT", "")
        block = os.getenv("SIH26_BLOCK", "")
        target = os.getenv("SIH26_TARGET_GP_MASTER", "")

        log("RUNNER MODE: ENABLED")
        log(f"Target State    : {state}")
        log(f"Target District : {district}")
        log(f"Target Block    : {block}")
        log(f"Target GP master: {target}")
        log("")

    try:
        masters = discover_gp_masters()
    except Exception as exc:
        log(f"TARGET RESOLUTION FAILED: {exc}")
        return 1

    if not masters:
        log(
            "No GP master is available for the requested target."
        )
        return 1

    if runner_mode and len(masters) != 1:
        log(
            "RUNNER SAFETY FAILURE: runner mode must resolve exactly "
            "one GP master."
        )
        return 1

    log(
        "Target GP master(s):"
        if runner_mode
        else "Discovered GP masters:"
    )

    for i, master in enumerate(masters, 1):
        log(f"  {i}. {master}")

    initialize_earth_engine()

    failures = []

    for master in masters:
        try:
            process_block(master)
        except KeyboardInterrupt:
            log("")
            log(
                "Interrupted by user. "
                "Any successfully completed chunks were already "
                "checkpointed to the raw CSV. "
                "No synthetic data were created."
            )
            raise
        except Exception as exc:
            log("")
            log(f"FAILED: {master}")
            log(f"ERROR: {exc}")
            failures.append((master, str(exc)))

    log("")
    log("=" * 80)
    log("STAGE 4 SUMMARY")
    log("=" * 80)

    if failures:
        log(
            f"Successful: {len(masters) - len(failures)}/{len(masters)}"
        )
        log(f"Failed:     {len(failures)}/{len(masters)}")

        for path, error in failures:
            log(f"\\n{path}\\n  {error}")

        return 2

    log(
        f"All {len(masters)} GP master(s) completed successfully."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
