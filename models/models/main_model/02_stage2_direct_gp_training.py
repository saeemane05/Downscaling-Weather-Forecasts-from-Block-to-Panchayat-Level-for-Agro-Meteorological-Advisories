"""
SIH26 — Stage 2 Direct GP Spatial Forecast Correction
======================================================

Dynamic block-level script.

Usage:
    python 02_stage2_direct_gp_training.py --state Maharashtra --district Nashik --block Sinnar

Historical GP reference target:
    Google Earth Engine -> ECMWF/ERA5_LAND/DAILY_AGGR

Stage-2 variables:
    temperature, humidity, precipitation, wind

Intentionally removed:
    pressure

Cloud cover is not trained because the selected ERA5-Land Daily Aggregated
bands do not provide a directly compatible daily cloud-cover target.

Important scientific point:
ERA5-Land is a reanalysis/reference field, NOT independent GP station truth.
Therefore Stage-2 metrics mean "correction toward ERA5-Land", not "station
accuracy at Gram Panchayat level".

Training strategy: four feature-specific multi-output models total. Each model receives the complete D1-D7 forecast vector for its own weather variable plus static GP spatial predictors and simultaneously learns D1-D7 ERA5-Land-minus-block-forecast residuals. Precipitation uses shared multi-output occurrence and amount models. The script does not modify the existing GP collection/merge pipeline.
It downloads the historical GP reference target directly inside Stage 2.
Historical ERA5-Land data are cached locally and fetched incrementally:
only missing GP/date combinations are requested on later runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Earth Engine API is not installed.\n"
        "Install with: pip install earthengine-api"
    ) from exc

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score


# =====================================================================
# CONFIGURATION
# =====================================================================

# Resolve from this file so the existing Stage-2 methodology can be invoked
# from any checkout, rather than only from the original developer path.
ROOT = Path(__file__).resolve().parents[2]
MAIN_MODEL_DIR = ROOT / "models" / "main_model"

START_DATE = pd.Timestamp("2024-01-19")
GEE_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
GEE_SCALE_M = 11132

# Incremental historical-target cache:
# previously downloaded ERA5-Land GP/day records are reused.
CACHE_FILENAME = "gp_era5_land_reference_targets.csv"
CACHE_META_FILENAME = "gp_era5_land_cache_metadata.json"

# Only request dates missing from the cache. This keeps later runs fast and
# avoids repeatedly querying Earth Engine for already stored historical data.

FORECAST_DAYS = range(1, 8)

# One model per weather variable. Each model learns across D1-D7.
# This replaces 28 independent horizon-specific models with 4 feature-specific
# residual models, while retaining all seven forecast horizons as predictors.
TRAINABLE_VARIABLES = [
    "temperature",
    "humidity",
    "precipitation",
    "wind",
]

FORECAST_COLUMN_PATTERNS = {
    "temperature": "temp_mean_forecast_d{d}",
    "humidity": "rh_mean_forecast_d{d}",
    "precipitation": "precip_forecast_d{d}",
    "wind": "wind_speed_forecast_d{d}",
    "cloud": "cloud_cover_forecast_d{d}",
}

CURRENT_FORECAST_COLUMNS = {
    "temperature": "gfs_temp_mean_c",
    "humidity": "gfs_rh_mean_pct",
    "precipitation": "gfs_precipitation_sum_mm",
    "wind": "gfs_wind_speed_mean_ms",
    "cloud": "gfs_cloud_cover_mean_pct",
}

GEE_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

TARGET_COLUMNS = {
    "temperature": "era5_temperature_c",
    "humidity": "era5_relative_humidity_pct",
    "precipitation": "era5_precipitation_mm",
    "wind": "era5_wind_speed_ms",
}


# =====================================================================
# CLI / PATHS
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH26 Stage 2 GP spatial forecast correction"
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--district", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument(
        "--gee-project",
        default=None,
        help="Optional Google Cloud project used by Earth Engine.",
    )
    return parser.parse_args()


def build_paths(state: str, district: str, block: str) -> Dict[str, Path]:
    gp_dir = ROOT / "datasets" / "cleaned" / state / district / block / "gp"
    block_dir = ROOT / "block" / state / district / block / "processed" / "merge"
    output = MAIN_MODEL_DIR / "stage2_direct_gp_training" / state / district / block

    return {
        "gp_features": gp_dir / "gp_merged_features.csv",
        "block_training": block_dir / "block_forecast_training_base.csv",
        "current_forecast": block_dir / "block_current_forecast.csv",
        "output": output,
    }


# =====================================================================
# BASIC UTILITIES
# =====================================================================

def normalize_gp_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def derive_relative_humidity(temp_c, dewpoint_c) -> np.ndarray:
    """
    RH from 2 m temperature and dew point using Magnus approximation.
    """
    temp_c = np.asarray(temp_c, dtype=float)
    dewpoint_c = np.asarray(dewpoint_c, dtype=float)

    a = 17.625
    b = 243.04

    es_t = np.exp((a * temp_c) / (b + temp_c))
    es_td = np.exp((a * dewpoint_c) / (b + dewpoint_c))

    rh = 100.0 * es_td / np.maximum(es_t, 1e-9)
    return np.clip(rh, 0.0, 100.0)


def initialize_earth_engine(project: str | None = None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("Earth Engine initialized.")
        return
    except Exception as first_error:
        print("Earth Engine initialization failed.")
        print(f"Reason: {first_error}")
        raise RuntimeError(
            "Could not initialize Earth Engine.\n"
            "Run `earthengine authenticate` once, then retry this Stage-2 command."
        ) from first_error


# =====================================================================
# INPUT DETECTION
# =====================================================================

def detect_gp_coordinates(gp_df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = [
        "landcover__centroid_lat",
        "centroid_lat",
        "latitude",
        "lat",
        "gp_lat",
    ]
    lon_candidates = [
        "landcover__centroid_lon",
        "centroid_lon",
        "longitude",
        "lon",
        "gp_lon",
    ]

    lat_col = next((c for c in lat_candidates if c in gp_df.columns), None)
    lon_col = next((c for c in lon_candidates if c in gp_df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find GP centroid latitude/longitude columns."
        )

    return lat_col, lon_col


def build_forecast_contract(block_df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    contract = {}

    for d in FORECAST_DAYS:
        contract[d] = {}
        missing = []

        for variable in TRAINABLE_VARIABLES:
            col = FORECAST_COLUMN_PATTERNS[variable].format(d=d)

            if col not in block_df.columns:
                missing.append((variable, col))
            else:
                contract[d][variable] = col

        if missing:
            raise ValueError(
                f"D{d}: missing required forecast columns: {missing}"
            )

    return contract


def select_static_gp_features(gp_df: pd.DataFrame) -> List[str]:
    """
    Stable spatial predictors only.

    Included:
      terrain, landcover, soil, coordinates and stable spatial numeric fields.

    Excluded:
      actual weather, forecast weather, targets, dates, metadata, LST and
      temporally varying satellite fields whose acquisition cutoff is not
      available in the merged GP table.
    """

    excluded_tokens = [
        "actual",
        "forecast",
        "target",
        "pressure",
        "lst__",
        "date",
        "time",
        "timestamp",
        "year",
        "month",
        "day",
        "issue",
        "retriev",
        "source",
        "model",
        "gp_id",
        "gp_name",
        "block_lgd",
        "block_code",
        "state",
        "district",
    ]

    spatial_prefixes = (
        "terrain__",
        "landcover__",
        "soil__",
        "elevation",
        "slope",
        "aspect",
        "curvature",
        "tpi",
        "tri",
    )

    selected = []

    for col in gp_df.columns:
        low = str(col).lower()

        if any(token in low for token in excluded_tokens):
            continue

        is_coordinate = low in {
            "latitude",
            "longitude",
            "lat",
            "lon",
            "gp_lat",
            "gp_lon",
            "landcover__centroid_lat",
            "landcover__centroid_lon",
        }

        is_spatial = low.startswith(spatial_prefixes)

        # Sentinel features are deliberately NOT included unless they have
        # been explicitly frozen with valid temporal provenance. The current
        # merged GP table does not provide that acquisition cutoff.
        if not (is_coordinate or is_spatial):
            continue

        if not pd.api.types.is_numeric_dtype(gp_df[col]):
            continue

        # Remove heavily incomplete predictors.
        if float(gp_df[col].isna().mean()) > 0.20:
            continue

        selected.append(col)

    if not selected:
        raise ValueError("No valid static GP spatial predictors remained.")

    # Preserve order while guaranteeing unique feature names.
    selected = list(
        dict.fromkeys(selected)
    )

    return selected


# =====================================================================
# GOOGLE EARTH ENGINE TARGET EXTRACTION
# =====================================================================

def make_gp_points(
    gp_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> ee.FeatureCollection:

    features = []

    for _, row in gp_df.iterrows():
        gp_id = normalize_gp_id(row["gp_id"])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue

        features.append(
            ee.Feature(
                ee.Geometry.Point([lon, lat]),
                {"gp_id": gp_id},
            )
        )

    if not features:
        raise ValueError("No valid GP coordinates found.")

    return ee.FeatureCollection(features)


def get_gee_latest_date() -> pd.Timestamp:
    collection = ee.ImageCollection(GEE_DATASET)
    latest = ee.Image(
        collection.sort("system:time_start", False).first()
    )

    millis = latest.get("system:time_start").getInfo()

    if millis is None:
        raise RuntimeError(
            "Could not determine the latest ERA5-Land date in Earth Engine."
        )

    return pd.to_datetime(int(millis), unit="ms").normalize()


def extract_gee_month(
    points: ee.FeatureCollection,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:

    collection = (
        ee.ImageCollection(GEE_DATASET)
        .filterDate(
            start.strftime("%Y-%m-%d"),
            end_exclusive.strftime("%Y-%m-%d"),
        )
        .select(GEE_BANDS)
    )

    def process_image(image):
        date_string = ee.Date(
            image.get("system:time_start")
        ).format("YYYY-MM-dd")

        reduced = image.reduceRegions(
            collection=points,
            reducer=ee.Reducer.first(),
            scale=GEE_SCALE_M,
        )

        return reduced.map(
            lambda feature: feature.set("date", date_string)
        )

    table = ee.FeatureCollection(
        collection.map(process_image).flatten()
    )

    info = table.getInfo()

    rows = [
        feature.get("properties", {})
        for feature in info.get("features", [])
    ]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def clean_gee_target(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    required = ["gp_id", "date"] + GEE_BANDS
    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(
            f"GEE extraction is missing expected fields: {missing}"
        )

    df = raw.copy()

    df["era5_temperature_c"] = (
        pd.to_numeric(df["temperature_2m"], errors="coerce")
        - 273.15
    )

    df["era5_dewpoint_c"] = (
        pd.to_numeric(df["dewpoint_temperature_2m"], errors="coerce")
        - 273.15
    )

    df["era5_relative_humidity_pct"] = derive_relative_humidity(
        df["era5_temperature_c"].to_numpy(),
        df["era5_dewpoint_c"].to_numpy(),
    )

    # ERA5-Land precipitation is metres of water depth.
    # Small negative values can occur because of GRIB packing; those are
    # physically impossible and are clipped to zero.
    precip_m = pd.to_numeric(
        df["total_precipitation_sum"],
        errors="coerce",
    )

    df["era5_precipitation_mm"] = np.maximum(precip_m, 0.0) * 1000.0

    u = pd.to_numeric(
        df["u_component_of_wind_10m"],
        errors="coerce",
    )
    v = pd.to_numeric(
        df["v_component_of_wind_10m"],
        errors="coerce",
    )

    df["era5_wind_speed_ms"] = np.sqrt(u * u + v * v)

    df["gp_id"] = df["gp_id"].map(normalize_gp_id)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    out = df[
        [
            "gp_id",
            "date",
            "era5_temperature_c",
            "era5_dewpoint_c",
            "era5_relative_humidity_pct",
            "era5_precipitation_mm",
            "era5_wind_speed_ms",
        ]
    ].copy()

    out.loc[
        ~out["era5_temperature_c"].between(-60, 60),
        "era5_temperature_c",
    ] = np.nan

    out.loc[
        ~out["era5_relative_humidity_pct"].between(0, 100),
        "era5_relative_humidity_pct",
    ] = np.nan

    out.loc[
        ~out["era5_precipitation_mm"].between(0, 1000),
        "era5_precipitation_mm",
    ] = np.nan

    out.loc[
        ~out["era5_wind_speed_ms"].between(0, 100),
        "era5_wind_speed_ms",
    ] = np.nan

    return (
        out.dropna(subset=["gp_id", "date"])
        .drop_duplicates(["gp_id", "date"])
        .sort_values(["date", "gp_id"])
        .reset_index(drop=True)
    )


def fetch_gp_historical_era5(
    gp_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    start_date: pd.Timestamp,
    requested_end_date: pd.Timestamp,
    output_dir: Path,
) -> pd.DataFrame:

    latest_available = get_gee_latest_date()
    actual_end = min(requested_end_date, latest_available)

    if actual_end < start_date:
        raise RuntimeError(
            "ERA5-Land does not cover the requested training period."
        )

    cache_path = output_dir / CACHE_FILENAME
    meta_path = output_dir / CACHE_META_FILENAME

    print()
    print("=" * 80)
    print("GEE ERA5-LAND HISTORICAL GP TARGET — INCREMENTAL MODE")
    print("=" * 80)
    print(f"Dataset              : {GEE_DATASET}")
    print(f"GP locations         : {gp_df['gp_id'].nunique()}")
    print(f"Requested start      : {start_date.date()}")
    print(f"Requested end        : {requested_end_date.date()}")
    print(f"Latest GEE date      : {latest_available.date()}")
    print(f"Usable target end    : {actual_end.date()}")
    print(f"Cache                : {cache_path}")

    # ---------------------------------------------------------------
    # Load existing cache
    # ---------------------------------------------------------------

    if cache_path.exists():
        print()
        print("Existing ERA5-Land cache found. Loading...")
        cached = pd.read_csv(cache_path)

        if not cached.empty:
            cached["gp_id"] = cached["gp_id"].map(normalize_gp_id)
            cached["date"] = pd.to_datetime(
                cached["date"],
                errors="coerce",
            )

            # The cache stores the CLEAN target schema, not the raw GEE
            # band schema. Older runs may also have raw GEE bands, so support
            # both forms safely.
            cached_target_columns = [
                "gp_id",
                "date",
                "era5_temperature_c",
                "era5_dewpoint_c",
                "era5_relative_humidity_pct",
                "era5_precipitation_mm",
                "era5_wind_speed_ms",
            ]

            if all(
                c in cached.columns
                for c in cached_target_columns
            ):
                cached = cached[cached_target_columns].copy()

                for c in cached_target_columns[2:]:
                    cached[c] = pd.to_numeric(
                        cached[c],
                        errors="coerce",
                    )

            else:
                raw_required = [
                    "gp_id",
                    "date",
                    *GEE_BANDS,
                ]

                if all(
                    c in cached.columns
                    for c in raw_required
                ):
                    cached = clean_gee_target(cached)
                else:
                    print(
                        "Existing cache has an unsupported schema. "
                        "It will be rebuilt from GEE."
                    )
                    cached = pd.DataFrame()

            expected_gps = set(
                gp_df["gp_id"].map(normalize_gp_id)
            )

            cached_gps = set(
                cached["gp_id"].dropna().unique()
            )

            # Keep only the current block's GPs and requested period.
            cached = cached[
                cached["gp_id"].isin(expected_gps)
                & cached["date"].between(
                    start_date,
                    actual_end,
                )
            ].copy()

            print(
                f"Cached rows available : {len(cached):,}"
            )
            print(
                f"Cached GP count       : "
                f"{cached['gp_id'].nunique():,}"
            )

        else:
            cached = pd.DataFrame()

    else:
        print()
        print("No existing ERA5-Land cache found.")
        cached = pd.DataFrame()

    # ---------------------------------------------------------------
    # Determine exact missing GP/date combinations
    # ---------------------------------------------------------------

    all_dates = pd.date_range(
        start=start_date,
        end=actual_end,
        freq="D",
    )

    expected_count = (
        gp_df["gp_id"].nunique()
        * len(all_dates)
    )

    if cached.empty:
        missing_dates = all_dates
        missing_gp_ids = set(
            gp_df["gp_id"].map(normalize_gp_id)
        )
        missing_pairs = expected_count
    else:
        cached_keys = set(
            zip(
                cached["gp_id"],
                cached["date"],
            )
        )

        expected_keys = (
            (gp_id, date)
            for gp_id in gp_df["gp_id"].map(normalize_gp_id)
            for date in all_dates
        )

        missing_keys = [
            (gp_id, date)
            for gp_id, date in expected_keys
            if (gp_id, date) not in cached_keys
        ]

        missing_pairs = len(missing_keys)

        missing_dates = sorted(
            {
                date
                for _, date in missing_keys
            }
        )

        missing_gp_ids = {
            gp_id
            for gp_id, _ in missing_keys
        }

    print()
    print(
        f"Expected GP-day records : {expected_count:,}"
    )
    print(
        f"Missing GP-day records  : {missing_pairs:,}"
    )

    if missing_pairs == 0:
        print("CACHE COMPLETE — no new Earth Engine extraction required.")

        final = cached.sort_values(
            ["date", "gp_id"]
        ).reset_index(drop=True)

        final.to_csv(
            cache_path,
            index=False,
        )

        return final

    print()
    print(
        "Only missing historical GP/day records will be downloaded."
    )

    # ---------------------------------------------------------------
    # GEE point collection
    # ---------------------------------------------------------------

    points = make_gp_points(
        gp_df[
            gp_df["gp_id"].isin(missing_gp_ids)
        ].copy(),
        lat_col,
        lon_col,
    )

    # ---------------------------------------------------------------
    # Extract missing dates in monthly chunks
    # ---------------------------------------------------------------

    new_chunks = []

    missing_dates = pd.DatetimeIndex(
        missing_dates
    )

    if missing_dates.empty:
        raise RuntimeError(
            "Missing GP/day records were detected but no missing dates "
            "were identified."
        )

    cursor = missing_dates.min()

    while cursor <= missing_dates.max():

        month_end = (
            cursor.to_period("M").end_time.normalize()
            + pd.Timedelta(days=1)
        )

        chunk_end = min(
            month_end,
            actual_end + pd.Timedelta(days=1),
        )

        print(
            f"Extracting missing period "
            f"{cursor.date()} -> "
            f"{(chunk_end - pd.Timedelta(days=1)).date()} ..."
        )

        last_error = None

        for attempt in range(1, 4):
            try:
                chunk = extract_gee_month(
                    points,
                    cursor,
                    chunk_end,
                )

                if not chunk.empty:
                    chunk = clean_gee_target(
                        chunk
                    )

                    # Keep only the exact requested GP/day range.
                    chunk = chunk[
                        chunk["gp_id"].isin(
                            missing_gp_ids
                        )
                        & chunk["date"].isin(
                            missing_dates
                        )
                    ].copy()

                print(
                    f"  New rows returned: "
                    f"{len(chunk):,}"
                )

                if not chunk.empty:
                    new_chunks.append(chunk)

                last_error = None
                break

            except Exception as exc:
                last_error = exc
                print(
                    f"  Attempt {attempt}/3 failed: {exc}"
                )

                if attempt < 3:
                    time.sleep(3 * attempt)

        if last_error is not None:
            raise RuntimeError(
                f"GEE extraction failed for "
                f"{cursor.date()} -> "
                f"{(chunk_end - pd.Timedelta(days=1)).date()}"
            ) from last_error

        cursor = chunk_end

    # ---------------------------------------------------------------
    # Merge cache + new data
    # ---------------------------------------------------------------

    if new_chunks:
        new_data = pd.concat(
            new_chunks,
            ignore_index=True,
        )
    else:
        new_data = pd.DataFrame()

    pieces = []

    if not cached.empty:
        pieces.append(cached)

    if not new_data.empty:
        pieces.append(new_data)

    if not pieces:
        raise RuntimeError(
            "No cached or newly downloaded ERA5-Land target data available."
        )

    combined = pd.concat(
        pieces,
        ignore_index=True,
    )

    combined["gp_id"] = combined[
        "gp_id"
    ].map(normalize_gp_id)

    combined["date"] = pd.to_datetime(
        combined["date"],
        errors="coerce",
    )

    combined = combined[
        combined["gp_id"].isin(
            set(gp_df["gp_id"].map(normalize_gp_id))
        )
        & combined["date"].between(
            start_date,
            actual_end,
        )
    ].copy()

    combined = (
        combined
        .drop_duplicates(
            ["gp_id", "date"],
            keep="last",
        )
        .sort_values(
            ["date", "gp_id"]
        )
        .reset_index(drop=True)
    )

    # Atomic-ish replacement: write a temporary file first.
    temp_path = cache_path.with_suffix(
        ".tmp.csv"
    )

    combined.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(
        cache_path
    )

    metadata = {
        "dataset": GEE_DATASET,
        "gp_count": int(
            gp_df["gp_id"].nunique()
        ),
        "first_date": (
            combined["date"].min().strftime("%Y-%m-%d")
            if not combined.empty
            else None
        ),
        "last_date": (
            combined["date"].max().strftime("%Y-%m-%d")
            if not combined.empty
            else None
        ),
        "rows": int(len(combined)),
        "latest_gee_date": latest_available.strftime(
            "%Y-%m-%d"
        ),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "incremental": True,
    }

    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Cache updated : {cache_path}"
    )
    print(
        f"Total cached rows : {len(combined):,}"
    )
    print(
        f"Coverage : "
        f"{combined['date'].min().date()} -> "
        f"{combined['date'].max().date()}"
    )

    return combined


# =====================================================================
# TRAINING FRAME
# =====================================================================

def create_training_frame(
    gp_df: pd.DataFrame,
    block_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    static_features: List[str],
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:

    if "target_date" in block_df.columns:
        block_df["date"] = pd.to_datetime(
            block_df["target_date"],
            errors="coerce",
        )
    else:
        block_df["date"] = pd.to_datetime(
            block_df["date"],
            errors="coerce",
        )

    d1_temp = FORECAST_COLUMN_PATTERNS["temperature"].format(d=1)

    forecast_dates = (
        block_df.loc[
            block_df[d1_temp].notna(),
            "date",
        ]
        .dropna()
        .drop_duplicates()
    )

    # Deduplicate columns because lat/lon may already be present inside
    # static_features. Duplicate dataframe column names make X[col] return a
    # DataFrame instead of a Series and cause pd.to_numeric() to fail.
    gp_columns = list(
        dict.fromkeys(
            ["gp_id", lat_col, lon_col] + static_features
        )
    )

    gp_static = gp_df[
        gp_columns
    ].copy()

    gp_static["gp_id"] = gp_static["gp_id"].map(normalize_gp_id)
    gp_static["_join"] = 1

    dates = pd.DataFrame({"date": forecast_dates})
    dates["_join"] = 1

    frame = gp_static.merge(
        dates,
        on="_join",
        how="inner",
    ).drop(columns="_join")

    block_cols = ["date"]

    for d in FORECAST_DAYS:
        for variable in TRAINABLE_VARIABLES:
            col = FORECAST_COLUMN_PATTERNS[variable].format(d=d)
            if col in block_df.columns:
                block_cols.append(col)

    block_small = (
        block_df[block_cols]
        .drop_duplicates("date")
    )

    frame = frame.merge(
        block_small,
        on="date",
        how="left",
    )

    frame = frame.merge(
        era5_df,
        on=["gp_id", "date"],
        how="left",
    )

    return frame


def chronological_split(frame: pd.DataFrame) -> Dict[str, pd.Index]:
    years = frame["date"].dt.year

    split = {
        "train": frame.index[years == 2024],
        "validation": frame.index[years == 2025],
        "test": frame.index[years == 2026],
    }

    for name, idx in split.items():
        if len(idx) == 0:
            raise ValueError(
                f"Chronological {name} split contains no rows."
            )

    return split


# =====================================================================
# MODELING — FOUR FEATURE-SPECIFIC MULTI-OUTPUT RESIDUAL MODELS
# =====================================================================

def make_multioutput_feature_matrix(
    frame: pd.DataFrame,
    indices: pd.Index,
    variable: str,
    spatial_features: List[str],
) -> pd.DataFrame:
    """
    One feature-specific model receives the complete D1-D7 forecast vector
    for its own weather variable plus static GP spatial predictors.
    """

    forecast_columns = [
        FORECAST_COLUMN_PATTERNS[variable].format(d=d)
        for d in FORECAST_DAYS
    ]

    # Keep each input column exactly once. This is important because the
    # coordinate columns can also be members of spatial_features.
    columns = list(
        dict.fromkeys(
            forecast_columns + spatial_features
        )
    )

    X = frame.loc[indices, columns].copy()

    if X.columns.duplicated().any():
        duplicated = X.columns[
            X.columns.duplicated()
        ].tolist()
        raise ValueError(
            "Duplicate feature columns reached the model input: "
            f"{duplicated}"
        )

    for col in columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    return X


def make_multioutput_residual_targets(
    frame: pd.DataFrame,
    indices: pd.Index,
    variable: str,
) -> pd.DataFrame:
    """
    Construct seven residual targets:

        residual_Dd = ERA5-Land GP target - block forecast_Dd

    One model therefore learns seven horizon-specific corrections at once.
    """

    target_col = TARGET_COLUMNS[variable]

    target_frame = pd.DataFrame(
        index=indices
    )

    if target_col not in frame.columns:
        raise ValueError(
            f"Missing target column for {variable}: {target_col}"
        )

    if frame.columns.tolist().count(target_col) > 1:
        raise ValueError(
            f"Duplicate target column detected: {target_col}"
        )

    for d in FORECAST_DAYS:
        forecast_col = FORECAST_COLUMN_PATTERNS[
            variable
        ].format(d=d)

        target_frame[
            f"residual_d{d}"
        ] = (
            pd.to_numeric(
                frame.loc[
                    indices,
                    target_col,
                ],
                errors="coerce",
            )
            -
            pd.to_numeric(
                frame.loc[
                    indices,
                    forecast_col,
                ],
                errors="coerce",
            )
        )

    return target_frame


def train_multioutput_regressor(
    X: pd.DataFrame,
    Y: pd.DataFrame,
):
    """
    One ExtraTrees model with seven output dimensions.

    sklearn ExtraTreesRegressor natively supports multi-output regression.
    """

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imp = imputer.fit_transform(X)

    model = ExtraTreesRegressor(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_imp,
        Y.to_numpy(dtype=float),
    )

    return model, imputer


def predict_multioutput(
    model_bundle,
    X: pd.DataFrame,
) -> np.ndarray:

    model, imputer = model_bundle

    return model.predict(
        imputer.transform(X)
    )


def train_feature_model(
    frame: pd.DataFrame,
    variable: str,
    spatial_features: List[str],
    split: Dict[str, pd.Index],
):
    """
    Train exactly one multi-output residual model for a weather variable.

    Example for temperature:

        X =
          temp D1..D7 forecasts
          + GP terrain
          + GP landcover
          + GP soil
          + GP coordinates

        Y =
          ERA5_GP - temp_D1
          ERA5_GP - temp_D2
          ...
          ERA5_GP - temp_D7
    """

    forecast_columns = [
        FORECAST_COLUMN_PATTERNS[variable].format(d=d)
        for d in FORECAST_DAYS
    ]

    target_col = TARGET_COLUMNS[variable]

    required = forecast_columns + [target_col]

    usable_mask = frame[
        required
    ].notna().all(axis=1)

    usable = set(
        frame.index[usable_mask]
    )

    train_idx = pd.Index(
        [
            i
            for i in split["train"]
            if i in usable
        ]
    )

    val_idx = pd.Index(
        [
            i
            for i in split["validation"]
            if i in usable
        ]
    )

    test_idx = pd.Index(
        [
            i
            for i in split["test"]
            if i in usable
        ]
    )

    if len(train_idx) < 50:
        raise ValueError(
            f"{variable}: only "
            f"{len(train_idx)} usable training rows."
        )

    X_train = make_multioutput_feature_matrix(
        frame,
        train_idx,
        variable,
        spatial_features,
    )

    Y_train = make_multioutput_residual_targets(
        frame,
        train_idx,
        variable,
    )

    model = train_multioutput_regressor(
        X_train,
        Y_train,
    )

    metrics = {
        "variable": variable,
        "model_type": (
            "feature_specific_multioutput_residual_model"
        ),
        "target_definition": (
            "ERA5-Land GP reference minus "
            "same-horizon block forecast"
        ),
        "target_source": (
            "ERA5-Land Daily Aggregated reanalysis"
        ),
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "n_estimators": 100,
        "n_spatial_features": len(
            spatial_features
        ),
        "n_forecast_features": len(
            forecast_columns
        ),
    }

    # ---------------------------------------------------------------
    # Horizon-specific validation/test evaluation
    # ---------------------------------------------------------------

    for split_name, idx in [
        ("validation", val_idx),
        ("test", test_idx),
    ]:

        if len(idx) == 0:
            continue

        X_split = make_multioutput_feature_matrix(
            frame,
            idx,
            variable,
            spatial_features,
        )

        predicted_residuals = predict_multioutput(
            model,
            X_split,
        )

        y_true = frame.loc[
            idx,
            target_col,
        ].astype(float).to_numpy()

        for position, d in enumerate(
            FORECAST_DAYS
        ):

            forecast_col = FORECAST_COLUMN_PATTERNS[
                variable
            ].format(d=d)

            raw = frame.loc[
                idx,
                forecast_col,
            ].astype(float).to_numpy()

            corrected = (
                raw
                + predicted_residuals[:, position]
            )

            raw_score = rmse(
                y_true,
                raw,
            )

            corrected_score = rmse(
                y_true,
                corrected,
            )

            metrics[
                f"{split_name}_d{d}_raw_rmse"
            ] = raw_score

            metrics[
                f"{split_name}_d{d}_corrected_rmse"
            ] = corrected_score

            metrics[
                f"{split_name}_d{d}_raw_mae"
            ] = float(
                mean_absolute_error(
                    y_true,
                    raw,
                )
            )

            metrics[
                f"{split_name}_d{d}_corrected_mae"
            ] = float(
                mean_absolute_error(
                    y_true,
                    corrected,
                )
            )

            metrics[
                f"{split_name}_d{d}_rmse_improvement_pct"
            ] = (
                (raw_score - corrected_score)
                / raw_score
                * 100.0
                if raw_score > 0
                else np.nan
            )

    return model, metrics


# =====================================================================
# PRECIPITATION — ONE MULTI-OUTPUT MODEL
# =====================================================================

def train_precipitation_feature_model(
    frame: pd.DataFrame,
    spatial_features: List[str],
    split: Dict[str, pd.Index],
):
    """
    One precipitation model across D1-D7.

    Because precipitation is zero-inflated, we retain a two-stage structure:
      1. occurrence model
      2. conditional amount model

    Each model is multi-output across D1-D7.
    """

    variable = "precipitation"

    forecast_columns = [
        FORECAST_COLUMN_PATTERNS[variable].format(d=d)
        for d in FORECAST_DAYS
    ]

    target_col = TARGET_COLUMNS[variable]

    required = forecast_columns + [target_col]

    usable_mask = frame[
        required
    ].notna().all(axis=1)

    usable = set(
        frame.index[usable_mask]
    )

    train_idx = pd.Index(
        [
            i
            for i in split["train"]
            if i in usable
        ]
    )

    val_idx = pd.Index(
        [
            i
            for i in split["validation"]
            if i in usable
        ]
    )

    test_idx = pd.Index(
        [
            i
            for i in split["test"]
            if i in usable
        ]
    )

    if len(train_idx) < 50:
        raise ValueError(
            "precipitation: insufficient training rows."
        )

    X_train = make_multioutput_feature_matrix(
        frame,
        train_idx,
        variable,
        spatial_features,
    )

    Y_train = make_multioutput_residual_targets(
        frame,
        train_idx,
        variable,
    )

    # Multi-output occurrence targets.
    occurrence_Y = pd.DataFrame(
        index=Y_train.index
    )

    # Multi-output amount targets.
    amount_Y = pd.DataFrame(
        index=Y_train.index
    )

    for position, d in enumerate(
        FORECAST_DAYS
    ):

        forecast_col = FORECAST_COLUMN_PATTERNS[
            variable
        ].format(d=d)

        y_abs = frame.loc[
            train_idx,
            target_col,
        ].astype(float)

        occurrence_Y[
            f"occurrence_d{d}"
        ] = (
            y_abs > 0.1
        ).astype(int)

        # Amount model learns the positive ERA5 reference amount.
        amount_Y[
            f"amount_d{d}"
        ] = y_abs

    # ---------------------------------------------------------------
    # Occurrence model
    # ---------------------------------------------------------------

    occurrence_imputer = SimpleImputer(
        strategy="median"
    )

    X_occ = occurrence_imputer.fit_transform(
        X_train
    )

    occurrence_model = ExtraTreesClassifier(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    occurrence_model.fit(
        X_occ,
        occurrence_Y.to_numpy(
            dtype=int
        ),
    )

    # ---------------------------------------------------------------
    # Conditional amount model
    # ---------------------------------------------------------------

    # Keep all rows for a stable multi-output amount model, but clip
    # predictions to non-negative values later.
    amount_imputer = SimpleImputer(
        strategy="median"
    )

    X_amount = amount_imputer.fit_transform(
        X_train
    )

    amount_model = ExtraTreesRegressor(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    amount_model.fit(
        X_amount,
        amount_Y.to_numpy(
            dtype=float
        ),
    )

    model = {
        "occurrence_model": occurrence_model,
        "occurrence_imputer": occurrence_imputer,
        "amount_model": amount_model,
        "amount_imputer": amount_imputer,
    }

    metrics = {
        "variable": "precipitation",
        "model_type": (
            "feature_specific_multioutput_occurrence_amount_model"
        ),
        "target_source": (
            "ERA5-Land Daily Aggregated reanalysis"
        ),
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "n_estimators": 100,
        "n_spatial_features": len(
            spatial_features
        ),
        "n_forecast_features": len(
            forecast_columns
        ),
    }

    def predict_precipitation(X):
        occurrence_probability = occurrence_model.predict_proba(
            occurrence_imputer.transform(X)
        )

        # For multi-output binary classification, sklearn returns a list
        # of arrays, one array per output.
        if isinstance(
            occurrence_probability,
            list,
        ):
            wet_probability = np.column_stack(
                [
                    probabilities[:, 1]
                    if probabilities.shape[1] > 1
                    else probabilities[:, 0]
                    for probabilities in occurrence_probability
                ]
            )
        else:
            # Defensive fallback for unusual sklearn versions.
            probabilities = np.asarray(
                occurrence_probability
            )

            if probabilities.ndim == 3:
                wet_probability = probabilities[:, :, 1]
            else:
                wet_probability = probabilities

        amount = np.maximum(
            amount_model.predict(
                amount_imputer.transform(X)
            ),
            0.0,
        )

        return wet_probability * amount

    for split_name, idx in [
        ("validation", val_idx),
        ("test", test_idx),
    ]:

        if len(idx) == 0:
            continue

        X_split = make_multioutput_feature_matrix(
            frame,
            idx,
            variable,
            spatial_features,
        )

        predicted = predict_precipitation(
            X_split
        )

        y_true = frame.loc[
            idx,
            target_col,
        ].astype(float).to_numpy()

        for position, d in enumerate(
            FORECAST_DAYS
        ):

            forecast_col = FORECAST_COLUMN_PATTERNS[
                variable
            ].format(d=d)

            raw = frame.loc[
                idx,
                forecast_col,
            ].astype(float).to_numpy()

            corrected = predicted[
                :,
                position,
            ]

            raw_score = rmse(
                y_true,
                raw,
            )

            corrected_score = rmse(
                y_true,
                corrected,
            )

            metrics[
                f"{split_name}_d{d}_raw_rmse"
            ] = raw_score

            metrics[
                f"{split_name}_d{d}_corrected_rmse"
            ] = corrected_score

            metrics[
                f"{split_name}_d{d}_rmse_improvement_pct"
            ] = (
                (raw_score - corrected_score)
                / raw_score
                * 100.0
                if raw_score > 0
                else np.nan
            )

    return model, metrics


# =====================================================================
# CURRENT FORECAST — SAME D1-D7 INPUT CONTRACT
# =====================================================================

def predict_precipitation_multioutput(
    model,
    X: pd.DataFrame,
) -> np.ndarray:

    occurrence_probability = model[
        "occurrence_model"
    ].predict_proba(
        model[
            "occurrence_imputer"
        ].transform(X)
    )

    if isinstance(
        occurrence_probability,
        list,
    ):
        wet_probability = np.column_stack(
            [
                probabilities[:, 1]
                if probabilities.shape[1] > 1
                else probabilities[:, 0]
                for probabilities in occurrence_probability
            ]
        )
    else:
        probabilities = np.asarray(
            occurrence_probability
        )

        if probabilities.ndim == 3:
            wet_probability = probabilities[:, :, 1]
        else:
            wet_probability = probabilities

    amount = np.maximum(
        model[
            "amount_model"
        ].predict(
            model[
                "amount_imputer"
            ].transform(X)
        ),
        0.0,
    )

    return wet_probability * amount


def predict_feature_model(
    model,
    X: pd.DataFrame,
    variable: str,
) -> np.ndarray:

    if variable == "precipitation":
        return predict_precipitation_multioutput(
            model,
            X,
        )

    return predict_multioutput(
        model,
        X,
    )


def build_current_horizon_vector(
    current_df: pd.DataFrame,
    variable: str,
    target_position: int,
) -> Dict[str, float]:
    """
    Build exactly the same D1-D7 forecast feature structure used during
    training.

    current_df is expected to contain one row for each target date in the
    current 7-day forecast.
    """

    values = {}

    for d in FORECAST_DAYS:

        if d <= len(current_df):

            row = current_df.iloc[d - 1]

            current_col = CURRENT_FORECAST_COLUMNS[
                variable
            ]

            value = pd.to_numeric(
                row.get(
                    current_col,
                    np.nan,
                ),
                errors="coerce",
            )

        else:
            value = np.nan

        values[
            FORECAST_COLUMN_PATTERNS[
                variable
            ].format(d=d)
        ] = value

    return values



def apply_current_forecast(
    gp_df: pd.DataFrame,
    current_df: pd.DataFrame,
    static_features: List[str],
    models: Dict[str, object],
) -> pd.DataFrame:
    """
    Apply the SAME feature contract used during training to the current
    7-day block forecast.

    Each trained model is keyed by weather feature and produces seven
    residual outputs simultaneously.  The d-th output is added to the
    current block forecast for horizon Dd.

    Important safeguards:
      - exactly one inference implementation
      - feature columns are aligned to the fitted imputer's training schema
      - duplicate GP columns are rejected
      - non-finite predictions are rejected
      - NaN corrected forecasts are never silently written
    """

    current_df = current_df.copy()

    if "target_date" in current_df.columns:
        current_df["target_date"] = pd.to_datetime(
            current_df["target_date"], errors="coerce"
        )
    elif "date" in current_df.columns:
        current_df["target_date"] = pd.to_datetime(
            current_df["date"], errors="coerce"
        )
    else:
        raise ValueError(
            "Current forecast is missing both 'target_date' and 'date'."
        )

    if gp_df.columns.duplicated().any():
        duplicates = gp_df.columns[gp_df.columns.duplicated()].tolist()
        raise ValueError(
            f"GP input contains duplicate columns before inference: {duplicates}"
        )

    missing_static = [c for c in static_features if c not in gp_df.columns]
    if missing_static:
        raise ValueError(
            "Current GP input is missing static model features: "
            f"{missing_static}"
        )

    rows = []

    for d in FORECAST_DAYS:
        if d > len(current_df):
            raise ValueError(
                f"Current forecast has only {len(current_df)} rows; D{d} is missing."
            )

        fc = current_df.iloc[d - 1]

        for _, gp in gp_df.iterrows():

            record = {
                "gp_id": normalize_gp_id(gp["gp_id"]),
                "target_date": fc["target_date"],
                "forecast_horizon_day": d,
            }

            for variable in TRAINABLE_VARIABLES:

                current_col = CURRENT_FORECAST_COLUMNS[variable]

                raw_value = pd.to_numeric(
                    fc.get(current_col, np.nan),
                    errors="coerce",
                )

                record[f"block_{variable}_raw"] = raw_value

                if variable not in models:
                    raise RuntimeError(
                        f"Trained model for '{variable}' is missing."
                    )

                model_bundle = models[variable]

                # Build exactly the D1-D7 forecast vector used in training.
                values = build_current_horizon_vector(
                    current_df=current_df,
                    variable=variable,
                    target_position=d,
                )

                # Add the same static GP predictors used by training.
                for feature in static_features:
                    value = gp[feature]

                    # A duplicated GP field would make pandas return a DataFrame
                    # instead of a scalar; reject it explicitly.
                    if isinstance(value, pd.Series):
                        raise ValueError(
                            f"GP feature '{feature}' is duplicated in the input."
                        )

                    values[feature] = pd.to_numeric(
                        value,
                        errors="coerce",
                    )

                X = pd.DataFrame([values])

                # Numeric conversion.
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

                # ----------------------------------------------------------
                # CRITICAL: use the exact training feature schema.
                # The fitted SimpleImputer stores feature_names_in_.
                # Reindexing prevents order/name mismatch between training
                # and current inference.
                # ----------------------------------------------------------
                if variable == "precipitation":
                    imputer = model_bundle["occurrence_imputer"]
                else:
                    _, imputer = model_bundle

                if not hasattr(imputer, "feature_names_in_"):
                    raise RuntimeError(
                        f"{variable}: fitted imputer has no feature_names_in_. "
                        "Cannot safely guarantee training/inference alignment."
                    )

                training_columns = list(imputer.feature_names_in_)

                X = X.reindex(
                    columns=training_columns
                )

                if X.columns.tolist() != training_columns:
                    raise RuntimeError(
                        f"{variable}: failed to align current features to "
                        "the fitted training feature schema."
                    )

                # ----------------------------------------------------------
                # Predict with the exact fitted model bundle.
                # ----------------------------------------------------------
                prediction_matrix = predict_feature_model(
                    model_bundle,
                    X,
                    variable,
                )

                prediction_matrix = np.asarray(
                    prediction_matrix,
                    dtype=float,
                )

                if prediction_matrix.ndim == 1:
                    prediction_matrix = prediction_matrix.reshape(1, -1)

                if prediction_matrix.shape[0] != 1:
                    raise RuntimeError(
                        f"{variable}: unexpected prediction shape "
                        f"{prediction_matrix.shape}."
                    )

                if prediction_matrix.shape[1] < d:
                    raise RuntimeError(
                        f"{variable}: model returned only "
                        f"{prediction_matrix.shape[1]} outputs; D{d} requested."
                    )

                prediction = float(
                    prediction_matrix[0, d - 1]
                )

                if not np.isfinite(prediction):
                    raise RuntimeError(
                        f"{variable} D{d} GP {record['gp_id']}: "
                        f"model produced non-finite residual {prediction}."
                    )

                if not np.isfinite(raw_value):
                    raise RuntimeError(
                        f"{variable} D{d}: current block forecast "
                        f"'{current_col}' is non-finite."
                    )

                corrected = float(
                    raw_value + prediction
                )

                if not np.isfinite(corrected):
                    raise RuntimeError(
                        f"{variable} D{d} GP {record['gp_id']}: "
                        "corrected forecast is non-finite."
                    )

                record[
                    f"gp_{variable}_corrected"
                ] = corrected

            rows.append(record)

    result = pd.DataFrame(rows)

    # Physical constraints.
    result["gp_temperature_corrected"] = result[
        "gp_temperature_corrected"
    ].clip(-60, 60)

    result["gp_humidity_corrected"] = result[
        "gp_humidity_corrected"
    ].clip(0, 100)

    result["gp_precipitation_corrected"] = result[
        "gp_precipitation_corrected"
    ].clip(0, None)

    result["gp_wind_corrected"] = result[
        "gp_wind_corrected"
    ].clip(0, None)

    # Final hard audit.
    corrected_columns = [
        "gp_temperature_corrected",
        "gp_humidity_corrected",
        "gp_precipitation_corrected",
        "gp_wind_corrected",
    ]

    for col in corrected_columns:
        n_bad = int(
            (~np.isfinite(
                pd.to_numeric(result[col], errors="coerce")
            )).sum()
        )
        if n_bad:
            raise RuntimeError(
                f"Final GP forecast audit failed: {col} contains "
                f"{n_bad} non-finite values."
            )

    expected_rows = gp_df["gp_id"].nunique() * len(FORECAST_DAYS)

    if len(result) != expected_rows:
        raise RuntimeError(
            f"Final GP forecast row-count audit failed: "
            f"{len(result)} rows, expected {expected_rows}."
        )

    if result["gp_id"].nunique() != gp_df["gp_id"].nunique():
        raise RuntimeError(
            "Final GP forecast GP-count audit failed."
        )

    return result

# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    args = parse_args()

    paths = build_paths(
        args.state,
        args.district,
        args.block,
    )

    print("=" * 80)
    print("SIH26 — STAGE 2 DIRECT GP SPATIAL FORECAST CORRECTION")
    print("=" * 80)
    print(f"State    : {args.state}")
    print(f"District : {args.district}")
    print(f"Block    : {args.block}")
    print()
    print("Historical GP target : GEE ERA5-Land Daily Aggregated")
    print("Trainable variables  : temperature, humidity, precipitation, wind")
    print("Cloud cover          : NOT TRAINED")
    print("Pressure             : REMOVED")
    print()

    for key in [
        "gp_features",
        "block_training",
        "current_forecast",
    ]:
        if not paths[key].exists():
            raise FileNotFoundError(
                f"Required file not found:\n{paths[key]}"
            )

    paths["output"].mkdir(
        parents=True,
        exist_ok=True,
    )

    gp_df = pd.read_csv(
        paths["gp_features"]
    )

    block_df = pd.read_csv(
        paths["block_training"]
    )

    current_df = pd.read_csv(
        paths["current_forecast"]
    )

    print(
        f"GP dataset shape              : {gp_df.shape}"
    )
    print(
        f"Block forecast training shape : {block_df.shape}"
    )
    print(
        f"Current block forecast shape  : {current_df.shape}"
    )

    if "gp_id" not in gp_df.columns:
        raise ValueError(
            "GP dataset must contain gp_id."
        )

    gp_df["gp_id"] = gp_df[
        "gp_id"
    ].map(normalize_gp_id)

    gp_df = gp_df.drop_duplicates(
        "gp_id"
    ).reset_index(drop=True)

    lat_col, lon_col = detect_gp_coordinates(
        gp_df
    )

    print()
    print(
        f"GP count      : {gp_df['gp_id'].nunique()}"
    )
    print(
        f"Latitude      : {lat_col}"
    )
    print(
        f"Longitude     : {lon_col}"
    )

    if "target_date" in block_df.columns:
        block_df["date"] = pd.to_datetime(
            block_df["target_date"],
            errors="coerce",
        )
    else:
        block_df["date"] = pd.to_datetime(
            block_df["date"],
            errors="coerce",
        )

    block_df = block_df.dropna(
        subset=["date"]
    ).copy()

    forecast_contract = build_forecast_contract(
        block_df
    )

    print()
    print("FORECAST COLUMN CONTRACT")
    print("-" * 70)

    for d in FORECAST_DAYS:
        print(f"D{d}")
        for variable, column in forecast_contract[d].items():
            print(
                f"  {variable:<16} -> {column}"
            )

    static_features = select_static_gp_features(
        gp_df
    )

    print()
    print("STATIC GP SPATIAL FEATURE CONTRACT")
    print("-" * 70)
    print(
        f"Static spatial feature count : "
        f"{len(static_features)}"
    )

    for feature in static_features:
        print(
            f"  {feature}"
        )

    # Historical forecast period.
    historical_start = max(
        START_DATE,
        block_df["date"].min(),
    )

    historical_end = block_df[
        "date"
    ].max()

    print()
    print(
        f"Historical forecast dates : "
        f"{historical_start.date()} -> "
        f"{historical_end.date()}"
    )

    # ---------------------------------------------------------------
    # Earth Engine
    # ---------------------------------------------------------------

    initialize_earth_engine(
        args.gee_project
    )

    era5_df = fetch_gp_historical_era5(
        gp_df=gp_df,
        lat_col=lat_col,
        lon_col=lon_col,
        start_date=historical_start,
        requested_end_date=historical_end,
        output_dir=paths["output"],
    )

    era5_path = (
        paths["output"]
        / "gp_era5_land_reference_targets.csv"
    )

    era5_df.to_csv(
        era5_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Training frame
    # ---------------------------------------------------------------

    frame = create_training_frame(
        gp_df=gp_df,
        block_df=block_df,
        era5_df=era5_df,
        static_features=static_features,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    frame_path = (
        paths["output"]
        / "stage2_training_frame.csv"
    )

    frame.to_csv(
        frame_path,
        index=False,
    )

    print()
    print(
        f"Stage-2 training frame shape : "
        f"{frame.shape}"
    )

    # Critical input audit. Duplicate pandas column names were the cause of
    # the previous "arg must be a list, tuple, 1-d array, or Series" error:
    # pd.to_numeric() received a DataFrame when X[col] was duplicated.
    duplicate_columns = frame.columns[
        frame.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        print()
        print("DUPLICATE TRAINING-FRAME COLUMNS DETECTED:")
        for column in duplicate_columns:
            print(f"  {column}")
        raise ValueError(
            "Training frame contains duplicate column names. "
            f"Duplicates: {duplicate_columns}"
        )

    print(
        "Training-frame column uniqueness : PASS"
    )

    split = chronological_split(
        frame
    )

    print()
    print("CHRONOLOGICAL SPLIT")
    print("-" * 70)

    for name, indices in split.items():
        dates = frame.loc[
            indices,
            "date",
        ]

        print(
            f"{name:<12}: "
            f"{len(indices):,} rows | "
            f"{dates.min().date()} -> "
            f"{dates.max().date()}"
        )

    # ---------------------------------------------------------------
    # Train exactly one model per weather feature
    # ---------------------------------------------------------------

    models = {}
    metrics = []

    print()
    print("=" * 80)
    print("STAGE 2 — FOUR FEATURE-SPECIFIC MULTI-OUTPUT MODELS")
    print("=" * 80)
    print(
        "Model count: 4"
    )
    print(
        "Each model learns D1-D7 corrections simultaneously."
    )
    print(
        "No separate model is trained for each horizon."
    )

    for variable in TRAINABLE_VARIABLES:

        print()
        print(
            f"Training {variable} model..."
        )

        try:

            if variable == "precipitation":

                model, metric = (
                    train_precipitation_feature_model(
                        frame=frame,
                        spatial_features=static_features,
                        split=split,
                    )
                )

            else:

                model, metric = train_feature_model(
                    frame=frame,
                    variable=variable,
                    spatial_features=static_features,
                    split=split,
                )

            models[
                variable
            ] = model

            metrics.append(
                metric
            )

            print(
                f"  {variable}: COMPLETE"
            )

            for d in FORECAST_DAYS:

                key = (
                    f"test_d{d}_rmse_improvement_pct"
                )

                if key in metric:

                    print(
                        f"  D{d} test RMSE improvement: "
                        f"{metric[key]:.2f}%"
                    )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            metrics.append(
                {
                    "variable": variable,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

    if len(models) != len(TRAINABLE_VARIABLES):

        failed = [
            variable
            for variable in TRAINABLE_VARIABLES
            if variable not in models
        ]

        raise RuntimeError(
            "Not all Stage-2 feature models trained successfully. "
            f"Failed: {failed}"
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_path = (
        paths["output"]
        / "stage2_era5_reference_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Current 7-day GP forecast
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("APPLYING STAGE 2 TO CURRENT BLOCK FORECAST")
    print("=" * 80)

    # IMPORTANT:
    # models are keyed by weather feature (temperature/humidity/precipitation/wind).
    # The feature-specific multi-output apply_current_forecast() above is used.
    # A previous duplicate legacy function expected (horizon, feature) keys and
    # silently produced NaN outputs because those keys do not exist.
    print()
    print("=" * 80)
    print("CURRENT GP INFERENCE AUDIT")
    print("=" * 80)

    current_gp = apply_current_forecast(
        gp_df=gp_df,
        current_df=current_df,
        static_features=static_features,
        models=models,
    )

    print(f"GP rows generated : {len(current_gp):,}")
    print(f"Unique GPs        : {current_gp['gp_id'].nunique():,}")
    print(f"Forecast horizons : {sorted(current_gp['forecast_horizon_day'].unique())}")

    for variable in TRAINABLE_VARIABLES:
        col = f"gp_{variable}_corrected"
        print(
            f"{variable:<16}: "
            f"{current_gp[col].notna().sum():,}/{len(current_gp):,} finite"
        )

    print("Current GP inference : PASS")


    corrected_cols = [
        "gp_temperature_corrected",
        "gp_humidity_corrected",
        "gp_precipitation_corrected",
        "gp_wind_corrected",
    ]
    for col in corrected_cols:
        if col in current_gp.columns and current_gp[col].isna().any():
            raise RuntimeError(
                f"Current GP inference produced NaN values in {col}. "
                "Stopping instead of writing an incomplete forecast."
            )

    current_output_path = (
        paths["output"]
        / "stage2_current_gp_forecast.csv"
    )

    current_gp.to_csv(
        current_output_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------

    manifest = {
        "project": "SIH26",
        "stage": "Stage 2",
        "state": args.state,
        "district": args.district,
        "block": args.block,
        "method": (
            "Feature-specific residual correction of block NWP forecasts "
            "using static GP spatial predictors and "
            "ERA5-Land reanalysis reference targets"
        ),
        "gp_count": int(
            gp_df["gp_id"].nunique()
        ),
        "historical_target_source": GEE_DATASET,
        "historical_target_type": (
            "ERA5-Land reanalysis reference; "
            "not independent station truth"
        ),
        "historical_target_resolution_m": GEE_SCALE_M,
        "trainable_variables": TRAINABLE_VARIABLES,
        "model_count": len(models),
        "model_strategy": "four feature-specific multi-output residual models",
        "model_strategy": "one model per weather feature across D1-D7",
        "cloud_status": (
            "not trained because a compatible daily "
            "ERA5-Land cloud target was not used"
        ),
        "pressure_status": "removed",
        "synthetic_target": False,
        "interpolated_target": False,
        "block_actual_copied_to_gp": False,
        "lst_target_month_used": False,
        "chronological_split": {
            "train": 2024,
            "validation": 2025,
            "test": 2026,
        },
        "copernicus_acknowledgement": (
            "Generated using Copernicus Climate Change "
            "Service Information."
        ),
        "outputs": {
            "era5_reference": str(
                era5_path
            ),
            "training_frame": str(
                frame_path
            ),
            "metrics": str(
                metrics_path
            ),
            "current_gp_forecast": str(
                current_output_path
            ),
        },
    }

    manifest_path = (
        paths["output"]
        / "stage2_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print(
        f"ERA5 reference targets : {era5_path}"
    )
    print(
        f"Training frame         : {frame_path}"
    )
    print(
        f"Metrics                : {metrics_path}"
    )
    print(
        f"Current GP forecast    : "
        f"{current_output_path}"
    )
    print(
        f"Manifest               : {manifest_path}"
    )
    print()
    print(
        "Scientific note:"
    )
    print(
        "Stage-2 metrics measure correction toward "
        "ERA5-Land reanalysis, not independent GP "
        "station-observation accuracy."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 80)
        print("STAGE 2 FAILED")
        print("=" * 80)
        print(str(exc))
        sys.exit(1)
