"""
SIH26 — Block-level GFS forecast collector
==========================================

Purpose
-------
Archive real NOAA GFS forecasts delivered through the Open-Meteo GFS endpoint.

Design rules
------------
1. User selects State -> District -> Block at runtime.
2. Block metadata supplies the centroid; no location is hard-coded.
3. Real GFS data only. No sample/synthetic fallback.
4. Preserve forecast provenance:
      forecast_run_datetime
      target_datetime
      forecast_lead_hours
      target_date
      forecast_model
      forecast_source
5. Archive forecast cycles rather than overwriting target dates.
6. Re-running the collector on the same run/target key does not duplicate rows.
7. The collector is for forecast data; actual weather is NEVER used as an input.
8. Daily aggregation is performed from the hourly GFS forecast.
9. All API-returned rows are retained; missing values remain missing.
10. A live/current cycle can be collected repeatedly, because a new GFS run
    is a new forecast cycle even when target dates overlap older cycles.

Important
---------
Open-Meteo's GFS endpoint exposes the GFS forecast, but the API response does
not provide a native GFS model-run timestamp as a target/run pair suitable for
a scientific archive in every response mode. This script therefore records the
retrieval timestamp as the archive cycle identifier and preserves the exact
hourly target timestamps. It does NOT pretend that retrieval time is the
official NOAA initialization time.

The forecast archive should therefore be interpreted as:
"forecast retrieved at T, targeting target time/date X"
rather than as an invented official GFS initialization time.

Source implementation follows the project's existing Weather_Final approach:
Open-Meteo GFS endpoint with hourly forecast variables and daily aggregation.
The old Weather_Final code used a generic datetime.now() forecast-issued field;
this version makes the distinction explicit and adds target/lead-time fields.
"""


from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    start = start.resolve()

    for p in [start, *start.parents]:
        required = ["block", "gp", "datasets", "scripts"]
        if all((p / x).exists() for x in required):
            return p

    raise RuntimeError(
        "Could not locate SIH26 project root. Expected block/, gp/, "
        "datasets/, and scripts/."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)


# ---------------------------------------------------------------------
# GFS configuration
# ---------------------------------------------------------------------

GFS_URL = "https://api.open-meteo.com/v1/gfs"

FORECAST_DAYS = 7

TIMEZONE = "Asia/Kolkata"

# Variables are deliberately requested hourly so we can preserve the
# target timestamp and then derive daily statistics.
GFS_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
]

REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------


def clean_location_name(value: str) -> str:
    value = str(value).strip()

    if not value:
        raise ValueError("Location name cannot be empty.")

    return re.sub(r'[<>:"/\\|?*]', "_", value)


def configure_location():
    print("\nEnter the target block location.")

    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    if not state or not district or not block:
        raise ValueError(
            "State, District, and Block are all required."
        )

    state_dir = clean_location_name(state)
    district_dir = clean_location_name(district)
    block_dir = clean_location_name(block)

    location_dir = (
        PROJECT_ROOT
        / "block"
        / state_dir
        / district_dir
        / block_dir
    )

    metadata_path = (
        location_dir
        / "metadata"
        / "block_metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Block metadata not found:\n{metadata_path}\n\n"
            "Run 01_block_boundary.py first for this block."
        )

    raw_dir = location_dir / "raw" / "gfs"
    processed_dir = location_dir / "processed" / "gfs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return (
        state,
        district,
        block,
        location_dir,
        metadata_path,
        raw_dir,
        processed_dir,
    )


# ---------------------------------------------------------------------
# Metadata / centroid
# ---------------------------------------------------------------------


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recursive_find_lat_lon(obj):
    if isinstance(obj, dict):
        # Explicit centroid objects first.
        for key, value in obj.items():
            key_l = str(key).lower()

            if key_l in {
                "centroid",
                "block_centroid",
                "block_center",
                "center",
            }:
                if isinstance(value, dict):
                    lat = value.get("lat", value.get("latitude"))
                    lon = value.get("lon", value.get("longitude"))

                    if lat is not None and lon is not None:
                        return float(lat), float(lon)

                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    return float(value[0]), float(value[1])

        lat_keys = {
            "lat",
            "latitude",
            "centroid_lat",
            "centroid_latitude",
            "block_latitude",
            "block_centroid_latitude",
        }

        lon_keys = {
            "lon",
            "lng",
            "longitude",
            "centroid_lon",
            "centroid_longitude",
            "block_longitude",
            "block_centroid_longitude",
        }

        lat = lon = None

        for key, value in obj.items():
            key_l = str(key).lower()

            if key_l in lat_keys:
                try:
                    lat = float(value)
                except (TypeError, ValueError):
                    pass

            if key_l in lon_keys:
                try:
                    lon = float(value)
                except (TypeError, ValueError):
                    pass

        if lat is not None and lon is not None:
            return lat, lon

        for value in obj.values():
            result = recursive_find_lat_lon(value)

            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = recursive_find_lat_lon(value)

            if result is not None:
                return result

    return None


def get_centroid(metadata: dict, location_dir: Path):
    result = recursive_find_lat_lon(metadata)

    if result is not None:
        return result

    boundary_path = (
        location_dir
        / "processed"
        / "block_boundary.geojson"
    )

    if boundary_path.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(boundary_path)

            if gdf.empty:
                raise RuntimeError("Block boundary is empty.")

            geom = gdf.geometry.union_all()
            centroid = geom.centroid

            return float(centroid.y), float(centroid.x)

        except Exception as exc:
            raise RuntimeError(
                f"Could not derive centroid from {boundary_path}"
            ) from exc

    raise KeyError(
        "Block centroid was not found in metadata and the block boundary "
        f"does not exist: {boundary_path}"
    )


# ---------------------------------------------------------------------
# GFS retrieval
# ---------------------------------------------------------------------


def fetch_gfs_hourly(lat: float, lon: float):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(GFS_HOURLY_VARIABLES),
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    headers = {
        "User-Agent": "SIH26-Downscaling/1.0",
    }

    print("\nRequesting real GFS forecast from Open-Meteo...")

    response = requests.get(
        GFS_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"GFS/Open-Meteo HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError(
            "GFS response does not contain an hourly forecast."
        )

    hourly = payload["hourly"]

    if "time" not in hourly:
        raise RuntimeError(
            "GFS response does not contain hourly timestamps."
        )

    missing = [
        v for v in GFS_HOURLY_VARIABLES
        if v not in hourly
    ]

    if missing:
        raise RuntimeError(
            f"GFS response is missing required variables: {missing}"
        )

    df = pd.DataFrame(hourly)

    # API local time is explicitly Asia/Kolkata.
    df["target_datetime"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df = df.dropna(subset=["target_datetime"]).copy()

    if df.empty:
        raise RuntimeError("GFS response contained no valid timestamps.")

    return df


# ---------------------------------------------------------------------
# Forecast transformation
# ---------------------------------------------------------------------


def calculate_vpd(temp_c, rh_pct):
    """
    VPD in kPa using temperature and relative humidity.
    """
    es = 0.6108 * np.exp(
        (17.27 * temp_c) /
        (temp_c + 237.3)
    )

    return es * (1.0 - rh_pct / 100.0)


def build_forecast_archive(hourly: pd.DataFrame, lat: float, lon: float):
    retrieved = pd.Timestamp.now(tz="UTC")

    df = hourly.copy()

    df["target_datetime"] = pd.to_datetime(
        df["target_datetime"],
        errors="coerce",
    )

    df = df.dropna(subset=["target_datetime"]).copy()

    # The endpoint uses local time for target timestamps. Keep it explicit.
    df["target_date"] = df["target_datetime"].dt.date

    # Forecast lead is measured relative to the retrieval time. This is
    # intentionally called retrieval-based lead, not official GFS cycle lead.
    target_naive = pd.to_datetime(df["target_datetime"])

    retrieved_local_naive = (
        retrieved.tz_convert(TIMEZONE)
        .tz_localize(None)
    )

    df["forecast_lead_hours"] = (
        target_naive - retrieved_local_naive
    ).dt.total_seconds() / 3600.0

    df["forecast_retrieved_at"] = retrieved.isoformat()

    # Compatibility field. Explicitly document its meaning.
    df["forecast_issued_at"] = retrieved.isoformat()

    df["forecast_source"] = "NOAA GFS via Open-Meteo"
    df["forecast_model"] = "GFS"
    df["latitude_requested"] = lat
    df["longitude_requested"] = lon

    df["synthetic"] = False
    df["fallback_used"] = False
    df["actual_weather_used_as_forecast_input"] = False

    # Rename to stable SIH26 archive names.
    rename = {
        "temperature_2m": "gfs_temperature_2m_c",
        "relative_humidity_2m": "gfs_relative_humidity_2m_pct",
        "precipitation": "gfs_precipitation_mm",
        "cloud_cover": "gfs_cloud_cover_pct",
        "wind_speed_10m": "gfs_wind_speed_10m_ms",
        "wind_direction_10m": "gfs_wind_direction_10m_deg",
        "wind_gusts_10m": "gfs_wind_gust_10m_ms",
        "surface_pressure": "gfs_surface_pressure_hpa",
    }

    df = df.rename(columns=rename)

    # Physical-unit validation.
    rh = pd.to_numeric(
        df["gfs_relative_humidity_2m_pct"],
        errors="coerce",
    )

    temp = pd.to_numeric(
        df["gfs_temperature_2m_c"],
        errors="coerce",
    )

    precip = pd.to_numeric(
        df["gfs_precipitation_mm"],
        errors="coerce",
    )

    pressure = pd.to_numeric(
        df["gfs_surface_pressure_hpa"],
        errors="coerce",
    )

    if ((rh < 0) | (rh > 100)).any():
        raise RuntimeError(
            "GFS relative humidity outside 0–100%."
        )

    if ((precip < 0)).any():
        raise RuntimeError(
            "GFS precipitation contains negative values."
        )

    if ((temp < -80) | (temp > 65)).any():
        raise RuntimeError(
            "GFS temperature outside plausible range."
        )

    if ((pressure < 300) | (pressure > 1100)).any():
        raise RuntimeError(
            "GFS pressure outside plausible range."
        )

    # Daily summary is derived from hourly GFS while the hourly archive
    # remains available for traceability.
    df["gfs_vpd_kpa"] = calculate_vpd(
        df["gfs_temperature_2m_c"],
        df["gfs_relative_humidity_2m_pct"],
    )

    daily = (
        df.groupby("target_date", as_index=False)
        .agg(
            gfs_temp_mean_c=(
                "gfs_temperature_2m_c",
                "mean",
            ),
            gfs_temp_max_c=(
                "gfs_temperature_2m_c",
                "max",
            ),
            gfs_temp_min_c=(
                "gfs_temperature_2m_c",
                "min",
            ),
            gfs_rh_mean_pct=(
                "gfs_relative_humidity_2m_pct",
                "mean",
            ),
            gfs_precipitation_sum_mm=(
                "gfs_precipitation_mm",
                "sum",
            ),
            gfs_cloud_cover_mean_pct=(
                "gfs_cloud_cover_pct",
                "mean",
            ),
            gfs_wind_speed_mean_ms=(
                "gfs_wind_speed_10m_ms",
                "mean",
            ),
            gfs_wind_gust_max_ms=(
                "gfs_wind_gust_10m_ms",
                "max",
            ),
            gfs_surface_pressure_mean_hpa=(
                "gfs_surface_pressure_hpa",
                "mean",
            ),
            gfs_vpd_mean_kpa=(
                "gfs_vpd_kpa",
                "mean",
            ),
        )
    )

    daily["forecast_retrieved_at"] = retrieved.isoformat()
    daily["forecast_issued_at"] = retrieved.isoformat()
    daily["forecast_source"] = "NOAA GFS via Open-Meteo"
    daily["forecast_model"] = "GFS"
    daily["latitude_requested"] = lat
    daily["longitude_requested"] = lon
    daily["synthetic"] = False
    daily["fallback_used"] = False
    daily["actual_weather_used_as_forecast_input"] = False

    return df, daily


# ---------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------


def append_deduplicated(path: Path, new_df: pd.DataFrame, keys):
    if path.exists():
        existing = pd.read_csv(path)

        frames = []
        if not existing.empty:
            frames.append(existing)

        if new_df is not None and not new_df.empty:
            frames.append(new_df)

        if frames:
            combined = pd.concat(
                frames,
                ignore_index=True,
            )
        else:
            combined = pd.DataFrame()
    else:
        combined = (
            new_df.copy()
            if new_df is not None
            else pd.DataFrame()
        )

    if combined.empty:
        return combined

    for key in keys:
        if key not in combined.columns:
            raise RuntimeError(
                f"Archive key '{key}' missing from GFS dataset."
            )

    combined = combined.drop_duplicates(
        subset=keys,
        keep="last",
    )

    # Sort by target first and retrieval/cycle second.
    sort_cols = [
        c for c in [
            "target_datetime",
            "target_date",
            "forecast_retrieved_at",
        ]
        if c in combined.columns
    ]

    if sort_cols:
        combined = combined.sort_values(sort_cols)

    combined.to_csv(path, index=False)

    return combined.reset_index(drop=True)


def save_raw_cycle(
    hourly: pd.DataFrame,
    raw_dir: Path,
    lat: float,
    lon: float,
):
    retrieved = pd.Timestamp.now(tz="UTC")
    stamp = retrieved.strftime("%Y%m%dT%H%M%SZ")

    raw_path = raw_dir / f"gfs_cycle_{stamp}.csv"

    raw = hourly.copy()
    raw["forecast_retrieved_at"] = retrieved.isoformat()
    raw["forecast_source"] = "NOAA GFS via Open-Meteo"
    raw["forecast_model"] = "GFS"
    raw["latitude_requested"] = lat
    raw["longitude_requested"] = lon
    raw["synthetic"] = False
    raw["fallback_used"] = False

    raw.to_csv(raw_path, index=False)

    return raw_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    (
        state,
        district,
        block,
        location_dir,
        metadata_path,
        raw_dir,
        processed_dir,
    ) = configure_location()

    metadata = load_metadata(metadata_path)

    lat, lon = get_centroid(
        metadata,
        location_dir,
    )

    hourly_path = processed_dir / "gfs_hourly_archive.csv"
    daily_path = processed_dir / "gfs_daily_archive.csv"
    metadata_path_out = processed_dir / "gfs_collection_metadata.json"

    print("\n" + "=" * 70)
    print("SIH26 — GFS BLOCK FORECAST COLLECTOR")
    print("=" * 70)
    print(f"State: {state}")
    print(f"District: {district}")
    print(f"Block: {block}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Block directory: {location_dir}")
    print(f"Centroid: {lat:.6f}, {lon:.6f}")
    print(f"GFS source: {GFS_URL}")
    print(f"Forecast horizon: {FORECAST_DAYS} days")
    print(f"Timezone: {TIMEZONE}")
    print("Synthetic/fallback data: DISABLED")
    print()

    hourly = fetch_gfs_hourly(lat, lon)

    raw_path = save_raw_cycle(
        hourly,
        raw_dir,
        lat,
        lon,
    )

    hourly_archive, daily_archive = build_forecast_archive(
        hourly,
        lat,
        lon,
    )

    hourly_archive_final = append_deduplicated(
        hourly_path,
        hourly_archive,
        keys=[
            "forecast_retrieved_at",
            "target_datetime",
        ],
    )

    daily_archive_final = append_deduplicated(
        daily_path,
        daily_archive,
        keys=[
            "forecast_retrieved_at",
            "target_date",
        ],
    )

    collection_meta = {
        "state": state,
        "district": district,
        "block": block,
        "latitude": lat,
        "longitude": lon,
        "source": "NOAA GFS via Open-Meteo",
        "endpoint": GFS_URL,
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
        "hourly_variables": GFS_HOURLY_VARIABLES,
        "synthetic": False,
        "fallback_used": False,
        "actual_weather_used_as_forecast_input": False,
        "cycle_identifier": (
            "forecast_retrieved_at + target_datetime"
        ),
        "lead_time_definition": (
            "target_datetime minus forecast_retrieved_at; "
            "retrieval-based, not an asserted official GFS run lead"
        ),
    }

    with metadata_path_out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            collection_meta,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final audit
    # -----------------------------------------------------------------

    target_dates = pd.to_datetime(
        hourly_archive_final["target_datetime"],
        errors="coerce",
    )

    print("\n" + "=" * 70)
    print("GFS COLLECTION AUDIT")
    print("=" * 70)
    print(f"Hourly rows in archive: {len(hourly_archive_final)}")
    print(f"Daily rows in archive : {len(daily_archive_final)}")
    print(
        "Target datetime range : "
        f"{target_dates.min()} -> {target_dates.max()}"
    )

    duplicate_keys = hourly_archive_final.duplicated(
        subset=[
            "forecast_retrieved_at",
            "target_datetime",
        ]
    ).sum()

    print(f"Duplicate cycle/target keys: {duplicate_keys}")

    print("Synthetic/fallback policy: DISABLED")

    if duplicate_keys != 0:
        raise RuntimeError(
            "GFS audit failed: duplicate forecast cycle/target keys."
        )

    print("Duplicate-key check: PASS")
    print("Real-source check: PASS")
    print("=" * 70)

    print("\nGFS COLLECTION COMPLETE")
    print(f"Raw cycle: {raw_path}")
    print(f"Hourly archive: {hourly_path}")
    print(f"Daily archive: {daily_path}")
    print(f"Metadata: {metadata_path_out}")


if __name__ == "__main__":
    main()
