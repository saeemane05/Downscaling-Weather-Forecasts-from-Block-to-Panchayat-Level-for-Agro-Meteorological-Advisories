"""
SIH26 — STAGE 09: GP NOAA GFS FORECASTS — LOCAL + INCREMENTAL

Purpose
-------
Collect NOAA/NCEP GFS 0.25-degree historical forecast data through
Google Earth Engine, but DOWNLOAD THE RESULT DIRECTLY TO LOCAL STORAGE.

This version does NOT create Google Drive export tasks.

It is incremental:
    1. Reads the project's existing block forecast training dates.
    2. Uses only those dates (2024 onward).
    3. Stores completed local chunks under:
         gp/<State>/<District>/<Block>/raw/gfs_forecasts/
    4. On the next run, checks existing local chunk files and skips them.
    5. Only missing date/lead chunks are requested.
    6. Builds/updates:
         gp/<State>/<District>/<Block>/processed/gp_gfs_forecast_training_base.csv

GFS:
    Earth Engine catalog: NOAA/GFS0P25
    Resolution: 0.25 degree
    Cycle used: 00 UTC
    Leads: D1-D7

No synthetic, interpolated, or fabricated weather values are created.
Unavailable GFS dates remain absent and are recorded in metadata.

Requirements:
    pip install earthengine-api geopandas pandas requests

One-time authentication:
    earthengine authenticate
"""

from __future__ import annotations

import io
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# RUNTIME SAFETY GUARD
# ---------------------------------------------------------------------------
_THIS_SOURCE = Path(__file__).resolve()
_THIS_SOURCE_TEXT = _THIS_SOURCE.read_text(encoding="utf-8")
_FORBIDDEN_GFS_TERMS = ("pressure" + "_" + "surface",
                        "surface" + "_" + "pressure" + "_" + "forecast")
_found_forbidden = [x for x in _FORBIDDEN_GFS_TERMS if x in _THIS_SOURCE_TEXT]
if _found_forbidden:
    raise RuntimeError(
        "FATAL: forbidden GFS variable reference found in the file being run: "
        + ", ".join(_found_forbidden)
        + f"\\nFile: {_THIS_SOURCE}"
    )

print(f"Stage 09 source: {_THIS_SOURCE}")
print("GFS variables: temperature_2m_above_ground, relative_humidity_2m_above_ground, "
      "total_precipitation_surface, u_component_of_wind_10m_above_ground, "
      "v_component_of_wind_10m_above_ground")
print("Pressure variable: DISABLED")


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GFS_DATASET = "NOAA/GFS0P25"
START_DATE = date(2024, 1, 1)

# Conservative local-download chunks.
DATE_CHUNK_DAYS = 3
REQUEST_DELAY_SECONDS = 2.0
HTTP_TIMEOUT = 180

# Four final dynamic weather variables: temperature, RH, precipitation, wind speed.
SELECTORS = [
    "gp_id",
    "gp_name",
    "latitude",
    "longitude",
    "date",
    "forecast_issued_date",
    "forecast_lead_days",
    "forecast_issue_cycle",
    "forecast_source",
    "forecast_model",
    "temp_mean_forecast",
    "rh_mean_forecast",
    "precip_forecast",
    "wind_speed_forecast",
]

BASE_COLUMNS = [
    "gp_id",
    "gp_name",
    "latitude",
    "longitude",
    "date",
    "forecast_issued_date",
    "forecast_lead_days",
    "forecast_issue_cycle",
    "forecast_source",
    "forecast_model",
    "temp_mean_forecast",
    "rh_mean_forecast",
    "precip_forecast",
    "wind_speed_forecast",
]


def clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def resolve_gp_dir(state: str, district: str, block: str) -> Path:
    root = PROJECT_ROOT / "gp"

    direct = root / state / district / block
    if direct.exists():
        return direct

    if not root.exists():
        raise FileNotFoundError(f"Missing GP root: {root}")

    for s in root.iterdir():
        if not s.is_dir() or s.name.lower() != state.lower():
            continue
        for d in s.iterdir():
            if not d.is_dir() or d.name.lower() != district.lower():
                continue
            for b in d.iterdir():
                if b.is_dir() and b.name.lower() == block.lower():
                    return b

    raise FileNotFoundError(
        f"Could not resolve GP directory for {state}/{district}/{block}"
    )


def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception:
        print("\nEarth Engine is not authenticated.")
        print("Run once:")
        print("    earthengine authenticate")
        raise


def get_training_dates(gp_dir: Path) -> tuple[list[str], str]:
    state = gp_dir.parent.parent.name
    district = gp_dir.parent.name
    block = gp_dir.name

    candidates = [
        PROJECT_ROOT
        / "datasets"
        / "engineered"
        / state
        / district
        / block
        / "block"
        / "block_forecast_training_base_engineered.csv",
        PROJECT_ROOT
        / "datasets"
        / "cleaned"
        / state
        / district
        / block
        / "block"
        / "block_forecast_training_base.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path, usecols=["date"])
        dates = (
            pd.to_datetime(df["date"], errors="coerce")
            .dropna()
            .dt.date
            .drop_duplicates()
            .sort_values()
        )

        today = date.today()
        dates = [
            d.strftime("%Y-%m-%d")
            for d in dates
            if START_DATE <= d <= today
        ]

        if dates:
            return dates, f"training dates from {path}"

    raise FileNotFoundError(
        "Could not find the existing block forecast training table. "
        "This collector deliberately does not invent a calendar of target dates."
    )


def load_gp_points(gp_master: Path) -> ee.FeatureCollection:
    gdf = gpd.read_file(gp_master)

    required = {"gp_id", "gp_name"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(
            f"gp_master.geojson is missing required columns: {sorted(missing)}"
        )

    # Representative points are calculated in a projected CRS to avoid
    # geographic-CRS centroid/point warnings.
    projected = gdf.to_crs(gdf.estimate_utm_crs())
    points = projected.representative_point().to_crs(4326)

    features = []

    for (_, row), point in zip(gdf.iterrows(), points):
        features.append(
            ee.Feature(
                ee.Geometry.Point([float(point.x), float(point.y)]),
                {
                    "gp_id": str(row["gp_id"]),
                    "gp_name": str(row["gp_name"]),
                    "latitude": float(point.y),
                    "longitude": float(point.x),
                },
            )
        )

    return ee.FeatureCollection(features)


def date_chunks(dates: list[str]) -> list[list[str]]:
    """
    Split the actual requested dates into <= DATE_CHUNK_DAYS calendar-day
    windows. Missing calendar dates are NOT added.
    """
    parsed = sorted(
        {
            datetime.strptime(x, "%Y-%m-%d").date()
            for x in dates
        }
    )

    if not parsed:
        return []

    chunks = []
    start = 0

    while start < len(parsed):
        first = parsed[start]
        last = first + timedelta(days=DATE_CHUNK_DAYS - 1)

        chunk = []
        i = start
        while i < len(parsed) and parsed[i] <= last:
            chunk.append(parsed[i].isoformat())
            i += 1

        chunks.append(chunk)
        start = i

    return chunks


def gfs_daily_collection(valid_date: str, lead: int) -> ee.ImageCollection:
    """
    Select the GFS run issued at 00 UTC exactly `lead` days before the
    valid date and its forecast hours covering that valid UTC day.
    """
    valid = datetime.strptime(valid_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    issue = valid - timedelta(days=lead)

    issue_ms = int(issue.timestamp() * 1000)
    issue_end_ms = issue_ms + 6 * 60 * 60 * 1000

    valid_start_ms = int(valid.timestamp() * 1000)
    valid_end_ms = valid_start_ms + 24 * 60 * 60 * 1000

    return (
        ee.ImageCollection(GFS_DATASET)
        .filter(ee.Filter.gte("creation_time", issue_ms))
        .filter(ee.Filter.lt("creation_time", issue_end_ms))
        .filter(ee.Filter.gte("forecast_time", valid_start_ms))
        .filter(ee.Filter.lt("forecast_time", valid_end_ms))
    )


def build_daily_fc(
    gp_points: ee.FeatureCollection,
    valid_date: str,
    lead: int,
) -> ee.FeatureCollection:
    """
    Sample the selected GFS forecast at every GP representative point for
    all forecast hours belonging to the valid day, then aggregate to daily
    values.
    """

    collection = gfs_daily_collection(valid_date, lead)

    def add_wind(image):
        u = image.select("u_component_of_wind_10m_above_ground")
        v = image.select("v_component_of_wind_10m_above_ground")
        wind = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed_10m")
        return image.addBands(wind)

    collection = collection.map(add_wind)

    # ONLY bands present in the GFS collection are selected.
    # wind_speed_10m is derived from the two available wind components.
    bands = [
        "temperature_2m_above_ground",
        "relative_humidity_2m_above_ground",
        "total_precipitation_surface",
        "wind_speed_10m",
    ]

    def sample(image):
        return (
            image.select(bands)
            .sampleRegions(
                collection=gp_points,
                properties=[
                    "gp_id",
                    "gp_name",
                    "latitude",
                    "longitude",
                ],
                scale=27830,
                geometries=False,
            )
            .map(
                lambda f: f.set(
                    {
                        "date": valid_date,
                        "forecast_lead_days": lead,
                        "forecast_issued_date": (
                            datetime.strptime(valid_date, "%Y-%m-%d")
                            - timedelta(days=lead)
                        ).strftime("%Y-%m-%d"),
                    }
                )
            )
        )

    sampled = collection.map(sample).flatten()

    # Reduce hourly samples to one GP/day row.
    gp_ids = gp_points.aggregate_array("gp_id")

    def reduce_gp(gp_id):
        gp = gp_points.filter(ee.Filter.eq("gp_id", gp_id)).first()
        subset = sampled.filter(ee.Filter.eq("gp_id", gp_id))

        return ee.Feature(
            None,
            {
                "gp_id": gp_id,
                "gp_name": gp.get("gp_name"),
                "latitude": gp.get("latitude"),
                "longitude": gp.get("longitude"),
                "date": valid_date,
                "forecast_issued_date": (
                    datetime.strptime(valid_date, "%Y-%m-%d")
                    - timedelta(days=lead)
                ).strftime("%Y-%m-%d"),
                "forecast_lead_days": lead,
                "forecast_issue_cycle": "00Z",
                "forecast_source": "NOAA/NCEP GFS",
                "forecast_model": "GFS 0.25-degree",
                "temp_mean_forecast": subset.aggregate_mean(
                    "temperature_2m_above_ground"
                ),
                "rh_mean_forecast": subset.aggregate_mean(
                    "relative_humidity_2m_above_ground"
                ),
                "precip_forecast": subset.aggregate_sum(
                    "total_precipitation_surface"
                ),
                "wind_speed_forecast": subset.aggregate_mean(
                    "wind_speed_10m"
                ),
                    },
        )

    return ee.FeatureCollection(gp_ids.map(reduce_gp))


def build_chunk_fc(
    gp_points: ee.FeatureCollection,
    chunk_dates: list[str],
    lead: int,
) -> ee.FeatureCollection:
    collections = [
        build_daily_fc(gp_points, d, lead)
        for d in chunk_dates
    ]

    result = collections[0]
    for item in collections[1:]:
        result = result.merge(item)

    return result


def download_feature_collection(
    fc: ee.FeatureCollection,
    output_csv: Path,
) -> int:
    """
    Download a FeatureCollection directly to local storage.

    Earth Engine generates a temporary table download URL. No Google Drive
    export task is created.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("      Creating local Earth Engine download URL...")
    url = fc.getDownloadURL(
        filetype="CSV",
        selectors=SELECTORS,
    )

    print("      Downloading to local storage...")
    response = requests.get(
        url,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()

    output_csv.write_bytes(response.content)

    df = pd.read_csv(io.BytesIO(response.content))

    # Normalize the expected date type immediately.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    df.to_csv(output_csv, index=False)

    return len(df)


def chunk_key(chunk_dates: list[str], lead: int) -> str:
    return (
        f"d{lead}_"
        f"{chunk_dates[0].replace('-', '')}_"
        f"{chunk_dates[-1].replace('-', '')}"
    )


def existing_chunk_files(raw_dir: Path, lead: int) -> set[str]:
    pattern = f"gfs_d{lead}_*.csv"
    return {p.stem for p in raw_dir.glob(pattern)}


def validate_chunk(path: Path, expected_dates: list[str]) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False

        df = pd.read_csv(path)

        required = {
            "gp_id",
            "date",
            "forecast_lead_days",
            "temp_mean_forecast",
            "rh_mean_forecast",
            "precip_forecast",
            "wind_speed_forecast",
                }

        if not required.issubset(df.columns):
            return False

        if df.empty:
            return False

        found = set(
            pd.to_datetime(df["date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
        )

        # We accept partial GFS coverage, but require at least one requested
        # date to have actually returned.
        return bool(found.intersection(expected_dates))

    except Exception:
        return False


def merge_local_chunks(
    raw_dir: Path,
    processed_path: Path,
) -> pd.DataFrame:
    files = sorted(raw_dir.glob("gfs_d*.csv"))

    if not files:
        return pd.DataFrame(columns=BASE_COLUMNS)

    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"  WARNING: could not read {path.name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=BASE_COLUMNS)

    df = pd.concat(frames, ignore_index=True)

    # Normalize and deduplicate on the true dynamic key.
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    df["gp_id"] = df["gp_id"].astype(str)
    df["forecast_lead_days"] = pd.to_numeric(
        df["forecast_lead_days"],
        errors="coerce",
    ).astype("Int64")

    df = df.dropna(subset=["gp_id", "date", "forecast_lead_days"])

    df = df.drop_duplicates(
        subset=["gp_id", "date", "forecast_lead_days"],
        keep="last",
    )

    df = df.sort_values(
        ["date", "gp_id", "forecast_lead_days"]
    ).reset_index(drop=True)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    return df


RUN_VERSION = "STAGE09_REQUIRED_FEATURES_ONLY_V2"

def main():
    print("=" * 88)
    print("SIH26 — STAGE 09: GP NOAA GFS FORECASTS — REQUIRED FEATURES ONLY")
    print("=" * 88)

    state = clean_name(input("Enter State    : "))
    district = clean_name(input("Enter District : "))
    block = clean_name(input("Enter Block    : "))

    gp_dir = resolve_gp_dir(state, district, block)
    gp_master = gp_dir / "processed" / "gp_master.geojson"

    if not gp_master.exists():
        raise FileNotFoundError(
            f"GP master not found: {gp_master}"
        )

    raw_dir = gp_dir / "raw" / "gfs_forecasts"
    processed_dir = gp_dir / "processed"
    processed_path = (
        processed_dir / "gp_gfs_forecast_training_base.csv"
    )
    metadata_path = (
        gp_dir / "metadata" / "gp_gfs_forecast_metadata.json"
    )

    dates, date_mode = get_training_dates(gp_dir)
    chunks = date_chunks(dates)

    gdf = gpd.read_file(gp_master)

    print("=" * 88)
    print(f"Location          : {state} / {district} / {block}")
    print(f"GP master         : {gp_master}")
    print(f"GP count          : {len(gdf)}")
    print(f"Date source       : {date_mode}")
    print(f"Target dates      : {len(dates)}")
    print(f"Target period     : {dates[0]} -> {dates[-1]}")
    print(f"Date chunk        : {DATE_CHUNK_DAYS} days")
    print("Forecast leads    : D1-D7")
    print("GFS cycle         : 00 UTC")
    print("Resolution        : 0.25 degree")
    print("Storage           : LOCAL")
    print(f"Raw directory     : {raw_dir}")
    print(f"Processed output  : {processed_path}")
    print("=" * 88)

    initialize_ee()
    gp_points = load_gp_points(gp_master)

    raw_dir.mkdir(parents=True, exist_ok=True)
    (gp_dir / "metadata").mkdir(parents=True, exist_ok=True)

    completed = []
    requested = 0
    skipped = 0
    failed = 0

    for lead in range(1, 8):
        for chunk_no, chunk_dates in enumerate(chunks, start=1):
            requested += 1

            key = chunk_key(chunk_dates, lead)
            output = raw_dir / f"gfs_{key}.csv"

            if validate_chunk(output, chunk_dates):
                print(
                    f"SKIP  D{lead} chunk {chunk_no}/{len(chunks)} "
                    f"{chunk_dates[0]} -> {chunk_dates[-1]} "
                    f"(already downloaded)"
                )
                skipped += 1
                completed.append(output.name)
                continue

            print(
                f"\nFETCH D{lead} chunk {chunk_no}/{len(chunks)}: "
                f"{chunk_dates[0]} -> {chunk_dates[-1]}"
            )

            try:
                fc = build_chunk_fc(
                    gp_points,
                    chunk_dates,
                    lead,
                )

                rows = download_feature_collection(
                    fc,
                    output,
                )

                if rows == 0:
                    print(
                        "      No rows returned. "
                        "Chunk retained as unavailable/no-data."
                    )
                    if output.exists():
                        output.unlink()
                    failed += 1
                else:
                    print(
                        f"      Saved locally: {output.name} "
                        f"({rows:,} rows)"
                    )
                    completed.append(output.name)

            except Exception as exc:
                failed += 1
                if output.exists():
                    output.unlink()

                print(
                    f"      FAILED D{lead} "
                    f"{chunk_dates[0]} -> {chunk_dates[-1]}"
                )
                print(f"      Reason: {exc}")
                print(
                    "      Continuing with the next chunk. "
                    "No synthetic data were created."
                )

            if REQUEST_DELAY_SECONDS:
                time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 88)
    print("BUILDING LOCAL MERGED GP GFS DATASET")
    print("=" * 88)

    merged = merge_local_chunks(
        raw_dir,
        processed_path,
    )

    metadata = {
        "stage": "09_gp_gfs_forecasts_local_incremental",
        "state": state,
        "district": district,
        "block": block,
        "gp_count": len(gdf),
        "dataset": GFS_DATASET,
        "source": "NOAA/NCEP GFS via Google Earth Engine",
        "storage": "local",
        "date_source": date_mode,
        "requested_date_count": len(dates),
        "requested_start": dates[0],
        "requested_end": dates[-1],
        "date_chunk_days": DATE_CHUNK_DAYS,
        "leads": list(range(1, 8)),
        "issue_cycle": "00Z",
        "resolution": "0.25 degree",
        "requested_chunks": requested,
        "skipped_existing_chunks": skipped,
        "failed_chunks": failed,
        "completed_chunk_count": len(completed),
        "merged_rows": int(len(merged)),
        "merged_unique_gps": (
            int(merged["gp_id"].nunique())
            if not merged.empty
            else 0
        ),
        "merged_unique_dates": (
            int(merged["date"].nunique())
            if not merged.empty
            else 0
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "incremental_rule": (
            "Existing valid local chunk CSVs are skipped. "
            "Missing/invalid chunks are downloaded."
        ),
        "no_synthetic_data": True,
        "no_interpolation": True,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Rows in merged dataset : {len(merged):,}")
    print(
        f"Unique GP IDs          : "
        f"{merged['gp_id'].nunique() if not merged.empty else 0}"
    )
    print(
        f"Unique dates           : "
        f"{merged['date'].nunique() if not merged.empty else 0}"
    )
    print(f"Output                  : {processed_path}")
    print(f"Metadata                : {metadata_path}")
    print("=" * 88)
    print("STAGE 09 LOCAL/INCREMENTAL RUN COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
