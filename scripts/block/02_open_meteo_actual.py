"""
SIH26 Weather Downscaling
=========================

02 - Open-Meteo Historical Actual Weather Collector

COLLECTION POLICY
-----------------
1. First run:
       Fetch data from exactly 5 years before today
       through today.

2. Subsequent runs:
       Read the existing processed CSV.
       Find the latest collected date.
       Fetch only dates after that date through today.

3. Existing records are never duplicated.

4. Raw API responses are retained.

5. No synthetic/fallback values are generated.

6. If the source has not yet published today's data,
   only the dates actually returned by the source are saved.

TIMEZONE
--------
Asia/Kolkata

DATA TYPE
---------
Historical actual/reanalysis weather.

This is NOT the historical forecast dataset.
"""

from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import json
import sys
import time

import pandas as pd
import geopandas as gpd
import requests


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.common.geo import clean_name, save_json


# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

PROJECT_TIMEZONE = "Asia/Kolkata"

# Exactly 5 years back from today.
HISTORY_YEARS = 5

# API request chunk size.
CHUNK_DAYS = 180

REQUEST_TIMEOUT = 120

REQUEST_SLEEP_SECONDS = 0.5


# ============================================================
# WEATHER VARIABLES
# ============================================================

DAILY_VARIABLES = [

    # Temperature
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",

    # Relative humidity
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",

    # Precipitation
    "precipitation_sum",
    "rain_sum",

    # Wind
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",

    # Radiation
    "shortwave_radiation_sum",

    # Cloud
    "cloud_cover_mean",

    # Pressure
    "surface_pressure_mean",

    # Evapotranspiration
    "et0_fao_evapotranspiration",
]


# ============================================================
# CURRENT DATE
# ============================================================

def get_today():
    """
    Get today's date in the project timezone.

    This avoids depending on the timezone of the machine
    running the script.
    """

    return datetime.now(
        ZoneInfo(PROJECT_TIMEZONE)
    ).date()


def get_five_year_start(today):
    """
    Return the date exactly 5 calendar years before today.

    Handles leap-year cases safely.
    """

    try:

        return today.replace(
            year=today.year - HISTORY_YEARS
        )

    except ValueError:

        # Example:
        # 2024-02-29 -> 2019-02-28

        return today.replace(
            year=today.year - HISTORY_YEARS,
            day=28
        )


# ============================================================
# DATE CHUNKS
# ============================================================

def generate_chunks(
    start_date,
    end_date,
    chunk_days=CHUNK_DAYS
):
    """
    Generate inclusive date chunks.
    """

    current = start_date

    while current <= end_date:

        chunk_end = min(
            current + timedelta(
                days=chunk_days - 1
            ),
            end_date
        )

        yield current, chunk_end

        current = (
            chunk_end +
            timedelta(days=1)
        )


# ============================================================
# PATHS
# ============================================================

def get_block_paths(
    state,
    district,
    block
):
    """
    Construct all paths for this Block.
    """

    base = (
        Path("block")
        / clean_name(state)
        / clean_name(district)
        / clean_name(block)
    )

    return {

        "base":
            base,

        "boundary":
            base
            / "processed"
            / "block_boundary.geojson",

        "raw":
            base
            / "raw"
            / "open_meteo",

        "processed":
            base
            / "processed",

        "metadata":
            base
            / "metadata",

        "dataset":
            base
            / "processed"
            / "open_meteo_actual_daily.csv",

        "metadata_file":
            base
            / "metadata"
            / "open_meteo_actual_metadata.json",
    }


# ============================================================
# EXISTING DATASET
# ============================================================

def load_existing_dataset(
    dataset_path
):
    """
    Load existing processed dataset if it exists.

    Returns:
        dataframe or None
    """

    if not dataset_path.exists():

        return None

    print(
        "\nExisting dataset detected."
    )

    df = pd.read_csv(
        dataset_path
    )

    if "date" not in df.columns:

        raise ValueError(
            "Existing dataset does not contain "
            "'date' column."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.normalize()

    if df["date"].isna().any():

        raise ValueError(
            "Existing dataset contains invalid dates."
        )

    return df


# ============================================================
# DETERMINE COLLECTION WINDOW
# ============================================================

def determine_collection_window(
    existing_df,
    today
):
    """
    Determine which dates need to be downloaded.

    First run:
        today - 5 years -> today

    Later runs:
        latest_existing_date + 1 -> today
    """

    initial_start = get_five_year_start(
        today
    )

    # --------------------------------------------------------
    # First run
    # --------------------------------------------------------

    if existing_df is None:

        return (
            initial_start,
            today,
            "initial_5_year_collection"
        )

    # --------------------------------------------------------
    # Existing data
    # --------------------------------------------------------

    latest_date = (
        existing_df["date"]
        .max()
        .date()
    )

    next_date = (
        latest_date +
        timedelta(days=1)
    )

    # --------------------------------------------------------
    # Already completely up to date
    # --------------------------------------------------------

    if next_date > today:

        return (
            None,
            None,
            "already_up_to_date"
        )

    return (
        next_date,
        today,
        "incremental_update"
    )


# ============================================================
# BLOCK BOUNDARY
# ============================================================

def load_block_boundary(
    boundary_path
):
    """
    Load the processed Block boundary generated
    by 01_block_boundary.py.
    """

    if not boundary_path.exists():

        raise FileNotFoundError(
            "\nBlock boundary not found:\n"
            f"{boundary_path}\n\n"
            "Run 01_block_boundary.py first."
        )

    gdf = gpd.read_file(
        boundary_path
    )

    if gdf.empty:

        raise ValueError(
            "Block boundary file is empty."
        )

    if gdf.crs is None:

        raise ValueError(
            "Block boundary has no CRS."
        )

    return gdf


# ============================================================
# BLOCK CENTROID
# ============================================================

def get_block_centroid(
    block_gdf
):
    """
    Calculate Block centroid in projected CRS
    and return WGS84 coordinates.

    This avoids calculating a centroid directly
    in geographic coordinates.
    """

    # UTM 43N for Maharashtra / Pune region.
    projected = block_gdf.to_crs(
        "EPSG:32643"
    )

    geometry = projected.geometry.union_all()

    centroid_projected = (
        geometry.centroid
    )

    centroid_gdf = gpd.GeoDataFrame(
        geometry=[
            centroid_projected
        ],
        crs="EPSG:32643"
    )

    centroid_wgs84 = (
        centroid_gdf
        .to_crs("EPSG:4326")
        .geometry.iloc[0]
    )

    latitude = float(
        centroid_wgs84.y
    )

    longitude = float(
        centroid_wgs84.x
    )

    return latitude, longitude


# ============================================================
# OPEN-METEO REQUEST
# ============================================================

def request_open_meteo(
    latitude,
    longitude,
    start_date,
    end_date
):
    """
    Request historical weather from Open-Meteo.
    """

    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "start_date":
            start_date.isoformat(),

        "end_date":
            end_date.isoformat(),

        "daily":
            ",".join(
                DAILY_VARIABLES
            ),

        "timezone":
            "UTC",

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm",

        "format":
            "json",
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    try:

        data = response.json()

    except Exception as exc:

        raise RuntimeError(
            "Open-Meteo returned a non-JSON response.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: {response.text[:1000]}"
        ) from exc

    return data


# ============================================================
# SAVE RAW RESPONSE
# ============================================================

def save_raw_response(
    data,
    raw_directory,
    chunk_start,
    chunk_end
):
    """
    Save original API response.

    Existing raw files are never overwritten.
    """

    raw_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        "open_meteo_"
        f"{chunk_start.isoformat()}_"
        f"{chunk_end.isoformat()}.json"
    )

    path = (
        raw_directory /
        filename
    )

    # Normally this will not exist because we only request
    # dates that have not already been collected.
    if path.exists():

        print(
            f"    Raw file already exists: "
            f"{path.name}"
        )

        return path

    save_json(
        data,
        path
    )

    return path


# ============================================================
# PARSE RESPONSE
# ============================================================

def parse_daily_response(
    data,
    latitude,
    longitude
):
    """
    Convert Open-Meteo daily response to DataFrame.
    """

    if "daily" not in data:

        raise RuntimeError(
            "Open-Meteo response does not contain "
            "'daily' data."
        )

    daily = data["daily"]

    if "time" not in daily:

        raise RuntimeError(
            "Open-Meteo response has no date axis."
        )

    dates = pd.to_datetime(
        daily["time"],
        errors="coerce"
    )

    if dates.isna().any():

        raise RuntimeError(
            "Invalid dates returned by Open-Meteo."
        )

    df = pd.DataFrame(
        {
            "date": dates
        }
    )

    for variable in DAILY_VARIABLES:

        values = daily.get(
            variable
        )

        if values is None:

            # Keep missing source variables as NaN.
            df[variable] = pd.NA

        else:

            if len(values) != len(df):

                raise RuntimeError(
                    f"Length mismatch for {variable}."
                )

            df[variable] = values

    # --------------------------------------------------------
    # Location / provenance
    # --------------------------------------------------------

    df["latitude"] = latitude

    df["longitude"] = longitude

    df["source"] = (
        "Open-Meteo Historical Archive"
    )

    df["timezone"] = (
        data.get("timezone")
    )

    return df


# ============================================================
# MERGE WITHOUT DUPLICATES
# ============================================================

def append_new_records(
    existing_df,
    new_df
):
    """
    Append only genuinely new dates.

    Existing dates are preserved.

    The unique key for this Block dataset is:
        date
    """

    if new_df is None or new_df.empty:

        return existing_df, 0

    new_df = new_df.copy()

    new_df["date"] = pd.to_datetime(
        new_df["date"]
    ).dt.normalize()

    # --------------------------------------------------------
    # Remove duplicates inside new data
    # --------------------------------------------------------

    new_df = (
        new_df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
    )

    # --------------------------------------------------------
    # First collection
    # --------------------------------------------------------

    if existing_df is None:

        final_df = new_df

        return final_df, len(new_df)

    existing_df = existing_df.copy()

    existing_df["date"] = pd.to_datetime(
        existing_df["date"]
    ).dt.normalize()

    existing_dates = set(
        existing_df["date"]
    )

    genuinely_new = new_df[
        ~new_df["date"].isin(
            existing_dates
        )
    ].copy()

    if genuinely_new.empty:

        return existing_df, 0

    # --------------------------------------------------------
    # Append
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            existing_df,
            genuinely_new
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final safety deduplication
    # --------------------------------------------------------

    final_df = (
        final_df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return final_df, len(genuinely_new)


# ============================================================
# QUALITY AUDIT
# ============================================================

def create_quality_audit(
    df
):
    """
    Create a non-destructive quality audit.

    No imputation is performed.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    date_range = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    missing_dates = (
        date_range
        .difference(
            df["date"]
        )
    )

    duplicate_dates = int(
        df["date"]
        .duplicated()
        .sum()
    )

    completeness = {}

    for variable in DAILY_VARIABLES:

        if variable not in df.columns:

            missing = len(df)

        else:

            missing = int(
                df[variable]
                .isna()
                .sum()
            )

        completeness[variable] = {

            "rows":
                int(len(df)),

            "missing":
                missing,

            "missing_percent":
                round(
                    missing /
                    len(df) *
                    100,
                    4
                )
        }

    return {

        "rows":
            int(len(df)),

        "date_start":
            df["date"]
            .min()
            .date()
            .isoformat(),

        "date_end":
            df["date"]
            .max()
            .date()
            .isoformat(),

        "missing_dates":
            int(len(missing_dates)),

        "duplicate_dates":
            duplicate_dates,

        "variable_completeness":
            completeness
    }


# ============================================================
# MAIN COLLECTION
# ============================================================

def collect_weather(
    state,
    district,
    block
):
    """
    Collect or incrementally update Block weather data.
    """

    paths = get_block_paths(
        state,
        district,
        block
    )

    # --------------------------------------------------------
    # Load boundary
    # --------------------------------------------------------

    block_gdf = load_block_boundary(
        paths["boundary"]
    )

    latitude, longitude = (
        get_block_centroid(
            block_gdf
        )
    )

    print()
    print(
        f"Block centroid:"
        f"\n  Latitude : {latitude:.6f}"
        f"\n  Longitude: {longitude:.6f}"
    )

    # --------------------------------------------------------
    # Today's date
    # --------------------------------------------------------

    today = get_today()

    five_year_start = (
        get_five_year_start(
            today
        )
    )

    print()
    print(
        f"Today: {today}"
    )

    print(
        f"Five-year historical start: "
        f"{five_year_start}"
    )

    # --------------------------------------------------------
    # Existing dataset
    # --------------------------------------------------------

    existing_df = load_existing_dataset(
        paths["dataset"]
    )

    # --------------------------------------------------------
    # Determine collection window
    # --------------------------------------------------------

    (
        collection_start,
        collection_end,
        collection_mode
    ) = determine_collection_window(
        existing_df,
        today
    )

    # --------------------------------------------------------
    # Already up to date
    # --------------------------------------------------------

    if collection_mode == "already_up_to_date":

        print()
        print(
            "Dataset is already up to date."
        )

        print(
            f"Latest date: "
            f"{existing_df['date'].max().date()}"
        )

        print(
            "No API request is required."
        )

        return existing_df

    # --------------------------------------------------------
    # Collection mode
    # --------------------------------------------------------

    if collection_mode == "initial_5_year_collection":

        print()
        print(
            "Collection mode:"
        )

        print(
            "  INITIAL 5-YEAR COLLECTION"
        )

    else:

        latest_existing = (
            existing_df["date"]
            .max()
            .date()
        )

        print()
        print(
            "Collection mode:"
        )

        print(
            "  INCREMENTAL UPDATE"
        )

        print(
            f"  Existing latest date: "
            f"{latest_existing}"
        )

    print()
    print(
        f"Fetching:"
        f"\n  From: {collection_start}"
        f"\n  To  : {collection_end}"
    )

    total_days = (
        collection_end -
        collection_start
    ).days + 1

    print(
        f"  Days: {total_days}"
    )

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    paths["raw"].mkdir(
        parents=True,
        exist_ok=True
    )

    paths["processed"].mkdir(
        parents=True,
        exist_ok=True
    )

    paths["metadata"].mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # API chunks
    # --------------------------------------------------------

    chunks = list(
        generate_chunks(
            collection_start,
            collection_end
        )
    )

    print(
        f"API chunks: {len(chunks)}"
    )

    all_new_frames = []

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    for index, (
        chunk_start,
        chunk_end
    ) in enumerate(
        chunks,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk_start} → {chunk_end}"
        )

        data = request_open_meteo(
            latitude,
            longitude,
            chunk_start,
            chunk_end
        )

        raw_path = save_raw_response(
            data,
            paths["raw"],
            chunk_start,
            chunk_end
        )

        print(
            f"    Raw response: "
            f"{raw_path.name}"
        )

        frame = parse_daily_response(
            data,
            latitude,
            longitude
        )

        print(
            f"    Rows received: "
            f"{len(frame)}"
        )

        all_new_frames.append(
            frame
        )

        if index < len(chunks):

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

    # --------------------------------------------------------
    # Combine newly downloaded data
    # --------------------------------------------------------

    if all_new_frames:

        new_df = pd.concat(
            all_new_frames,
            ignore_index=True
        )

    else:

        new_df = pd.DataFrame()

    # --------------------------------------------------------
    # Append only new dates
    # --------------------------------------------------------

    final_df, added_rows = (
        append_new_records(
            existing_df,
            new_df
        )
    )

    # --------------------------------------------------------
    # Quality audit
    # --------------------------------------------------------

    audit = create_quality_audit(
        final_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        paths["dataset"],
        index=False
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "dataset":
            "Open-Meteo Historical Actual Weather",

        "source":
            "Open-Meteo Historical Archive",

        "source_url":
            OPEN_METEO_ARCHIVE_URL,

        "state":
            state,

        "district":
            district,

        "block":
            block,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "collection_timezone":
            PROJECT_TIMEZONE,

        "history_years":
            HISTORY_YEARS,

        "today":
            today.isoformat(),

        "collection_mode":
            collection_mode,

        "collection_start":
            collection_start.isoformat(),

        "collection_end":
            collection_end.isoformat(),

        "rows_added_this_run":
            int(added_rows),

        "total_rows":
            int(len(final_df)),

        "daily_variables":
            DAILY_VARIABLES,

        "quality_audit":
            audit,

        "synthetic_values_used":
            False,

        "duplicate_records_added":
            0,

        "notes": [

            "First run collects five years "
            "back from execution date.",

            "Later runs collect only dates "
            "after the existing latest date.",

            "Existing records are preserved.",

            "Duplicate dates are not appended.",

            "No synthetic or fallback values "
            "are generated.",

            "The source may lag the current day; "
            "only returned observations are stored.",

            "Raw API responses are retained "
            "for reproducibility."
        ]
    }

    save_json(
        metadata,
        paths["metadata_file"]
    )

    return final_df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SIH26 — OPEN-METEO HISTORICAL "
        "ACTUAL WEATHER"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Administrative input ONLY
    # --------------------------------------------------------

    state = input(
        "\nState: "
    ).strip()

    district = input(
        "District: "
    ).strip()

    block = input(
        "Block: "
    ).strip()

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    df = collect_weather(
        state,
        district,
        block
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "OPEN-METEO COLLECTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total rows       : "
        f"{len(df):,}"
    )

    print(
        f"Date range       : "
        f"{df['date'].min().date()} "
        f"→ "
        f"{df['date'].max().date()}"
    )

    print(
        "\nExisting data was preserved."
    )

    print(
        "Only genuinely new dates were appended."
    )

    print("=" * 70)


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nProcess cancelled."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "ERROR:"
        )
        print(exc)

        sys.exit(1)