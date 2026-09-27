"""
SIH26 — Block-level ERA5-Land collector
Exact-date-safe, incremental, real-source-only version.

Rules:
- First run: approximately last 5 years through today.
- Later runs: latest valid stored date + 1 through the REAL source frontier.
- Maximum 2 ERA5-Land nodes/day: 00:00 and 12:00 UTC.
- CDS requests are MONTH-SAFE because separate year/month/day fields can
  otherwise create unintended Cartesian date combinations.
- No synthetic/fallback values.
- Returned rows are strictly filtered to the requested interval.
- Daily values are the mean of the two available synoptic nodes.
- Existing valid data are preserved and duplicates are removed.
- A final audit reports missing dates, duplicate dates, and out-of-range rows.

NOTE:
ERA5-Land is a historical/reanalysis source and may lag the current date.
The collector never fabricates unavailable dates.
"""

from __future__ import annotations

import calendar
import json
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    import cdsapi
except ImportError:
    cdsapi = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Location is supplied by the user at runtime.
# Nothing below is tied to Baramati, Pune, or Maharashtra.
ERA5_VARIABLES = [
    "2m_dewpoint_temperature",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
]

# Explicitly capped at two nodes/day.
TIMES = ["00:00", "12:00"]

DATASET = "reanalysis-era5-land"

# CDS accepts area requests; nearest returned grid cell is selected locally.
AREA_HALF_DEG = 0.1

# Keep concurrency modest and deterministic.
MAX_WORKERS = 2

# First-run history.
HISTORY_YEARS = 5

# Request today's date, but do not fabricate unavailable source dates.
REQUEST_END = date.today()

# ---------------------------------------------------------------------
# Project root / paths
# ---------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]

    for p in candidates:
        required = ["block", "gp", "datasets", "scripts"]
        if all((p / x).exists() for x in required):
            return p

    raise RuntimeError(
        "Could not locate SIH26 project root. Expected a directory containing "
        "block/, gp/, datasets/, and scripts/."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)

LOCATION_DIR = None
METADATA_PATH = None
RAW_DIR = None
PROCESSED_DIR = None
OUTPUT_CSV = None
COLLECTION_META_PATH = None


def clean_location_name(value: str) -> str:
    """Create a safe directory name without changing the user's location."""
    value = str(value).strip()
    if not value:
        raise ValueError("Location name cannot be empty.")
    return re.sub(r'[<>:"/\\|?*]', "_", value)


def configure_location():
    """
    Ask the user for State -> District -> Block and resolve the existing
    SIH26 block directory.

    The script therefore works for any block that already has metadata
    created by the earlier block-boundary collector.
    """
    print("\nEnter the target block location.")
    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    if not state or not district or not block:
        raise ValueError(
            "State, District, and Block are all required."
        )

    safe_state = clean_location_name(state)
    safe_district = clean_location_name(district)
    safe_block = clean_location_name(block)

    location_dir = (
        PROJECT_ROOT
        / "block"
        / safe_state
        / safe_district
        / safe_block
    )

    metadata_path = location_dir / "metadata" / "block_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Block metadata not found for the requested location:\n"
            f"{metadata_path}\n\n"
            "Run 01_block_boundary.py first for this "
            "State/District/Block."
        )

    raw_dir = location_dir / "raw" / "era5_land"
    processed_dir = location_dir / "processed" / "era5_land"
    output_csv = processed_dir / "era5_land_daily.csv"
    collection_meta_path = processed_dir / "collection_metadata.json"

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
        output_csv,
        collection_meta_path,
    )


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Block metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recursive_find_lat_lon(obj):
    """
    Search metadata recursively for a sensible latitude/longitude pair.
    This avoids assuming one exact block_metadata.json schema.
    """
    if isinstance(obj, dict):
        # Prefer an explicit centroid object.
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in {"centroid", "block_centroid", "block_center", "center"}:
                if isinstance(value, dict):
                    lat = value.get("lat", value.get("latitude"))
                    lon = value.get("lon", value.get("longitude"))
                    if lat is not None and lon is not None:
                        return float(lat), float(lon)
                elif isinstance(value, (list, tuple)) and len(value) >= 2:
                    return float(value[0]), float(value[1])

        # Then look for paired scalar keys.
        lat_keys = {"lat", "latitude", "centroid_lat", "centroid_latitude",
                    "block_latitude", "block_centroid_latitude"}
        lon_keys = {"lon", "lng", "longitude", "centroid_lon",
                    "centroid_longitude", "block_longitude",
                    "block_centroid_longitude"}

        lat = lon = None
        for key, value in obj.items():
            kl = str(key).lower()
            if kl in lat_keys:
                try:
                    lat = float(value)
                except (TypeError, ValueError):
                    pass
            if kl in lon_keys:
                try:
                    lon = float(value)
                except (TypeError, ValueError):
                    pass

        if lat is not None and lon is not None:
            return lat, lon

        for value in obj.values():
            result = _recursive_find_lat_lon(value)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = _recursive_find_lat_lon(value)
            if result is not None:
                return result

    return None


def get_centroid(metadata: dict) -> tuple[float, float]:
    """
    Resolve the block centroid without hard-coding Baramati.

    Priority:
      1. Explicit centroid/lat-lon in block_metadata.json.
      2. Centroid of the generated block_boundary.geojson.

    The second option is important because the existing project metadata
    schema does not necessarily store latitude/longitude under a single
    fixed key.
    """
    result = _recursive_find_lat_lon(metadata)

    if result is not None:
        return result

    boundary_path = (
        LOCATION_DIR / "processed" / "block_boundary.geojson"
    )

    if boundary_path.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(boundary_path)

            if gdf.empty:
                raise RuntimeError("Block boundary GeoJSON is empty.")

            # Boundary is stored in WGS84. Project centroid is only used
            # to select the nearest ERA5-Land grid cell, so geographic
            # centroid is sufficient here.
            geom = gdf.geometry.union_all()
            c = geom.centroid

            return float(c.y), float(c.x)

        except Exception as exc:
            raise RuntimeError(
                "Could not resolve centroid from metadata, and reading "
                f"the block boundary failed: {boundary_path}"
            ) from exc

    raise KeyError(
        "Could not find block centroid latitude/longitude in "
        "block_metadata.json and block_boundary.geojson was not found. "
        f"Expected boundary: {boundary_path}"
    )


# ---------------------------------------------------------------------
# Date logic
# ---------------------------------------------------------------------


def get_existing_latest_date() -> date | None:
    if not OUTPUT_CSV.exists():
        return None

    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=["date"])
    except Exception as exc:
        raise RuntimeError(
            f"Existing ERA5-Land output could not be read: {OUTPUT_CSV}"
        ) from exc

    if df.empty:
        return None

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()

    if dates.empty:
        return None

    return dates.dt.date.max()


def determine_collection_period() -> tuple[date, date, str, date]:
    """
    Determine the incremental request window and the permanent historical
    anchor for this block.

    The first run establishes a permanent collection_start. Later runs use
    latest stored valid date + 1. This prevents today's date from silently
    changing the historical boundary.
    """
    latest = get_existing_latest_date()

    collection_start = None
    if COLLECTION_META_PATH is not None and COLLECTION_META_PATH.exists():
        try:
            with COLLECTION_META_PATH.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("collection_start"):
                collection_start = datetime.strptime(
                    meta["collection_start"], "%Y-%m-%d"
                ).date()
        except Exception:
            collection_start = None

    if collection_start is None:
        collection_start = REQUEST_END.replace(
            year=REQUEST_END.year - HISTORY_YEARS
        )

        # Persist the original historical anchor. It must not drift on
        # subsequent runs.
        if COLLECTION_META_PATH is not None:
            COLLECTION_META_PATH.parent.mkdir(parents=True, exist_ok=True)
            with COLLECTION_META_PATH.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "collection_start": collection_start.isoformat(),
                        "history_years_initial": HISTORY_YEARS,
                        "source": DATASET,
                        "nodes_per_day": len(TIMES),
                        "times_utc": TIMES,
                        "synthetic": False,
                    },
                    f,
                    indent=2,
                )

    if latest is None:
        start = collection_start
        mode = "INITIAL 5-YEAR COLLECTION"
    else:
        start = max(latest + timedelta(days=1), collection_start)
        mode = "INCREMENTAL UPDATE"

    return start, REQUEST_END, mode, collection_start


def month_batches(start: date, end: date):
    """
    Yield exact, calendar-month-safe requests.

    This is deliberately NOT an annual batch. CDS year/month/day fields can
    otherwise represent combinations rather than an arbitrary date interval.
    """
    cursor = start.replace(day=1)

    while cursor <= end:
        month_end = date(
            cursor.year,
            cursor.month,
            calendar.monthrange(cursor.year, cursor.month)[1],
        )

        batch_start = max(start, cursor)
        batch_end = min(end, month_end)

        yield batch_start, batch_end

        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


# ---------------------------------------------------------------------
# CDS request
# ---------------------------------------------------------------------


def build_request(
    batch_start: date,
    batch_end: date,
    lat: float,
    lon: float,
    times: list[str] | None = None,
) -> dict:
    # Because batch_start/end are guaranteed to be in the SAME month,
    # year/month/day lists are unambiguous.
    days = [
        f"{d:02d}"
        for d in range(batch_start.day, batch_end.day + 1)
    ]

    if times is None:
        times = TIMES

    return {
        "variable": ERA5_VARIABLES,
        "year": [str(batch_start.year)],
        "month": [f"{batch_start.month:02d}"],
        "day": days,
        "time": times,
        "area": [
            lat + AREA_HALF_DEG,
            lon - AREA_HALF_DEG,
            lat - AREA_HALF_DEG,
            lon + AREA_HALF_DEG,
        ],
        # CDS now expects data_format rather than the deprecated format key.
        "data_format": "netcdf",
    }


def parse_latest_available_datetime(error: Exception):
    """
    Extract the source availability timestamp from the CDS error message.

    Example:
      The latest date available for this dataset is:
      2026-09-11 07:00
    """
    text = str(error)
    match = re.search(
        r"latest date available for this dataset is:\s*"
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}):?(\d{2})?",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    try:
        return datetime.strptime(
            f"{match.group(1)} {match.group(2)}",
            "%Y-%m-%d %H",
        )
    except ValueError:
        return None


def raw_filename(batch_start: date, batch_end: date) -> Path:
    return RAW_DIR / (
        f"era5_land_{batch_start:%Y%m%d}_{batch_end:%Y%m%d}.nc"
    )


def download_batch(
    batch_start: date,
    batch_end: date,
    lat: float,
    lon: float,
) -> tuple[date, date, Path] | None:
    if cdsapi is None:
        raise RuntimeError(
            "cdsapi is not installed. Install/configure the real CDS API; "
            "no synthetic fallback is permitted."
        )

    client = cdsapi.Client()

    # First try the complete requested batch with both daily nodes.
    attempts = [
        (batch_start, batch_end, TIMES),
    ]

    # If CDS reports that the source has not reached the requested end,
    # automatically retry using the real source availability. This is
    # essential for incremental collection because ERA5-Land can lag today.
    tried = set()

    while attempts:
        current_start, current_end, current_times = attempts.pop(0)
        key = (current_start, current_end, tuple(current_times))
        if key in tried:
            continue
        tried.add(key)

        if current_start > current_end:
            continue

        target = raw_filename(current_start, current_end)

        # Do not reuse a file created by a different node configuration.
        # The filename is supplemented by a node-specific suffix when the
        # final day only has the 00 UTC node.
        if current_times == TIMES:
            target = RAW_DIR / (
                f"era5_land_{current_start:%Y%m%d}_{current_end:%Y%m%d}_00_12.nc"
            )
        else:
            target = RAW_DIR / (
                f"era5_land_{current_start:%Y%m%d}_{current_end:%Y%m%d}_00.nc"
            )

        if target.exists() and target.stat().st_size > 0:
            print(
                f"  Using existing CDS download: "
                f"{current_start} -> {current_end}"
                f" | nodes={','.join(current_times)}"
            )
            return current_start, current_end, target

        request = build_request(
            current_start,
            current_end,
            lat,
            lon,
            current_times,
        )

        print(
            f"  Downloading: {current_start} -> {current_end}"
            f" | {len(request['day'])} days × {len(current_times)} nodes"
        )

        try:
            client.retrieve(DATASET, request, str(target))

            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(
                    f"CDS reported success but the file is missing/empty: {target}"
                )

            return current_start, current_end, target

        except Exception as exc:
            available = parse_latest_available_datetime(exc)

            if available is None:
                raise

            available_date = available.date()
            available_hour = available.hour

            print(
                f"  CDS source availability reported as "
                f"{available:%Y-%m-%d %H:%M}."
            )

            # The source has not advanced to the requested incremental
            # start date. This is NOT an error. There is simply no new
            # real ERA5-Land data to collect yet.
            if available_date < current_start:
                print(
                    f"  No new ERA5-Land data available for "
                    f"{current_start} -> {current_end}."
                )
                print(
                    f"  Real source currently ends at "
                    f"{available:%Y-%m-%d %H:%M}."
                )
                print(
                    "  Keeping the existing dataset unchanged. "
                    "No synthetic values will be inserted."
                )
                return None

            # If the latest available date is inside this batch, retry only
            # through that real date. If the source has only reached before
            # 12 UTC on the latest date, request 00 UTC only for that date.
            safe_end = min(current_end, available_date)

            if available_hour < 12 and safe_end == available_date:
                # Request all earlier days with both nodes, then the latest
                # partial day with its one actually available node.
                previous_end = available_date - timedelta(days=1)

                if current_start <= previous_end:
                    attempts.insert(
                        0,
                        (current_start, previous_end, TIMES),
                    )

                attempts.append(
                    (available_date, available_date, ["00:00"])
                )
            else:
                attempts.append(
                    (current_start, safe_end, TIMES)
                )

            # If the original batch ended before the source availability,
            # the original request should not need another retry.
            continue

    raise RuntimeError(
        f"No ERA5-Land data could be collected for {batch_start} -> {batch_end}."
    )


# ---------------------------------------------------------------------
# NetCDF reader
# ---------------------------------------------------------------------


def read_netcdf(
    path: Path,
    requested_start: date,
    requested_end: date,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Read one raw CDS NetCDF using xarray.

    The import is kept local so the collector gives a clear dependency
    message if xarray/netCDF support is absent.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "xarray is required to process ERA5-Land NetCDF files."
        ) from exc

    ds = xr.open_dataset(path)

    try:
        # Normalize longitude convention if necessary.
        if "longitude" not in ds.coords or "latitude" not in ds.coords:
            raise RuntimeError(
                f"Unexpected ERA5-Land coordinate structure in {path}"
            )

        lat_name = "latitude"
        lon_name = "longitude"

        # Nearest grid cell to block centroid.
        ds_point = ds.sel(
            {
                lat_name: lat,
                lon_name: lon,
            },
            method="nearest",
        )

        df = ds_point.to_dataframe().reset_index()

    finally:
        ds.close()

    if "time" not in df.columns:
        raise RuntimeError(f"No time coordinate found in {path}")

    df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # CRITICAL: hard filter to the exact requested interval.
    df = df[
        (df["date"] >= requested_start)
        & (df["date"] <= requested_end)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Confirm no more than two source nodes/day survived.
    node_counts = df.groupby("date")["time"].nunique()
    if (node_counts > len(TIMES)).any():
        bad = node_counts[node_counts > len(TIMES)]
        raise RuntimeError(
            f"More than {len(TIMES)} ERA5-Land nodes/day found in {path}: "
            f"{bad.to_dict()}"
        )

    # Daily mean of available 00/12 UTC nodes.
    agg = {}

    if "2m_dewpoint_temperature" in df.columns:
        agg["2m_dewpoint_temperature"] = "mean"

    if "surface_pressure" in df.columns:
        agg["surface_pressure"] = "mean"

    if "volumetric_soil_water_layer_1" in df.columns:
        agg["volumetric_soil_water_layer_1"] = "mean"

    if not agg:
        raise RuntimeError(
            f"None of the requested ERA5-Land variables were found in {path}"
        )

    daily = (
        df.groupby("date", as_index=False)
        .agg(agg)
        .sort_values("date")
    )

    return daily


# ---------------------------------------------------------------------
# Output cleaning / audit
# ---------------------------------------------------------------------


def read_existing_output() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "volumetric_soil_water_layer_1",
            ]
        )

    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).copy()

    return df



def sanitize_existing_output(df: pd.DataFrame, collection_start: date) -> pd.DataFrame:
    """
    Remove rows that predate the permanent collection anchor.

    This is important for repairing the previously contaminated dataset,
    which contained 2021-09-01 onward even though the intended first date
    was 2021-09-15/16. No valid data on/after the anchor are modified.
    """
    if df is None or df.empty:
        return df

    before = len(df)
    df = df[df["date"] >= collection_start].copy()
    removed = before - len(df)

    if removed:
        print(
            f"  Removed {removed} existing rows before permanent "
            f"collection start {collection_start}."
        )
        print(
            "  These rows were outside the project's intended historical "
            "window and are not used."
        )

    return df

def merge_and_save(
    existing: pd.DataFrame,
    new_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    valid_new = [x for x in new_frames if x is not None and not x.empty]

    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    frames.extend(valid_new)

    if not frames:
        return existing

    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Date is the block-level ERA5-Land daily key.
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    # Stable column order.
    columns = [
        "date",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "volumetric_soil_water_layer_1",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[columns]

    df.to_csv(OUTPUT_CSV, index=False)

    return df


def final_audit(
    df: pd.DataFrame,
    requested_start: date,
    requested_end: date,
):
    print("\n" + "=" * 70)
    print("ERA5-LAND FINAL DATE / DATA AUDIT")
    print("=" * 70)

    if df.empty:
        print("No valid ERA5-Land rows available.")
        return

    dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.date

    actual_start = dates.min()
    actual_end = dates.max()

    print(f"Requested start : {requested_start}")
    print(f"Requested end   : {requested_end}")
    print(f"Actual start    : {actual_start}")
    print(f"Actual end      : {actual_end}")
    print(f"Rows            : {len(df)}")
    print(f"Unique dates    : {dates.nunique()}")

    out_of_range = df[
        (df["date"] < requested_start)
        | (df["date"] > requested_end)
    ]

    duplicate_dates = df[df.duplicated("date", keep=False)]

    expected_dates = set(
        pd.date_range(requested_start, requested_end, freq="D").date
    )
    actual_dates = set(dates)

    missing_dates = sorted(expected_dates - actual_dates)

    print(f"Out-of-range rows: {len(out_of_range)}")
    print(f"Duplicate dates : {len(duplicate_dates)}")
    print(f"Missing dates   : {len(missing_dates)}")

    if missing_dates:
        print(
            "Missing-date note: missing dates are NOT filled synthetically "
            "during extraction. They remain absent and will be handled later "
            "by the project's data-cleaning policy."
        )
        if len(missing_dates) <= 20:
            print("Missing:", ", ".join(map(str, missing_dates)))
        else:
            print(
                "First missing:",
                ", ".join(map(str, missing_dates[:10])),
                "...",
            )

    if len(out_of_range) > 0:
        raise RuntimeError(
            "AUDIT FAILED: out-of-range dates remain in the final dataset."
        )

    if len(duplicate_dates) > 0:
        raise RuntimeError(
            "AUDIT FAILED: duplicate daily dates remain in the final dataset."
        )

    print("Date-boundary check: PASS")
    print("Duplicate-date check: PASS")
    print("Synthetic/fallback policy: DISABLED")
    print("=" * 70)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    (
        state,
        district,
        block,
        location_dir,
        metadata_path,
        raw_dir,
        processed_dir,
        output_csv,
        collection_meta_path,
    ) = configure_location()

    # Configure the location-specific paths only after user selection.
    global LOCATION_DIR, METADATA_PATH, RAW_DIR, PROCESSED_DIR, OUTPUT_CSV, COLLECTION_META_PATH
    LOCATION_DIR = location_dir
    METADATA_PATH = metadata_path
    RAW_DIR = raw_dir
    PROCESSED_DIR = processed_dir
    OUTPUT_CSV = output_csv
    COLLECTION_META_PATH = collection_meta_path

    metadata = load_metadata()
    lat, lon = get_centroid(metadata)

    print("\n" + "=" * 70)
    print("SIH26 — ERA5-LAND BLOCK COLLECTOR")
    print("=" * 70)
    print(f"State: {state}")
    print(f"District: {district}")
    print(f"Block: {block}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Block directory: {LOCATION_DIR}")
    print(f"Centroid: {lat:.6f}, {lon:.6f}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Raw: {RAW_DIR}")
    print()

    start, end, mode, collection_start = determine_collection_period()

    print(f"Mode: {mode}")
    print(f"Permanent collection start: {collection_start}")
    print(f"Requested collection window: {start} -> {end}")
    print(
        "ERA5-Land nodes/day: "
        "2 (00:00 UTC, 12:00 UTC)"
    )
    print(
        "Batch strategy: exact calendar-month batches; "
        "maximum safe CDS payload per batch"
    )
    print(
        "Synthetic/fallback values: DISABLED"
    )

    if start > end:
        print(
            "\nDataset is already current for the requested source period."
        )
        existing = read_existing_output()
        if not existing.empty:
            # The audit here is for the permanent dataset boundary, not the
            # empty incremental window.
            final_audit(existing, collection_start, end)
            existing.to_csv(OUTPUT_CSV, index=False)
        return

    batches = list(month_batches(start, end))

    print(f"\nTotal exact-safe batches: {len(batches)}")

    existing = read_existing_output()
    existing = sanitize_existing_output(existing, collection_start)

    # IMPORTANT:
    # Do not use old contaminated annual raw files as new scientific data.
    # Only files whose filenames correspond to exact month-safe ranges are
    # considered by this version.
    #
    # Existing processed CSV is retained, but final merge is always
    # deduplicated and hard-audited.

    new_frames = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                download_batch,
                batch_start,
                batch_end,
                lat,
                lon,
            ): (batch_start, batch_end)
            for batch_start, batch_end in batches
        }

        completed = 0

        for future in as_completed(futures):
            batch_start, batch_end = futures[future]

            try:
                result = future.result()

                completed += 1

                if result is None:
                    print(
                        f"  Completed {completed}/{len(batches)}: "
                        f"{batch_start} -> {batch_end} | "
                        "0 new rows (source not yet available)"
                    )
                    continue

                bs, be, raw_path = result

                daily = read_netcdf(
                    raw_path,
                    requested_start=bs,
                    requested_end=be,
                    lat=lat,
                    lon=lon,
                )

                if not daily.empty:
                    new_frames.append(daily)

                print(
                    f"  Completed {completed}/{len(batches)}: "
                    f"{bs} -> {be} | {len(daily)} daily rows"
                )

            except Exception as exc:
                raise RuntimeError(
                    f"ERA5-Land batch failed: {batch_start} -> {batch_end}"
                ) from exc

    final_df = merge_and_save(existing, new_frames)

    final_audit(
        final_df,
        requested_start=collection_start,
        requested_end=end,
    )

    print("\nERA5-LAND COLLECTION COMPLETE")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(final_df)}")

    if not final_df.empty:
        print(
            "Date range:",
            final_df["date"].min(),
            "->",
            final_df["date"].max(),
        )


if __name__ == "__main__":
    main()
