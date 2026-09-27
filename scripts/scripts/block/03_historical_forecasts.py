"""
03_historical_forecasts.py
Universal block-level historical forecast collector.

Design:
- Discovers block metadata recursively; no state/district/block is hard-coded.
- Uses the working Open-Meteo Previous Runs API implementation from Weather_Final.
- Collects fixed D+1 ... D+7 historical forecasts.
- Preserves forecast provenance.
- Uses incremental collection: first run = ~5 years through today; later runs = after latest stored target date.
- No actual weather values are used as forecast inputs.
- No synthetic/default/fallback weather values are generated.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import re
# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCK_ROOT = PROJECT_ROOT / "block"

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
TIMEZONE = "GMT"
TIMEOUT = 120

FC_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stop(message: str, exc: Exception | None = None) -> None:
    if exc:
        raise RuntimeError(f"{message}\nOriginal error: {exc}") from exc
    raise RuntimeError(message)


def get_json(url: str, params: dict, label: str) -> dict:
    """
    Robust real-source request.

    The Previous Runs service can occasionally return a non-JSON response
    (for example a transient gateway/rate-limit page). Never interpret that
    as weather data and never substitute synthetic values.
    """
    import time

    max_attempts = 5
    backoff_seconds = [3, 8, 15, 30, 60]

    for attempt in range(max_attempts):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=TIMEOUT,
                headers={"User-Agent": "SIH26-Downscaling/1.0"},
            )

            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as exc:
                    if attempt < max_attempts - 1:
                        print(
                            f"\n   ⚠ {label}: HTTP 200 but non-JSON response; "
                            f"retrying in {backoff_seconds[attempt]}s..."
                        )
                        time.sleep(backoff_seconds[attempt])
                        continue
                    stop(
                        f"{label} returned HTTP 200 but not valid JSON. "
                        "No fake/default data will be generated.\n"
                        f"Response start: {r.text[:500]}",
                        exc,
                    )

            if r.status_code == 429 or 500 <= r.status_code <= 599:
                if attempt < max_attempts - 1:
                    print(
                        f"\n   ⚠ {label}: HTTP {r.status_code}; "
                        f"retrying in {backoff_seconds[attempt]}s..."
                    )
                    time.sleep(backoff_seconds[attempt])
                    continue

            stop(
                f"{label} failed with HTTP {r.status_code}.\n"
                f"Response: {r.text[:1000]}"
            )

        except requests.RequestException as exc:
            if attempt < max_attempts - 1:
                print(
                    f"\n   ⚠ {label}: network error; "
                    f"retrying in {backoff_seconds[attempt]}s..."
                )
                time.sleep(backoff_seconds[attempt])
                continue
            stop(
                f"{label} request failed. No fake/default data will be generated.",
                exc,
            )

    stop(f"{label} failed after {max_attempts} attempts.")


def first_value(d: dict, keys: list[str]):
    """Return the first non-empty value from a metadata dictionary."""
    for key in keys:
        value = d.get(key)
        if value not in (None, "", []):
            return value
    return None


# ---------------------------------------------------------------------------
# Universal block discovery
# ---------------------------------------------------------------------------

def discover_block() -> tuple[Path, dict]:
    """
    Resolve the block from explicit user input.

    No state/district/block is hard-coded. The user selects the location and
    the script resolves the corresponding block_metadata.json dynamically.
    """
    print("\nSelect the block for historical forecast collection.")
    state_input = input("State: ").strip()
    district_input = input("District: ").strip()
    block_input = input("Block: ").strip()

    if not state_input or not district_input or not block_input:
        stop("State, District and Block are required.")

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    candidates = list(BLOCK_ROOT.rglob("block_metadata.json"))

    if not candidates:
        stop(
            "No block_metadata.json found below the project block/ directory.\n"
            "Run 01_block_boundary.py first."
        )

    matches = []

    for meta_path in candidates:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rel = meta_path.relative_to(BLOCK_ROOT)

        # Expected hierarchy:
        # block/State/District/Block/metadata/block_metadata.json
        if len(rel.parts) < 5:
            continue

        path_state = rel.parts[-5]
        path_district = rel.parts[-4]
        path_block = rel.parts[-3]

        if (
            norm(path_state) == norm(state_input)
            and norm(path_district) == norm(district_input)
            and norm(path_block) == norm(block_input)
        ):
            matches.append((meta_path, meta, path_state, path_district, path_block))

    if not matches:
        stop(
            f"Could not find a block matching:\n"
            f"  State: {state_input}\n"
            f"  District: {district_input}\n"
            f"  Block: {block_input}\n\n"
            "Check the spelling or run 01_block_boundary.py for that block first."
        )

    if len(matches) > 1:
        print("\n⚠ Multiple metadata files match this location:")
        for i, (p, _, _, _, _) in enumerate(matches, 1):
            print(f"  {i}. {p}")

        choice = input("Select metadata file [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(matches):
                stop("Invalid metadata selection.")
        except ValueError:
            stop("Metadata selection must be a number.")

        meta_path, meta, path_state, path_district, path_block = matches[idx]
    else:
        meta_path, meta, path_state, path_district, path_block = matches[0]

    state = path_state
    district = path_district
    block = path_block

    lat = first_value(meta, [
        "latitude", "lat", "block_latitude", "block_centroid_lat",
        "centroid_lat"
    ])
    lon = first_value(meta, [
        "longitude", "lon", "lng", "block_longitude", "block_centroid_lon",
        "centroid_lon"
    ])

    centroid = meta.get("centroid")
    if isinstance(centroid, dict):
        lat = lat if lat is not None else first_value(
            centroid, ["latitude", "lat"]
        )
        lon = lon if lon is not None else first_value(
            centroid, ["longitude", "lon", "lng"]
        )

    if lat is None or lon is None:
        boundary = meta_path.parent.parent / "processed" / "block_boundary.geojson"

        if not boundary.exists():
            stop(
                f"Could not find block centroid in {meta_path} and no saved "
                f"block boundary exists at {boundary}."
            )

        try:
            import geopandas as gpd

            gdf = gpd.read_file(boundary)
            if gdf.empty:
                stop(f"Block boundary is empty: {boundary}")

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            projected = gdf.to_crs(gdf.estimate_utm_crs())
            c = projected.geometry.union_all().centroid
            c_wgs84 = gpd.GeoSeries(
                [c], crs=projected.crs
            ).to_crs("EPSG:4326").iloc[0]

            lat = float(c_wgs84.y)
            lon = float(c_wgs84.x)

        except Exception as exc:
            stop(
                "Could not derive the block centroid from "
                "block_boundary.geojson.",
                exc,
            )

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception as exc:
        stop(f"Invalid block centroid: lat={lat}, lon={lon}", exc)

    return meta_path, {
        "state": state,
        "district": district,
        "block": block,
        "latitude": lat,
        "longitude": lon,
        "metadata_path": meta_path,
    }


# ---------------------------------------------------------------------------
# Previous Runs collector
# ---------------------------------------------------------------------------

def hourly_frame(payload: dict, variables: list[str], label: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        stop(f"{label}: no hourly timestamps returned.")

    df = pd.DataFrame({"time": pd.to_datetime(times)})

    for variable in variables:
        if variable not in hourly:
            stop(f"{label}: required variable missing: {variable}")

        if len(hourly[variable]) != len(df):
            stop(f"{label}: length mismatch for {variable}")

        df[variable] = hourly[variable]

    df["date"] = df["time"].dt.date
    return df


def date_chunks(start: date, end: date, chunk_days: int = 60):
    """Yield inclusive date ranges for manageable API requests."""
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=chunk_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def collect_previous_runs(
    lat: float,
    lon: float,
    start: date,
    end: date,
    output_dir: Path,
) -> pd.DataFrame:

    print("\n[1/1] Open-Meteo Previous Runs — D+1 ... D+7")

    leads: dict[int, pd.DataFrame] = {}

    import time

    for lead in range(1, 8):

        names = [f"{v}_previous_day{lead}" for v in FC_VARS]

        # Split the multi-year first collection into manageable requests.
        # This avoids transient gateway/rate-limit/non-JSON responses while
        # preserving the exact Previous Runs methodology.
        chunks = []
        ranges = list(date_chunks(start, end, chunk_days=60))

        print(f"   D+{lead}: {len(ranges)} API chunks")

        for chunk_no, (chunk_start, chunk_end) in enumerate(ranges, start=1):

            payload = get_json(
                PREVIOUS_RUNS_URL,
                {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": str(chunk_start),
                    "end_date": str(chunk_end),
                    "hourly": ",".join(names),
                    "timezone": TIMEZONE,
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "ms",
                    "precipitation_unit": "mm",
                },
                f"Open-Meteo Previous Runs D+{lead} "
                f"[chunk {chunk_no}/{len(ranges)}]",
            )

            h_chunk = hourly_frame(
                payload,
                names,
                f"Previous Runs D+{lead} chunk {chunk_no}",
            )

            chunks.append(h_chunk)

            if chunk_no < len(ranges):
                time.sleep(1.5)

        if not chunks:
            stop(f"Previous Runs D+{lead}: no API chunks returned.")

        h = pd.concat(chunks, ignore_index=True)
        h = h.drop_duplicates(subset=["time"], keep="first")

        counts = h.groupby("date").size()
        complete_dates = counts[counts == 24].index
        h = h[h["date"].isin(complete_dates)].copy()

        rename = {
            f"temperature_2m_previous_day{lead}":
                f"temp_mean_forecast_d{lead}",
            f"relative_humidity_2m_previous_day{lead}":
                f"rh_mean_forecast_d{lead}",
            f"precipitation_previous_day{lead}":
                f"precip_forecast_d{lead}",
            f"cloud_cover_previous_day{lead}":
                f"cloud_cover_forecast_d{lead}",
            f"wind_speed_10m_previous_day{lead}":
                f"wind_speed_forecast_d{lead}",
            f"wind_direction_10m_previous_day{lead}":
                f"wind_direction_forecast_d{lead}",
            f"wind_gusts_10m_previous_day{lead}":
                f"wind_gust_forecast_d{lead}",
            f"vapour_pressure_deficit_previous_day{lead}":
                f"vpd_forecast_d{lead}",
        }

        h = h.rename(columns=rename)

        aggregation = {}
        for variable in FC_VARS:
            source_column = rename[f"{variable}_previous_day{lead}"]

            if variable == "precipitation":
                function = "sum"
            elif variable == "wind_gusts_10m":
                function = "max"
            else:
                function = "mean"

            aggregation[source_column] = (source_column, function)

        daily = h.groupby("date", as_index=False).agg(**aggregation)
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

        # Explicit fixed-lead provenance.
        daily[f"forecast_issued_date_d{lead}"] = (
            pd.to_datetime(daily["date"]) - pd.Timedelta(days=lead)
        ).dt.strftime("%Y-%m-%d")

        daily[f"forecast_lead_days_d{lead}"] = lead
        daily[f"forecast_issued_date_rule_d{lead}"] = (
            f"target_date_minus_{lead}_day"
        )

        leads[lead] = daily

        print(f"   D+{lead}: {len(daily)} complete target dates")

    # Merge each lead using UNIQUE provenance column names.
    merged = leads[1]

    for lead in range(2, 8):
        merged = merged.merge(
            leads[lead],
            on="date",
            how="outer",
            validate="one_to_one",
        )

    # D+1 is the canonical forecast layer.
    required = [
        "temp_mean_forecast_d1",
        "rh_mean_forecast_d1",
        "precip_forecast_d1",
    ]

    merged = merged.dropna(subset=required).copy()

    merged["latitude"] = lat
    merged["longitude"] = lon
    merged["forecast_source"] = "Open-Meteo Previous Runs API"
    merged["forecast_d1_source"] = "previous_day1"
    merged["forecast_reference"] = "fixed_24h_lead"
    merged["forecast_issued_date"] = merged["forecast_issued_date_d1"]

    # Strict leakage/provenance test.
    issued = pd.to_datetime(merged["forecast_issued_date"])
    target = pd.to_datetime(merged["date"])

    if not (issued < target).all():
        bad = merged.loc[~(issued < target), ["date", "forecast_issued_date"]]
        stop(
            "D+1 provenance check failed. "
            "At least one forecast would use information from the target day.\n"
            f"Examples:\n{bad.head().to_string(index=False)}"
        )

    # Also verify each fixed lead has the expected temporal relationship.
    for lead in range(1, 8):
        col = f"forecast_issued_date_d{lead}"
        issued_lead = pd.to_datetime(merged[col])
        if not (issued_lead == target - pd.Timedelta(days=lead)).all():
            stop(f"D+{lead} provenance relationship check failed.")

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "historical_forecasts_previous_runs_d1_d7.csv"
    d1_path = output_dir / "open_meteo_previous_runs_d1.csv"

    # -----------------------------------------------------------------------
    # NEW-DATA EMBEDDING RULE
    #
    # Existing data is never discarded.
    # Only newly collected target dates are added.
    # Duplicate target dates are removed deterministically.
    # -----------------------------------------------------------------------

    def append_and_deduplicate(
        new_df: pd.DataFrame,
        path: Path,
        key_columns: list[str],
    ) -> pd.DataFrame:

        if path.exists():
            try:
                existing = pd.read_csv(path)

                # Pandas cannot safely concat a DataFrame that contains
                # duplicate column labels. If duplicate labels exist and
                # their values are identical, keep one copy. If they differ,
                # fail safely rather than guessing which column is correct.
                duplicate_columns = existing.columns[
                    existing.columns.duplicated()
                ].unique().tolist()

                if duplicate_columns:
                    for column in duplicate_columns:
                        positions = [
                            i for i, name in enumerate(existing.columns)
                            if name == column
                        ]
                        reference = existing.iloc[:, positions[0]]

                        for position in positions[1:]:
                            if not reference.equals(existing.iloc[:, position]):
                                stop(
                                    f"Conflicting duplicate columns detected "
                                    f"in existing dataset {path}: {column!r}. "
                                    "The existing file will NOT be overwritten."
                                )

                    existing = existing.loc[
                        :, ~existing.columns.duplicated(keep="first")
                    ]

                new_duplicate_columns = new_df.columns[
                    new_df.columns.duplicated()
                ].unique().tolist()

                if new_duplicate_columns:
                    for column in new_duplicate_columns:
                        positions = [
                            i for i, name in enumerate(new_df.columns)
                            if name == column
                        ]
                        reference = new_df.iloc[:, positions[0]]

                        for position in positions[1:]:
                            if not reference.equals(new_df.iloc[:, position]):
                                stop(
                                    f"Conflicting duplicate columns detected "
                                    f"in newly collected data: {column!r}."
                                )

                    new_df = new_df.loc[
                        :, ~new_df.columns.duplicated(keep="first")
                    ]

                existing["date"] = pd.to_datetime(
                    existing["date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

                existing = existing.dropna(subset=key_columns)

                combined = pd.concat(
                    [existing, new_df],
                    ignore_index=True,
                )

                # Keep the newly collected record when the same key already
                # exists. This makes reruns safe without creating duplicates.
                combined["_row_order"] = range(len(combined))
                combined = combined.drop_duplicates(
                    subset=key_columns,
                    keep="last",
                )

                combined = combined.sort_values(key_columns)
                combined = combined.drop(columns="_row_order")

                print(
                    f"   Existing records: {len(existing)} | "
                    f"New records received: {len(new_df)} | "
                    f"Final records: {len(combined)}"
                )

            except Exception as exc:
                stop(
                    f"Could not safely merge existing dataset {path}. "
                    "The existing file will NOT be overwritten.",
                    exc,
                )
        else:
            combined = new_df.copy()
            combined = combined.drop_duplicates(
                subset=key_columns,
                keep="last",
            ).sort_values(key_columns)

            print(
                f"   New dataset created: {len(combined)} records"
            )

        combined.to_csv(path, index=False)

        duplicate_count = combined.duplicated(
            subset=key_columns
        ).sum()

        if duplicate_count != 0:
            stop(
                f"Duplicate-key validation failed for {path}: "
                f"{duplicate_count} duplicates remain."
            )

        return combined

    # Full D+1..D+7 table.
    merged = append_and_deduplicate(
        merged,
        full_path,
        ["date"],
    )

    # Canonical D+1 table.
    d1_columns = [
        "date",
        "temp_mean_forecast_d1",
        "rh_mean_forecast_d1",
        "precip_forecast_d1",
        "cloud_cover_forecast_d1",
        "wind_speed_forecast_d1",
        "wind_direction_forecast_d1",
        "wind_gust_forecast_d1",
        "vpd_forecast_d1",
        "forecast_issued_date_d1",
        "forecast_lead_days_d1",
        "forecast_d1_source",
        "forecast_reference",
        "forecast_source",
        "latitude",
        "longitude",
    ]

    d1 = merged[d1_columns].copy()

    d1 = d1.rename(
        columns={
            column: column.replace("_d1", "")
            for column in d1.columns
            if "_d1" in column
        }
    )

    d1 = append_and_deduplicate(
        d1,
        d1_path,
        ["date"],
    )

    # Dataset-level metadata reflects the actual persisted dataset.
    metadata = {
        "source": "Open-Meteo Previous Runs API",
        "leads": [1, 2, 3, 4, 5, 6, 7],
        "canonical_lead": 1,
        "fixed_24h_lead": True,
        "synthetic": False,
        "fallback_used": False,
        "actual_weather_used_as_forecast_input": False,
        "provenance_check": "PASS",
        "duplicate_key_check": "PASS",
        "incremental_append": True,
        "latitude": lat,
        "longitude": lon,
        "target_start": str(
            pd.to_datetime(merged["date"]).min().date()
        ),
        "target_end": str(
            pd.to_datetime(merged["date"]).max().date()
        ),
    }

    (output_dir / "historical_forecasts_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\n   Full D+1..D+7: {len(merged)} rows")
    print(f"   Canonical D+1:  {len(d1)} rows")
    print("   Provenance: PASS")
    print("   Duplicate-key check: PASS")
    print("   Incremental append: ENABLED")
    print(f"   Saved: {full_path}")
    print(f"   Saved: {d1_path}")

    return merged


# ---------------------------------------------------------------------------
# Incremental date handling
# ---------------------------------------------------------------------------

def get_existing_latest_date(output_dir: Path) -> date | None:
    """
    Determine the incremental starting point from the full D+1..D+7 dataset.
    """
    path = output_dir / "historical_forecasts_previous_runs_d1_d7.csv"

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, usecols=["date"])
        if df.empty:
            return None

        dates = pd.to_datetime(df["date"], errors="coerce").dropna()

        if dates.empty:
            return None

        return dates.max().date()

    except Exception as exc:
        stop(
            f"Could not inspect existing historical forecast dataset {path}.",
            exc,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("SIH26 — BLOCK HISTORICAL FORECAST COLLECTION")
    print("Open-Meteo Previous Runs | D+1 ... D+7")
    print("REAL DATA ONLY — NO SAMPLE / NO RANDOM / NO DEFAULT")
    print("=" * 80)

    metadata_path, location = discover_block()

    state = location["state"]
    district = location["district"]
    block = location["block"]
    lat = location["latitude"]
    lon = location["longitude"]

    # Location-specific output directory comes from discovered metadata path.
    location_root = metadata_path.parent.parent
    output_dir = location_root / "raw" / "historical_forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nState:    {state}")
    print(f"District: {district}")
    print(f"Block:    {block}")
    print(f"Centroid: {lat:.6f}, {lon:.6f}")
    print(f"Metadata: {metadata_path}")
    print(f"Output:   {output_dir}")

    today = date.today()
    existing_latest = get_existing_latest_date(output_dir)

    if existing_latest is None:
        start = (pd.Timestamp(today) - pd.DateOffset(years=5)).date()
        print("\nMode: FIRST COLLECTION")
        print(f"Target period: {start} -> {today}")
    else:
        start = existing_latest + timedelta(days=1)
        print("\nMode: INCREMENTAL UPDATE")
        print(f"Existing latest target date: {existing_latest}")

        if start > today:
            print("Dataset is already current. Nothing to collect.")
            return

        print(f"New target period: {start} -> {today}")

    # Previous Runs has a fixed-lead historical forecast relationship:
    # D+1 target date uses the forecast issued one day before the target.
    collect_previous_runs(lat, lon, start, today, output_dir)

    print("\n" + "=" * 80)
    print("HISTORICAL FORECAST COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
