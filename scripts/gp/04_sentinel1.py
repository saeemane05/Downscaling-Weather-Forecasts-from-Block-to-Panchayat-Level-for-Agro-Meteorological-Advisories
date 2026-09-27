"""
SIH26 — STAGE 5: SENTINEL-1 GP DATA EXTRACTION — V4
====================================================

Universal GP-level Sentinel-1 extraction for SIH26 downscaling.

Design:
- Automatically discovers every gp_master.geojson below <project_root>/gp/
- No state/district/block/path hardcoding.
- First run: previous 5 years through today.
- Incremental runs: latest successfully stored acquisition date + 1 day through today.
- Primary source: COPERNICUS/S1_GRD
- Filters: IW, dual-pol VV+VH, 10 m resolution.
- GP-first extraction architecture:
      one GP -> ImageCollection.map() -> reduceRegion() -> Image properties
  This deliberately avoids returning FeatureCollections from ImageCollection.map()
  and avoids ImageCollection.iterate() over FeatureCollections.
- Raw observations are appended only after a complete GP/date-chunk request succeeds.
- No synthetic observations, interpolation, or imputation.
- Ascending and descending observations remain explicitly identified.
- Raw observations and processed temporal summaries are separate.
- Missing periods remain missing and are reported.

Run from:
    E:\\SIH26_Downscaling

Example:
    python scripts\\gp\\04_sentinel1.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ee
import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

COLLECTION_ID = "COPERNICUS/S1_GRD"

INITIAL_YEARS = 5
DATE_CHUNK_DAYS = 365
GP_BATCH_SIZE = 8

SCALE_M = 10
TILE_SCALE = 4
MAX_PIXELS_PER_REGION = 10_000_000

# Sentinel-1 GRD requirements
INSTRUMENT_MODE = "IW"
REQUIRED_POLARIZATIONS = ("VV", "VH")
REQUIRED_RESOLUTION_METERS = 10

# Do not silently manufacture observations.
ALLOW_SYNTHETIC = False
ALLOW_IMPUTATION = False

RAW_FILENAME = "sentinel1_observations.csv"
RAW_METADATA_FILENAME = "sentinel1_source_metadata.json"

PROCESSED_FILENAME = "gp_sentinel1_features.csv"
PROCESSED_GEOJSON_FILENAME = "gp_sentinel1_features.geojson"

# Candidate temporal windows. These are descriptive features; later modelling
# must still enforce issuance-date availability to prevent leakage.
WINDOWS_DAYS = (7, 30, 90, 365)


# ---------------------------------------------------------------------------
# PATH / DISCOVERY
# ---------------------------------------------------------------------------

def project_root_from_script() -> Path:
    # scripts/gp/04_sentinel1.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def discover_gp_masters(project_root: Path) -> List[Path]:
    masters = sorted(
        project_root.joinpath("gp").glob("**/processed/gp_master.geojson")
    )
    return masters


def block_dirs_from_master(master_path: Path) -> Tuple[Path, Path]:
    processed_dir = master_path.parent
    raw_dir = processed_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


# ---------------------------------------------------------------------------
# EARTH ENGINE
# ---------------------------------------------------------------------------

def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate with "
            "`earthengine authenticate` / the configured EE credentials, "
            "then rerun."
        ) from exc


def clean_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        raise RuntimeError("GP master is empty.")

    if "gp_id" not in gdf.columns:
        raise RuntimeError("GP master does not contain required `gp_id`.")

    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise RuntimeError("GP master has no non-empty geometries.")

    if gdf.crs is None:
        raise RuntimeError("GP master has no CRS.")

    gdf = gdf.to_crs(4326)

    # Repair only invalid geometry representations for EE ingestion.
    # This does not invent area or alter the authoritative GP identity.
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    if (~gdf.geometry.is_valid).any():
        raise RuntimeError("GP master still contains invalid geometries after repair.")

    return gdf


def geodataframe_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []

    for _, row in gdf.iterrows():
        geom = row.geometry.__geo_interface__
        properties = {
            "gp_id": str(row["gp_id"]),
            "gp_name": str(row.get("gp_name", "")),
        }

        # Preserve useful source identifiers if present.
        for col in (
            "block_name",
            "district_name",
            "state_name",
            "block_lgd",
            "district_lgd",
            "state_lgd",
        ):
            if col in row.index and pd.notna(row[col]):
                properties[col] = str(row[col])

        features.append(ee.Feature(ee.Geometry(geom), properties))

    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------------
# SENTINEL-1 COLLECTION
# ---------------------------------------------------------------------------

def build_collection(
    start_date: str,
    end_date_exclusive: str,
    geometry: ee.Geometry,
) -> ee.ImageCollection:
    """
    Build a strict dual-pol IW 10 m Sentinel-1 GRD collection.

    End date is exclusive.
    """
    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(start_date, end_date_exclusive)
        .filterBounds(geometry)
        .filter(ee.Filter.eq("instrumentMode", INSTRUMENT_MODE))
        .filter(ee.Filter.eq("resolution_meters", REQUIRED_RESOLUTION_METERS))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
    )

    return collection


def prepare_image(image: ee.Image) -> ee.Image:
    """
    Prepare per-pixel variables.

    S1_GRD VV/VH are retained in the dataset's native dB representation.
    VV-VH is calculated in dB, which is the logarithmic polarization
    difference and avoids an invalid division of dB values.

    `valid_fraction` is based on the joint VV/VH mask.
    """
    image = ee.Image(image)

    vv = image.select("VV").rename("vv_db")
    vh = image.select("VH").rename("vh_db")
    angle = image.select("angle").rename("incidence_angle_deg")

    joint_valid = (
        vv.mask()
        .And(vh.mask())
        .And(angle.mask())
    )

    vv = vv.updateMask(joint_valid)
    vh = vh.updateMask(joint_valid)
    angle = angle.updateMask(joint_valid)

    vv_vh_diff = vv.subtract(vh).rename("vv_vh_diff_db")

    # Unmasked 0/1 validity layer for a true area-coverage fraction.
    valid_fraction_pixel = (
        joint_valid
        .unmask(0)
        .rename("valid_fraction_pixel")
        .toFloat()
    )

    return (
        ee.Image.cat(
            vv,
            vh,
            vv_vh_diff,
            angle,
            valid_fraction_pixel,
        )
    )


# ---------------------------------------------------------------------------
# GP-FIRST EXTRACTION
# ---------------------------------------------------------------------------

def extract_one_gp(
    gp_feature: ee.Feature,
    collection: ee.ImageCollection,
) -> ee.ImageCollection:
    """
    Return an ImageCollection in which each Sentinel-1 image carries
    GP-level statistics as image properties.

    This is the key architecture:
        ImageCollection.map() -> Image
        Image.reduceRegion() -> Dictionary
        Dictionary attached to Image properties

    It never returns a FeatureCollection from ImageCollection.map().
    """
    gp_feature = ee.Feature(gp_feature)
    gp_geometry = gp_feature.geometry()

    def image_to_feature_image(img):
        img = ee.Image(img)
        prepared = prepare_image(img)

        stats = prepared.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=gp_geometry,
            scale=SCALE_M,
            bestEffort=True,
            maxPixels=MAX_PIXELS_PER_REGION,
            tileScale=TILE_SCALE,
        )

        # Only keep images for which the core S1 variables produced values.
        # The image itself remains the mapped return type.
        return (
            img.set({
                "gp_id": gp_feature.get("gp_id"),
                "gp_name": gp_feature.get("gp_name"),
                "s1_vv_db": stats.get("vv_db"),
                "s1_vh_db": stats.get("vh_db"),
                "s1_vv_vh_diff_db": stats.get("vv_vh_diff_db"),
                "s1_incidence_angle_deg": stats.get("incidence_angle_deg"),
                "s1_valid_fraction": stats.get("valid_fraction_pixel"),
            })
        )

    return collection.map(image_to_feature_image)


def image_collection_to_rows(
    gp_id: str,
    gp_name: str,
    collection: ee.ImageCollection,
) -> List[Dict]:
    """
    Materialize the GP's image properties with one server request.
    """
    properties = [
        "system:index",
        "system:time_start",
        "orbitProperties_pass",
        "relativeOrbitNumber_start",
        "relativeOrbitNumber_stop",
        "absoluteOrbitNumber_start",
        "absoluteOrbitNumber_stop",
        "platform_number",
        "instrumentMode",
        "resolution_meters",
        "s1_vv_db",
        "s1_vh_db",
        "s1_vv_vh_diff_db",
        "s1_incidence_angle_deg",
        "s1_valid_fraction",
    ]

    rows = collection.aggregate_array("system:index").getInfo()
    if not rows:
        return []

    # A single getInfo for the complete set of per-image property records.
    records = collection.toList(collection.size()).map(
        lambda img: ee.Image(img).toDictionary(properties)
    ).getInfo()

    result = []
    for record in records:
        if not record:
            continue

        vv = record.get("s1_vv_db")
        vh = record.get("s1_vh_db")
        diff = record.get("s1_vv_vh_diff_db")

        # A GP/image row is usable only if the core radar observations exist.
        if vv is None or vh is None or diff is None:
            continue

        ts = record.get("system:time_start")
        if ts is None:
            continue

        acquisition = pd.to_datetime(ts, unit="ms", utc=True)

        result.append(
            {
                "gp_id": gp_id,
                "gp_name": gp_name,
                "image_id": record.get("system:index"),
                "acquisition_datetime_utc": acquisition.isoformat(),
                "acquisition_date": acquisition.date().isoformat(),
                "orbit_pass": record.get("orbitProperties_pass"),
                "relative_orbit_start": record.get("relativeOrbitNumber_start"),
                "relative_orbit_stop": record.get("relativeOrbitNumber_stop"),
                "absolute_orbit_start": record.get("absoluteOrbitNumber_start"),
                "absolute_orbit_stop": record.get("absoluteOrbitNumber_stop"),
                "platform_number": record.get("platform_number"),
                "instrument_mode": record.get("instrumentMode"),
                "resolution_meters": record.get("resolution_meters"),
                "vv_db_mean": vv,
                "vh_db_mean": vh,
                "vv_vh_diff_db_mean": diff,
                "incidence_angle_deg_mean": record.get(
                    "s1_incidence_angle_deg"
                ),
                "valid_pixel_fraction": record.get(
                    "s1_valid_fraction"
                ),
                "synthetic": False,
                "imputed": False,
            }
        )

    return result


def server_extract(
    gp_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date_exclusive: str,
) -> List[Dict]:
    """
    Extract all GP x S1 observations for a date chunk.

    GP-first is intentionally used because the EE Python API expects
    ImageCollection.map() callbacks to return Images, not FeatureCollections.
    """
    block_geometry = ee.Geometry(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [x, y]
                    for x, y in gp_gdf.to_crs(4326)
                    .geometry.unary_union.exterior.coords
                ]
            ],
        }
    )

    collection = build_collection(
        start_date,
        end_date_exclusive,
        block_geometry,
    )

    image_count = collection.size().getInfo()

    if image_count == 0:
        return []

    all_rows: List[Dict] = []

    for offset in range(0, len(gp_gdf), GP_BATCH_SIZE):
        batch = gp_gdf.iloc[offset: offset + GP_BATCH_SIZE].copy()
        gp_fc = geodataframe_to_ee_fc(batch)

        # One GP at a time. This is intentionally conservative to avoid
        # nested reduceRegions/reduceRegion aggregation failures.
        for _, gp_row in batch.iterrows():
            gp_id = str(gp_row["gp_id"])
            gp_name = str(gp_row.get("gp_name", ""))

            gp_feature = ee.Feature(
                ee.Geometry(gp_row.geometry.__geo_interface__),
                {
                    "gp_id": gp_id,
                    "gp_name": gp_name,
                },
            )

            gp_collection = extract_one_gp(
                gp_feature,
                collection,
            )

            rows = image_collection_to_rows(
                gp_id,
                gp_name,
                gp_collection,
            )

            all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# RAW STORAGE / QC
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "gp_id",
    "gp_name",
    "image_id",
    "acquisition_datetime_utc",
    "acquisition_date",
    "orbit_pass",
    "relative_orbit_start",
    "relative_orbit_stop",
    "absolute_orbit_start",
    "absolute_orbit_stop",
    "platform_number",
    "instrument_mode",
    "resolution_meters",
    "vv_db_mean",
    "vh_db_mean",
    "vv_vh_diff_db_mean",
    "incidence_angle_deg_mean",
    "valid_pixel_fraction",
    "synthetic",
    "imputed",
]


def read_raw(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = pd.read_csv(raw_path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Existing Sentinel-1 raw file is missing required columns: {missing}"
        )

    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    return df[RAW_COLUMNS].copy()


def deduplicate_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    key = ["gp_id", "image_id"]

    before = len(df)
    out = (
        df.sort_values(
            ["acquisition_datetime_utc", "gp_id", "image_id"]
        )
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )

    if len(out) != before:
        print(f"  Removed {before - len(out)} duplicate GP×image rows.")

    return out


def finite_or_nan(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def qc_raw(
    df: pd.DataFrame,
    expected_gp_ids: Sequence[str],
    chunk_start: str,
    chunk_end_inclusive: str,
) -> Dict:
    if df.empty:
        return {
            "status": "PASS_EMPTY_PERIOD",
            "rows": 0,
            "unique_gps": 0,
            "unique_images": 0,
            "duplicate_gp_image": 0,
            "out_of_range_rows": 0,
            "invalid_core_values": 0,
            "invalid_coverage": 0,
        }

    work = df.copy()

    work["acquisition_date"] = pd.to_datetime(
        work["acquisition_date"], errors="coerce"
    )

    start = pd.Timestamp(chunk_start)
    end = pd.Timestamp(chunk_end_inclusive)

    out_of_range = (
        work["acquisition_date"].isna()
        | (work["acquisition_date"] < start)
        | (work["acquisition_date"] > end)
    )

    core_cols = [
        "vv_db_mean",
        "vh_db_mean",
        "vv_vh_diff_db_mean",
        "incidence_angle_deg_mean",
        "valid_pixel_fraction",
    ]

    invalid_core = 0
    for col in core_cols:
        vals = finite_or_nan(work[col])
        invalid_core += int(vals.isna().sum())

    coverage = finite_or_nan(work["valid_pixel_fraction"])
    invalid_coverage = int(
        ((coverage < 0) | (coverage > 1)).fillna(True).sum()
    )

    duplicates = int(
        work.duplicated(["gp_id", "image_id"], keep=False).sum()
    )

    status = "PASS"
    if out_of_range.any() or invalid_core > 0 or invalid_coverage > 0:
        status = "FAIL"
    elif duplicates > 0:
        status = "FAIL"

    return {
        "status": status,
        "rows": int(len(work)),
        "unique_gps": int(work["gp_id"].nunique()),
        "unique_images": int(work["image_id"].nunique()),
        "duplicate_gp_image": duplicates,
        "out_of_range_rows": int(out_of_range.sum()),
        "invalid_core_values": invalid_core,
        "invalid_coverage": invalid_coverage,
        "expected_gp_count": len(expected_gp_ids),
        "missing_gps": sorted(
            set(map(str, expected_gp_ids))
            - set(work["gp_id"].astype(str))
        ),
    }


def append_raw(
    raw_path: Path,
    existing: pd.DataFrame,
    new_rows: List[Dict],
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows, columns=RAW_COLUMNS)

    combined = pd.concat(
        [existing, new_df],
        ignore_index=True,
    )

    combined = deduplicate_raw(combined)

    # Store deterministic ordering.
    combined["acquisition_datetime_utc"] = pd.to_datetime(
        combined["acquisition_datetime_utc"], utc=True, errors="coerce"
    )

    combined = combined.sort_values(
        ["acquisition_datetime_utc", "gp_id", "image_id"]
    ).reset_index(drop=True)

    # CSV cannot reliably retain timezone-aware dtype; write ISO strings.
    combined["acquisition_datetime_utc"] = combined[
        "acquisition_datetime_utc"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    combined.to_csv(raw_path, index=False)

    return combined


# ---------------------------------------------------------------------------
# PROCESSED FEATURES
# ---------------------------------------------------------------------------

def safe_stats(series: pd.Series, prefix: str) -> Dict:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_count": 0,
        }

    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_std": float(values.std(ddof=1))
        if len(values) > 1 else 0.0,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_count": int(len(values)),
    }


def build_processed_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    work = raw.copy()

    work["acquisition_date"] = pd.to_datetime(
        work["acquisition_date"],
        errors="coerce",
    )

    for col in (
        "vv_db_mean",
        "vh_db_mean",
        "vv_vh_diff_db_mean",
        "incidence_angle_deg_mean",
        "valid_pixel_fraction",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    output = []

    for gp_id, gp in work.groupby("gp_id", sort=True):
        gp_name = str(gp["gp_name"].iloc[0])

        row = {
            "gp_id": str(gp_id),
            "gp_name": gp_name,
            "s1_observation_count": int(len(gp)),
            "s1_unique_image_count": int(gp["image_id"].nunique()),
            "s1_first_date": (
                gp["acquisition_date"].min().date().isoformat()
                if gp["acquisition_date"].notna().any()
                else None
            ),
            "s1_last_date": (
                gp["acquisition_date"].max().date().isoformat()
                if gp["acquisition_date"].notna().any()
                else None
            ),
            "s1_ascending_count": int(
                (gp["orbit_pass"].astype(str).str.upper() == "ASCENDING").sum()
            ),
            "s1_descending_count": int(
                (gp["orbit_pass"].astype(str).str.upper() == "DESCENDING").sum()
            ),
        }

        for col, prefix in (
            ("vv_db_mean", "s1_vv_db"),
            ("vh_db_mean", "s1_vh_db"),
            ("vv_vh_diff_db_mean", "s1_vv_vh_diff_db"),
            ("incidence_angle_deg_mean", "s1_incidence_angle_deg"),
            ("valid_pixel_fraction", "s1_valid_fraction"),
        ):
            row.update(safe_stats(gp[col], prefix))

        # Orbit-separated long-history summaries.
        for orbit in ("ASCENDING", "DESCENDING"):
            sub = gp[
                gp["orbit_pass"].astype(str).str.upper() == orbit
            ]

            suffix = orbit.lower()
            row[f"s1_{suffix}_count"] = int(len(sub))

            for col, prefix in (
                ("vv_db_mean", f"s1_vv_db_{suffix}"),
                ("vh_db_mean", f"s1_vh_db_{suffix}"),
                ("vv_vh_diff_db_mean", f"s1_vv_vh_diff_db_{suffix}"),
                ("incidence_angle_deg_mean", f"s1_incidence_angle_deg_{suffix}"),
            ):
                row.update(safe_stats(sub[col], prefix))

        # Recent temporal summaries, calculated independently for each GP.
        # These are descriptive summaries; model-ready features must later be
        # generated relative to each forecast issuance timestamp.
        latest_date = gp["acquisition_date"].max()

        if pd.notna(latest_date):
            for days in WINDOWS_DAYS:
                cutoff = latest_date - pd.Timedelta(days=days)
                sub = gp[
                    (gp["acquisition_date"] >= cutoff)
                    & (gp["acquisition_date"] <= latest_date)
                ]

                row[f"s1_obs_count_last_{days}d"] = int(len(sub))

                for col, prefix in (
                    ("vv_db_mean", f"s1_vv_last_{days}d"),
                    ("vh_db_mean", f"s1_vh_last_{days}d"),
                    (
                        "vv_vh_diff_db_mean",
                        f"s1_vv_vh_diff_last_{days}d",
                    ),
                    (
                        "incidence_angle_deg_mean",
                        f"s1_incidence_angle_last_{days}d",
                    ),
                ):
                    values = pd.to_numeric(
                        sub[col], errors="coerce"
                    ).dropna()

                    row[f"{prefix}_mean"] = (
                        float(values.mean()) if not values.empty else math.nan
                    )
                    row[f"{prefix}_median"] = (
                        float(values.median()) if not values.empty else math.nan
                    )

        output.append(row)

    return pd.DataFrame(output).sort_values("gp_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# GEOJSON OUTPUT
# ---------------------------------------------------------------------------

def write_processed_geojson(
    features: pd.DataFrame,
    gp_master: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    if features.empty:
        return

    geometry_cols = ["gp_id", "geometry"]
    master_geometry = gp_master[geometry_cols].copy()

    merged = master_geometry.merge(
        features,
        on="gp_id",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        return

    out = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=gp_master.crs,
    )

    out.to_file(output_path, driver="GeoJSON")


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------

def write_metadata(
    path: Path,
    master_path: Path,
    raw_df: pd.DataFrame,
    fetch_mode: str,
    initial_or_incremental_start: str,
    fetch_end: str,
    status: str,
) -> None:
    metadata = {
        "stage": "SIH26_STAGE_5_SENTINEL1_GP",
        "collection": COLLECTION_ID,
        "master_path": str(master_path),
        "fetch_mode": fetch_mode,
        "fetch_start": initial_or_incremental_start,
        "fetch_end": fetch_end,
        "initial_years": INITIAL_YEARS,
        "incremental_rule": "latest stored acquisition date + 1 day",
        "date_chunk_days": DATE_CHUNK_DAYS,
        "gp_batch_size": GP_BATCH_SIZE,
        "scale_m": SCALE_M,
        "instrument_mode": INSTRUMENT_MODE,
        "required_polarizations": list(REQUIRED_POLARIZATIONS),
        "required_resolution_meters": REQUIRED_RESOLUTION_METERS,
        "synthetic": False,
        "imputation": False,
        "raw_rows": int(len(raw_df)),
        "unique_gps": int(raw_df["gp_id"].nunique()) if not raw_df.empty else 0,
        "unique_images": int(raw_df["image_id"].nunique()) if not raw_df.empty else 0,
        "last_successful_acquisition_date": (
            str(raw_df["acquisition_date"].max())
            if not raw_df.empty
            else None
        ),
        "status": status,
        "notes": [
            "Raw GP×image observations are retained.",
            "Ascending and descending orbit observations are not silently merged at acquisition level.",
            "No synthetic observations, interpolation, or imputation.",
            "Processed temporal summaries are descriptive and must be shifted to forecast issuance time during model feature construction.",
        ],
    }

    path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# DATE WINDOWS
# ---------------------------------------------------------------------------

def initial_start_date() -> date:
    # Exact calendar-year subtraction without external dateutil dependency.
    today = date.today()
    try:
        return today.replace(year=today.year - INITIAL_YEARS)
    except ValueError:
        return today.replace(
            year=today.year - INITIAL_YEARS,
            month=2,
            day=28,
        )


def determine_fetch_window(
    raw_df: pd.DataFrame,
) -> Tuple[str, str, str]:
    today = date.today()

    if raw_df.empty:
        start = initial_start_date()
        return start.isoformat(), today.isoformat(), "INITIAL_WINDOW"

    dates = pd.to_datetime(
        raw_df["acquisition_date"],
        errors="coerce",
    ).dropna()

    if dates.empty:
        start = initial_start_date()
        return start.isoformat(), today.isoformat(), "INITIAL_WINDOW"

    latest = dates.max().date()

    if latest > today:
        raise RuntimeError(
            f"Existing Sentinel-1 raw data has future acquisition date {latest}."
        )

    start = latest + timedelta(days=1)

    if start > today:
        return start.isoformat(), today.isoformat(), "UP_TO_DATE"

    return start.isoformat(), today.isoformat(), "INCREMENTAL"


def date_chunks(
    start: date,
    end_inclusive: date,
    chunk_days: int,
) -> Iterable[Tuple[date, date]]:
    current = start

    while current <= end_inclusive:
        chunk_end = min(
            current + timedelta(days=chunk_days - 1),
            end_inclusive,
        )
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ---------------------------------------------------------------------------
# BLOCK PROCESSING
# ---------------------------------------------------------------------------

def process_block(master_path: Path) -> None:
    print("\n" + "=" * 88)
    print(f"BLOCK / GP MASTER: {master_path}")
    print("=" * 88)

    gp_master = gpd.read_file(master_path)
    gp_master = clean_geometry(gp_master)

    gp_ids = gp_master["gp_id"].astype(str).tolist()

    raw_dir, processed_dir = block_dirs_from_master(master_path)

    raw_path = raw_dir / RAW_FILENAME
    metadata_path = raw_dir / RAW_METADATA_FILENAME
    processed_path = processed_dir / PROCESSED_FILENAME
    processed_geojson_path = processed_dir / PROCESSED_GEOJSON_FILENAME

    raw_df = read_raw(raw_path)

    print(f"GP count: {len(gp_master)}")
    print(f"Existing raw rows: {len(raw_df)}")

    fetch_start, fetch_end, mode = determine_fetch_window(raw_df)

    print(f"Fetch mode: {mode}")
    print(f"Fetch start: {fetch_start}")
    print(f"Fetch end:   {fetch_end}")

    if mode == "UP_TO_DATE":
        print("No new acquisition dates to fetch.")
        if not raw_df.empty:
            features = build_processed_features(raw_df)
            features.to_csv(processed_path, index=False)
            write_processed_geojson(
                features,
                gp_master,
                processed_geojson_path,
            )
            write_metadata(
                metadata_path,
                master_path,
                raw_df,
                mode,
                fetch_start,
                fetch_end,
                "UP_TO_DATE",
            )
        return

    start = date.fromisoformat(fetch_start)
    end = date.fromisoformat(fetch_end)

    any_new_observations = False

    for chunk_start, chunk_end in date_chunks(
        start,
        end,
        DATE_CHUNK_DAYS,
    ):
        print(
            f"\nDate chunk: {chunk_start.isoformat()} -> "
            f"{chunk_end.isoformat()}"
        )

        t0 = time.time()

        # One server-side collection check before doing GP requests.
        block_union = gp_master.geometry.unary_union
        block_fc = ee.FeatureCollection(
            geodataframe_to_ee_fc(gp_master)
        )

        block_geometry = ee.Geometry(
            block_union.__geo_interface__
        )

        collection = build_collection(
            chunk_start.isoformat(),
            (chunk_end + timedelta(days=1)).isoformat(),
            block_geometry,
        )

        image_count = collection.size().getInfo()

        print(f"  Sentinel-1 images intersecting block: {image_count}")

        if image_count == 0:
            print("  No Sentinel-1 acquisitions in this period.")
            print("  QC status: PASS_EMPTY_PERIOD")
            continue

        chunk_rows: List[Dict] = []

        # Process GP batches, but deliberately keep each GP's EE operation
        # independent. This is slower than nested server aggregation but much
        # more robust for the EE Python API and small/medium GP counts.
        batch_count = math.ceil(len(gp_master) / GP_BATCH_SIZE)

        for batch_idx, offset in enumerate(
            range(0, len(gp_master), GP_BATCH_SIZE),
            start=1,
        ):
            batch = gp_master.iloc[
                offset: offset + GP_BATCH_SIZE
            ].copy()

            print(
                f"  GP batch {batch_idx}/{batch_count} "
                f"({len(batch)} GPs)..."
            )

            batch_t0 = time.time()

            for _, gp_row in batch.iterrows():
                gp_id = str(gp_row["gp_id"])
                gp_name = str(gp_row.get("gp_name", ""))

                gp_feature = ee.Feature(
                    ee.Geometry(gp_row.geometry.__geo_interface__),
                    {
                        "gp_id": gp_id,
                        "gp_name": gp_name,
                    },
                )

                gp_collection = extract_one_gp(
                    gp_feature,
                    collection,
                )

                rows = image_collection_to_rows(
                    gp_id,
                    gp_name,
                    gp_collection,
                )

                chunk_rows.extend(rows)

            print(
                f"    rows={len(chunk_rows)} "
                f"time={time.time() - batch_t0:.1f}s"
            )

        chunk_df = pd.DataFrame(chunk_rows, columns=RAW_COLUMNS)

        # Validate only the newly extracted chunk before committing it.
        qc = qc_raw(
            chunk_df,
            gp_ids,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )

        print(f"  QC status: {qc['status']}")
        print(f"  Raw observations: {qc['rows']}")
        print(f"  Unique GPs represented: {qc['unique_gps']}")
        print(f"  Unique S1 images: {qc['unique_images']}")

        if qc["status"] == "FAIL":
            raise RuntimeError(
                f"Sentinel-1 QC failed for {master_path.name} "
                f"{chunk_start} -> {chunk_end}: {qc}"
            )

        # Empty periods are valid and are NOT written as fake observations.
        if chunk_df.empty:
            print("  No valid GP-level observations in this period.")
            continue

        raw_df = append_raw(
            raw_path,
            raw_df,
            chunk_rows,
        )

        any_new_observations = True

        write_metadata(
            metadata_path,
            master_path,
            raw_df,
            mode,
            fetch_start,
            fetch_end,
            "IN_PROGRESS",
        )

        print(
            f"  Raw checkpoint saved: {len(raw_df)} total rows."
        )
        print(
            f"  Chunk time: {time.time() - t0:.1f}s"
        )

    if raw_df.empty:
        write_metadata(
            metadata_path,
            master_path,
            raw_df,
            mode,
            fetch_start,
            fetch_end,
            "NO_OBSERVATIONS",
        )
        raise RuntimeError(
            "No Sentinel-1 observations were obtained for this block. "
            "The script did not create processed features and did not "
            "manufacture missing values."
        )

    # Final raw QC after all chunks.
    raw_df = deduplicate_raw(raw_df)

    if raw_df["synthetic"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("Synthetic flag detected in raw Sentinel-1 data.")

    if raw_df["imputed"].astype(str).str.lower().eq("true").any():
        raise RuntimeError("Imputation flag detected in raw Sentinel-1 data.")

    # Final processed GP features.
    features = build_processed_features(raw_df)

    if features.empty:
        raise RuntimeError(
            "Raw Sentinel-1 observations exist but processed features are empty."
        )

    if features["gp_id"].duplicated().any():
        raise RuntimeError(
            "Processed Sentinel-1 features contain duplicate GP IDs."
        )

    features.to_csv(processed_path, index=False)

    write_processed_geojson(
        features,
        gp_master,
        processed_geojson_path,
    )

    write_metadata(
        metadata_path,
        master_path,
        raw_df,
        mode,
        fetch_start,
        fetch_end,
        "SUCCESS",
    )

    print("\n  FINAL STATUS: SUCCESS")
    print(f"  Raw rows:       {len(raw_df)}")
    print(f"  Processed GPs:  {len(features)}")
    print(f"  Raw file:       {raw_path}")
    print(f"  Feature file:   {processed_path}")
    print(f"  GeoJSON file:   {processed_geojson_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    project_root = project_root_from_script()

    print("=" * 88)
    print("SIH26 — STAGE 5: SENTINEL-1 GP DATA EXTRACTION — V4")
    print("=" * 88)
    print(f"Project root: {project_root}")
    print(f"Collection:   {COLLECTION_ID}")
    print(f"Initial span: last {INITIAL_YEARS} years")
    print("Incremental:  latest stored acquisition + 1 day")
    print(f"GP batch:     {GP_BATCH_SIZE}")
    print(f"Date chunk:   {DATE_CHUNK_DAYS} days")
    print(f"Scale:        {SCALE_M} m")
    print("Synthetic:    DISABLED")
    print("Imputation:   DISABLED")
    print("Architecture: GP-first reduceRegion()")
    print()

    masters = discover_gp_masters(project_root)

    if not masters:
        raise RuntimeError(
            f"No gp_master.geojson files found below {project_root / 'gp'}"
        )

    print(f"Discovered GP masters: {len(masters)}")
    for master in masters:
        print(f"  - {master}")

    initialize_ee()

    for master in masters:
        process_block(master)

    print("\n" + "=" * 88)
    print("ALL DISCOVERED GP MASTERS COMPLETED SUCCESSFULLY")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
