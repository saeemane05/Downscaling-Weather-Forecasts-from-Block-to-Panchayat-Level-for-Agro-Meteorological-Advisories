from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import ee
import geopandas as gpd
import pandas as pd


# ============================================================================
# SIH26 — STAGE 7: GP LAND SURFACE TEMPERATURE — FAST / QUALITY CONTROLLED
# ============================================================================
#
# Design decision:
#   Do NOT reduce every Landsat scene separately.
#
#   Instead, create monthly Landsat-8/9 LST composites and reduce each
#   composite over all GPs in one Earth Engine request.
#
#   This changes ~356 scene-level requests for a 5-year Sinnar history into
#   ~60 monthly composites, while preserving the date/month and the list of
#   source scenes in metadata.
#
#   This is appropriate for the current downscaling feature set because LST
#   is a dynamic contextual predictor, not the target itself.
#
# Quality rules:
#   1. Landsat Collection 2 Level 2 Tier 1, L2SP only.
#   2. ST_B10 is scaled using the official C2 factor/offset.
#   3. QA_PIXEL masks fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
#   4. Invalid/zero ST_B10 pixels remain masked.
#   5. No synthetic values.
#   6. No interpolation.
#   7. No imputation.
#   8. A monthly composite is allowed to have missing pixels.
#   9. A GP is accepted only if it has at least one valid monthly LST value.
#
# Source:
#   USGS Landsat 8/9 Collection 2 Level 2 Tier 1, Google Earth Engine.
#
# Final compact GP features:
#   lst_mean_c
#   lst_median_c
#   lst_std_c
#   lst_valid_months
#
# The monthly table remains available for later leakage-safe lag construction.
# ============================================================================


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LST_SOURCE = (
    "USGS Landsat 8/9 Collection 2 Level 2 Tier 1, "
    "Google Earth Engine"
)

LST_RESOLUTION_M = 30
TILE_SCALE = 4
MAX_PIXELS_PER_REGION = 10_000_000

# Number of monthly composites evaluated in one EE request.
# 6 gives a good balance between speed and request size.
MONTH_BATCH_SIZE = 4

DEFAULT_HISTORY_YEARS = 5

L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

RAW_FILENAME = "lst_gp_monthly_observations.csv"
PROCESSED_FILENAME = "gp_lst_features.csv"
PROCESSED_GEOJSON_FILENAME = "gp_lst_features.geojson"
METADATA_FILENAME = "lst_source_metadata.json"


# ============================================================================
# LOCATION / PATHS
# ============================================================================

def norm(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_location() -> Tuple[str, str, str, bool]:
    parser = argparse.ArgumentParser(
        description="SIH26 universal GP Landsat LST extractor"
    )
    parser.add_argument("--state")
    parser.add_argument("--district")
    parser.add_argument("--block")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the full configured LST history.",
    )

    args = parser.parse_args()

    supplied = [args.state, args.district, args.block]

    if all(supplied):
        return (
            args.state.strip(),
            args.district.strip(),
            args.block.strip(),
            bool(args.force),
        )

    if any(supplied):
        raise SystemExit(
            "Provide all three: --state --district --block."
        )

    print("=" * 88)
    print("SIH26 — STAGE 7: GP LAND SURFACE TEMPERATURE")
    print("=" * 88)

    state = input("Enter State    : ").strip()
    district = input("Enter District : ").strip()
    block = input("Enter Block    : ").strip()

    if not all([state, district, block]):
        raise SystemExit("State, district and block are required.")

    return state, district, block, False


def find_gp_master(
    state: str,
    district: str,
    block: str,
) -> Path:
    gp_root = PROJECT_ROOT / "gp"

    state_n = norm(state)
    district_n = norm(district)
    block_n = norm(block)

    exact = (
        gp_root
        / state_n
        / district_n
        / block
        / "processed"
        / "gp_master.geojson"
    )

    if exact.exists():
        return exact

    matches = []
    for path in gp_root.glob("**/processed/gp_master.geojson"):
        parts = {norm(p) for p in path.parts}
        if (
            state_n in parts
            and district_n in parts
            and block_n in parts
        ):
            matches.append(path)

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"No GP master found for {state}/{district}/{block}."
        )

    raise RuntimeError(
        "Multiple GP masters match the requested location:\n"
        + "\n".join(str(p) for p in matches)
    )


def load_gp_master(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise RuntimeError(f"GP master is empty: {path}")

    if "gp_id" not in gdf.columns:
        raise RuntimeError(f"`gp_id` is missing from: {path}")

    if gdf.crs is None:
        raise RuntimeError(f"GP master has no CRS: {path}")

    gdf = gdf.to_crs(4326).copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    if (~gdf.geometry.is_valid).any():
        raise RuntimeError(
            "Some GP geometries remain invalid after geometry repair."
        )

    gdf["gp_id"] = gdf["gp_id"].astype(str)

    if gdf["gp_id"].duplicated().any():
        raise RuntimeError("GP master contains duplicate gp_id values.")

    return gdf


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []

    for _, row in gdf.iterrows():
        features.append(
            ee.Feature(
                ee.Geometry(row.geometry.__geo_interface__),
                {
                    "gp_id": str(row["gp_id"]),
                    "gp_name": str(row.get("gp_name", "")),
                },
            )
        )

    return ee.FeatureCollection(features)


# ============================================================================
# EARTH ENGINE
# ============================================================================

def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate Earth Engine."
        ) from exc


# ============================================================================
# DATE WINDOW
# ============================================================================

def infer_history_start(block_master_path: Path) -> str:
    block_root = block_master_path.parents[3]

    candidates = [
        block_root / "raw" / "02_open_meteo_actual.csv",
        block_root / "raw" / "open_meteo_actual.csv",
        block_root / "processed" / "weather_observations.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)

            for col in df.columns:
                if col.lower() in {
                    "date",
                    "target_date",
                    "valid_date",
                    "time",
                }:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().any():
                        return parsed.min().date().isoformat()
        except Exception:
            continue

    return (
        date.today() - timedelta(days=365 * DEFAULT_HISTORY_YEARS)
    ).isoformat()


def month_start(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize().replace(day=1)


def month_end_exclusive(start: pd.Timestamp) -> pd.Timestamp:
    return start + pd.offsets.MonthBegin(1)


def determine_fetch_window(
    raw_path: Path,
    history_start: str,
    force: bool,
) -> Tuple[str, str, bool]:
    end_date = date.today().isoformat()

    if force or not raw_path.exists():
        return history_start, end_date, True

    try:
        existing = pd.read_csv(raw_path)

        if "month_start" not in existing.columns:
            return history_start, end_date, True

        parsed = pd.to_datetime(
            existing["month_start"],
            errors="coerce",
        )

        if not parsed.notna().any():
            return history_start, end_date, True

        latest = parsed.max().date()
        next_month = (
            pd.Timestamp(latest) + pd.offsets.MonthBegin(1)
        ).date()

        if next_month > date.today():
            return next_month.isoformat(), end_date, False

        return next_month.isoformat(), end_date, False

    except Exception:
        return history_start, end_date, True


# ============================================================================
# LANDSAT PREPROCESSING
# ============================================================================

def mask_and_prepare(image: ee.Image) -> ee.Image:
    """
    Convert ST_B10 to Celsius and apply official QA_PIXEL masking.

    Collection 2 ST_B10:
        Kelvin = DN * 0.00341802 + 149.0
        Celsius = Kelvin - 273.15
    """
    qa = image.select("QA_PIXEL")

    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)  # fill
        .And(qa.bitwiseAnd(1 << 1).eq(0))  # dilated cloud
        .And(qa.bitwiseAnd(1 << 2).eq(0))  # cirrus
        .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
        .And(qa.bitwiseAnd(1 << 5).eq(0))  # snow
    )

    st = image.select("ST_B10")

    # DN=0 is fill/no temperature. Do not invent a value.
    mask = mask.And(st.gt(0))

    lst_c = (
        st
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("lst_c")
    )

    return (
        lst_c
        .updateMask(mask)
        .copyProperties(
            image,
            [
                "system:index",
                "system:time_start",
                "LANDSAT_PRODUCT_ID",
                "SPACECRAFT_ID",
                "DATE_ACQUIRED",
                "CLOUD_COVER",
                "PROCESSING_LEVEL",
            ],
        )
    )


def build_collection(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    end_exclusive = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    l8 = (
        ee.ImageCollection(L8_COLLECTION)
        .filterDate(start_date, end_exclusive)
        .filterBounds(region)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .filter(ee.Filter.lte("CLOUD_COVER", 60))
        .select(["ST_B10", "QA_PIXEL"])
    )

    l9 = (
        ee.ImageCollection(L9_COLLECTION)
        .filterDate(start_date, end_exclusive)
        .filterBounds(region)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .filter(ee.Filter.lte("CLOUD_COVER", 60))
        .select(["ST_B10", "QA_PIXEL"])
    )

    return (
        l8.merge(l9)
        .sort("system:time_start")
        .map(mask_and_prepare)
    )


# ============================================================================

# ============================================================================
# PREFLIGHT DIAGNOSTIC
# ============================================================================

def run_lst_preflight(
    gp_master: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """
    Run a small, bounded Earth Engine diagnostic before the full extraction.

    Purpose:
      - distinguish a source/QA/masking problem from a reduceRegions/property
        naming problem;
      - inspect one real Landsat image and one GP;
      - fail early instead of spending ~15 minutes on a known-empty run.

    No values produced here are written to the dataset.
    """
    print()
    print("=" * 88)
    print("LST PREFLIGHT DIAGNOSTIC")
    print("=" * 88)

    region = ee.Geometry(gp_master.iloc[0].geometry.__geo_interface__)
    gp_fc_one = gdf_to_ee_fc(gp_master.iloc[:1].copy())

    end_exclusive = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    def raw_collection(collection_id: str):
        return (
            ee.ImageCollection(collection_id)
            .filterDate(start_date, end_exclusive)
            .filterBounds(region)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
            .filter(ee.Filter.lte("CLOUD_COVER", 60))
            .select(["ST_B10", "QA_PIXEL"])
            .sort("system:time_start")
        )

    l8 = raw_collection(L8_COLLECTION)
    l9 = raw_collection(L9_COLLECTION)
    merged = l8.merge(l9).sort("system:time_start")

    counts = ee.Dictionary({
        "l8_count": l8.size(),
        "l9_count": l9.size(),
        "merged_count": merged.size(),
    }).getInfo()

    print(f"  L8 L2SP scenes:       {counts.get('l8_count', 0)}")
    print(f"  L9 L2SP scenes:       {counts.get('l9_count', 0)}")
    print(f"  Combined scenes:      {counts.get('merged_count', 0)}")

    if int(counts.get("merged_count", 0) or 0) == 0:
        raise RuntimeError(
            "LST PREFLIGHT FAILED: no Landsat 8/9 L2SP scenes satisfy the "
            "date, region, cloud-cover, and processing-level filters."
        )

    image = ee.Image(merged.first())
    image_info = image.select(["ST_B10", "QA_PIXEL"]).reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=region,
        scale=LST_RESOLUTION_M,
        bestEffort=True,
        maxPixels=MAX_PIXELS_PER_REGION,
    ).getInfo()

    print(f"  Example scene:         {image.get('LANDSAT_PRODUCT_ID').getInfo()}")
    print(f"  Example raw ST_B10:    {image_info.get('ST_B10')}")
    print(f"  Example raw QA_PIXEL:  {image_info.get('QA_PIXEL')}")

    # Earth Engine may type a mapped/cast server-side object as a generic
    # Element in Python. Explicitly cast it back to Image before Image methods.
    prepared = ee.Image(mask_and_prepare(ee.Image(image)))

    prepared_info = prepared.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=region,
        scale=LST_RESOLUTION_M,
        bestEffort=True,
        maxPixels=MAX_PIXELS_PER_REGION,
    ).getInfo()

    print(f"  Example masked LST °C:  {prepared_info.get('lst_c')}")

    # Test the exact reduceRegions path used by the full extractor.
    test_composite = ee.Image(prepared).rename("lst_median_c")
    test_reduced = test_composite.reduceRegions(
        collection=gp_fc_one,
        reducer=ee.Reducer.mean(),
        scale=LST_RESOLUTION_M,
        tileScale=TILE_SCALE,
        maxPixelsPerRegion=MAX_PIXELS_PER_REGION,
    )

    test_info = test_reduced.getInfo()
    features = test_info.get("features", [])

    if not features:
        raise RuntimeError(
            "LST PREFLIGHT FAILED: reduceRegions returned no feature."
        )

    props = features[0].get("properties", {})
    print(f"  reduceRegions properties: {sorted(props.keys())}")
    reduced_value = props.get("mean")
    print(f"  Example GP reduced LST:   {reduced_value}")

    if reduced_value is None:
        raise RuntimeError(
            "LST PREFLIGHT FAILED: reduceRegions returned no mean value for "
            "the test GP. The source image itself contains valid LST, so this "
            "indicates that the test GP has no unmasked thermal pixels for "
            "that scene."
        )

    print("  PREFLIGHT STATUS: PASS")
    print("  Source contains valid thermal pixels and reduceRegions returns LST.")
    print("=" * 88)
    print()

# MONTHLY COMPOSITE DEFINITIONS
# ============================================================================

def build_monthly_specs(
    start_date: str,
    end_date: str,
) -> List[Tuple[str, str, str]]:
    """
    Return:
        (month_start, month_end_exclusive, label)

    Only complete historical months are used.
    The current incomplete month is excluded because it would otherwise
    produce a partial temporal sample.
    """
    start = month_start(start_date)
    end = pd.Timestamp(end_date)

    # Do not include the current partial month.
    current_month = pd.Timestamp.today().normalize().replace(day=1)

    if end >= current_month:
        end = current_month

    specs = []

    cursor = start
    while cursor < end:
        nxt = month_end_exclusive(cursor)

        if cursor >= end:
            break

        actual_end = min(nxt, end)

        specs.append(
            (
                cursor.strftime("%Y-%m-%d"),
                actual_end.strftime("%Y-%m-%d"),
                cursor.strftime("%Y-%m"),
            )
        )

        cursor = nxt

    return specs


def monthly_composite(
    region: ee.Geometry,
    month_start_date: str,
    month_end_date: str,
) -> Tuple[ee.Image, ee.Number]:
    collection = build_collection(
        region,
        month_start_date,
        (
            pd.Timestamp(month_end_date)
            - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
    )

    scene_count = collection.size()

    # A month can legitimately contain zero usable Landsat L2SP scenes after
    # the source filters. Earth Engine's median()/mean() on an empty
    # collection produces a zero-band image, which cannot be passed to
    # reduceRegions(). Preserve that month explicitly as masked/no-data
    # instead of inventing a value.
    empty_template = (
        ee.Image.constant(0)
        .rename("lst_median_c")
        .updateMask(ee.Image.constant(0))
    )

    def non_empty_composite():
        return collection.median().rename("lst_median_c")

    composite = ee.Image(
        ee.Algorithms.If(
            scene_count.gt(0),
            ee.Image(non_empty_composite()),
            ee.Image(empty_template),
        )
    )

    return composite, scene_count


# ============================================================================
# BATCHED MONTHLY EXTRACTION
# ============================================================================

def extract_month_batch(
    specs: List[Tuple[str, str, str]],
    region: ee.Geometry,
    gp_fc: ee.FeatureCollection,
) -> Tuple[List[Dict], List[Dict]]:
    """
    One EE getInfo for several monthly composites.

    Each monthly composite is reduced over all GPs. Missing valid pixels remain
    null and are not filled.
    """
    feature_collections = []
    scene_meta = []

    for month_start_date, month_end_date, label in specs:
        composite, scene_count = monthly_composite(
            region,
            month_start_date,
            month_end_date,
        )

        def add_provenance(feature):
            return feature.set(
                {
                    "month_start": label,
                    "source_start": month_start_date,
                    "source_end_exclusive": month_end_date,
                    "source_scene_count": scene_count,
                }
            )

        reduced = (
            composite.reduceRegions(
                collection=gp_fc,
                reducer=ee.Reducer.mean(),
                scale=LST_RESOLUTION_M,
                tileScale=TILE_SCALE,
                maxPixelsPerRegion=MAX_PIXELS_PER_REGION,
            )
            .map(add_provenance)
        )

        feature_collections.append(reduced)

        scene_meta.append(
            {
                "month_start": label,
                "source_start": month_start_date,
                "source_end_exclusive": month_end_date,
                "scene_count": scene_count,
            }
        )

    merged = ee.FeatureCollection(feature_collections).flatten()
    result = merged.getInfo()

    rows = []

    for feature in result.get("features", []):
        props = feature.get("properties", {})

        rows.append(
            {
                "gp_id": str(props.get("gp_id", "")),
                "gp_name": props.get("gp_name", ""),
                "month_start": props.get("month_start"),
                "source_start": props.get("source_start"),
                "source_end_exclusive": props.get(
                    "source_end_exclusive"
                ),
                "source_scene_count": props.get(
                    "source_scene_count"
                ),
                # ee.Reducer.mean() returns the reducer statistic under
                # the property name "mean" for a single-band image.
                "lst_median_c": props.get("mean"),
            }
        )

    return rows, scene_meta


def extract_landsat_monthly(
    gp_master: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, List[Dict]]:
    region = ee.Geometry(
        gp_master.union_all().__geo_interface__
    )

    specs = build_monthly_specs(
        start_date,
        end_date,
    )

    print(f"Monthly periods available: {len(specs)}")

    if not specs:
        return pd.DataFrame(), []

    gp_fc = gdf_to_ee_fc(gp_master)

    all_rows = []
    all_meta = []

    for start in range(0, len(specs), MONTH_BATCH_SIZE):
        batch = specs[start:start + MONTH_BATCH_SIZE]

        print(
            f"  Processing months {start + 1}-{start + len(batch)} "
            f"of {len(specs)}..."
        )

        t0 = time.time()

        rows, meta = extract_month_batch(
            batch,
            region,
            gp_fc,
        )

        all_rows.extend(rows)
        all_meta.extend(meta)

        print(
            f"    Returned {len(rows)} GP-month rows "
            f"in {time.time() - t0:.1f}s"
        )

    return pd.DataFrame(all_rows), all_meta


# ============================================================================
# QC
# ============================================================================

def validate_monthly(
    raw: pd.DataFrame,
    gp_master: gpd.GeoDataFrame,
) -> Dict:
    expected = set(gp_master["gp_id"].astype(str))

    if raw.empty:
        return {
            "status": "FAIL",
            "expected_gp_count": len(expected),
            "observed_gp_count": 0,
            "duplicate_gp_month_rows": 0,
            "valid_lstm_month_values": 0,
            "g_empty_gp_count": len(expected),
            "synthetic": False,
            "imputation": False,
        }

    raw["gp_id"] = raw["gp_id"].astype(str)

    observed = set(raw["gp_id"])

    duplicate_rows = int(
        raw.duplicated(
            subset=["gp_id", "month_start"],
            keep=False,
        ).sum()
    )

    # Strict decoded-temperature plausibility check.
    # These are source values, so out-of-range values are rejected rather
    # than clipped or replaced.
    numeric_lsts = pd.to_numeric(
        raw["lst_median_c"],
        errors="coerce",
    )
    invalid_range = (
        numeric_lsts.notna()
        & ((numeric_lsts < -50.0) | (numeric_lsts > 80.0))
    )
    invalid_range_count = int(invalid_range.sum())

    valid_values = int(numeric_lsts.notna().sum())

    valid_per_gp = (
        raw.groupby("gp_id")["lst_median_c"]
        .apply(lambda x: int(x.notna().sum()))
    )

    empty_gps = [
        gp_id
        for gp_id in expected
        if int(valid_per_gp.get(gp_id, 0)) == 0
    ]

    status = (
        observed == expected
        and duplicate_rows == 0
        and len(empty_gps) == 0
        and invalid_range_count == 0
    )

    return {
        "status": "PASS" if status else "FAIL",
        "expected_gp_count": len(expected),
        "observed_gp_count": len(observed),
        "duplicate_gp_month_rows": duplicate_rows,
        "valid_lst_month_values": valid_values,
        "invalid_physical_range_values": invalid_range_count,
        "accepted_temperature_range_c": [-50.0, 80.0],
        "empty_gp_count": len(empty_gps),
        "empty_gp_ids": sorted(empty_gps),
        "min_valid_months_per_gp": (
            int(valid_per_gp.min())
            if len(valid_per_gp)
            else 0
        ),
        "max_valid_months_per_gp": (
            int(valid_per_gp.max())
            if len(valid_per_gp)
            else 0
        ),
        "synthetic": False,
        "imputation": False,
    }


# ============================================================================
# PROCESSED FEATURES
# ============================================================================

def build_processed(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    rows = []

    for gp_id, group in raw.groupby("gp_id", sort=False):
        median_values = pd.to_numeric(
            group["lst_median_c"],
            errors="coerce",
        ).dropna()

        if median_values.empty:
            continue

        rows.append(
            {
                "gp_id": str(gp_id),
                "gp_name": str(
                    group["gp_name"].dropna().iloc[0]
                    if group["gp_name"].notna().any()
                    else ""
                ),
                "lst_mean_c": float(median_values.mean()),
                "lst_median_c": float(median_values.median()),
                "lst_std_c": (
                    float(median_values.std(ddof=0))
                    if len(median_values) > 1
                    else 0.0
                ),
                "lst_valid_months": int(len(median_values)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# OUTPUTS
# ============================================================================

def write_outputs(
    gp_master: gpd.GeoDataFrame,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
    metadata: Dict,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / RAW_FILENAME
    processed_path = processed_dir / PROCESSED_FILENAME
    geojson_path = processed_dir / PROCESSED_GEOJSON_FILENAME
    metadata_path = raw_dir / METADATA_FILENAME

    raw.to_csv(raw_path, index=False)
    processed.to_csv(processed_path, index=False)

    geo = gp_master[["gp_id", "geometry"]].copy()
    geo["gp_id"] = geo["gp_id"].astype(str)

    merged = geo.merge(
        processed,
        on="gp_id",
        how="left",
        validate="one_to_one",
    )

    gdf = gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=gp_master.crs,
    )

    gdf.to_file(
        geojson_path,
        driver="GeoJSON",
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Raw data:       {raw_path}")
    print(f"Features CSV:   {processed_path}")
    print(f"Features GIS:   {geojson_path}")
    print(f"Metadata:       {metadata_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    started = time.time()

    state, district, block, force = parse_location()

    master_path = find_gp_master(
        state,
        district,
        block,
    )

    raw_dir = master_path.parent.parent / "raw"
    processed_dir = master_path.parent

    raw_path = raw_dir / RAW_FILENAME
    processed_path = processed_dir / PROCESSED_FILENAME
    geojson_path = processed_dir / PROCESSED_GEOJSON_FILENAME
    metadata_path = raw_dir / METADATA_FILENAME

    history_start = infer_history_start(master_path)

    fetch_start, fetch_end, full_refresh = determine_fetch_window(
        raw_path,
        history_start,
        force,
    )

    print()
    print("=" * 88)
    print("SIH26 — STAGE 7: GP LAND SURFACE TEMPERATURE")
    print("=" * 88)
    print(f"Location:       {state} / {district} / {block}")
    print(f"GP master:      {master_path}")
    print(f"Source:         {LST_SOURCE}")
    print(f"Resolution:     {LST_RESOLUTION_M} m")
    print("Sensors:        Landsat 8 + Landsat 9")
    print("Product:        Collection 2 Level 2 Tier 1")
    print("Band:           ST_B10")
    print("Method:         MONTHLY VALID-PIXEL COMPOSITES")
    print("Synthetic:      DISABLED")
    print("Imputation:     DISABLED")
    print(f"Force refresh:  {force}")
    print(f"GP count:       {len(load_gp_master(master_path))}")
    print(f"History start:  {history_start}")
    print(f"Fetch start:    {fetch_start}")
    print(f"Fetch end:      {fetch_end}")

    gp_master = load_gp_master(master_path)

    # Static/reusable output check.
    if (
        not force
        and raw_path.exists()
        and processed_path.exists()
        and geojson_path.exists()
        and metadata_path.exists()
    ):
        try:
            existing = pd.read_csv(processed_path)

            required = {
                "gp_id",
                "lst_mean_c",
                "lst_median_c",
                "lst_std_c",
                "lst_valid_months",
            }

            expected_ids = set(gp_master["gp_id"].astype(str))

            if (
                required.issubset(existing.columns)
                and set(existing["gp_id"].astype(str))
                == expected_ids
                and len(existing) == len(expected_ids)
                and (existing["lst_valid_months"] > 0).all()
            ):
                print()
                print("EXISTING VALID LST DATASET DETECTED")
                print("Mode: REUSE EXISTING DATA")
                print(f"GP rows: {len(existing)}")
                print("No Landsat request was made.")
                print(f"Total time: {time.time() - started:.1f}s")
                return 0

        except Exception as exc:
            print(
                "Existing-output validation failed; refreshing. "
                f"Reason: {exc}"
            )

    if not full_refresh and fetch_start > fetch_end:
        print()
        print("NO NEW COMPLETE MONTH AVAILABLE")
        print("Existing LST data are current.")
        return 0

    initialize_ee()

    # Fail fast if the source/QA/reduction path cannot produce one real value.
    # This avoids spending many minutes on a full empty extraction.
    run_lst_preflight(
        gp_master,
        fetch_start,
        fetch_end,
    )

    print()
    print("Building Landsat monthly composite collection...")
    print("Quality filters:")
    print("  - L2SP processing level only")
    print("  - CLOUD_COVER <= 60%")
    print("  - QA_PIXEL fill/cloud/cirrus/shadow/snow masked")
    print("  - ST_B10 fill pixels masked")
    print("  - No imputation/interpolation")

    t0 = time.time()

    raw_new, scene_meta = extract_landsat_monthly(
        gp_master,
        fetch_start,
        fetch_end,
    )

    print(
        f"Extraction completed in {time.time() - t0:.1f}s"
    )

    if raw_new.empty:
        raise RuntimeError(
            "No monthly LST data were returned."
        )

    raw_new["gp_id"] = raw_new["gp_id"].astype(str)

    qc_result = validate_monthly(
        raw_new,
        gp_master,
    )

    print()
    print("MONTHLY LST QC")
    print(f"  Status:                 {qc_result['status']}")
    print(f"  GP count:               {qc_result['observed_gp_count']}")
    print(f"  GP-month rows:          {len(raw_new)}")
    print(
        f"  Valid monthly LST:      "
        f"{qc_result['valid_lst_month_values']}"
    )
    print(
        f"  Duplicate GP-month:     "
        f"{qc_result['duplicate_gp_month_rows']}"
    )
    print(
        f"  Min valid months/GP:    "
        f"{qc_result.get('min_valid_months_per_gp', 0)}"
    )
    print(
        f"  Max valid months/GP:    "
        f"{qc_result.get('max_valid_months_per_gp', 0)}"
    )
    print("  Synthetic:              PASS (disabled)")
    print("  Imputation:             PASS (disabled)")

    if qc_result["status"] != "PASS":
        raise RuntimeError(
            "LST QC failed. No synthetic or imputed values will be created:\n"
            + json.dumps(qc_result, indent=2)
        )

    # Incremental merge.
    if (
        not full_refresh
        and raw_path.exists()
    ):
        old = pd.read_csv(raw_path)
        old["gp_id"] = old["gp_id"].astype(str)

        combined = pd.concat(
            [old, raw_new],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=["gp_id", "month_start"],
            keep="last",
        )
    else:
        combined = raw_new.copy()

    combined["month_start"] = pd.to_datetime(
        combined["month_start"],
        errors="coerce",
    ).dt.strftime("%Y-%m")

    processed = build_processed(combined)

    expected_ids = set(gp_master["gp_id"].astype(str))
    processed_ids = set(processed["gp_id"].astype(str))

    if expected_ids != processed_ids:
        missing = sorted(expected_ids - processed_ids)
        raise RuntimeError(
            "Processed LST features are missing GPs:\n"
            + "\n".join(missing)
        )

    model_columns = [
        "lst_mean_c",
        "lst_median_c",
        "lst_std_c",
        "lst_valid_months",
    ]

    if processed[model_columns].isna().any().any():
        raise RuntimeError(
            "Final LST feature table contains missing model features. "
            "No imputation will be performed."
        )

    metadata = {
        "stage": "SIH26_STAGE_7_GP_LST",
        "location": {
            "state": state,
            "district": district,
            "block": block,
        },
        "source": LST_SOURCE,
        "resolution_m": LST_RESOLUTION_M,
        "sensors": ["Landsat 8", "Landsat 9"],
        "collections": [
            L8_COLLECTION,
            L9_COLLECTION,
        ],
        "processing_level": "L2SP",
        "band": "ST_B10",
        "scaling": {
            "kelvin": "DN * 0.00341802 + 149.0",
            "celsius": "kelvin - 273.15",
        },
        "quality_mask": [
            "QA_PIXEL fill",
            "QA_PIXEL dilated cloud",
            "QA_PIXEL cirrus",
            "QA_PIXEL cloud",
            "QA_PIXEL cloud shadow",
            "QA_PIXEL snow",
            "ST_B10 fill",
        ],
        "cloud_cover_filter_pct": 60,
        "method": (
            "Monthly median valid-pixel LST composites, "
            "then GP polygon zonal statistics."
        ),
        "monthly_rows": int(len(combined)),
        "processed_rows": int(len(processed)),
        "model_features": model_columns,
        "synthetic": False,
        "imputation": False,
        "interpolation": False,
        "source_scene_metadata": scene_meta,
        "qc": qc_result,
        "notes": [
            "Monthly median composites are used instead of scene-by-scene GP reductions "
            "to reduce Earth Engine request count and runtime.",
            "Masked/cloud-contaminated pixels are not filled.",
            "Only GPs with at least one valid monthly LST observation are accepted.",
            "The monthly table is retained for future leakage-safe lag construction.",
            "Current incomplete month is excluded from the historical composite set.",
        ],
    }

    write_outputs(
        gp_master,
        combined,
        processed,
        raw_dir,
        processed_dir,
        metadata,
    )

    print()
    print("=" * 88)
    print("FINAL STATUS: SUCCESS")
    print("=" * 88)
    print(f"GPs:            {len(gp_master)}")
    print(f"Monthly rows:   {len(combined)}")
    print(f"Processed rows: {len(processed)}")
    print("Model features: 4")
    print(f"Total time:     {time.time() - started:.1f}s")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
