from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import ee
import geopandas as gpd
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOIL_RESOLUTION_M = 250
TILE_SCALE = 4
MAX_PIXELS_PER_REGION = 10_000_000

SOIL_SOURCE = "ISRIC SoilGrids 2.0 (250 m), Google Earth Engine"

# IMPORTANT:
# Only the soil variables that have a clear role in the current
# GP-level agro-meteorological downscaling methodology are extracted.
#
# Included:
#   clay  -> soil texture / water holding behaviour
#   sand  -> soil texture / infiltration behaviour
#   soc   -> soil organic carbon / water retention and thermal behaviour
#   bdod  -> bulk density / soil physical properties
#
# Deliberately excluded from this stage:
#   cec, cfvo, nitrogen, ocd, ocs, phh2o
#
# These are useful for agronomic/soil studies but are not necessary for the
# first weather-downscaling feature set and would increase dimensionality and
# processing complexity.
SOIL_ASSETS = {
    "clay": "projects/soilgrids-isric/clay_mean",
    "sand": "projects/soilgrids-isric/sand_mean",
    "soc": "projects/soilgrids-isric/soc_mean",
    "bdod": "projects/soilgrids-isric/bdod_mean",
}

# Only the 0-30 cm root zone is retained.
# SoilGrids provides three standard intervals covering this zone.
DEPTHS = [
    "0-5cm",
    "5-15cm",
    "15-30cm",
]

# SoilGrids mapped-unit -> conventional-unit conversion factors.
CONVERSION = {
    "clay": 10.0,       # mapped value -> %
    "sand": 10.0,       # mapped value -> %
    "soc": 10.0,        # mapped value -> g/kg
    "bdod": 100.0,      # mapped value -> kg/dm3
}

RAW_FILENAME = "soil_gp_raw.csv"
PROCESSED_FILENAME = "gp_soil_features.csv"
PROCESSED_GEOJSON_FILENAME = "gp_soil_features.geojson"
METADATA_FILENAME = "soil_source_metadata.json"


# ============================================================================
# LOCATION
# ============================================================================

# ============================================================================

def norm(value: str) -> str:
    return (
        str(value).strip().lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_location() -> Tuple[str, str, str, bool]:
    parser = argparse.ArgumentParser(
        description="SIH26 universal GP SoilGrids extractor"
    )
    parser.add_argument("--state")
    parser.add_argument("--district")
    parser.add_argument("--block")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract static soil data even if valid output exists.",
    )
    args = parser.parse_args()

    supplied = [args.state, args.district, args.block]

    if all(supplied):
        return (
            args.state.strip(),
            args.district.strip(),
            args.block.strip(),
            args.force,
        )

    if any(supplied):
        raise SystemExit(
            "Provide all three: --state, --district and --block."
        )

    print("=" * 88)
    print("SIH26 — STAGE 6: GP SOIL FEATURES")
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

    # Exact expected location.
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

    # Universal fallback for capitalization/naming differences.
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
            f"No GP master found for {state}/{district}/{block} "
            f"below {gp_root}."
        )

    raise RuntimeError(
        "Multiple GP masters match the requested location:\n"
        + "\n".join(str(p) for p in matches)
    )


# ============================================================================
# EARTH ENGINE / GP MASTER
# ============================================================================

def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate Earth Engine "
            "before running this script."
        ) from exc


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
            "Some GP geometries remain invalid after repair."
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
# LOCATION
# ============================================================================

def norm(value: str) -> str:
    return (
        str(value).strip().lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_location() -> Tuple[str, str, str, bool]:
    parser = argparse.ArgumentParser(
        description="SIH26 universal GP SoilGrids extractor"
    )
    parser.add_argument("--state")
    parser.add_argument("--district")
    parser.add_argument("--block")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract static soil data even if valid output exists.",
    )
    args = parser.parse_args()

    supplied = [args.state, args.district, args.block]

    if all(supplied):
        return (
            args.state.strip(),
            args.district.strip(),
            args.block.strip(),
            args.force,
        )

    if any(supplied):
        raise SystemExit(
            "Provide all three: --state, --district and --block."
        )

    print("=" * 88)
    print("SIH26 — STAGE 6: GP SOIL FEATURES")
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

    # Exact expected location.
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

    # Universal fallback for capitalization/naming differences.
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
            f"No GP master found for {state}/{district}/{block} "
            f"below {gp_root}."
        )

    raise RuntimeError(
        "Multiple GP masters match the requested location:\n"
        + "\n".join(str(p) for p in matches)
    )


# ============================================================================
# EARTH ENGINE / GP MASTER
# ============================================================================

def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate Earth Engine "
            "before running this script."
        ) from exc


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
            "Some GP geometries remain invalid after repair."
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
# SOILGRIDS IMAGE STACK
# ============================================================================

def build_soil_stack() -> ee.Image:
    """
    Build only the minimal static SoilGrids stack.

    Four properties x three depth intervals:
        clay, sand, soc, bdod
        0-5, 5-15, 15-30 cm

    No water-content assets are queried here. This avoids unnecessary
    dependencies and keeps Stage 6 focused on the soil variables required
    by the current weather-downscaling methodology.
    """
    bands = []

    for prop, asset_id in SOIL_ASSETS.items():
        image = ee.Image(asset_id)

        for depth in DEPTHS:
            source_band = f"{prop}_{depth}_mean"
            output_band = f"soil_{prop}_{depth}"

            band = (
                image
                .select(source_band)
                .divide(CONVERSION[prop])
                .rename(output_band)
            )

            bands.append(band)

    return ee.Image.cat(bands).float()


# ============================================================================
# EXTRACTION
# ============================================================================

def extract_soil(gp_master: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract GP-level zonal means for the four selected soil properties.

    Only mean values are retained. Standard deviation is deliberately not
    extracted because it is not part of the current model feature set.

    Missing source values remain missing and are never imputed.
    """
    image = build_soil_stack()
    gp_fc = gdf_to_ee_fc(gp_master)

    reduced = image.reduceRegions(
        collection=gp_fc,
        reducer=ee.Reducer.mean(),
        scale=SOIL_RESOLUTION_M,
        tileScale=TILE_SCALE,
        maxPixelsPerRegion=MAX_PIXELS_PER_REGION,
    )

    data = reduced.getInfo()

    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})

        row = {
            "gp_id": str(props.get("gp_id", "")),
            "gp_name": props.get("gp_name", ""),
        }

        for key, value in props.items():
            if key not in ("gp_id", "gp_name"):
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# QC
# ============================================================================

def qc(
    raw: pd.DataFrame,
    gp_master: gpd.GeoDataFrame,
) -> Dict:
    expected = set(gp_master["gp_id"].astype(str))
    observed = set(raw["gp_id"].astype(str))

    duplicate_rows = int(
        raw["gp_id"].astype(str).duplicated().sum()
    )

    expected_numeric_cols = [
        f"soil_{prop}_{depth}"
        for prop in SOIL_ASSETS
        for depth in DEPTHS
    ]

    missing_columns = [
        c for c in expected_numeric_cols
        if c not in raw.columns
    ]

    if missing_columns:
        return {
            "status": "FAIL",
            "expected_gp_count": len(expected),
            "observed_gp_count": len(observed),
            "missing_gp_ids": sorted(expected - observed),
            "unexpected_gp_ids": sorted(observed - expected),
            "duplicate_gp_rows": duplicate_rows,
            "missing_source_cells": None,
            "missing_source_columns": missing_columns,
            "synthetic": False,
            "imputation": False,
        }

    missing_source_cells = int(
        raw[expected_numeric_cols].isna().sum().sum()
    )

    return {
        "status": (
            "PASS"
            if (
                expected == observed
                and duplicate_rows == 0
                and missing_source_cells == 0
            )
            else "FAIL"
        ),
        "expected_gp_count": len(expected),
        "observed_gp_count": len(observed),
        "missing_gp_ids": sorted(expected - observed),
        "unexpected_gp_ids": sorted(observed - expected),
        "duplicate_gp_rows": duplicate_rows,
        "missing_source_cells": missing_source_cells,
        "missing_source_columns": [],
        "synthetic": False,
        "imputation": False,
    }


# ============================================================================
# PROCESSED FEATURES
# ============================================================================

def weighted_depth_value(
    row: pd.Series,
    property_name: str,
    intervals: List[Tuple[str, float]],
) -> float:
    """
    Thickness-weighted mean across the requested soil depth intervals.

    For this stage the intervals are exactly 0-5, 5-15 and 15-30 cm,
    producing a 0-30 cm root-zone value.
    """
    values = []

    for depth, weight in intervals:
        col = f"soil_{property_name}_{depth}"
        value = pd.to_numeric(
            pd.Series([row.get(col)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            return float("nan")

        values.append((float(value), weight))

    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return float("nan")

    return sum(value * weight for value, weight in values) / total_weight


def build_processed(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the compact model-ready soil table.

    Final soil predictors:
        soil_clay_rootzone_0_30cm_pct
        soil_sand_rootzone_0_30cm_pct
        soil_soc_rootzone_0_30cm_g_kg
        soil_bdod_rootzone_0_30cm_kg_dm3

    These four values are static GP-level predictors.
    """
    if raw.empty:
        return pd.DataFrame()

    root = [
        ("0-5cm", 5.0),
        ("5-15cm", 10.0),
        ("15-30cm", 15.0),
    ]

    rows = []

    for _, row in raw.iterrows():
        out = {
            "gp_id": str(row["gp_id"]),
            "gp_name": row.get("gp_name", ""),
        }

        out["soil_clay_rootzone_0_30cm_pct"] = weighted_depth_value(
            row, "clay", root
        )
        out["soil_sand_rootzone_0_30cm_pct"] = weighted_depth_value(
            row, "sand", root
        )
        out["soil_soc_rootzone_0_30cm_g_kg"] = weighted_depth_value(
            row, "soc", root
        )
        out["soil_bdod_rootzone_0_30cm_kg_dm3"] = weighted_depth_value(
            row, "bdod", root
        )

        rows.append(out)

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

    print()
    print("=" * 88)
    print("SIH26 — STAGE 6: GP SOIL FEATURES")
    print("=" * 88)
    print(f"Location:       {state} / {district} / {block}")
    print(f"GP master:      {master_path}")
    print(f"Source:         {SOIL_SOURCE}")
    print(f"Resolution:     {SOIL_RESOLUTION_M} m")
    print("Selected:       clay, sand, SOC, bulk density")
    print("Depth:          0-30 cm root zone")
    print("Synthetic:      DISABLED")
    print("Imputation:     DISABLED")
    print(f"Force refresh:  {force}")
    print()

    # Soil is static. Reuse an existing complete dataset.
    if (
        not force
        and raw_path.exists()
        and processed_path.exists()
        and geojson_path.exists()
        and metadata_path.exists()
    ):
        try:
            existing = pd.read_csv(processed_path)
            gp_master = load_gp_master(master_path)

            expected_features = {
                "gp_id",
                "soil_clay_rootzone_0_30cm_pct",
                "soil_sand_rootzone_0_30cm_pct",
                "soil_soc_rootzone_0_30cm_g_kg",
                "soil_bdod_rootzone_0_30cm_kg_dm3",
            }

            if (
                expected_features.issubset(existing.columns)
                and existing["gp_id"].astype(str).nunique()
                == len(gp_master)
                and set(existing["gp_id"].astype(str))
                == set(gp_master["gp_id"].astype(str))
                and not existing[
                    list(expected_features - {"gp_id"})
                ].isna().any().any()
            ):
                print("EXISTING SOIL DATASET DETECTED")
                print("Mode: REUSE EXISTING DATA")
                print(f"GP rows: {len(existing)}")
                print("No SoilGrids request was made.")
                print(f"Elapsed: {time.time() - started:.1f}s")
                return 0
        except Exception as exc:
            print(
                "Existing output validation failed; "
                f"re-extraction will be performed. Reason: {exc}"
            )

    gp_master = load_gp_master(master_path)

    print(f"GP count:       {len(gp_master)}")

    initialize_ee()

    print("Building minimal SoilGrids image stack...")
    print("  Properties: clay, sand, SOC, bulk density")
    print("  Depths:     0-5, 5-15, 15-30 cm")
    print("  Water-content assets: NOT USED")

    t0 = time.time()
    raw = extract_soil(gp_master)

    print(
        f"Extraction returned {len(raw)} GP rows "
        f"in {time.time() - t0:.1f}s."
    )

    if raw.empty:
        raise RuntimeError(
            "SoilGrids returned zero GP rows. "
            "No synthetic/imputed values will be created."
        )

    raw["gp_id"] = raw["gp_id"].astype(str)

    if raw["gp_id"].duplicated().any():
        raise RuntimeError(
            "Soil extraction returned duplicate GP IDs."
        )

    qc_result = qc(raw, gp_master)

    print()
    print("QC")
    print(f"  GP count:              {qc_result['status']}")
    print(f"  Expected GPs:          {qc_result['expected_gp_count']}")
    print(f"  Observed GPs:          {qc_result['observed_gp_count']}")
    print(f"  Duplicate GP rows:     {qc_result['duplicate_gp_rows']}")
    print(
        f"  Missing source cells:  "
        f"{qc_result['missing_source_cells']}"
    )
    print("  Synthetic:             PASS (disabled)")
    print("  Imputation:            PASS (disabled)")

    if qc_result["status"] != "PASS":
        raise RuntimeError(
            "Soil QC failed. Missing source values are not filled:\n"
            + json.dumps(qc_result, indent=2)
        )

    print()
    print("Building processed GP soil features...")

    processed = build_processed(raw)

    if processed.empty:
        raise RuntimeError(
            "Processed soil feature table is empty."
        )

    if len(processed) != len(gp_master):
        raise RuntimeError(
            "Processed GP count does not match GP master."
        )

    model_columns = [
        "soil_clay_rootzone_0_30cm_pct",
        "soil_sand_rootzone_0_30cm_pct",
        "soil_soc_rootzone_0_30cm_g_kg",
        "soil_bdod_rootzone_0_30cm_kg_dm3",
    ]

    if processed[model_columns].isna().any().any():
        raise RuntimeError(
            "Processed soil features contain missing values. "
            "No imputation will be performed."
        )

    metadata = {
        "stage": "SIH26_STAGE_6_GP_SOIL",
        "location": {
            "state": state,
            "district": district,
            "block": block,
        },
        "source": SOIL_SOURCE,
        "resolution_m": SOIL_RESOLUTION_M,
        "selected_properties": [
            "clay",
            "sand",
            "soc",
            "bdod",
        ],
        "excluded_properties": [
            "cec",
            "cfvo",
            "nitrogen",
            "ocd",
            "ocs",
            "phh2o",
        ],
        "depth_intervals": DEPTHS,
        "processed_feature_depth": "0-30 cm root zone",
        "synthetic": False,
        "imputation": False,
        "gp_count": len(gp_master),
        "raw_rows": len(raw),
        "processed_rows": len(processed),
        "qc": qc_result,
        "method": (
            "250 m SoilGrids polygon zonal means for four selected "
            "soil properties across 0-5, 5-15 and 15-30 cm; "
            "processed values are thickness-weighted 0-30 cm means."
        ),
        "model_features": model_columns,
        "notes": [
            "Only soil variables with a direct role in the current "
            "agro-meteorological downscaling feature set are retained.",
            "No water-content assets are queried in this stage.",
            "No synthetic data, interpolation or imputation is used.",
            "Missing source values cause QC failure rather than being filled.",
            "Soil is static and validated outputs are reused unless --force is supplied.",
        ],
    }

    write_outputs(
        gp_master,
        raw,
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
    print(f"Raw rows:       {len(raw)}")
    print(f"Processed rows: {len(processed)}")
    print("Model features: 4")
    print(f"Total time:     {time.time() - started:.1f}s")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
