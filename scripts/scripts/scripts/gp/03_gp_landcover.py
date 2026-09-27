"""
SIH26 — Stage 3: GP Land Cover
================================

Universal GP land-cover extraction pipeline.

IMPORTANT:
-----------
This script contains NO hard-coded state, district, block, or user
computer path.

It discovers the SIH26 project root from this file's location:

    <project_root>/
        scripts/gp/03_gp_landcover.py

and automatically discovers every:

    gp/<state>/<district>/<block>/processed/gp_master.geojson

Therefore the same script can be run for Sinnar, Baramati, Bicholim,
or any future block without editing the code.

Usage:
    python scripts/gp/03_gp_landcover.py

Optional:
    python scripts/gp/03_gp_landcover.py --block Sinnar
    python scripts/gp/03_gp_landcover.py --block Baramati

If no --block is supplied, ALL discovered GP masters are processed.

Data source:
    Google Earth Engine Dynamic World V1
    GOOGLE/DYNAMICWORLD/V1

Resolution:
    10 m

Reference period:
    2021 calendar year

Outputs for each block:
    gp/<state>/<district>/<block>/processed/gp_landcover_features.csv
    gp/<state>/<district>/<block>/processed/gp_landcover_features.geojson
    gp/<state>/<district>/<block>/raw/landcover/landcover_source_metadata.json

No synthetic values.
No imputation.
No silent GP deletion.
Existing GP geometry is not modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd


# =====================================================================
# UNIVERSAL CONFIGURATION
# =====================================================================

# scripts/gp/03_gp_landcover.py
# parents[0] = gp
# parents[1] = scripts
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GP_ROOT = PROJECT_ROOT / "gp"

EE_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

BASELINE_YEAR = 2021
START_DATE = f"{BASELINE_YEAR}-01-01"
END_DATE = f"{BASELINE_YEAR + 1}-01-01"

SCALE_M = 10
TILE_SCALE = 4
CLASS_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

OUTPUT_NAMES = {
    "water": "water_fraction",
    "trees": "trees_fraction",
    "grass": "grass_fraction",
    "flooded_vegetation": "flooded_vegetation_fraction",
    "crops": "crops_fraction",
    "shrub_and_scrub": "shrub_fraction",
    "built": "built_fraction",
    "bare": "bare_fraction",
    "snow_and_ice": "snow_ice_fraction",
}


# =====================================================================
# HELPERS
# =====================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def detect_gp_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Detect the GP identifier from the actual GP master schema."""

    candidates = [
        "gp_code",
        "gp_lgd_code",
        "gp_lgdcode",
        "gp_lgd_cod",
        "gplgdcode",
        "gpcode",
        "gp_code11",
        "gpcode11",
        "gp_lgd",
        "lgd_gp_code",
        "lgdcode",
        "gp_id",
        "GP_CODE",
        "GP_LGD_CODE",
        "GP_LGDCODE",
    ]

    normalized = {
        str(col).strip().lower(): col
        for col in gdf.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    # Safe fallback.
    candidates_found = []

    for col in gdf.columns:
        low = str(col).strip().lower()

        if low == "geometry":
            continue

        if (
            "gp" in low
            and (
                "code" in low
                or "lgd" in low
                or low.endswith("_id")
            )
        ):
            candidates_found.append(col)

    if len(candidates_found) == 1:
        return candidates_found[0]

    raise ValueError(
        "Could not safely identify the GP ID column.\n"
        f"Available columns: {list(gdf.columns)}\n"
        "The GP master must contain an identifiable GP/LGD code field."
    )


def detect_gp_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "gp_name",
        "gpname",
        "gram_panchayat_name",
        "gram_panchayat",
        "name",
        "GP_NAME",
    ]

    normalized = {
        str(col).strip().lower(): col
        for col in gdf.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def validate_gp_master(
    gdf: gpd.GeoDataFrame,
    id_col: str,
) -> None:

    if gdf.empty:
        raise ValueError("GP master contains zero records.")

    if gdf.crs is None:
        raise ValueError("GP master has no CRS.")

    if gdf.geometry.isna().any():
        raise ValueError("GP master contains missing geometries.")

    if gdf.geometry.is_empty.any():
        raise ValueError("GP master contains empty geometries.")

    allowed = {"Polygon", "MultiPolygon"}

    bad_types = set(gdf.geometry.geom_type) - allowed

    if bad_types:
        raise ValueError(
            f"GP master contains unsupported geometry types: {bad_types}"
        )

    ids = gdf[id_col].astype(str).str.strip()

    invalid_ids = ids.eq("") | ids.str.lower().isin(
        {"nan", "none", "null"}
    )

    if invalid_ids.any():
        raise ValueError(
            f"GP ID column '{id_col}' contains empty/invalid IDs."
        )

    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].unique().tolist()

        raise ValueError(
            "Duplicate GP IDs exist in gp_master. "
            "Resolve the boundary/master dataset first; this script "
            f"will not silently collapse them.\nDuplicates: {duplicates[:20]}"
        )


def gdf_to_ee_features(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    name_col: str | None,
):
    features = []

    for _, row in gdf.iterrows():

        geometry = row.geometry.__geo_interface__

        properties = {
            "gp_uid": str(row[id_col]).strip()
        }

        if name_col:
            value = row[name_col]

            if pd.notna(value):
                properties["gp_name"] = str(value).strip()
            else:
                properties["gp_name"] = ""

        features.append(
            ee.Feature(
                ee.Geometry(geometry),
                properties,
            )
        )

    return features


def discover_gp_masters(block_filter: str | None):
    """
    Discover all GP master files.

    Expected universal structure:
        gp/<state>/<district>/<block>/processed/gp_master.geojson
    """

    if not GP_ROOT.exists():
        raise FileNotFoundError(
            f"GP directory not found:\n{GP_ROOT}\n"
            "The script must be located inside <project_root>/scripts/gp/."
        )

    masters = sorted(
        GP_ROOT.glob(
            "*/ */*/processed/gp_master.geojson"
        )
    )

    # pathlib glob cannot use spaces as intended in the above pattern,
    # so use recursive search as the authoritative discovery method.
    masters = sorted(
        p for p in GP_ROOT.rglob("gp_master.geojson")
        if p.parent.name == "processed"
    )

    if block_filter:
        masters = [
            p for p in masters
            if p.parent.parent.name.lower() == block_filter.lower()
        ]

    if not masters:
        filter_text = (
            f" for block '{block_filter}'"
            if block_filter
            else ""
        )

        raise FileNotFoundError(
            f"No gp_master.geojson files were discovered{filter_text}.\n"
            f"Searched under:\n{GP_ROOT}\n"
            "Expected structure:\n"
            "<project_root>/gp/<state>/<district>/<block>/processed/gp_master.geojson"
        )

    return masters


def shannon_entropy(values):
    p = np.asarray(values, dtype=float)

    if not np.isfinite(p).all():
        return np.nan

    p = np.clip(p, 0.0, 1.0)

    total = p.sum()

    if total <= 0:
        return np.nan

    p = p / total
    positive = p[p > 0]

    return float(-(positive * np.log(positive)).sum())


# =====================================================================
# PROCESS ONE BLOCK
# =====================================================================

def process_block(gp_master_path: Path):

    # Extract state/district/block from the directory structure.
    processed_dir = gp_master_path.parent
    block_dir = processed_dir.parent
    district_dir = block_dir.parent
    state_dir = district_dir.parent

    state = state_dir.name
    district = district_dir.name
    block = block_dir.name

    output_dir = processed_dir
    raw_dir = block_dir / "raw" / "landcover"

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("STAGE 3 — GP LAND COVER")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"State: {state}")
    print(f"District: {district}")
    print(f"Block: {block}")
    print(f"GP master: {gp_master_path}")
    print(f"Source: {EE_COLLECTION}")
    print(f"Baseline: {BASELINE_YEAR}")
    print(f"Resolution: {SCALE_M} m")
    print()

    # ---------------------------------------------------------------
    # Load GP master
    # ---------------------------------------------------------------

    gdf = gpd.read_file(gp_master_path)

    print(f"GP records loaded: {len(gdf)}")

    id_col = detect_gp_id_column(gdf)
    name_col = detect_gp_name_column(gdf)

    print(f"Detected GP ID column: {id_col}")

    if name_col:
        print(f"Detected GP name column: {name_col}")
    else:
        print("GP name column: not found")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    validate_gp_master(gdf, id_col)

    expected_ids = (
        gdf[id_col]
        .astype(str)
        .str.strip()
        .tolist()
    )

    expected_id_set = set(expected_ids)

    # ---------------------------------------------------------------
    # Block geometry for collection filtering
    # ---------------------------------------------------------------

    # GeoPandas 1.x: union_all() replaces the deprecated unary_union.
    block_union = gdf.geometry.union_all()
    block_geometry = ee.Geometry(
        block_union.__geo_interface__
    )

    # ---------------------------------------------------------------
    # Dynamic World
    # ---------------------------------------------------------------

    collection = (
        ee.ImageCollection(EE_COLLECTION)
        .filterDate(START_DATE, END_DATE)
        .filterBounds(block_geometry)
    )

    image_count = int(collection.size().getInfo())

    if image_count == 0:
        raise RuntimeError(
            f"No Dynamic World images found for {block} "
            f"during {BASELINE_YEAR}."
        )

    print(f"Dynamic World images found: {image_count}")

    # Annual mean of probability bands.
    probability_image = collection.select(
        CLASS_BANDS
    ).mean()

    available_bands = probability_image.bandNames().getInfo()

    missing_bands = [
        band
        for band in CLASS_BANDS
        if band not in available_bands
    ]

    if missing_bands:
        raise RuntimeError(
            "Dynamic World probability bands are missing:\n"
            f"{missing_bands}\n"
            f"Available bands: {available_bands}"
        )

    # Every probability band must be available.
    valid_pixel = (
        probability_image
        .select(CLASS_BANDS)
        .reduce(ee.Reducer.min())
        .mask()
        .rename("valid_pixel")
    )

    image_for_reduce = probability_image.addBands(valid_pixel)

    # ---------------------------------------------------------------
    # GP FeatureCollection
    # ---------------------------------------------------------------

    ee_features = gdf_to_ee_features(
        gdf,
        id_col,
        name_col,
    )

    gp_fc = ee.FeatureCollection(ee_features)

    # ---------------------------------------------------------------
    # Zonal statistics
    # ---------------------------------------------------------------
    # NOTE: ee.Image.reduceRegions() accepts scale/tileScale here, but
    # does NOT accept maxPixels. maxPixels was therefore intentionally
    # removed from this call.

    result_fc = image_for_reduce.reduceRegions(
        collection=gp_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
        tileScale=TILE_SCALE,
    )

    result = result_fc.getInfo()

    if "features" not in result:
        raise RuntimeError(
            "Earth Engine returned no feature collection."
        )

    rows = []

    for feature in result["features"]:

        props = feature.get("properties", {})

        gp_uid = str(
            props.get("gp_uid", "")
        ).strip()

        if not gp_uid:
            raise RuntimeError(
                "Earth Engine returned a feature without gp_uid."
            )

        row = {
            "gp_uid": gp_uid,
        }

        if name_col:
            row["gp_name"] = str(
                props.get("gp_name", "")
            ).strip()

        for band in CLASS_BANDS:
            row[band] = props.get(band)

        row["valid_pixel_fraction"] = props.get(
            "valid_pixel"
        )

        rows.append(row)

    lc = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Record matching
    # ---------------------------------------------------------------

    if lc.empty:
        raise RuntimeError(
            "Dynamic World returned zero GP records."
        )

    returned_ids = (
        lc["gp_uid"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    returned_id_set = set(returned_ids)

    missing_ids = sorted(
        expected_id_set - returned_id_set
    )

    unexpected_ids = sorted(
        returned_id_set - expected_id_set
    )

    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} GP(s) missing from Dynamic World results:\n"
            f"{missing_ids[:20]}"
        )

    if unexpected_ids:
        raise RuntimeError(
            "Unexpected GP IDs returned by Earth Engine:\n"
            f"{unexpected_ids[:20]}"
        )

    if lc["gp_uid"].duplicated().any():
        duplicates = (
            lc.loc[
                lc["gp_uid"].duplicated(keep=False),
                "gp_uid"
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate GP IDs returned by Earth Engine: {duplicates}"
        )

    # ---------------------------------------------------------------
    # Numeric conversion and strict QC
    # ---------------------------------------------------------------

    numeric_columns = CLASS_BANDS + [
        "valid_pixel_fraction"
    ]

    for col in numeric_columns:
        lc[col] = pd.to_numeric(
            lc[col],
            errors="coerce"
        )

    if lc[numeric_columns].isna().any().any():

        bad = lc.loc[
            lc[numeric_columns].isna().any(axis=1),
            ["gp_uid"] + numeric_columns
        ]

        raise RuntimeError(
            "Missing Dynamic World values detected.\n"
            "No imputation is permitted.\n\n"
            f"{bad.to_string(index=False)}"
        )

    values = lc[CLASS_BANDS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Non-finite Dynamic World probability detected."
        )

    if (values < -1e-6).any() or (
        values > 1.000001
    ).any():
        raise RuntimeError(
            "Dynamic World probability outside [0, 1]."
        )

    coverage = lc[
        "valid_pixel_fraction"
    ].to_numpy(dtype=float)

    if not np.isfinite(coverage).all():
        raise RuntimeError(
            "Non-finite valid-pixel coverage detected."
        )

    if (coverage < 0).any() or (coverage > 1).any():
        raise RuntimeError(
            "valid_pixel_fraction outside [0, 1]."
        )

    # Dynamic World probability bands are expected to sum to 1.
    probability_sums = values.sum(axis=1)

    max_probability_error = float(
        np.max(np.abs(probability_sums - 1.0))
    )

    if max_probability_error > 0.02:
        raise RuntimeError(
            "Dynamic World probability sums failed QC.\n"
            f"Maximum deviation from 1: {max_probability_error:.6f}"
        )

    # Require nearly complete valid coverage.
    low_coverage = lc[
        lc["valid_pixel_fraction"] < 0.99
    ]

    if not low_coverage.empty:
        raise RuntimeError(
            "GP(s) have less than 99% valid Dynamic World coverage.\n"
            "No imputation or synthetic filling will be performed.\n\n"
            f"{low_coverage[['gp_uid', 'valid_pixel_fraction']].to_string(index=False)}"
        )

    # ---------------------------------------------------------------
    # Derived land-cover features
    # ---------------------------------------------------------------

    dominant_index = values.argmax(axis=1)

    lc["dominant_landcover"] = [
        CLASS_BANDS[i]
        for i in dominant_index
    ]

    lc["dominant_probability"] = values.max(
        axis=1
    )

    lc["landcover_entropy"] = [
        shannon_entropy(row)
        for row in values
    ]

    lc["landcover_probability_sum"] = (
        probability_sums
    )

    # ---------------------------------------------------------------
    # Rename probability features
    # ---------------------------------------------------------------

    lc = lc.rename(
        columns=OUTPUT_NAMES
    )

    feature_columns = list(
        OUTPUT_NAMES.values()
    ) + [
        "dominant_landcover",
        "dominant_probability",
        "landcover_entropy",
        "valid_pixel_fraction",
        "landcover_probability_sum",
    ]

    # ---------------------------------------------------------------
    # Merge with GP master
    # ---------------------------------------------------------------

    base = gdf.copy()

    base["_gp_join_id"] = (
        base[id_col]
        .astype(str)
        .str.strip()
    )

    lc["_gp_join_id"] = (
        lc["gp_uid"]
        .astype(str)
        .str.strip()
    )

    merge_columns = [
        "_gp_join_id"
    ] + feature_columns

    merged = base.merge(
        lc[merge_columns],
        on="_gp_join_id",
        how="left",
        validate="one_to_one",
    )

    merged = merged.drop(
        columns=["_gp_join_id"]
    )

    # ---------------------------------------------------------------
    # Final QC after merge
    # ---------------------------------------------------------------

    if len(merged) != len(gdf):
        raise RuntimeError(
            "GP record count changed during merge.\n"
            f"Input: {len(gdf)}\n"
            f"Output: {len(merged)}"
        )

    final_ids = (
        merged[id_col]
        .astype(str)
        .str.strip()
    )

    if final_ids.duplicated().any():
        raise RuntimeError(
            "Duplicate GP IDs appeared after merge."
        )

    if merged[feature_columns].isna().any().any():
        bad_columns = (
            merged[feature_columns]
            .columns[
                merged[feature_columns]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise RuntimeError(
            "Missing land-cover values after merge:\n"
            f"{bad_columns}"
        )

    numeric_final = merged[
        feature_columns
    ].select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric_final.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite numeric values found after merge."
        )

    fraction_columns = list(
        OUTPUT_NAMES.values()
    )

    fraction_values = merged[
        fraction_columns
    ].to_numpy(dtype=float)

    if (fraction_values < -1e-6).any() or (
        fraction_values > 1.000001
    ).any():
        raise RuntimeError(
            "Final land-cover fractions outside [0, 1]."
        )

    final_sums = merged[
        "landcover_probability_sum"
    ].to_numpy(dtype=float)

    if not np.all(
        np.abs(final_sums - 1.0) <= 0.02
    ):
        raise RuntimeError(
            "Final land-cover probability sums failed QC."
        )

    if not np.isfinite(
        merged["landcover_entropy"]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Land-cover entropy contains non-finite values."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    csv_path = (
        output_dir /
        "gp_landcover_features.csv"
    )

    geojson_path = (
        output_dir /
        "gp_landcover_features.geojson"
    )

    metadata_path = (
        raw_dir /
        "landcover_source_metadata.json"
    )

    merged.drop(
        columns="geometry"
    ).to_csv(
        csv_path,
        index=False
    )

    merged.to_file(
        geojson_path,
        driver="GeoJSON"
    )

    metadata = {
        "stage": "03_gp_landcover",
        "status": "PASS",
        "project_root": str(PROJECT_ROOT),
        "state": state,
        "district": district,
        "block": block,
        "gp_master": str(gp_master_path),
        "gp_id_column": str(id_col),
        "gp_name_column": (
            str(name_col)
            if name_col
            else None
        ),
        "gp_count_input": int(len(gdf)),
        "gp_count_output": int(len(merged)),
        "earth_engine_collection": EE_COLLECTION,
        "baseline_year": BASELINE_YEAR,
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "dynamic_world_image_count": image_count,
        "spatial_resolution_m": SCALE_M,
        "features": feature_columns,
        "quality_control": {
            "gp_count_match": True,
            "duplicate_gp_ids": False,
            "missing_values": False,
            "non_finite_values": False,
            "probabilities_in_range": True,
            "probability_sum_max_deviation": max_probability_error,
            "minimum_valid_pixel_fraction": 0.99,
            "synthetic_values": False,
            "imputation": False,
            "geometry_modified": False,
            "silent_gp_deletion": False,
        },
        "outputs": {
            "csv": str(csv_path),
            "geojson": str(geojson_path),
            "metadata": str(metadata_path),
        },
        "created_utc": utc_now(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print()
    print("-" * 72)
    print(f"LAND COVER COMPLETE — {block}")
    print("-" * 72)
    print(f"GP count input:        {len(gdf)}")
    print(f"GP count output:       {len(merged)}")
    print(f"GP count match:        PASS")
    print(f"Duplicate GP IDs:      0")
    print(f"Missing values:        0")
    print(f"Non-finite values:     0")
    print(f"Probability range:     PASS")
    print(f"Probability sums:      PASS")
    print(f"Valid coverage:        PASS")
    print(f"Synthetic values:      FALSE")
    print(f"Imputation:            FALSE")
    print()
    print(f"CSV:       {csv_path}")
    print(f"GeoJSON:   {geojson_path}")
    print(f"Metadata:  {metadata_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Universal SIH26 GP Dynamic World land-cover extraction."
        )
    )

    parser.add_argument(
        "--block",
        default=None,
        help=(
            "Optional block name. If omitted, every discovered "
            "gp_master.geojson is processed."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print("SIH26 — UNIVERSAL GP LAND-COVER PIPELINE")
    print("=" * 72)
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Project root:    {PROJECT_ROOT}")
    print(f"GP root:         {GP_ROOT}")
    print()

    masters = discover_gp_masters(
        args.block
    )

    print(
        f"GP master(s) discovered: {len(masters)}"
    )

    for path in masters:
        print(f"  - {path}")

    print()

    # Authenticate/init once.
    try:
        ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Google Earth Engine initialization failed.\n"
            "Authenticate Earth Engine before running this stage."
        ) from exc

    print("Earth Engine initialized: PASS")

    failures = []

    for master in masters:

        try:
            process_block(master)

        except Exception as exc:

            failures.append(
                {
                    "gp_master": str(master),
                    "error": str(exc),
                }
            )

            print()
            print("ERROR")
            print("=" * 72)
            print(str(exc))
            print("=" * 72)

            # Continue to the next discovered block instead of allowing
            # one bad block to prevent all other blocks from processing.
            continue

    print()
    print("=" * 72)
    print("STAGE 3 — FINAL PIPELINE REPORT")
    print("=" * 72)

    successful = len(masters) - len(failures)

    print(f"Discovered blocks:  {len(masters)}")
    print(f"Successful blocks:  {successful}")
    print(f"Failed blocks:      {len(failures)}")

    if failures:

        print()
        print("Failures:")

        for failure in failures:
            print(
                f"- {failure['gp_master']}"
            )
            print(
                f"  {failure['error']}"
            )

        raise RuntimeError(
            f"{len(failures)} block(s) failed. "
            "See the detailed errors above."
        )

    print()
    print("ALL DISCOVERED BLOCKS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
