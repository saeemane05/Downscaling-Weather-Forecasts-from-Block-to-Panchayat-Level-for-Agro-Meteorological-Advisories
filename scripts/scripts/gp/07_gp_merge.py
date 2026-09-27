"""
SIH26 — GP FEATURE MERGER
Stage 8: Merge all completed GP-level spatial predictors into one
universal, leakage-safe, provenance-preserving dataset.

INPUT
-----
<project_root>/gp/<State>/<District>/<Block>/processed/gp_master.geojson

Processed GP feature sources:
    gp_terrain_features.csv
    gp_landcover_features.csv
    gp_sentinel2_features.csv
    gp_sentinel1_features.csv
    gp_soil_features.csv
    gp_lst_features.csv

OUTPUT
------
<block>/processed/
    gp_merged_features.csv
    gp_merged_features.geojson
    gp_merge_metadata.json

DESIGN PRINCIPLES
-----------------
1. State / district / block are supplied by the user.
2. No state, district, block, GP ID, or file path is hard-coded.
3. gp_master.geojson is the authoritative GP identity + geometry source.
4. Every feature source must contain exactly one row per GP ID.
5. All joins are strict one-to-one joins.
6. No fillna(), interpolation, synthetic values, or imputation.
7. Missing GP data causes the merge to STOP rather than fabricate data.
8. Geometry is taken ONLY from gp_master.geojson.
9. Raw observation/acquisition tables are NOT merged here.
10. Existing outputs are not treated as inputs to the merge.
11. A failed QC run does not write the final dataset.
12. Provenance and QC are written to metadata.

This stage is intentionally separate from the one-click pipeline runner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_SOURCES = {
    "terrain": "gp_terrain_features.csv",
    "landcover": "gp_landcover_features.csv",
    "sentinel2": "gp_sentinel2_features.csv",
    "sentinel1": "gp_sentinel1_features.csv",
    "soil": "gp_soil_features.csv",
    "lst": "gp_lst_features.csv",
}

OUTPUT_CSV = "gp_merged_features.csv"
OUTPUT_GEOJSON = "gp_merged_features.geojson"
OUTPUT_METADATA = "gp_merge_metadata.json"

# Common accepted GP identity names. Current SIH26 outputs use gp_id.
GP_ID_CANDIDATES = (
    "gp_id",
    "GP_ID",
    "gp_code",
    "gpcode",
    "gp_lgd_code",
    "gp_lgd",
)

GP_NAME_CANDIDATES = (
    "gp_name",
    "GP_NAME",
    "gpname",
    "name",
)


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_location(value: str) -> str:
    """
    Convert a user-supplied location component into the directory naming
    convention used by the SIH26 GP pipeline.

    This is deliberately NOT used for identity comparison; it only resolves
    the existing project directory.
    """
    value = str(value).strip()
    value = re.sub(r"\s+", "_", value)
    return value


def project_root() -> Path:
    # scripts/gp/08_gp_merge.py -> project root is ../../
    return Path(__file__).resolve().parents[2]


def resolve_gp_project(state: str, district: str, block: str) -> Path:
    return (
        project_root()
        / "gp"
        / clean_location(state)
        / clean_location(district)
        / str(block).strip()
    )


def find_column(
    columns: List[str],
    candidates: Tuple[str, ...],
    label: str,
) -> str:
    """
    Find a column without silently guessing an unrelated column.
    Matching is exact first, then case-insensitive.
    """
    column_set = list(columns)

    for candidate in candidates:
        if candidate in column_set:
            return candidate

    lower_map = {str(c).lower(): c for c in column_set}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"{label} column not found. "
        f"Expected one of: {list(candidates)}. "
        f"Available columns: {column_set}"
    )


def canonicalize_gp_id(series: pd.Series) -> pd.Series:
    """
    Convert IDs to strings for safe cross-file joining.

    Important:
    - IDs are identifiers, not measurements.
    - Missing IDs remain missing.
    - No numeric rounding or invented IDs are performed.
    """
    result = series.astype("string").str.strip()

    # Handle common CSV artifacts such as 123.0 created when an integer
    # identifier was accidentally serialized as a floating-point value.
    result = result.str.replace(r"^(\d+)\.0$", r"\1", regex=True)

    return result


def load_master(master_path: Path) -> Tuple[gpd.GeoDataFrame, str, str]:
    if not master_path.exists():
        raise FileNotFoundError(f"GP master not found: {master_path}")

    gdf = gpd.read_file(master_path)

    if gdf.empty:
        raise ValueError("GP master is empty.")

    if gdf.crs is None:
        raise ValueError("GP master has no CRS.")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    id_col = find_column(list(gdf.columns), GP_ID_CANDIDATES, "GP ID")
    name_col = find_column(list(gdf.columns), GP_NAME_CANDIDATES, "GP name")

    gdf[id_col] = canonicalize_gp_id(gdf[id_col])

    if gdf[id_col].isna().any() or (gdf[id_col] == "").any():
        raise ValueError("GP master contains missing/empty GP IDs.")

    duplicate_ids = gdf.loc[gdf[id_col].duplicated(keep=False), id_col].unique()
    if len(duplicate_ids):
        raise ValueError(
            "GP master contains duplicate GP IDs. "
            f"Duplicate IDs: {duplicate_ids.tolist()}"
        )

    if gdf.geometry.is_empty.any():
        raise ValueError("GP master contains empty geometries.")

    if (~gdf.geometry.is_valid).any():
        raise ValueError(
            "GP master contains invalid geometries. "
            "Fix Stage 1 before running the GP merger."
        )

    return gdf, id_col, name_col


def load_feature_table(
    path: Path,
    source_name: str,
    master_ids: set[str],
) -> Tuple[pd.DataFrame, str, dict]:
    """
    Load one processed GP feature table and perform strict identity QC.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required processed feature file is missing for {source_name}: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Processed feature file is empty for {source_name}: {path}"
        )

    id_col = find_column(list(df.columns), GP_ID_CANDIDATES, f"{source_name} GP ID")
    df[id_col] = canonicalize_gp_id(df[id_col])

    missing_id = int(df[id_col].isna().sum() + (df[id_col] == "").sum())

    if missing_id:
        raise ValueError(
            f"{source_name}: {missing_id} row(s) have missing/empty GP IDs."
        )

    duplicate_mask = df[id_col].duplicated(keep=False)
    duplicate_ids = sorted(df.loc[duplicate_mask, id_col].unique().tolist())

    if duplicate_ids:
        raise ValueError(
            f"{source_name}: duplicate GP IDs detected: {duplicate_ids}"
        )

    source_ids = set(df[id_col].tolist())

    missing_from_source = sorted(master_ids - source_ids)
    unexpected_ids = sorted(source_ids - master_ids)

    if missing_from_source:
        raise ValueError(
            f"{source_name}: missing {len(missing_from_source)} GP(s) from "
            f"the authoritative gp_master. Missing IDs: {missing_from_source}"
        )

    if unexpected_ids:
        raise ValueError(
            f"{source_name}: contains {len(unexpected_ids)} GP ID(s) not present "
            f"in gp_master. Unexpected IDs: {unexpected_ids}"
        )

    if len(df) != len(master_ids):
        raise ValueError(
            f"{source_name}: row count {len(df)} does not equal "
            f"gp_master GP count {len(master_ids)}."
        )

    # A processed feature table should be attributes, not geometry.
    # If a geometry column exists, it is ignored deliberately. Geometry will
    # come only from gp_master.
    geometry_like = [
        c for c in df.columns
        if str(c).lower() in {"geometry", "geom", "wkt"}
    ]

    df = df.drop(columns=geometry_like, errors="ignore")

    # Rename identity to canonical gp_id.
    if id_col != "gp_id":
        df = df.rename(columns={id_col: "gp_id"})

    # Detect identity/name collisions before prefixing.
    feature_columns = [c for c in df.columns if c not in {"gp_id", "gp_name"}]
    # GP name is authoritative in gp_master.
# Feature tables contribute only their actual predictor columns.
    df = df.drop(columns=["gp_name"], errors="ignore")

    if not feature_columns:
        raise ValueError(
            f"{source_name}: no feature columns remain after identity cleanup."
        )

    # Prefix source-specific feature columns to prevent accidental collisions
    # across sources while retaining the original feature meaning.
    rename_map = {
        c: f"{source_name}__{c}"
        for c in feature_columns
    }
    df = df.rename(columns=rename_map)

    # Check for duplicate column names after transformation.
    if df.columns.duplicated().any():
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"{source_name}: duplicate columns after processing: "
            f"{duplicated_columns}"
        )

    qc = {
        "source": source_name,
        "file": str(path),
        "rows": int(len(df)),
        "gp_id_column_original": id_col,
        "gp_count": int(df["gp_id"].nunique()),
        "duplicate_gp_ids": len(duplicate_ids),
        "missing_from_master": len(missing_from_source),
        "unexpected_gp_ids": len(unexpected_ids),
        "feature_count": len(feature_columns),
        "feature_columns_original": feature_columns,
        "geometry_columns_ignored": geometry_like,
    }

    return df, id_col, qc


def check_numeric_quality(df: pd.DataFrame) -> dict:
    """
    Report missing/non-finite values without imputing them.

    Missing values are allowed because the source stages legitimately have
    different observation coverage. The merger never fills them.
    """
    feature_cols = [c for c in df.columns if c not in {"gp_id", "gp_name", "geometry"}]

    missing_counts = {}
    nonfinite_counts = {}

    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        missing_counts[col] = int(numeric.isna().sum())

        # Object/string columns can legitimately be categorical, so only
        # inspect non-finite values where numeric conversion is meaningful.
        original_non_null = df[col].notna()
        if original_non_null.any():
            nonfinite = np.isinf(numeric.fillna(0).to_numpy(dtype=float))
            nonfinite_counts[col] = int(nonfinite.sum())
        else:
            nonfinite_counts[col] = 0

    return {
        "feature_columns": len(feature_cols),
        "columns_with_missing_values": {
            k: v for k, v in missing_counts.items() if v > 0
        },
        "total_missing_cells": int(sum(missing_counts.values())),
        "columns_with_nonfinite_values": {
            k: v for k, v in nonfinite_counts.items() if v > 0
        },
        "total_nonfinite_cells": int(sum(nonfinite_counts.values())),
    }


def merge_source(
    base: pd.DataFrame,
    source_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Strict one-to-one merge.

    validate='one_to_one' is deliberately used so a future source bug cannot
    silently multiply GP rows.
    """
    overlap = (
        set(base.columns)
        & set(source_df.columns)
    ) - {"gp_id"}

    if overlap:
        raise ValueError(
            f"{source_name}: unexpected column collision with merged table: "
            f"{sorted(overlap)}"
        )

    merged = base.merge(
        source_df,
        on="gp_id",
        how="left",
        validate="one_to_one",
        indicator=False,
    )

    if len(merged) != len(base):
        raise RuntimeError(
            f"{source_name}: merge changed GP row count from "
            f"{len(base)} to {len(merged)}."
        )

    return merged


# ============================================================================
# MAIN MERGE
# ============================================================================

def run_merge(
    state: str,
    district: str,
    block: str,
) -> None:

    start_time = datetime.now(timezone.utc)

    gp_project = resolve_gp_project(state, district, block)
    processed_dir = gp_project / "processed"
    master_path = processed_dir / "gp_master.geojson"

    output_csv = processed_dir / OUTPUT_CSV
    output_geojson = processed_dir / OUTPUT_GEOJSON
    output_metadata = processed_dir / OUTPUT_METADATA

    print("=" * 90)
    print("SIH26 — STAGE 8: GP FEATURE MERGE")
    print("=" * 90)
    print(f"State:          {state}")
    print(f"District:       {district}")
    print(f"Block:          {block}")
    print(f"Project:        {gp_project}")
    print(f"GP master:      {master_path}")
    print()
    print("Merge policy:")
    print("  • gp_master.geojson = authoritative GP identity + geometry")
    print("  • all six processed feature sources are required")
    print("  • strict one-to-one GP ID joins")
    print("  • no synthetic values")
    print("  • no imputation/interpolation")
    print("  • geometry comes only from gp_master")
    print()

    if not gp_project.exists():
        raise FileNotFoundError(
            f"GP project directory does not exist: {gp_project}"
        )

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory does not exist: {processed_dir}"
        )

    # ---------------------------------------------------------------------
    # 1. Load authoritative GP master
    # ---------------------------------------------------------------------
    print("[1/4] Loading authoritative GP master...")
    master_gdf, master_id_col, master_name_col = load_master(master_path)

    master_ids = set(master_gdf[master_id_col].tolist())

    print(f"  GP count:       {len(master_gdf)}")
    print(f"  GP ID column:   {master_id_col}")
    print(f"  GP name column: {master_name_col}")
    print("  Geometry:       PASS")
    print("  Unique GP IDs:  PASS")

    # Start final table with ONLY authoritative identity.
    merged = pd.DataFrame({
        "gp_id": master_gdf[master_id_col].astype("string"),
        "gp_name": master_gdf[master_name_col].astype("string"),
    })

    # ---------------------------------------------------------------------
    # 2. Load and validate all processed feature sources
    # ---------------------------------------------------------------------
    print()
    print("[2/4] Validating processed GP feature sources...")

    source_qc = {}
    source_paths = {}

    for source_name, filename in FEATURE_SOURCES.items():
        path = processed_dir / filename
        source_paths[source_name] = path

        print(f"\n  {source_name.upper()}")
        print(f"    File: {path}")

        source_df, _, qc = load_feature_table(
            path,
            source_name,
            master_ids,
        )

        print(f"    Rows:             {qc['rows']}")
        print(f"    Unique GP IDs:    {qc['gp_count']}")
        print(f"    Duplicate GP IDs: {qc['duplicate_gp_ids']}")
        print(f"    Feature columns:  {qc['feature_count']}")

        if qc["geometry_columns_ignored"]:
            print(
                "    Geometry columns ignored: "
                f"{qc['geometry_columns_ignored']}"
            )

        merged = merge_source(
            merged,
            source_df,
            source_name,
        )

        source_qc[source_name] = qc

        print("    One-to-one merge:  PASS")

    # ---------------------------------------------------------------------
    # 3. Final QC
    # ---------------------------------------------------------------------
    print()
    print("[3/4] Running final merged-dataset QC...")

    expected_gp_count = len(master_gdf)

    final_gp_count = len(merged)
    unique_gp_count = merged["gp_id"].nunique()
    duplicate_gp_count = int(merged["gp_id"].duplicated().sum())

    if final_gp_count != expected_gp_count:
        raise RuntimeError(
            f"Final GP row count mismatch: {final_gp_count} != "
            f"{expected_gp_count}"
        )

    if unique_gp_count != expected_gp_count:
        raise RuntimeError(
            f"Final unique GP count mismatch: {unique_gp_count} != "
            f"{expected_gp_count}"
        )

    if duplicate_gp_count != 0:
        raise RuntimeError(
            f"Final dataset contains {duplicate_gp_count} duplicate GP rows."
        )

    # Do NOT reject ordinary NaNs here. Different remote-sensing sources can
    # have legitimate missing observations, and the project policy is to
    # preserve those gaps rather than fabricate values.
    numeric_qc = check_numeric_quality(merged)

    print(f"  GP count match:       PASS ({final_gp_count})")
    print(f"  Unique GP IDs:        PASS ({unique_gp_count})")
    print(f"  Duplicate GP rows:    PASS ({duplicate_gp_count})")
    print(
        "  Missing values:       "
        f"{numeric_qc['total_missing_cells']} preserved; NO imputation"
    )
    print(
        "  Non-finite values:    "
        f"{numeric_qc['total_nonfinite_cells']}"
    )

    if numeric_qc["total_nonfinite_cells"] > 0:
        raise RuntimeError(
            "Final merged dataset contains non-finite numeric values. "
            "No output will be written."
        )

    # Verify that the GP master ordering is retained.
    expected_order = master_gdf[master_id_col].astype("string").tolist()
    actual_order = merged["gp_id"].astype("string").tolist()

    if actual_order != expected_order:
        raise RuntimeError(
            "Final GP ordering does not match gp_master ordering."
        )

    print("  GP ordering:          PASS")
    print("  Synthetic data:       DISABLED")
    print("  Imputation:           DISABLED")
    print("  Geometry authority:   gp_master")

    # ---------------------------------------------------------------------
    # 4. Save outputs
    # ---------------------------------------------------------------------
    print()
    print("[4/4] Writing final GP dataset...")

    # Build GIS output from gp_master geometry only.
    output_gdf = master_gdf[[master_id_col, master_name_col, "geometry"]].copy()

    output_gdf = output_gdf.rename(
        columns={
            master_id_col: "gp_id",
            master_name_col: "gp_name",
        }
    )

    # Avoid writing a second gp_id/gp_name from the feature table.
    attributes = merged.drop(columns=["gp_name"])

    output_gdf = output_gdf.merge(
        attributes,
        on="gp_id",
        how="left",
        validate="one_to_one",
    )

    if len(output_gdf) != expected_gp_count:
        raise RuntimeError(
            "GeoJSON merge changed GP count. Final outputs will not be written."
        )

    # CSV intentionally contains no geometry.
    merged.to_csv(output_csv, index=False)

    # GeoJSON contains authoritative GP geometry + merged attributes.
    output_gdf.to_file(
        output_geojson,
        driver="GeoJSON",
    )

    elapsed = (
        datetime.now(timezone.utc) - start_time
    ).total_seconds()

    metadata = {
        "pipeline": "SIH26",
        "stage": "Stage 8 — GP Feature Merge",
        "status": "SUCCESS",
        "created_utc": now_utc(),
        "location": {
            "state": str(state),
            "district": str(district),
            "block": str(block),
        },
        "project_directory": str(gp_project),
        "authoritative_gp_master": str(master_path),
        "gp_master_identity_columns": {
            "gp_id": master_id_col,
            "gp_name": master_name_col,
        },
        "gp_count": int(expected_gp_count),
        "merge_policy": {
            "join_key": "gp_id",
            "join_type": "left",
            "validation": "one_to_one",
            "geometry_source": "gp_master.geojson",
            "geometry_reused_from_master_only": True,
            "raw_temporal_tables_merged": False,
            "synthetic_data": False,
            "imputation": False,
            "interpolation": False,
        },
        "sources": source_qc,
        "final_dataset": {
            "row_count": int(len(merged)),
            "unique_gp_ids": int(unique_gp_count),
            "duplicate_gp_rows": int(duplicate_gp_count),
            "column_count_csv": int(len(merged.columns)),
            "feature_count_excluding_identity": int(
                len([c for c in merged.columns if c not in {"gp_id", "gp_name"}])
            ),
            "numeric_quality": numeric_qc,
        },
        "outputs": {
            "csv": str(output_csv),
            "geojson": str(output_geojson),
            "metadata": str(output_metadata),
        },
        "elapsed_seconds": float(elapsed),
    }

    output_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("FINAL STATUS: SUCCESS")
    print("=" * 90)
    print(f"GPs:              {expected_gp_count}")
    print(
        "Model attributes:  "
        f"{len([c for c in merged.columns if c not in {'gp_id', 'gp_name'}])}"
    )
    print(f"CSV columns:       {len(merged.columns)}")
    print(f"Missing cells:     {numeric_qc['total_missing_cells']} (preserved)")
    print(f"CSV:               {output_csv}")
    print(f"GeoJSON:           {output_geojson}")
    print(f"Metadata:          {output_metadata}")
    print(f"Elapsed:           {elapsed:.1f}s")
    print("=" * 90)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH26 universal GP feature merger."
    )

    parser.add_argument(
        "--state",
        help="State name, e.g. Maharashtra",
    )
    parser.add_argument(
        "--district",
        help="District name, e.g. Pune",
    )
    parser.add_argument(
        "--block",
        help="Block/Taluka name, e.g. Baramati",
    )

    return parser.parse_args()


def get_location_from_user(args: argparse.Namespace) -> Tuple[str, str, str]:
    state = args.state or input("Enter State    : ").strip()
    district = args.district or input("Enter District : ").strip()
    block = args.block or input("Enter Block    : ").strip()

    if not state or not district or not block:
        raise ValueError(
            "State, District, and Block are required."
        )

    return state, district, block


def main() -> None:
    args = parse_args()

    try:
        state, district, block = get_location_from_user(args)
        run_merge(state, district, block)

    except KeyboardInterrupt:
        print("\n\nMerge interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 90)
        print("FINAL STATUS: FAILED")
        print("=" * 90)
        print(str(exc))
        print()
        print("No final merged dataset was intentionally written after this failure.")
        print("=" * 90)
        sys.exit(1)


if __name__ == "__main__":
    main()
