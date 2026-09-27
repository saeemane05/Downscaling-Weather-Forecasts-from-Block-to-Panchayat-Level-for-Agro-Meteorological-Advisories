"""
SIH26 - Universal GP Terrain Feature Extraction
Stage 2: Terrain predictors for Gram Panchayat-level weather downscaling.

Inputs
------
<project_root>/gp/<State>/<District>/<Block>/processed/gp_master.geojson

Outputs
-------
gp/<State>/<District>/<Block>/
    raw/terrain/
        terrain_source_metadata.json
    processed/
        gp_terrain_features.csv
        gp_terrain_features.geojson

Method
------
DEM source:
    NASA/USGS SRTMGL1_003 via Google Earth Engine
    ~30 m / 1 arc-second

Terrain layers:
    elevation
    slope
    aspect
    laplacian_curvature
    TPI (300 m neighborhood)
    TRI (90 m local variability proxy)

For each GP, zonal statistics are calculated:
    mean, min, max, std

Aspect is treated as a circular variable:
    aspect_sin_mean
    aspect_cos_mean
    aspect_mean_deg

No synthetic/fallback terrain values are created.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEM_ID = "USGS/SRTMGL1_003"

# Terrain neighborhood definitions.
# 30 m SRTM pixels:
#   3 pixels ~= 90 m local terrain variability
#   10 pixels ~= 300 m local position
TRI_RADIUS_PIXELS = 3
TPI_RADIUS_PIXELS = 10

# Small buffer around the Block for neighborhood-derived terrain.
# The final zonal statistics are still restricted to GP polygons.
BLOCK_BUFFER_DEGREES = 0.01


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def project_root() -> Path:
    # scripts/gp/02_gp_terrain.py -> project root is ../../
    return Path(__file__).resolve().parents[2]


def resolve_gp_project(state: str, district: str, block: str) -> Path:
    root = project_root()
    return root / "gp" / state.strip().replace(" ", "_") / district.strip().replace(" ", "_") / block.strip()


# ---------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------

def initialize_gee() -> None:
    try:
        ee.Initialize()
    except Exception:
        print("\nGoogle Earth Engine is not initialized for this Python environment.")
        print("Run the normal Earth Engine authentication/setup for this machine,")
        print("then rerun this script.")
        raise


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def load_gp_master(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GP master not found: {path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError("GP master is empty.")

    if gdf.crs is None:
        raise ValueError("GP master has no CRS.")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if gdf.geometry.is_empty.any():
        raise ValueError("GP master contains empty geometries.")

    if (~gdf.geometry.is_valid).any():
        raise ValueError("GP master contains invalid geometries. Fix Stage 1 before terrain extraction.")

    return gdf


def find_column(gdf: gpd.GeoDataFrame, candidates: list[str], label: str) -> str:
    lookup = {str(c).lower(): c for c in gdf.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"Could not find {label} column. Tried: {candidates}")


def geodataframe_to_ee_features(gdf: gpd.GeoDataFrame, id_col: str, name_col: str) -> ee.FeatureCollection:
    features = []

    for _, row in gdf.iterrows():
        geom = row.geometry.__geo_interface__
        props = {
            "gp_id": str(row[id_col]),
            "gp_name": str(row[name_col]),
        }

        # Keep useful Stage-1 provenance fields if present.
        for col in [
            "resolution",
            "geometry_status",
            "source_feature_count",
            "source_area_km2",
            "analysis_area_km2",
        ]:
            if col in gdf.columns:
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, np.generic):
                        value = value.item()
                    props[col] = value

        features.append(ee.Feature(ee.Geometry(geom), props))

    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------
# Terrain image construction
# ---------------------------------------------------------------------

def build_terrain_image() -> ee.Image:
    dem = ee.Image(DEM_ID).select("elevation")

    terrain = ee.Algorithms.Terrain(dem)

    elevation = dem.rename("elevation")
    slope = terrain.select("slope").rename("slope")
    aspect = terrain.select("aspect").rename("aspect")

    # Aspect in radians for circular representation.
    aspect_rad = aspect.multiply(np.pi / 180.0)

    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")

    # Local terrain variability proxy:
    # standard deviation of elevation in a ~90 m neighborhood.
    tri = (
        dem.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(),
            kernel=ee.Kernel.circle(
                radius=TRI_RADIUS_PIXELS,
                units="pixels",
            ),
        )
        .rename("tri")
    )

    # TPI = local elevation - local mean elevation.
    local_mean = dem.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(
            radius=TPI_RADIUS_PIXELS,
            units="pixels",
        ),
    )

    tpi = dem.subtract(local_mean).rename("tpi")

    # Formal second-derivative terrain curvature:
    # a discrete 3x3 Laplacian of elevation.
    #
    # Kernel:
    #   [0  1  0]
    #   [1 -4  1]
    #   [0  1  0]
    #
    # This measures local convexity/concavity from second-order
    # elevation derivatives. It is intentionally named
    # "laplacian_curvature" rather than profile/plan curvature,
    # because profile and plan curvature require a directional
    # differential-geometry definition.
    curvature_kernel = ee.Kernel.fixed(
        3,
        3,
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        1,
        1,
        False,
    )

    laplacian_curvature = (
        dem.convolve(curvature_kernel)
        .rename("laplacian_curvature")
    )

    return ee.Image.cat(
        [
            elevation,
            slope,
            aspect_sin,
            aspect_cos,
            tri,
            tpi,
            laplacian_curvature,
        ]
    )


# ---------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------

def reduce_zonal_stats(
    terrain: ee.Image,
    gp_fc: ee.FeatureCollection,
) -> list[dict]:
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )

    reduced = terrain.reduceRegions(
        collection=gp_fc,
        reducer=reducer,
        scale=30,
        tileScale=4,
    )

    return reduced.getInfo()["features"]


def clean_reduced_records(records: list[dict]) -> pd.DataFrame:
    rows = []

    for feature in records:
        props = feature.get("properties", {})
        rows.append(props)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Earth Engine returned no GP terrain statistics.")

    return df


# ---------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------

def rename_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Earth Engine's combined reducer produces names such as:
      elevation_mean
      elevation_min
      elevation_max
      elevation_stdDev

    Normalize stdDev -> std.
    """
    rename = {}
    for col in df.columns:
        if col.endswith("_stdDev"):
            rename[col] = col[:-7] + "_std"

    return df.rename(columns=rename)


def add_circular_aspect(df: pd.DataFrame) -> pd.DataFrame:
    if "aspect_sin_mean" not in df.columns or "aspect_cos_mean" not in df.columns:
        raise ValueError("Aspect circular components are missing.")

    sin_mean = pd.to_numeric(df["aspect_sin_mean"], errors="coerce")
    cos_mean = pd.to_numeric(df["aspect_cos_mean"], errors="coerce")

    # atan2(sin, cos) -> radians -> degrees, normalized to [0, 360).
    angle = np.degrees(np.arctan2(sin_mean, cos_mean))
    df["aspect_mean_deg"] = np.mod(angle, 360.0)

    # Circular concentration / resultant length.
    df["aspect_resultant_length"] = np.sqrt(
        sin_mean**2 + cos_mean**2
    )

    return df


def validate_output(
    df: pd.DataFrame,
    expected_gp_count: int,
) -> dict:
    """Validate only actual terrain variables; never coerce metadata to numeric."""
    terrain_cols = [
        c for c in df.columns
        if (c.startswith("elevation_") or c.startswith("slope_")
            or c.startswith("aspect_sin_") or c.startswith("aspect_cos_")
            or c.startswith("tpi_") or c.startswith("tri_")
            or c.startswith("laplacian_curvature_")
            or c in {"aspect_mean_deg", "aspect_resultant_length"})
    ]
    if not terrain_cols:
        raise ValueError("No terrain feature columns were found.")

    numeric = df[terrain_cols].apply(pd.to_numeric, errors="coerce")
    missing_by_column = {c: int(numeric[c].isna().sum()) for c in terrain_cols if numeric[c].isna().any()}
    nonfinite_by_column = {}
    for c in terrain_cols:
        vals = numeric[c].to_numpy(dtype=float)
        n = int((~np.isfinite(vals)).sum())
        if n:
            nonfinite_by_column[c] = n

    problematic_cells = []
    for c in terrain_cols:
        vals = numeric[c].to_numpy(dtype=float)
        for i in np.where(~np.isfinite(vals))[0]:
            problematic_cells.append({
                "row": int(i),
                "gp_id": str(df.iloc[i]["gp_id"]),
                "gp_name": str(df.iloc[i]["gp_name"]),
                "feature": c,
                "value": repr(df.iloc[i][c]),
            })

    report = {
        "rows": int(len(df)),
        "expected_gp_count": int(expected_gp_count),
        "row_count_match": bool(len(df) == expected_gp_count),
        "duplicate_gp_ids": int(df["gp_id"].duplicated().sum()),
        "terrain_feature_columns": terrain_cols,
        "terrain_feature_count": len(terrain_cols),
        "all_terrain_values_finite": not nonfinite_by_column,
        "missing_values_by_terrain_column": missing_by_column,
        "nonfinite_values_by_terrain_column": nonfinite_by_column,
        "problematic_cells": problematic_cells,
    }
    if not report["row_count_match"]:
        raise ValueError("GP count mismatch: " + json.dumps(report, indent=2))
    if report["duplicate_gp_ids"]:
        raise ValueError("Duplicate GP IDs: " + json.dumps(report, indent=2))
    if nonfinite_by_column:
        raise ValueError("Missing/non-finite terrain values detected; no imputation performed.\n" + json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("SIH26 - UNIVERSAL GP TERRAIN FEATURE EXTRACTION")
    print("=" * 70)

    state = input("Enter State   : ").strip()
    district = input("Enter District: ").strip()
    block = input("Enter Block    : ").strip()

    gp_project = resolve_gp_project(state, district, block)
    processed_dir = gp_project / "processed"
    raw_terrain_dir = gp_project / "raw" / "terrain"

    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_terrain_dir.mkdir(parents=True, exist_ok=True)

    gp_master_path = processed_dir / "gp_master.geojson"

    print("\nProject directory:")
    print(f"  {gp_project}")

    print("\n[1/6] Loading Stage-1 GP master...")
    gdf = load_gp_master(gp_master_path)

    id_col = find_column(
        gdf,
        ["gp_id", "gp_code", "gpcode"],
        "GP ID",
    )
    name_col = find_column(
        gdf,
        ["gp_name", "gpname", "name"],
        "GP name",
    )

    print(f"  GP records: {len(gdf)}")
    print(f"  GP ID column: {id_col}")
    print(f"  GP name column: {name_col}")

    print("\n[2/6] Initializing Google Earth Engine...")
    initialize_gee()
    print("  Earth Engine initialized.")

    print("\n[3/6] Building terrain layers...")
    print(f"  DEM: {DEM_ID}")
    print("  Nominal DEM resolution: ~30 m")
    print(f"  TRI neighborhood: {TRI_RADIUS_PIXELS} pixels (~90 m)")
    print(f"  TPI neighborhood: {TPI_RADIUS_PIXELS} pixels (~300 m)")
    print("  Curvature: formal 3x3 discrete Laplacian of elevation")

    terrain = build_terrain_image()

    print("\n[4/6] Extracting GP zonal statistics...")
    gp_fc = geodataframe_to_ee_features(gdf, id_col, name_col)
    records = reduce_zonal_stats(terrain, gp_fc)

    df = clean_reduced_records(records)
    df = rename_stat_columns(df)
    df = add_circular_aspect(df)

    # Preserve stable ordering from gp_master.
    master_ids = gdf[id_col].astype(str).tolist()
    df["gp_id"] = df["gp_id"].astype(str)

    order = {gp_id: i for i, gp_id in enumerate(master_ids)}
    df["_order"] = df["gp_id"].map(order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # Put identity columns first.
    identity = ["gp_id", "gp_name"]
    other = [c for c in df.columns if c not in identity]
    df = df[identity + other]

    print(f"  Terrain records returned: {len(df)}")

    print("\n[5/6] Running QC...")
    qc = validate_output(df, len(gdf))

    print(f"  GP count match : {qc['row_count_match']}")
    print(f"  Duplicate IDs  : {qc['duplicate_gp_ids']}")
    print(f"  Finite values  : {qc['all_terrain_values_finite']}")

    # Save CSV.
    csv_path = processed_dir / "gp_terrain_features.csv"
    df.to_csv(csv_path, index=False)

    # Join terrain attributes back to the Stage-1 geometries.
    output_gdf = gdf.copy()
    output_gdf[id_col] = output_gdf[id_col].astype(str)

    terrain_join = df.copy().rename(
        columns={
            "gp_id": id_col,
            "gp_name": name_col,
        }
    )

    output_gdf = output_gdf.merge(
        terrain_join,
        on=[id_col, name_col],
        how="left",
        validate="one_to_one",
        suffixes=("", "_terrain"),
    )

    geojson_path = processed_dir / "gp_terrain_features.geojson"
    output_gdf.to_file(geojson_path, driver="GeoJSON")

    print("\n[6/6] Writing provenance metadata...")

    metadata = {
        "pipeline": "SIH26 Universal GP Terrain Feature Extraction",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "district": district,
        "block": block,
        "input_gp_master": str(gp_master_path),
        "dem_source": DEM_ID,
        "dem_nominal_resolution_m": 30,
        "earth_engine": True,
        "terrain_layers": [
            "elevation",
            "slope",
            "aspect_sin",
            "aspect_cos",
            "tri",
            "tpi",
            "laplacian_curvature",
        ],
        "tri_radius_pixels": TRI_RADIUS_PIXELS,
        "tpi_radius_pixels": TPI_RADIUS_PIXELS,
        "zonal_statistics": [
            "mean",
            "min",
            "max",
            "std",
        ],
        "aspect_method": "circular representation using mean sine/cosine and atan2-derived mean direction",
        "synthetic_values": False,
        "gp_count": int(len(df)),
        "qc": qc,
        "outputs": {
            "csv": str(csv_path),
            "geojson": str(geojson_path),
        },
    }

    metadata_path = raw_terrain_dir / "terrain_source_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\n  CSV     : {csv_path}")
    print(f"  GeoJSON : {geojson_path}")
    print(f"  Metadata: {metadata_path}")

    print("\n" + "=" * 70)
    print("GP TERRAIN EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
