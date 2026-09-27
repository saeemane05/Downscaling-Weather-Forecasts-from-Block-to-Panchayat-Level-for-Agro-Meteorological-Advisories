"""
SIH26 - GP DUPLICATE GEOMETRY DIAGNOSTIC

Purpose
-------
Investigate GP IDs that occur more than once in an official Gram Manchitra
raw GP boundary snapshot.

This script DOES NOT modify or delete the raw boundary data.
It reports:
  - duplicate GP IDs
  - geometry type
  - area in km²
  - centroid
  - bounds
  - geometry equality
  - intersection / overlap
  - containment
  - WKB signatures

Usage
-----
Run from the SIH26_Downscaling project:

    python scripts\\gp\\02_gp_duplicate_diagnostic.py

The script asks for State, District and Block, then automatically selects
the newest raw Gram Manchitra GP GeoJSON snapshot in that block's raw folder.
"""

from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------

print("=" * 70)
print("SIH26 - GP DUPLICATE GEOMETRY DIAGNOSTIC")
print("=" * 70)

state = input("Enter State   : ").strip()
district = input("Enter District: ").strip()
block = input("Enter Block    : ").strip()

state_dir = state.replace(" ", "_")
district_dir = district.replace(" ", "_")
block_dir = block.replace(" ", "_")

block_root = (
    PROJECT_ROOT
    / "gp"
    / state_dir
    / district_dir
    / block_dir
)

raw_boundary_dir = block_root / "raw" / "boundaries"
processed_dir = block_root / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

print("\nProject directory:")
print(f"  {block_root}")

print("\n[1/4] Looking for raw Gram Manchitra snapshots...")

snapshots = sorted(
    raw_boundary_dir.glob("gram_manchitra_gp_*.geojson"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not snapshots:
    raise FileNotFoundError(
        f"No raw Gram Manchitra GP snapshots found in:\n{raw_boundary_dir}"
    )

raw_file = snapshots[0]

print(f"  Snapshots found: {len(snapshots)}")
print(f"  Using newest   : {raw_file.name}")


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

print("\n[2/4] Loading raw official GP boundary snapshot...")

gdf = gpd.read_file(raw_file)

if gdf.empty:
    raise RuntimeError("The raw GP boundary file contains no records.")

print(f"  Records loaded: {len(gdf)}")
print(f"  CRS           : {gdf.crs}")

# Identify GP ID / name columns robustly.
def find_column(columns, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    # relaxed matching
    for c in columns:
        lc = str(c).strip().lower().replace(" ", "_")
        for candidate in candidates:
            cc = candidate.strip().lower().replace(" ", "_")
            if cc in lc or lc in cc:
                return c

    return None


gp_id_col = find_column(
    gdf.columns,
    [
        "gp_id",
        "gpcode",
        "gp_code",
        "gp_lgd",
        "gp_lgd_code",
        "gplgdcode",
    ],
)

gp_name_col = find_column(
    gdf.columns,
    [
        "gp_name",
        "gpname",
        "gram_panchayat_name",
        "panchayat_name",
    ],
)

if gp_id_col is None:
    raise RuntimeError(
        "Could not identify the GP ID column. "
        f"Available columns: {list(gdf.columns)}"
    )

if gp_name_col is None:
    raise RuntimeError(
        "Could not identify the GP name column. "
        f"Available columns: {list(gdf.columns)}"
    )

gdf["_diag_gp_id"] = (
    gdf[gp_id_col]
    .astype(str)
    .str.strip()
)

gdf["_diag_gp_name"] = (
    gdf[gp_name_col]
    .astype(str)
    .str.strip()
)

duplicate_counts = (
    gdf["_diag_gp_id"]
    .value_counts()
)

duplicate_ids = duplicate_counts[
    duplicate_counts > 1
].index.tolist()

print(f"  GP ID column  : {gp_id_col}")
print(f"  GP name column: {gp_name_col}")
print(f"  Duplicate IDs : {len(duplicate_ids)}")

if not duplicate_ids:
    print("\nNo duplicate GP IDs were found.")
    print("The raw snapshot is already unique by GP ID.")
    raise SystemExit(0)


# ---------------------------------------------------------------------
# METRIC CRS
# ---------------------------------------------------------------------

print("\n[3/4] Analysing duplicate geometries...")

if gdf.crs is None:
    raise RuntimeError("Raw GP layer has no CRS.")

# Reproject to a suitable local projected CRS for area/distance metrics.
centroid = gdf.to_crs(4326).geometry.union_all().centroid

lon = float(centroid.x)
lat = float(centroid.y)

utm_zone = int((lon + 180) // 6) + 1
utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone

metric_gdf = gdf.to_crs(epsg=utm_epsg)


def geometry_signature(geom):
    if geom is None or geom.is_empty:
        return ""

    try:
        normalized = geom.normalize()
        return hashlib.sha256(normalized.wkb).hexdigest()
    except Exception:
        return hashlib.sha256(geom.wkb).hexdigest()


def safe_overlap_area(a, b):
    try:
        return float(a.intersection(b).area)
    except Exception:
        return float("nan")


def safe_union_area(a, b):
    try:
        return float(a.union(b).area)
    except Exception:
        return float("nan")


details = []
summary = []

for gp_id in duplicate_ids:

    idxs = list(
        gdf.index[gdf["_diag_gp_id"] == gp_id]
    )

    # Sort deterministically so record numbering is reproducible.
    idxs = sorted(idxs, key=lambda x: str(x))

    signatures = []
    names = []

    for record_no, idx in enumerate(idxs, start=1):

        geom = gdf.loc[idx, "geometry"]
        metric_geom = metric_gdf.loc[idx, "geometry"]
        name = gdf.loc[idx, "_diag_gp_name"]

        sig = geometry_signature(geom)

        signatures.append(sig)
        names.append(name)

        if geom is None or geom.is_empty:
            geom_type = "EMPTY"
            valid = False
            area_km2 = float("nan")
            centroid_lat = float("nan")
            centroid_lon = float("nan")
            minx = miny = maxx = maxy = float("nan")
        else:
            geom_type = geom.geom_type
            valid = bool(geom.is_valid)
            area_km2 = float(metric_geom.area / 1_000_000.0)

            centroid_wgs = gpd.GeoSeries(
                [metric_geom],
                crs=f"EPSG:{utm_epsg}",
            ).to_crs(4326).iloc[0].centroid

            centroid_lat = float(centroid_wgs.y)
            centroid_lon = float(centroid_wgs.x)

            bounds = geom.bounds
            minx, miny, maxx, maxy = map(float, bounds)

        details.append(
            {
                "gp_id": gp_id,
                "gp_name": name,
                "record_no": record_no,
                "source_index": idx,
                "geometry_type": geom_type,
                "geometry_valid": valid,
                "area_km2": area_km2,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "minx": minx,
                "miny": miny,
                "maxx": maxx,
                "maxy": maxy,
                "geometry_signature": sig,
            }
        )

    # Pairwise spatial diagnostics.
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):

            idx_a = idxs[i]
            idx_b = idxs[j]

            a = gdf.loc[idx_a, "geometry"]
            b = gdf.loc[idx_b, "geometry"]

            if a is None or b is None or a.is_empty or b.is_empty:
                equals = False
                intersects = False
                contains_ab = False
                contains_ba = False
                overlap_area_km2 = float("nan")
                overlap_pct_a = float("nan")
                overlap_pct_b = float("nan")
            else:
                equals = bool(a.equals(b))
                intersects = bool(a.intersects(b))
                contains_ab = bool(a.contains(b))
                contains_ba = bool(b.contains(a))

                overlap_area = safe_overlap_area(
                    metric_gdf.loc[idx_a, "geometry"],
                    metric_gdf.loc[idx_b, "geometry"],
                )

                area_a = float(
                    metric_gdf.loc[idx_a, "geometry"].area
                )
                area_b = float(
                    metric_gdf.loc[idx_b, "geometry"].area
                )

                overlap_area_km2 = overlap_area / 1_000_000.0

                overlap_pct_a = (
                    100.0 * overlap_area / area_a
                    if area_a > 0 else float("nan")
                )

                overlap_pct_b = (
                    100.0 * overlap_area / area_b
                    if area_b > 0 else float("nan")
                )

            spatial_relationship = "DISJOINT"

            if equals:
                spatial_relationship = "IDENTICAL"
            elif contains_ab:
                spatial_relationship = "A_CONTAINS_B"
            elif contains_ba:
                spatial_relationship = "B_CONTAINS_A"
            elif intersects:
                spatial_relationship = "OVERLAPPING"

            summary.append(
                {
                    "gp_id": gp_id,
                    "record_a": i + 1,
                    "record_b": j + 1,
                    "name_a": names[i],
                    "name_b": names[j],
                    "geometry_equal": equals,
                    "intersects": intersects,
                    "a_contains_b": contains_ab,
                    "b_contains_a": contains_ba,
                    "spatial_relationship": spatial_relationship,
                    "overlap_area_km2": overlap_area_km2,
                    "overlap_pct_of_a": overlap_pct_a,
                    "overlap_pct_of_b": overlap_pct_b,
                }
            )


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

details_df = pd.DataFrame(details)
summary_df = pd.DataFrame(summary)

details_file = processed_dir / "duplicate_gp_geometry_details.csv"
summary_file = processed_dir / "duplicate_gp_spatial_relationships.csv"

details_df.to_csv(details_file, index=False)
summary_df.to_csv(summary_file, index=False)


# ---------------------------------------------------------------------
# CREATE A MAP OF CONFLICTING RECORDS
# ---------------------------------------------------------------------

conflict_indices = []

for gp_id in duplicate_ids:
    conflict_indices.extend(
        gdf.index[gdf["_diag_gp_id"] == gp_id].tolist()
    )

conflict_gdf = gdf.loc[conflict_indices].copy()

# Remove diagnostic columns from map attributes.
map_drop = [
    c for c in ["_diag_gp_id", "_diag_gp_name"]
    if c in conflict_gdf.columns
]
conflict_gdf = conflict_gdf.drop(columns=map_drop)

geojson_file = processed_dir / "duplicate_gp_conflicts.geojson"
conflict_gdf.to_file(
    geojson_file,
    driver="GeoJSON",
)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "district": district,
    "block": block,
    "raw_snapshot": str(raw_file),
    "records_in_snapshot": int(len(gdf)),
    "duplicate_gp_ids": int(len(duplicate_ids)),
    "duplicate_gp_id_values": [str(x) for x in duplicate_ids],
    "gp_id_column": str(gp_id_col),
    "gp_name_column": str(gp_name_col),
    "metric_epsg": int(utm_epsg),
    "outputs": {
        "geometry_details": str(details_file),
        "spatial_relationships": str(summary_file),
        "conflicting_geometries": str(geojson_file),
    },
}

report_file = processed_dir / "duplicate_gp_diagnostic_metadata.json"

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)


# ---------------------------------------------------------------------
# CONSOLE SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE GEOMETRY DIAGNOSTIC COMPLETE")
print("=" * 70)

print(f"\nBlock:")
print(f"  {state} / {district} / {block}")

print("\nRaw snapshot:")
print(f"  {raw_file}")

print("\nDuplicate GP IDs:")
for gp_id in duplicate_ids:
    count = int(duplicate_counts.loc[gp_id])
    print(f"  {gp_id}: {count} records")

print("\nDetailed geometry records:")
print(details_df.to_string(index=False))

print("\nSpatial relationships:")
print(summary_df.to_string(index=False))

print("\nOutputs:")
print(f"  Geometry details       : {details_file}")
print(f"  Spatial relationships  : {summary_file}")
print(f"  Conflict GeoJSON       : {geojson_file}")
print(f"  Diagnostic metadata    : {report_file}")

print("\nIMPORTANT:")
print("  No duplicate record was deleted.")
print("  No geometry was merged.")
print("  No geometry was selected as 'correct'.")
print("  Raw official data remains untouched.")

print("\nNext step:")
print("  Inspect the spatial relationships before defining the")
print("  universal duplicate-GP handling rule.")
print("=" * 70)
