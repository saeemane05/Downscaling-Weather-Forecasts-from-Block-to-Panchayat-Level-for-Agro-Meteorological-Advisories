"""
SIH26 - GP DUPLICATE / BLOCK CONSISTENCY DIAGNOSTIC

Purpose:
    Investigate duplicate GP IDs returned by official Gram Manchitra.

Checks:
    1. Whether duplicate GP geometry parts are inside the official Block.
    2. Whether duplicate parts overlap each other.
    3. Whether duplicate parts overlap OTHER GP IDs.
    4. What a dissolve-by-GP-ID would produce.

Scientific safeguards:
    - Raw GP data is never modified.
    - No duplicate is deleted.
    - No geometry is selected as "correct".
    - Dissolve is simulated only.
"""

from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
import requests
from shapely.ops import unary_union


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------
# OFFICIAL GRAM MANCHITRA BLOCK LAYER
# ---------------------------------------------------------------------

BLOCK_QUERY_URL = (
    "https://grammanchitragis.nic.in/grammanchitra/rest/services/"
    "panchayat/adminpanch/MapServer/2/query"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def norm(value):
    """Normalize administrative names for comparison."""
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .upper()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(columns, candidates):
    """Find a column using exact then relaxed normalized matching."""
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    for c in columns:
        lc = str(c).strip().lower().replace(" ", "_")
        for candidate in candidates:
            cc = candidate.strip().lower().replace(" ", "_")
            if cc in lc or lc in cc:
                return c

    return None


def geometry_signature(geom):
    """Stable signature used only for diagnostics."""
    if geom is None or geom.is_empty:
        return ""
    try:
        return hashlib.sha256(geom.normalize().wkb).hexdigest()
    except Exception:
        return hashlib.sha256(geom.wkb).hexdigest()


def safe_intersection_area(a, b):
    try:
        return float(a.intersection(b).area)
    except Exception:
        return float("nan")


def get_utm_epsg(gdf):
    """Choose a local UTM CRS from the dataset centroid."""
    wgs = gdf.to_crs(4326)

    valid = [
        g for g in wgs.geometry
        if g is not None and not g.is_empty
    ]

    if not valid:
        raise RuntimeError("No valid geometry available for UTM selection.")

    centroid = unary_union(valid).centroid
    lon = float(centroid.x)
    lat = float(centroid.y)

    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def fetch_block_records():
    """Retrieve official Block Panchayat records in pages."""
    params = {
        "where": "1=1",
        "outFields": (
            "block_name,state,district,block_lgd,blkcode11,"
            "dtcode11,stcode11"
        ),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 1000,
    }

    records = []
    offset = 0

    while True:
        params["resultOffset"] = offset

        response = requests.get(
            BLOCK_QUERY_URL,
            params=params,
            timeout=120,
            verify=False,
        )
        response.raise_for_status()

        data = response.json()

        features = data.get("features", [])

        if not features:
            break

        records.extend(features)
        print(f"  Retrieved block records: {len(records)}")

        if len(features) < 1000:
            break

        offset += 1000

    return records


def resolve_block(records, state, district, block):
    """Resolve selected Block locally from official returned records."""
    matched = []

    for feature in records:
        props = feature.get("properties", {})

        if (
            norm(props.get("state")) == norm(state)
            and norm(props.get("district")) == norm(district)
            and norm(props.get("block_name")) == norm(block)
        ):
            matched.append(feature)

    if not matched:
        raise RuntimeError(
            f"Could not resolve Block: {state} / {district} / {block}"
        )

    return gpd.GeoDataFrame.from_features(
        matched,
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------

print("=" * 70)
print("SIH26 - GP DUPLICATE / BLOCK CONSISTENCY DIAGNOSTIC")
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


# ---------------------------------------------------------------------
# 1. LOAD GP SNAPSHOT
# ---------------------------------------------------------------------

print("\n[1/6] Loading newest official GP snapshot...")

snapshots = sorted(
    raw_boundary_dir.glob("gram_manchitra_gp_*.geojson"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not snapshots:
    raise FileNotFoundError(
        f"No raw GP snapshot found in:\n{raw_boundary_dir}"
    )

gp_file = snapshots[0]
gp = gpd.read_file(gp_file)

if gp.empty:
    raise RuntimeError("GP snapshot is empty.")

if gp.crs is None:
    raise RuntimeError("GP snapshot has no CRS.")

gp_id_col = find_column(
    gp.columns,
    ["gp_id", "gpcode", "gp_code", "gp_lgd", "gp_lgd_code", "gplgdcode"],
)

gp_name_col = find_column(
    gp.columns,
    ["gp_name", "gpname", "gram_panchayat_name", "panchayat_name"],
)

if gp_id_col is None or gp_name_col is None:
    raise RuntimeError(
        "Could not identify GP ID/name columns.\n"
        f"Columns: {list(gp.columns)}"
    )

gp["_gp_id_diag"] = gp[gp_id_col].astype(str).str.strip()
gp["_gp_name_diag"] = gp[gp_name_col].astype(str).str.strip()

counts = gp["_gp_id_diag"].value_counts()
duplicate_ids = counts[counts > 1].index.tolist()

print(f"  Snapshot: {gp_file.name}")
print(f"  Records: {len(gp)}")
print(f"  GP ID column: {gp_id_col}")
print(f"  GP name column: {gp_name_col}")
print(f"  Duplicate GP IDs: {len(duplicate_ids)}")

if not duplicate_ids:
    print("\nNo duplicate GP IDs found.")
    raise SystemExit(0)


# ---------------------------------------------------------------------
# 2. RETRIEVE OFFICIAL BLOCK
# ---------------------------------------------------------------------

print("\n[2/6] Retrieving official Block boundary from Gram Manchitra...")
print("  Source: official Gram Manchitra Block Panchayat layer")

records = fetch_block_records()

print(f"  Total Block records retrieved: {len(records)}")

block_gdf = resolve_block(
    records,
    state,
    district,
    block,
)

print(f"  Matching Block records: {len(block_gdf)}")

block_lgd_values = []
if "block_lgd" in block_gdf.columns:
    block_lgd_values = sorted(
        set(str(x) for x in block_gdf["block_lgd"].dropna())
    )

print(f"  Block LGD values: {block_lgd_values}")

block_union_wgs84 = unary_union(
    [
        g for g in block_gdf.geometry
        if g is not None and not g.is_empty
    ]
)

if block_union_wgs84.is_empty:
    raise RuntimeError("Resolved Block geometry is empty.")

block_snapshot = (
    raw_boundary_dir
    / (
        "gram_manchitra_block_"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.geojson"
    )
)

block_gdf.to_file(block_snapshot, driver="GeoJSON")

print(f"  Block snapshot saved: {block_snapshot}")


# ---------------------------------------------------------------------
# 3. PREPARE METRIC CRS
# ---------------------------------------------------------------------

print("\n[3/6] Preparing spatial analysis...")

gp_wgs84 = gp.to_crs(4326)

block_union_gdf = gpd.GeoDataFrame(
    {"geometry": [block_union_wgs84]},
    crs="EPSG:4326",
)

utm_epsg = get_utm_epsg(block_union_gdf)

gp_m = gp_wgs84.to_crs(utm_epsg)
block_union_m = (
    block_union_gdf.to_crs(utm_epsg)
    .geometry
    .iloc[0]
)

print(f"  Metric CRS: EPSG:{utm_epsg}")


# ---------------------------------------------------------------------
# 4. BLOCK CONTAINMENT + SAME-ID RELATIONSHIPS
# ---------------------------------------------------------------------

print("\n[4/6] Checking duplicate parts against Block boundary...")

part_rows = []
same_id_rows = []

for gp_id in duplicate_ids:

    indices = list(
        gp_m.index[
            gp_m["_gp_id_diag"] == gp_id
        ]
    )

    for record_no, idx in enumerate(indices, start=1):

        geom = gp_m.loc[idx, "geometry"]

        if geom is None or geom.is_empty:
            continue

        area_m2 = float(geom.area)

        inside_m2 = safe_intersection_area(
            geom,
            block_union_m,
        )

        pct_inside = (
            100.0 * inside_m2 / area_m2
            if area_m2 > 0 and inside_m2 == inside_m2
            else float("nan")
        )

        part_rows.append(
            {
                "gp_id": gp_id,
                "gp_name": gp_m.loc[idx, "_gp_name_diag"],
                "record_no": record_no,
                "source_index": idx,
                "geometry_type": geom.geom_type,
                "geometry_valid": bool(geom.is_valid),
                "area_km2": area_m2 / 1_000_000.0,
                "inside_block_area_km2": (
                    inside_m2 / 1_000_000.0
                    if inside_m2 == inside_m2
                    else float("nan")
                ),
                "pct_inside_block": pct_inside,
                "fully_within_block": bool(
                    geom.within(block_union_m)
                ),
                "intersects_block": bool(
                    geom.intersects(block_union_m)
                ),
                "geometry_signature": geometry_signature(geom),
            }
        )

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):

            a = gp_m.loc[indices[i], "geometry"]
            b = gp_m.loc[indices[j], "geometry"]

            overlap = safe_intersection_area(a, b)

            if a.equals(b):
                relation = "IDENTICAL"
            elif a.contains(b):
                relation = "A_CONTAINS_B"
            elif b.contains(a):
                relation = "B_CONTAINS_A"
            elif a.intersects(b):
                relation = "OVERLAPPING"
            else:
                relation = "DISJOINT"

            same_id_rows.append(
                {
                    "gp_id": gp_id,
                    "record_a": i + 1,
                    "record_b": j + 1,
                    "relation": relation,
                    "overlap_area_km2": (
                        overlap / 1_000_000.0
                        if overlap == overlap
                        else float("nan")
                    ),
                }
            )


# ---------------------------------------------------------------------
# 5. CROSS-GP OVERLAP
# ---------------------------------------------------------------------

print("\n[5/6] Checking duplicate parts against OTHER GPs...")

cross_rows = []

for gp_id in duplicate_ids:

    indices = list(
        gp_m.index[
            gp_m["_gp_id_diag"] == gp_id
        ]
    )

    for idx in indices:

        geom = gp_m.loc[idx, "geometry"]

        if geom is None or geom.is_empty:
            continue

        candidate_indices = list(
            gp_m.sindex.query(
                geom,
                predicate="intersects",
            )
        )

        for other_idx in candidate_indices:

            if other_idx == idx:
                continue

            other_id = gp_m.loc[
                other_idx,
                "_gp_id_diag",
            ]

            if other_id == gp_id:
                continue

            other_geom = gp_m.loc[
                other_idx,
                "geometry",
            ]

            overlap = safe_intersection_area(
                geom,
                other_geom,
            )

            if overlap != overlap or overlap <= 0:
                continue

            area_a = float(geom.area)
            area_b = float(other_geom.area)

            cross_rows.append(
                {
                    "gp_id": gp_id,
                    "gp_name": gp_m.loc[
                        idx,
                        "_gp_name_diag",
                    ],
                    "source_index": idx,
                    "other_gp_id": other_id,
                    "other_gp_name": gp_m.loc[
                        other_idx,
                        "_gp_name_diag",
                    ],
                    "overlap_area_km2": (
                        overlap / 1_000_000.0
                    ),
                    "overlap_pct_of_duplicate_part": (
                        100.0 * overlap / area_a
                        if area_a > 0 else float("nan")
                    ),
                    "overlap_pct_of_other_gp": (
                        100.0 * overlap / area_b
                        if area_b > 0 else float("nan")
                    ),
                }
            )


# ---------------------------------------------------------------------
# 6. DISSOLVE SIMULATION
# ---------------------------------------------------------------------

print("\n[6/6] Simulating dissolve by GP ID...")

dissolve_rows = []

for gp_id in duplicate_ids:

    indices = list(
        gp_m.index[
            gp_m["_gp_id_diag"] == gp_id
        ]
    )

    geoms = [
        gp_m.loc[idx, "geometry"]
        for idx in indices
        if (
            gp_m.loc[idx, "geometry"] is not None
            and not gp_m.loc[idx, "geometry"].is_empty
        )
    ]

    if not geoms:
        continue

    dissolved = unary_union(geoms)

    if dissolved.geom_type == "Polygon":
        component_count = 1
    elif dissolved.geom_type == "MultiPolygon":
        component_count = len(dissolved.geoms)
    else:
        component_count = None

    dissolve_rows.append(
        {
            "gp_id": gp_id,
            "gp_name": gp_m.loc[
                indices[0],
                "_gp_name_diag",
            ],
            "source_feature_count": len(geoms),
            "dissolved_geometry_type": dissolved.geom_type,
            "dissolved_polygon_parts": component_count,
            "dissolved_area_km2": (
                dissolved.area / 1_000_000.0
            ),
            "dissolved_within_block": bool(
                dissolved.within(block_union_m)
            ),
            "dissolved_intersects_block": bool(
                dissolved.intersects(block_union_m)
            ),
        }
    )


# ---------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------

part_df = pd.DataFrame(part_rows)
same_id_df = pd.DataFrame(same_id_rows)
cross_df = pd.DataFrame(cross_rows)
dissolve_df = pd.DataFrame(dissolve_rows)

part_file = processed_dir / "gp_duplicate_block_check.csv"
same_id_file = (
    processed_dir / "gp_duplicate_same_id_relationships.csv"
)
cross_file = (
    processed_dir / "gp_duplicate_cross_overlap.csv"
)
dissolve_file = (
    processed_dir / "gp_duplicate_dissolve_simulation.csv"
)

part_df.to_csv(part_file, index=False)
same_id_df.to_csv(same_id_file, index=False)
cross_df.to_csv(cross_file, index=False)
dissolve_df.to_csv(dissolve_file, index=False)

conflict_indices = []

for gp_id in duplicate_ids:
    conflict_indices.extend(
        gp.index[
            gp["_gp_id_diag"] == gp_id
        ].tolist()
    )

conflict_gdf = gp.loc[conflict_indices].copy()

for col in ["_gp_id_diag", "_gp_name_diag"]:
    if col in conflict_gdf.columns:
        conflict_gdf = conflict_gdf.drop(columns=[col])

diagnostic_geojson = (
    processed_dir / "gp_duplicate_block_diagnostic.geojson"
)

conflict_gdf.to_file(
    diagnostic_geojson,
    driver="GeoJSON",
)

metadata = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "district": district,
    "block": block,
    "gp_snapshot": str(gp_file),
    "block_snapshot": str(block_snapshot),
    "block_source": BLOCK_QUERY_URL,
    "gp_records": int(len(gp)),
    "duplicate_gp_ids": [str(x) for x in duplicate_ids],
    "duplicate_gp_count": len(duplicate_ids),
    "metric_epsg": int(utm_epsg),
    "outputs": {
        "block_check": str(part_file),
        "same_id_relationships": str(same_id_file),
        "cross_gp_overlap": str(cross_file),
        "dissolve_simulation": str(dissolve_file),
        "diagnostic_geojson": str(diagnostic_geojson),
    },
}

metadata_file = (
    processed_dir / "gp_duplicate_block_metadata.json"
)

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("GP DUPLICATE / BLOCK CONSISTENCY DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\nBlock containment:")
if part_df.empty:
    print("  None")
else:
    print(
        part_df[
            [
                "gp_id",
                "gp_name",
                "record_no",
                "area_km2",
                "pct_inside_block",
                "fully_within_block",
            ]
        ].to_string(index=False)
    )

print("\nSame-ID spatial relationships:")
if same_id_df.empty:
    print("  None")
else:
    print(same_id_df.to_string(index=False))

print("\nOverlap with OTHER GP IDs:")
if cross_df.empty:
    print("  No positive-area overlap detected.")
else:
    print(cross_df.to_string(index=False))

print("\nDissolve simulation:")
if dissolve_df.empty:
    print("  None")
else:
    print(dissolve_df.to_string(index=False))

print("\nOutputs:")
print(f"  {part_file}")
print(f"  {same_id_file}")
print(f"  {cross_file}")
print(f"  {dissolve_file}")
print(f"  {diagnostic_geojson}")
print(f"  {metadata_file}")

print("\nIMPORTANT:")
print("  Raw GP data was not modified.")
print("  No duplicate was deleted.")
print("  No geometry was selected as correct.")
print("  Dissolve was simulated only.")

print("=" * 70)
