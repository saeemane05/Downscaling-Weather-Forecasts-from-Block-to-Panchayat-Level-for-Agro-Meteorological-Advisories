"""
SIH26 Weather Downscaling
=========================

01 - Universal Block Boundary Collector

Input:
    State
    District
    Block

Process:
    1. Resolve Block LGD code.
    2. Retrieve all GP polygons belonging to the Block.
    3. Preserve those GP polygons as raw Block-member geometry.
    4. Dissolve GP polygons into a Block boundary.
    5. Calculate Block area and centroid.
    6. Save provenance metadata.

No synthetic geometry is created.
"""

from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd
import requests
import urllib3

# ------------------------------------------------------------
# Allow imports from project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from scripts.common.geo import (
    GRAM_MANCHITRA_GP_QUERY_URL,
    clean_name,
    normalize_text,
    safe_get_json,
    save_json,
    validate_geometry,
    repair_invalid_geometry,
    dissolve_to_single_polygon,
    add_polygon_metrics,
    get_bounds,
)


urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# BLOCK RESOLUTION
# ============================================================

def resolve_block(
    state,
    district,
    block
):
    """
    Resolve State + District + Block to a unique
    Block LGD code.
    """

    params = {

        "where": (
            "UPPER(blkname) LIKE "
            f"'%{normalize_text(block)}%'"
        ),

        "outFields": (
            "blkname,"
            "blklgdcode,"
            "blk_lgdcod,"
            "dtname,"
            "dtcode11,"
            "stname"
        ),

        "returnGeometry": "false",

        "f": "json"
    }

    data = safe_get_json(
        GRAM_MANCHITRA_GP_QUERY_URL,
        params
    )

    features = data.get(
        "features",
        []
    )

    if not features:

        raise RuntimeError(
            f"No Block found matching "
            f"'{block}'."
        )

    records = []

    for feature in features:

        attrs = feature.get(
            "attributes",
            {}
        )

        records.append({

            "block_name":
                attrs.get("blkname"),

            "block_lgd_code":
                attrs.get("blklgdcode"),

            "block_lgd_code_alt":
                attrs.get("blk_lgdcod"),

            "district_name":
                attrs.get("dtname"),

            "district_code":
                attrs.get("dtcode11"),

            "state_name":
                attrs.get("stname")
        })

    df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Remove duplicate representations of same LGD block.
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["block_lgd_code"]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # State filter
    # --------------------------------------------------------

    if state:

        mask = (
            df["state_name"]
            .fillna("")
            .str.upper()
            .str.contains(
                normalize_text(state),
                regex=False
            )
        )

        filtered = df[mask]

        if not filtered.empty:

            df = filtered

    # --------------------------------------------------------
    # District filter
    # --------------------------------------------------------

    if district:

        mask = (
            df["district_name"]
            .fillna("")
            .str.upper()
            .str.contains(
                normalize_text(district),
                regex=False
            )
        )

        filtered = df[mask]

        if not filtered.empty:

            df = filtered

    # --------------------------------------------------------
    # Exact Block match
    # --------------------------------------------------------

    exact = df[
        df["block_name"]
        .fillna("")
        .str.upper()
        .eq(
            normalize_text(block)
        )
    ]

    if len(exact) == 1:

        return exact.iloc[0].to_dict()

    # --------------------------------------------------------
    # Multiple matches
    # --------------------------------------------------------

    if len(df) > 1:

        print(
            "\nMultiple Blocks matched:\n"
        )

        for i, row in df.iterrows():

            print(
                f"[{i + 1}] "
                f"{row['block_name']} | "
                f"LGD={row['block_lgd_code']} | "
                f"District={row['district_name']} | "
                f"State={row['state_name']}"
            )

        while True:

            choice = input(
                "\nSelect correct Block: "
            ).strip()

            try:

                index = (
                    int(choice) - 1
                )

                return df.iloc[
                    index
                ].to_dict()

            except (
                ValueError,
                IndexError
            ):

                print(
                    "Invalid selection. "
                    "Try again."
                )

    return df.iloc[0].to_dict()


# ============================================================
# DOWNLOAD GP POLYGONS
# ============================================================

def download_gp_polygons(
    block_lgd_code
):
    """
    Retrieve GP polygons belonging to the Block.
    """

    params = {

        "where": (
            f"blklgdcode = '{block_lgd_code}' "
            "AND gp_code <> ''"
        ),

        "outFields": (
            "gp_code,"
            "gp_name,"
            "blklgdcode,"
            "blk_lgdcod,"
            "blkname,"
            "dtname,"
            "stname"
        ),

        "returnGeometry": "true",

        "outSR": "4326",

        "f": "geojson"
    }

    response = requests.get(
        GRAM_MANCHITRA_GP_QUERY_URL,
        params=params,
        timeout=120,
        verify=False
    )

    response.raise_for_status()

    data = response.json()

    features = data.get(
        "features",
        []
    )

    if not features:

        raise RuntimeError(
            "The Block returned zero GP "
            "polygon features."
        )

    gdf = (
        gpd.GeoDataFrame
        .from_features(
            features,
            crs="EPSG:4326"
        )
    )

    return gdf


# ============================================================
# SAVE
# ============================================================

def save_block_outputs(
    state,
    district,
    block,
    block_info,
    gp_gdf,
    block_gdf
):

    base = (
        Path("block")
        / clean_name(state)
        / clean_name(district)
        / clean_name(block)
    )

    raw_dir = (
        base /
        "raw" /
        "boundary"
    )

    processed_dir = (
        base /
        "processed"
    )

    metadata_dir = (
        base /
        "metadata"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # RAW GP polygons used to construct Block
    # --------------------------------------------------------

    gp_path = (
        raw_dir /
        "block_member_gp_boundaries.geojson"
    )

    gp_gdf.to_file(
        gp_path,
        driver="GeoJSON"
    )

    # --------------------------------------------------------
    # Processed Block boundary
    # --------------------------------------------------------

    block_path = (
        processed_dir /
        "block_boundary.geojson"
    )

    block_gdf.to_file(
        block_path,
        driver="GeoJSON"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    bounds = get_bounds(
        block_gdf
    )

    metadata = {

        "project":
            "SIH26 Weather Downscaling",

        "state":
            state,

        "district":
            district,

        "block_requested":
            block,

        "resolved_block_name":
            block_info.get(
                "block_name"
            ),

        "block_lgd_code":
            str(
                block_info.get(
                    "block_lgd_code"
                )
            ),

        "district_name":
            block_info.get(
                "district_name"
            ),

        "district_code":
            block_info.get(
                "district_code"
            ),

        "state_name":
            block_info.get(
                "state_name"
            ),

        "number_of_gp_polygons":
            int(len(gp_gdf)),

        "area_km2":
            float(
                block_gdf[
                    "area_km2"
                ].iloc[0]
            ),

        "centroid_lat":
            float(
                block_gdf[
                    "centroid_lat"
                ].iloc[0]
            ),

        "centroid_lon":
            float(
                block_gdf[
                    "centroid_lon"
                ].iloc[0]
            ),

        "bounds":
            bounds,

        "crs":
            "EPSG:4326",

        "source":
            "Gram Manchitra",

        "source_url":
            GRAM_MANCHITRA_GP_QUERY_URL,

        "construction":
            "Block geometry created by "
            "dissolving GP polygons belonging "
            "to the resolved Block LGD code.",

        "synthetic_values_used":
            False
    }

    metadata_path = (
        metadata_dir /
        "block_metadata.json"
    )

    save_json(
        metadata,
        metadata_path
    )

    return base


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SIH26 — UNIVERSAL BLOCK "
        "BOUNDARY COLLECTOR"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    state = input(
        "\nState: "
    ).strip()

    district = input(
        "District: "
    ).strip()

    block = input(
        "Block: "
    ).strip()

    if not state:

        raise ValueError(
            "State cannot be empty."
        )

    if not district:

        raise ValueError(
            "District cannot be empty."
        )

    if not block:

        raise ValueError(
            "Block cannot be empty."
        )

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    print(
        "\n[1/4] Resolving Block..."
    )

    block_info = resolve_block(
        state,
        district,
        block
    )

    print(
        f"    State   : "
        f"{block_info['state_name']}"
    )

    print(
        f"    District: "
        f"{block_info['district_name']}"
    )

    print(
        f"    Block   : "
        f"{block_info['block_name']}"
    )

    print(
        f"    LGD Code: "
        f"{block_info['block_lgd_code']}"
    )

    # --------------------------------------------------------
    # DOWNLOAD GP POLYGONS
    # --------------------------------------------------------

    print(
        "\n[2/4] Downloading GP polygons..."
    )

    gp_gdf = download_gp_polygons(
        block_info[
            "block_lgd_code"
        ]
    )

    print(
        f"    GP polygons received: "
        f"{len(gp_gdf)}"
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print(
        "\n[3/4] Validating geometry..."
    )

    validate_geometry(
        gp_gdf,
        require_polygon=True
    )

    gp_gdf = repair_invalid_geometry(
        gp_gdf
    )

    # Add GP metrics as useful provenance
    gp_gdf = add_polygon_metrics(
        gp_gdf
    )

    unique_gp = (
        gp_gdf[
            "gp_code"
        ]
        .dropna()
        .astype(str)
        .nunique()
    )

    print(
        f"    Unique GP IDs: "
        f"{unique_gp}"
    )

    if unique_gp != len(gp_gdf):

        print(
            "    WARNING: duplicate GP IDs "
            "were returned."
        )

    # --------------------------------------------------------
    # DISSOLVE
    # --------------------------------------------------------

    print(
        "\n    Creating Block boundary..."
    )

    block_gdf = dissolve_to_single_polygon(
        gp_gdf
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print(
        "\n[4/4] Saving outputs..."
    )

    output_dir = save_block_outputs(
        state=state,
        district=district,
        block=block,
        block_info=block_info,
        gp_gdf=gp_gdf,
        block_gdf=block_gdf
    )

    print()
    print("=" * 70)
    print("BLOCK BOUNDARY COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Block       : "
        f"{block_info['block_name']}"
    )

    print(
        f"Block LGD   : "
        f"{block_info['block_lgd_code']}"
    )

    print(
        f"GP polygons : "
        f"{len(gp_gdf)}"
    )

    print(
        f"Area        : "
        f"{block_gdf['area_km2'].iloc[0]:.3f} km²"
    )

    print(
        f"Centroid    : "
        f"{block_gdf['centroid_lat'].iloc[0]:.6f}, "
        f"{block_gdf['centroid_lon'].iloc[0]:.6f}"
    )

    print(
        f"\nSaved under:\n"
        f"{output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nProcess cancelled."
        )

        sys.exit(1)

    except Exception as exc:

        print(
            "\nERROR:"
        )

        print(exc)

        sys.exit(1)