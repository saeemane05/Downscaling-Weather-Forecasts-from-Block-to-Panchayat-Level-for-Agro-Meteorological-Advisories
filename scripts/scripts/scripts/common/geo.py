"""
SIH26 Weather Downscaling
Common Geographic Utilities

These functions are shared by Block and GP pipelines.

Important principles:
- No hard-coded State/District/Block.
- No synthetic spatial values.
- CRS is always explicit.
- Metric calculations use a dynamically selected UTM CRS.
"""

from pathlib import Path
import json
import re

import geopandas as gpd
import pandas as pd
import requests


# ============================================================
# OFFICIAL GRAM MANCHITRA SERVICE
# ============================================================

GRAM_MANCHITRA_GP_QUERY_URL = (
    "https://grammanchitragis.nic.in/"
    "grammanchitra/rest/services/"
    "panchayat/panchayat_admin/"
    "MapServer/3/query"
)


# ============================================================
# TEXT / FILE UTILITIES
# ============================================================

def clean_name(value: str) -> str:
    """
    Make a string safe for use as a Windows/Linux folder name.
    """

    value = str(value).strip()

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value
    )

    value = re.sub(
        r"\s+",
        "_",
        value
    )

    return value


def normalize_text(value) -> str:
    """
    Normalize administrative names for comparison.
    """

    if value is None:
        return ""

    return str(value).strip().upper()


# ============================================================
# HTTP
# ============================================================

def safe_get_json(
    url,
    params,
    timeout=120,
    verify=False
):
    """
    Request JSON from an API and fail explicitly if the
    response is not valid JSON.
    """

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        verify=verify
    )

    response.raise_for_status()

    try:
        return response.json()

    except Exception as exc:

        raise RuntimeError(
            "Expected JSON response but received "
            f"something else from:\n{url}\n\n"
            f"HTTP status: {response.status_code}\n"
            f"Response preview:\n"
            f"{response.text[:1000]}"
        ) from exc


# ============================================================
# JSON
# ============================================================

def save_json(data, path):
    """
    Save JSON with UTF-8 encoding.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# SPATIAL DATA LOADING
# ============================================================

def load_vector(path):
    """
    Load a vector dataset and validate its basic structure.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Spatial file does not exist:\n{path}"
        )

    gdf = gpd.read_file(path)

    if gdf.empty:

        raise ValueError(
            f"Spatial file is empty:\n{path}"
        )

    if gdf.crs is None:

        raise ValueError(
            f"Spatial file has no CRS:\n{path}"
        )

    return gdf


# ============================================================
# GEOMETRY VALIDATION
# ============================================================

def validate_geometry(
    gdf,
    require_polygon=False
):
    """
    Validate geometry without modifying it.
    """

    if gdf.empty:

        raise ValueError(
            "GeoDataFrame is empty."
        )

    if gdf.geometry.isna().any():

        raise ValueError(
            "Dataset contains missing geometries."
        )

    if gdf.geometry.is_empty.any():

        raise ValueError(
            "Dataset contains empty geometries."
        )

    if require_polygon:

        allowed = {
            "Polygon",
            "MultiPolygon"
        }

        bad = ~gdf.geometry.geom_type.isin(
            allowed
        )

    if bad.any():

        bad_types = sorted(
            gdf.loc[
                bad,
                "geometry"
            ].geom_type.unique()
        )

        raise ValueError(
            "Polygon geometry required. "
            f"Found: {bad_types}"
        )

    return {
        "rows": int(len(gdf)),
        "crs": str(gdf.crs),
        "invalid_geometries": int(
            (~gdf.geometry.is_valid).sum()
        ),
        "empty_geometries": int(
            gdf.geometry.is_empty.sum()
        )
    }


def repair_invalid_geometry(gdf):
    """
    Repair geometrically invalid polygons.

    IMPORTANT:
    Missing/empty geometry is never fabricated.
    """

    gdf = gdf.copy()

    if gdf.geometry.isna().any():

        raise ValueError(
            "Cannot repair missing geometry."
        )

    if gdf.geometry.is_empty.any():

        raise ValueError(
            "Cannot repair empty geometry."
        )

    invalid = ~gdf.geometry.is_valid

    if invalid.any():

        gdf.loc[
            invalid,
            "geometry"
        ] = (
            gdf.loc[
                invalid,
                "geometry"
            ].buffer(0)
        )

    if (~gdf.geometry.is_valid).any():

        raise ValueError(
            "Some geometries remain invalid "
            "after repair."
        )

    return gdf


# ============================================================
# DYNAMIC UTM
# ============================================================

def get_utm_epsg(gdf):
    """
    Determine an appropriate UTM CRS from the geometry centroid.

    Works across India instead of assuming one fixed UTM zone.
    """

    wgs84 = gdf.to_crs("EPSG:4326")

    centroid = wgs84.geometry.union_all().centroid

    longitude = centroid.x
    latitude = centroid.y

    zone = int(
        (longitude + 180) // 6 + 1
    )

    if latitude >= 0:

        epsg = 32600 + zone

    else:

        epsg = 32700 + zone

    return epsg


# ============================================================
# METRIC ATTRIBUTES
# ============================================================

def add_polygon_metrics(
    gdf,
    metric_epsg=None
):
    """
    Add:
        area_km2
        centroid_lat
        centroid_lon

    Source geometry remains in its original CRS.
    """

    gdf = gdf.copy()

    if metric_epsg is None:

        metric_epsg = get_utm_epsg(
            gdf
        )

    metric = gdf.to_crs(
        epsg=metric_epsg
    )

    gdf["area_km2"] = (
        metric.geometry.area
        / 1_000_000
    )

    centroids = (
        metric.geometry.centroid
    )

    centroid_wgs84 = (
        gpd.GeoSeries(
            centroids,
            crs=f"EPSG:{metric_epsg}"
        )
        .to_crs("EPSG:4326")
    )

    gdf["centroid_lat"] = (
        centroid_wgs84.y.values
    )

    gdf["centroid_lon"] = (
        centroid_wgs84.x.values
    )

    return gdf


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bounds(gdf):
    """
    Return WGS84 bounding box.
    """

    wgs84 = gdf.to_crs(
        "EPSG:4326"
    )

    minx, miny, maxx, maxy = (
        wgs84.total_bounds
    )

    return {
        "min_lon": float(minx),
        "min_lat": float(miny),
        "max_lon": float(maxx),
        "max_lat": float(maxy)
    }


# ============================================================
# DISSOLVE
# ============================================================

def dissolve_to_single_polygon(
    gdf,
    metric_epsg=None
):
    """
    Dissolve multiple polygons into a single
    Block-level polygon.
    """

    validate_geometry(
        gdf,
        require_polygon=True
    )

    gdf = repair_invalid_geometry(
        gdf
    )

    if metric_epsg is None:

        metric_epsg = get_utm_epsg(
            gdf
        )

    metric = gdf.to_crs(
        epsg=metric_epsg
    )

    geometry = (
        metric.geometry.union_all()
    )

    if geometry.is_empty:

        raise ValueError(
            "Dissolved geometry is empty."
        )

    if not geometry.is_valid:

        geometry = geometry.buffer(
            0
        )

    result = gpd.GeoDataFrame(
        {
            "geometry": [geometry]
        },
        crs=f"EPSG:{metric_epsg}"
    )

    result = result.to_crs(
        "EPSG:4326"
    )

    result = add_polygon_metrics(
        result,
        metric_epsg=metric_epsg
    )

    return result


# ============================================================
# GRAM MANCHITRA GP QUERY
# ============================================================

def query_block_gps(
    block_lgd_code
):
    """
    Download all GP polygons belonging to a Block.
    """

    if not block_lgd_code:

        raise ValueError(
            "Block LGD code is required."
        )

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

    data = safe_get_json(
        GRAM_MANCHITRA_GP_QUERY_URL,
        params
    )

    features = data.get(
        "features",
        []
    )

    if not features:

        raise ValueError(
            "No GP polygons returned for "
            f"Block LGD code "
            f"{block_lgd_code}."
        )

    gdf = (
        gpd.GeoDataFrame
        .from_features(
            features,
            crs="EPSG:4326"
        )
    )

    validate_geometry(
        gdf,
        require_polygon=True
    )

    return gdf