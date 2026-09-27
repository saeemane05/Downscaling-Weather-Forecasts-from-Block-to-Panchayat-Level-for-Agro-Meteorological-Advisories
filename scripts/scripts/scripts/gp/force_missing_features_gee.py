# SIH26 - Force-recover missing GEE features
# Run from project root:
#   python scripts\gp\force_missing_features_gee.py
#
# This script NEVER modifies existing datasets. It writes only to:
# gp/<State>/<District>/<Block>/recovery/

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"

S2_CLOUD_PERCENT = int(os.getenv("S2_CLOUD_PERCENT", "20"))
S2_SCALE_M = int(os.getenv("S2_SCALE_M", "20"))
S1_SCALE_M = int(os.getenv("S1_SCALE_M", "10"))
TILE_SCALE = int(os.getenv("GEE_TILE_SCALE", "4"))
S1_YEARS = int(os.getenv("S1_RECOVERY_YEARS", "5"))


def norm(x):
    return "".join(c.lower() for c in str(x) if c.isalnum())


def resolve_master(state, district, block):
    """
    Resolve the GP master generically.

    Does NOT assume fixed Path.parts positions.
    Matches the three directory names immediately above:
        processed/gp_master.geojson

    Spaces / underscores / hyphens are treated as equivalent.
    """

    def norm(x):
        return "".join(
            c.lower()
            for c in str(x)
            if c.isalnum()
        )

    wanted_state = norm(state)
    wanted_district = norm(district)
    wanted_block = norm(block)

    candidates = sorted(
        PROJECT_ROOT.glob("gp/**/processed/gp_master.geojson")
    )

    matches = []

    for path in candidates:
        # Expected:
        # .../gp/<state>/<district>/<block>/processed/gp_master.geojson
        if path.parent.name.lower() != "processed":
            continue

        block_dir = path.parent.parent
        district_dir = block_dir.parent
        state_dir = district_dir.parent

        if (
            norm(state_dir.name) == wanted_state
            and norm(district_dir.name) == wanted_district
            and norm(block_dir.name) == wanted_block
        ):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "\nCould not find GP master.\n"
            f"Requested:\n"
            f"  State   : {state}\n"
            f"  District: {district}\n"
            f"  Block   : {block}\n\n"
            f"Searched under:\n"
            f"  {PROJECT_ROOT / 'gp'}\n\n"
            "Available GP masters:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple matching GP masters found:\n"
            + "\n".join(f"  {p}" for p in matches)
        )

    return matches[0]

def read_master(path):
    gdf = gpd.read_file(path)
    if gdf.empty or gdf.crs is None:
        raise ValueError(f"Invalid GP master: {path}")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    id_col = next(
        (c for c in ["gp_id", "gp_code", "LGD_GP_CODE", "lgd_gp_code"]
         if c in gdf.columns),
        None,
    )
    if id_col is None:
        raise ValueError(f"No GP identifier in {path}")

    gdf = gdf.copy()
    gdf["gp_id"] = gdf[id_col].astype(str).str.strip()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf["gp_id"].duplicated().any():
        raise ValueError("Duplicate GP IDs found; recovery will not silently alter them.")

    return gdf


def init_ee():
    try:
        ee.Initialize()
    except Exception as e:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate with your existing EE setup."
        ) from e


def mask_s2(img):
    scl = img.select("SCL")
    clear = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
        .And(scl.neq(10)).And(scl.neq(11))
    )
    edge = img.select("B8A").mask().And(img.select("B9").mask())
    return img.updateMask(clear.And(edge))


def add_ndvi(img):
    b4 = img.select("B4").multiply(0.0001)
    b8 = img.select("B8").multiply(0.0001)
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    return img.addBands(ndvi)


def clean_numbers(values):
    out = []
    for x in values:
        if x is None:
            continue
        try:
            x = float(x)
            if np.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return out


def recover_s2(gdf, start, end):
    block_geom = ee.Geometry(gdf.geometry.unary_union.__geo_interface__)

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(block_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_PERCENT))
        .map(mask_s2)
        .map(add_ndvi)
    )

    n_images = int(collection.size().getInfo())
    print(f"S2 candidate images: {n_images}")

    if n_images == 0:
        return pd.DataFrame(), {
            "candidate_images": 0,
            "gps_with_valid_change": 0,
            "status": "NO_S2_IMAGES",
        }

    end_d = date.fromisoformat(end)
    current_start = end_d - timedelta(days=30)
    previous_start = current_start - timedelta(days=30)

    rows = []

    for i, (_, gp) in enumerate(gdf.iterrows(), 1):
        geom = ee.Geometry(gp.geometry.__geo_interface__)

        def zonal(img):
            value = img.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=S2_SCALE_M,
                bestEffort=True,
                maxPixels=10_000_000,
                tileScale=TILE_SCALE,
            ).get("NDVI")
            return img.set("gp_ndvi", value)

        current = collection.filterDate(
            current_start.isoformat(), end
        ).map(zonal)
        previous = collection.filterDate(
            previous_start.isoformat(), current_start.isoformat()
        ).map(zonal)

        cur = clean_numbers(current.aggregate_array("gp_ndvi").getInfo())
        prev = clean_numbers(previous.aggregate_array("gp_ndvi").getInfo())

        change = float(np.mean(cur) - np.mean(prev)) if cur and prev else np.nan

        rows.append({
            "gp_id": str(gp["gp_id"]),
            "gp_name": str(gp.get("gp_name", "")),
            "current_30d_ndvi_mean": float(np.mean(cur)) if cur else np.nan,
            "previous_30d_ndvi_mean": float(np.mean(prev)) if prev else np.nan,
            "NDVI_30d_change": change,
            "current_observation_count": len(cur),
            "previous_observation_count": len(prev),
            "synthetic": False,
            "imputed": False,
        })

        print(
            f"S2 GP {i}/{len(gdf)}: {gp['gp_id']} | "
            f"current={len(cur)} previous={len(prev)}"
        )

    result = pd.DataFrame(rows)
    usable = int(result["NDVI_30d_change"].notna().sum())

    return result, {
        "candidate_images": n_images,
        "gps_with_valid_change": usable,
        "status": "DATA_AVAILABLE" if usable else "NO_USABLE_NDVI_30D_CHANGE",
    }


def prepare_s1(img):
    vv = img.select("VV")
    vh = img.select("VH")
    angle = img.select("angle")
    valid = vv.mask().And(vh.mask()).And(angle.mask())
    vv = vv.updateMask(valid)
    vh = vh.updateMask(valid)
    angle = angle.updateMask(valid)
    diff = vv.subtract(vh).rename("vv_vh_diff_db")
    return ee.Image.cat(vv, vh, diff, angle)


def recover_s1(gdf, start, end):
    block_geom = ee.Geometry(gdf.geometry.unary_union.__geo_interface__)

    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(block_geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VV"
        ))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VH"
        ))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    )

    n_images = int(collection.size().getInfo())
    print(f"S1 ASCENDING candidate images: {n_images}")

    if n_images == 0:
        return pd.DataFrame(), {
            "candidate_images": 0,
            "gps_with_ascending_data": 0,
            "status": "NO_ASCENDING_S1_IMAGES",
        }

    rows = []

    for i, (_, gp) in enumerate(gdf.iterrows(), 1):
        geom = ee.Geometry(gp.geometry.__geo_interface__)

        def zonal(img):
            stats = prepare_s1(img).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=S1_SCALE_M,
                bestEffort=True,
                maxPixels=10_000_000,
                tileScale=TILE_SCALE,
            )
            return img.set({
                "gp_vv": stats.get("VV"),
                "gp_vh": stats.get("VH"),
                "gp_diff": stats.get("vv_vh_diff_db"),
                "gp_angle": stats.get("angle"),
            })

        c = collection.map(zonal)
        vv = clean_numbers(c.aggregate_array("gp_vv").getInfo())
        vh = clean_numbers(c.aggregate_array("gp_vh").getInfo())
        diff = clean_numbers(c.aggregate_array("gp_diff").getInfo())
        angle = clean_numbers(c.aggregate_array("gp_angle").getInfo())

        def stat(a, fn):
            return float(fn(a)) if a else np.nan

        rows.append({
            "gp_id": str(gp["gp_id"]),
            "gp_name": str(gp.get("gp_name", "")),
            "s1_ascending_count": len(vv),
            "s1_vv_db_ascending_mean": stat(vv, np.mean),
            "s1_vv_db_ascending_median": stat(vv, np.median),
            "s1_vv_db_ascending_std": stat(vv, np.std),
            "s1_vv_db_ascending_min": stat(vv, np.min),
            "s1_vv_db_ascending_max": stat(vv, np.max),
            "s1_vh_db_ascending_mean": stat(vh, np.mean),
            "s1_vh_db_ascending_median": stat(vh, np.median),
            "s1_vh_db_ascending_std": stat(vh, np.std),
            "s1_vh_db_ascending_min": stat(vh, np.min),
            "s1_vh_db_ascending_max": stat(vh, np.max),
            "s1_vv_vh_diff_db_ascending_mean": stat(diff, np.mean),
            "s1_vv_vh_diff_db_ascending_median": stat(diff, np.median),
            "s1_vv_vh_diff_db_ascending_std": stat(diff, np.std),
            "s1_vv_vh_diff_db_ascending_min": stat(diff, np.min),
            "s1_vv_vh_diff_db_ascending_max": stat(diff, np.max),
            "s1_incidence_angle_deg_ascending_mean": stat(angle, np.mean),
            "s1_incidence_angle_deg_ascending_median": stat(angle, np.median),
            "s1_incidence_angle_deg_ascending_std": stat(angle, np.std),
            "s1_incidence_angle_deg_ascending_min": stat(angle, np.min),
            "s1_incidence_angle_deg_ascending_max": stat(angle, np.max),
            "synthetic": False,
            "imputed": False,
        })

        print(f"S1 GP {i}/{len(gdf)}: {gp['gp_id']} | ascending={len(vv)}")

    result = pd.DataFrame(rows)
    usable = int(result["s1_ascending_count"].gt(0).sum())

    return result, {
        "candidate_images": n_images,
        "gps_with_ascending_data": usable,
        "status": "DATA_AVAILABLE" if usable else "NO_USABLE_ASCENDING_DATA",
    }


def main():
    print("=" * 90)
    print("SIH26 - FORCE MISSING FEATURES DIRECTLY FROM GEE")
    print("=" * 90)

    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    master = resolve_master(state, district, block)
    gdf = read_master(master)

    print(f"GP master: {master}")
    print(f"GP count: {len(gdf)}")

    init_ee()

    today = date.today()
    end = (today + timedelta(days=1)).isoformat()

    s2_start = (today - timedelta(days=60)).isoformat()

    try:
        s1_start = today.replace(year=today.year - S1_YEARS).isoformat()
    except ValueError:
        s1_start = today.replace(
            year=today.year - S1_YEARS, day=28
        ).isoformat()

    recovery = master.parent.parent / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)

    s2, s2_report = recover_s2(gdf, s2_start, end)
    s1, s1_report = recover_s1(gdf, s1_start, end)

    s2_path = recovery / "force_sentinel2_NDVI_30d_change.csv"
    s1_path = recovery / "force_sentinel1_ascending_features.csv"
    report_path = recovery / "force_recovery_report.json"

    s2.to_csv(s2_path, index=False)
    s1.to_csv(s1_path, index=False)

    report = {
        "state": state,
        "district": district,
        "block": block,
        "gp_master": str(master),
        "gp_count": len(gdf),
        "sentinel2": s2_report,
        "sentinel1": s1_report,
        "synthetic_data": False,
        "imputation": False,
        "existing_dataset_modified": False,
        "outputs": {
            "sentinel2": str(s2_path),
            "sentinel1": str(s1_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("RECOVERY COMPLETE")
    print("=" * 90)
    print("S2:", s2_report)
    print("S1:", s1_report)
    print(f"S2 output: {s2_path}")
    print(f"S1 output: {s1_path}")
    print(f"Report:    {report_path}")
    print("Existing datasets were NOT modified.")


if __name__ == "__main__":
    main()
