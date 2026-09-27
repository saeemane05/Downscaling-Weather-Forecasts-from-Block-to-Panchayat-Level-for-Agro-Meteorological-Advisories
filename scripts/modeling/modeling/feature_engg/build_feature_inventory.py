#!/usr/bin/env python3
"""
SIH26 — FEATURE INVENTORY / FEATURE DICTIONARY BUILDER

Purpose:
    Inspect the ACTUAL cleaned datasets for one State/District/Block and
    produce a structured inventory of every column before feature engineering.

This script does NOT create engineered features.
It does NOT modify source or cleaned datasets.

It classifies each actual column using:
    - dataset
    - source level
    - dtype
    - missingness
    - constant status
    - likely metadata/identifier status
    - likely temporal/static status
    - preliminary modeling role

Important:
    The preliminary role is deliberately conservative. It is a review
    starting point, not an automatic final modeling decision.

Outputs:
    datasets/feature_inventory/<State>/<District>/<Block>/
        feature_inventory.csv
        feature_inventory.json
        feature_inventory_summary.json
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_ROOT = PROJECT_ROOT / "datasets" / "cleaned"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "feature_inventory"


def norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent, requested):
    wanted = norm(requested)
    exact = parent / requested
    if exact.exists() and exact.is_dir():
        return exact

    matches = [
        p for p in parent.iterdir()
        if p.is_dir() and norm(p.name) == wanted
    ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve '{requested}' under {parent}"
        )
    raise RuntimeError(f"Ambiguous directory: {matches}")


def resolve_location(root, state, district, block):
    return resolve_child(
        resolve_child(
            resolve_child(root, state),
            district,
        ),
        block,
    )


def preliminary_classification(column, series, dataset_name):
    c = norm(column)
    name = str(column).lower()

    identifier_tokens = [
        "id", "uuid", "code", "name", "geometry", "wkt", "polygon",
        "source", "timezone", "retrieval", "timestamp", "run_timestamp",
        "issue_date", "target_date", "forecast_source", "model",
    ]

    temporal_tokens = [
        "date", "time", "lag", "rolling", "lead", "previous",
        "days_since", "acquisition",
    ]

    spatial_tokens = [
        "latitude", "longitude", "lat", "lon", "elevation", "slope",
        "aspect", "distance", "terrain", "landcover", "soil",
        "ndvi", "evi", "savi", "ndmi", "nbr", "vv", "vh", "lst",
    ]

    target_tokens = [
        "target", "actual", "rainfall", "precip", "precipitation",
    ]

    if any(t in c for t in ["synthetic", "interpolation", "fallback"]):
        return "provenance", "metadata", "exclude_initial"

    if any(t in c for t in identifier_tokens):
        return "metadata_or_identifier", "metadata", "exclude_initial"

    if any(t in c for t in target_tokens):
        # Actual weather columns are not automatically targets; they need
        # explicit temporal/forecast-role review.
        return "weather_or_target_candidate", "predictor_or_target", "review"

    if any(t in c for t in temporal_tokens):
        return "temporal", "temporal", "review"

    if any(t in c for t in spatial_tokens):
        return "spatial_or_remote_sensing", "predictor", "keep_review"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric_environmental_or_derived", "predictor", "keep_review"

    if pd.api.types.is_bool_dtype(series):
        return "boolean_indicator", "predictor_or_metadata", "review"

    return "categorical_or_text", "categorical_or_metadata", "review"


def inspect_file(path, dataset_name, level):
    df = pd.read_csv(path, low_memory=False)

    records = []

    for col in df.columns:
        s = df[col]
        n = len(s)
        missing = int(s.isna().sum())
        non_missing = n - missing

        if non_missing:
            nunique = int(s.nunique(dropna=True))
        else:
            nunique = 0

        all_missing = n > 0 and missing == n
        constant = non_missing > 0 and nunique <= 1

        classification, role, action = preliminary_classification(
            col, s, dataset_name
        )

        record = {
            "dataset": dataset_name,
            "level": level,
            "column": str(col),
            "dtype": str(s.dtype),
            "rows": int(n),
            "missing_count": missing,
            "missing_percent": round(
                100.0 * missing / n, 4
            ) if n else None,
            "non_missing_count": non_missing,
            "unique_non_null": nunique,
            "all_missing_100pct": bool(all_missing),
            "constant_non_null": bool(constant),
            "classification": classification,
            "preliminary_role": role,
            "preliminary_action": action,
            "availability_rule": "",
            "final_role": "",
            "final_action": "",
            "leakage_risk": "",
            "scientific_reason": "",
        }

        records.append(record)

    return records


def main():
    print("=" * 90)
    print("SIH26 — ACTUAL FEATURE INVENTORY / FEATURE DICTIONARY")
    print("=" * 90)

    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    location = resolve_location(CLEANED_ROOT, state, district, block)

    print(f"\nCleaned location: {location}")

    files = [
        (
            "block_observations_daily",
            location / "block" / "block_observations_daily.csv",
            "block",
        ),
        (
            "block_forecast_training_base",
            location / "block" / "block_forecast_training_base.csv",
            "block",
        ),
        (
            "block_current_forecast",
            location / "block" / "block_current_forecast.csv",
            "block",
        ),
        (
            "gp_merged_features",
            location / "gp" / "gp_merged_features.csv",
            "gp",
        ),
    ]

    all_records = []
    dataset_summary = {}

    for dataset_name, path, level in files:
        if not path.exists():
            print(f"\n[SKIP] {path}")
            continue

        print(f"\n[READ] {path}")

        records = inspect_file(path, dataset_name, level)
        all_records.extend(records)

        df = pd.read_csv(path, low_memory=False)

        dataset_summary[dataset_name] = {
            "file": str(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "100pct_missing_columns": int(
                sum(r["all_missing_100pct"] for r in records)
            ),
            "constant_columns": int(
                sum(r["constant_non_null"] for r in records)
            ),
            "partially_missing_columns": int(
                sum(
                    0 < r["missing_percent"] < 100
                    for r in records
                )
            ),
        }

        print(
            f"       rows={len(df)}, columns={len(df.columns)}, "
            f"100% missing={dataset_summary[dataset_name]['100pct_missing_columns']}"
        )

    if not all_records:
        raise RuntimeError("No cleaned datasets were found.")

    out_dir = (
        OUTPUT_ROOT / state / district / block
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_df = pd.DataFrame(all_records)

    csv_path = out_dir / "feature_inventory.csv"
    json_path = out_dir / "feature_inventory.json"
    summary_path = out_dir / "feature_inventory_summary.json"

    inventory_df.to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps(all_records, indent=2),
        encoding="utf-8",
    )

    summary = {
        "state": state,
        "district": district,
        "block": block,
        "purpose": (
            "Actual column inventory before feature engineering. "
            "No engineered features were created."
        ),
        "total_feature_records": len(all_records),
        "datasets": dataset_summary,
        "classification_is_preliminary": True,
        "final_modeling_decisions_required": True,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("FEATURE INVENTORY COMPLETE")
    print("=" * 90)
    print(f"Features inspected: {len(all_records)}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        "IMPORTANT: Preliminary classifications are NOT final modeling "
        "decisions. We will review the actual inventory before engineering."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
