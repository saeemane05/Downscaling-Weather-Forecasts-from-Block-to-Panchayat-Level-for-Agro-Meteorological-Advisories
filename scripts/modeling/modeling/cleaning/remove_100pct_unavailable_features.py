#!/usr/bin/env python3
"""
SIH26 - REMOVE ONLY 100% UNAVAILABLE FEATURES

Universal model-feature cleaning rule:
  - Remove only columns that are 100% unavailable (all values NaN).
  - Never delete rows / GPs.
  - Never impute, interpolate, or fill missing values.
  - Never overwrite raw/processed source datasets.
  - Preserve a manifest of every removed feature.

Actual block pipeline layout:
  block/<State>/<District>/<Block>/processed/merge/*.csv

GP pipeline layout:
  gp/<State>/<District>/<Block>/processed/gp_merged_features.csv

Outputs:
  datasets/cleaned/<State>/<District>/<Block>/block/
  datasets/cleaned/<State>/<District>/<Block>/gp/
"""

from pathlib import Path
import json
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLOCK_ROOT = PROJECT_ROOT / "block"
GP_ROOT = PROJECT_ROOT / "gp"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent, requested):
    wanted = norm(requested)

    exact = parent / requested
    if exact.exists() and exact.is_dir():
        return exact

    if not parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {parent}")

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

    raise RuntimeError(
        f"Ambiguous directory '{requested}' under {parent}: {matches}"
    )


def resolve_location(root, state, district, block):
    return resolve_child(
        resolve_child(
            resolve_child(root, state),
            district,
        ),
        block,
    )


def clean_one(input_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)

    removed = []

    for col in df.columns:
        total = len(df)
        missing_count = int(df[col].isna().sum())

        if total > 0 and missing_count == total:
            removed.append({
                "column": str(col),
                "rows": total,
                "missing_count": missing_count,
                "missing_percent": 100.0,
                "reason": "100_percent_unavailable",
            })

    kept_columns = [
        col for col in df.columns
        if str(col) not in {x["column"] for x in removed}
    ]

    cleaned = df.loc[:, kept_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": int(len(df)),
        "output_rows": int(len(cleaned)),
        "input_columns": int(len(df.columns)),
        "output_columns": int(len(cleaned.columns)),
        "removed_columns": removed,
        "removed_count": len(removed),
        "row_deletion_count": 0,
        "imputation_performed": False,
        "interpolation_performed": False,
    }


def main():
    print("=" * 90)
    print("SIH26 - REMOVE ONLY 100% UNAVAILABLE FEATURES")
    print("=" * 90)

    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    block_dir = resolve_location(BLOCK_ROOT, state, district, block)
    gp_dir = resolve_location(GP_ROOT, state, district, block)

    print(f"\nBlock directory: {block_dir}")
    print(f"GP directory   : {gp_dir}")

    # IMPORTANT:
    # Block merge outputs live under processed/merge.
    block_processed = block_dir / "processed" / "merge"
    gp_processed = gp_dir / "processed"

    sources = [
        (
            "block_observations_daily",
            block_processed / "block_observations_daily.csv",
            "block",
        ),
        (
            "block_forecast_training_base",
            block_processed / "block_forecast_training_base.csv",
            "block",
        ),
        (
            "block_current_forecast",
            block_processed / "block_current_forecast.csv",
            "block",
        ),
        (
            "gp_merged_features",
            gp_processed / "gp_merged_features.csv",
            "gp",
        ),
    ]

    output_root = (
        DATASETS_ROOT
        / "cleaned"
        / state
        / district
        / block
    )

    manifest = {
        "rule": (
            "Remove only columns that are 100% unavailable. "
            "Never delete rows/GPs. Never impute or interpolate."
        ),
        "state": state,
        "district": district,
        "block": block,
        "source_files_untouched": True,
        "row_deletion_performed": False,
        "imputation_performed": False,
        "interpolation_performed": False,
        "datasets": {},
    }

    for name, source, dataset_type in sources:
        if not source.exists():
            print(f"\n[SKIP] Missing source: {source}")
            continue

        print(f"\n[PROCESS] {name}")
        print(f"  Source: {source}")

        output_dir = output_root / dataset_type
        output_path = output_dir / f"{name}_model_features.csv"

        result = clean_one(source, output_path)
        manifest["datasets"][name] = result

        print(
            f"  Rows    : {result['input_rows']} -> "
            f"{result['output_rows']}"
        )
        print(
            f"  Columns : {result['input_columns']} -> "
            f"{result['output_columns']}"
        )
        print(f"  Removed : {result['removed_count']}")

        for item in result["removed_columns"]:
            print(f"    - {item['column']}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "100pct_unavailable_feature_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("CLEANING COMPLETE")
    print("=" * 90)
    print(f"Output root: {output_root}")
    print(f"Manifest:    {manifest_path}")
    print("\nRules enforced:")
    print("  [OK] Only 100% missing columns removed")
    print("  [OK] No rows / GPs removed")
    print("  [OK] No imputation")
    print("  [OK] No interpolation")
    print("  [OK] Source datasets untouched")
    print("=" * 90)


if __name__ == "__main__":
    main()
