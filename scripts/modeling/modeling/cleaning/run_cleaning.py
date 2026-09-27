"""
SIH26 — UNIVERSAL MODELING DATA CLEANING MAIN RUNNER
-----------------------------------------------------

Enter State / District / Block ONCE.

Pipeline:
    1. Non-destructive audit
    2. Block model-data cleaning
    3. GP model-data cleaning
    4. Remove ONLY 100%-unavailable feature columns from separate
       model-feature copies

Important:
    - Original block/ and gp/ datasets are NEVER overwritten.
    - No rows / GPs are deleted.
    - No imputation.
    - No interpolation.
    - No synthetic values.
    - No outlier deletion.
    - No feature engineering.
    - Partially missing features are retained.
    - Only columns with 100% NaN are excluded from the model-feature copies.

The 100%-unavailable stage is integrated into THIS MAIN FILE.
No separate script needs to be run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
BLOCK_ROOT = PROJECT_ROOT / "block"
GP_ROOT = PROJECT_ROOT / "gp"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent: Path, requested: str) -> Path:
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


def resolve_location(root: Path, state: str, district: str, block: str) -> Path:
    return resolve_child(
        resolve_child(
            resolve_child(root, state),
            district,
        ),
        block,
    )


def run_stage(script: Path, state: str, district: str, block: str):
    command = [
        sys.executable,
        str(script),
        state,
        district,
        block,
    ]

    print("\n" + "#" * 80)
    print(f"RUNNING: {script.name}")
    print("#" * 80)

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {script.name} "
            f"(exit code {result.returncode})"
        )


def remove_only_100pct_unavailable(
    input_path: Path,
    output_path: Path,
):
    """
    Remove ONLY columns where every row is NaN.

    This function intentionally does NOT:
      - delete rows
      - delete GPs
      - impute
      - interpolate
      - replace missing values
      - remove partially missing columns

    The authoritative source file is read and left untouched.
    """

    df = pd.read_csv(input_path, low_memory=False)

    removed = []

    for col in df.columns:
        total = len(df)
        missing_count = int(df[col].isna().sum())

        if total > 0 and missing_count == total:
            removed.append(
                {
                    "column": str(col),
                    "rows": total,
                    "missing_count": missing_count,
                    "missing_percent": 100.0,
                    "reason": "100_percent_unavailable",
                }
            )

    removed_names = {item["column"] for item in removed}

    kept_columns = [
        col for col in df.columns
        if str(col) not in removed_names
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
        "source_untouched": True,
    }


def run_100pct_unavailable_stage(
    state: str,
    district: str,
    block: str,
):
    """
    Integrated final cleaning stage.

    It uses the authoritative processed datasets directly and writes
    separate *_model_features.csv files.

    This is intentionally separate from 01_clean_block.py / 02_clean_gp.py
    outputs so that the original datasets remain authoritative and the
    100%-availability rule is transparent.
    """

    print("\n" + "#" * 80)
    print("RUNNING: 100%-UNAVAILABLE FEATURE REMOVAL")
    print("#" * 80)

    block_dir = resolve_location(BLOCK_ROOT, state, district, block)
    gp_dir = resolve_location(GP_ROOT, state, district, block)

    print(f"Block directory: {block_dir}")
    print(f"GP directory   : {gp_dir}")

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
        / block_dir.parent.parent.name
        / block_dir.parent.name
        / block_dir.name
    )

    manifest = {
        "rule": (
            "Remove only columns that are 100% unavailable "
            "(all values are NaN)."
        ),
        "state": state,
        "district": district,
        "block": block,
        "source_files_untouched": True,
        "row_deletion_performed": False,
        "imputation_performed": False,
        "interpolation_performed": False,
        "partial_missing_features_retained": True,
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

        result = remove_only_100pct_unavailable(
            source,
            output_path,
        )

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

    manifest_path = (
        output_root
        / "100pct_unavailable_feature_manifest.json"
    )

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "-" * 80)
    print("100%-UNAVAILABLE FEATURE REMOVAL COMPLETE")
    print("-" * 80)
    print(f"Output root: {output_root}")
    print(f"Manifest:    {manifest_path}")
    print("  [OK] Only 100% missing columns removed")
    print("  [OK] No rows / GPs removed")
    print("  [OK] No imputation")
    print("  [OK] No interpolation")
    print("  [OK] Partially missing features retained")
    print("  [OK] Source datasets untouched")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?")
    parser.add_argument("district", nargs="?")
    parser.add_argument("block", nargs="?")
    args = parser.parse_args()

    state = args.state or input("State: ").strip()
    district = args.district or input("District: ").strip()
    block = args.block or input("Block: ").strip()

    print("=" * 80)
    print("SIH26 — MODELING DATA CLEANING")
    print("=" * 80)
    print(f"State   : {state}")
    print(f"District: {district}")
    print(f"Block   : {block}")
    print()
    print("Policy:")
    print("  • source datasets are NOT overwritten")
    print("  • no imputation")
    print("  • no interpolation")
    print("  • no synthetic values")
    print("  • no outlier deletion")
    print("  • no feature engineering")
    print("  • exact duplicate rows may be removed from CLEANED copies only")
    print("  • +/-inf is converted to NaN in CLEANED copies")
    print("  • only 100%-unavailable columns are removed from MODEL copies")
    print("  • partially missing columns are retained")
    print("  • no rows / GPs are removed by the 100% rule")
    print("=" * 80)

    # Existing cleaning/audit stages.
    run_stage(
        SCRIPT_DIR / "00_data_audit.py",
        state,
        district,
        block,
    )

    run_stage(
        SCRIPT_DIR / "01_clean_block.py",
        state,
        district,
        block,
    )

    run_stage(
        SCRIPT_DIR / "02_clean_gp.py",
        state,
        district,
        block,
    )

    # NEW: integrated 100%-unavailable feature stage.
    run_100pct_unavailable_stage(
        state,
        district,
        block,
    )

    print("\n" + "=" * 80)
    print("SIH26 MODELING DATA CLEANING COMPLETED")
    print("=" * 80)
    print("Original block/ and gp/ data were preserved.")
    print("Cleaned datasets are under:")
    print("  datasets/cleaned/<State>/<District>/<Block>/")
    print()
    print("Model-feature copies additionally exclude ONLY")
    print("100%-unavailable feature columns.")
    print("=" * 80)


if __name__ == "__main__":
    main()
