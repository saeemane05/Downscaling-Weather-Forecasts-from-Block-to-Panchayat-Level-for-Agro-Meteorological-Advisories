"""
SIH26 — BLOCK MODEL-DATA CLEANER
--------------------------------
Cleans ONLY the model-ready block datasets.

Source files are NEVER overwritten.

Cleaning performed:
    1. Detect/collapse identical duplicate column labels.
       Conflicting duplicate labels cause a safe failure.
    2. Trim surrounding whitespace from string cells.
    3. Convert +/-inf to NaN.
    4. Remove exact duplicate rows.
    5. Preserve all legitimate missing values.
    6. Preserve the original feature values; no imputation/interpolation.
    7. Write a cleaning report.

No outlier deletion, scaling, normalization, or feature engineering is done here.
Those belong to later modeling stages.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLOCK_ROOT = PROJECT_ROOT / "block"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent: Path, requested: str) -> Path:
    wanted = norm(requested)
    exact = parent / requested
    if exact.exists():
        return exact
    matches = [p for p in parent.iterdir() if p.is_dir() and norm(p.name) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not resolve '{requested}' under {parent}")
    raise RuntimeError(f"Ambiguous directory '{requested}' under {parent}: {matches}")


def resolve_block(state: str, district: str, block: str) -> Path:
    return resolve_child(
        resolve_child(resolve_child(BLOCK_ROOT, state), district),
        block,
    )


def collapse_duplicate_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()

    for name in duplicated:
        positions = [i for i, c in enumerate(df.columns) if c == name]
        first = df.iloc[:, positions[0]]

        for pos in positions[1:]:
            other = df.iloc[:, pos]
            if not first.equals(other):
                raise RuntimeError(
                    f"Conflicting duplicate column label '{name}' in {source}. "
                    "The source file was not modified."
                )

    if duplicated:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df


def clean_dataframe(df: pd.DataFrame, source: Path):
    before_rows = len(df)
    before_missing = int(df.isna().sum().sum())

    df = collapse_duplicate_columns(df, source)

    string_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in string_columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        inf_mask = np.isinf(values.to_numpy(dtype=float, copy=False))
        inf_count += int(inf_mask.sum())
        if inf_mask.any():
            df.loc[inf_mask, col] = np.nan

    duplicate_rows = int(df.duplicated(keep="first").sum())
    if duplicate_rows:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)

    after_missing = int(df.isna().sum().sum())

    report = {
        "source": str(source),
        "rows_before": before_rows,
        "rows_after": len(df),
        "exact_duplicate_rows_removed": duplicate_rows,
        "infinite_numeric_values_converted_to_nan": inf_count,
        "missing_cells_before": before_missing,
        "missing_cells_after": after_missing,
        "imputation": False,
        "interpolation": False,
        "outlier_removal": False,
        "feature_engineering": False,
    }

    return df, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?")
    parser.add_argument("district", nargs="?")
    parser.add_argument("block", nargs="?")
    args = parser.parse_args()

    state = args.state or input("State: ").strip()
    district = args.district or input("District: ").strip()
    block = args.block or input("Block: ").strip()

    block_dir = resolve_block(state, district, block)

    sources = {
        "block_observations_daily": block_dir / "processed" / "merge" / "block_observations_daily.csv",
        "block_forecast_training_base": block_dir / "processed" / "merge" / "block_forecast_training_base.csv",
        "block_current_forecast": block_dir / "processed" / "merge" / "block_current_forecast.csv",
    }

    output_root = (
        DATASETS_ROOT / "cleaned"
        / block_dir.parent.parent.name
        / block_dir.parent.name
        / block_dir.name
        / "block"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for name, source in sources.items():
        if not source.exists():
            print(f"[SKIP] Missing source: {source}")
            continue

        print(f"[CLEAN] {name}")
        df = pd.read_csv(source, low_memory=False)
        cleaned, report = clean_dataframe(df, source)

        output = output_root / f"{name}.csv"
        cleaned.to_csv(output, index=False)

        report["output"] = str(output)
        all_reports.append(report)
        print(f"        {len(df)} -> {len(cleaned)} rows")
        print(f"        output: {output}")

    report_path = output_root / "block_cleaning_report.json"
    report_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "state": state,
                "district": district,
                "block": block,
                "reports": all_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("BLOCK CLEANING COMPLETE")
    print("=" * 72)
    print(f"Output root: {output_root}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()
