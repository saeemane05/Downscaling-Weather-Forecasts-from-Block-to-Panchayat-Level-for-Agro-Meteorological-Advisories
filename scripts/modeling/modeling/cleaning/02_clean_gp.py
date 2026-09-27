"""
SIH26 — GP MODEL-DATA CLEANER
-----------------------------
Cleans the final GP feature table without modifying the authoritative
GP boundary or any raw observations.

Cleaning performed:
    - duplicate-column safety check
    - string whitespace trimming
    - +/-inf -> NaN
    - exact duplicate row removal
    - preservation of legitimate missing values

No imputation/interpolation/outlier removal/feature engineering is performed.
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
GP_ROOT = PROJECT_ROOT / "gp"
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


def resolve_gp(state: str, district: str, block: str) -> Path:
    return resolve_child(
        resolve_child(resolve_child(GP_ROOT, state), district),
        block,
    )


def collapse_duplicate_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()

    for name in duplicated:
        positions = [i for i, c in enumerate(df.columns) if c == name]
        first = df.iloc[:, positions[0]]
        for pos in positions[1:]:
            if not first.equals(df.iloc[:, pos]):
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

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    inf_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(df[col], errors="coerce")
        mask = np.isinf(values.to_numpy(dtype=float, copy=False))
        inf_count += int(mask.sum())
        if mask.any():
            df.loc[mask, col] = np.nan

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

    gp_dir = resolve_gp(state, district, block)

    source = gp_dir / "processed" / "gp_merged_features.csv"
    if not source.exists():
        raise FileNotFoundError(f"Required GP dataset not found: {source}")

    output_root = (
        DATASETS_ROOT / "cleaned"
        / gp_dir.parent.parent.name
        / gp_dir.parent.name
        / gp_dir.name
        / "gp"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[CLEAN] {source}")
    df = pd.read_csv(source, low_memory=False)
    cleaned, report = clean_dataframe(df, source)

    output = output_root / "gp_merged_features.csv"
    cleaned.to_csv(output, index=False)

    report["output"] = str(output)

    report_path = output_root / "gp_cleaning_report.json"
    report_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "state": state,
                "district": district,
                "block": block,
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("GP CLEANING COMPLETE")
    print("=" * 72)
    print(f"Rows:       {len(cleaned)}")
    print(f"Columns:    {len(cleaned.columns)}")
    print(f"Output:     {output}")
    print(f"Report:     {report_path}")


if __name__ == "__main__":
    main()
