"""
SIH26 — MODELING DATA AUDIT
--------------------------------
Purpose:
    Non-destructive audit of the model-ready datasets for one target block.

This script DOES NOT:
    - impute missing values
    - interpolate values
    - remove outliers
    - scale/normalize features
    - overwrite source data
    - train a model

It reports:
    - shape/schema
    - duplicate column labels
    - exact duplicate rows
    - missingness
    - non-finite numeric values
    - likely date/time columns
    - constant columns
    - numeric ranges

Usage:
    python 00_data_audit.py
or:
    python 00_data_audit.py "Maharashtra" "Nashik" "Sinnar"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
BLOCK_ROOT = PROJECT_ROOT / "block"
GP_ROOT = PROJECT_ROOT / "gp"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent: Path, requested: str) -> Path:
    if not parent.exists():
        raise FileNotFoundError(parent)
    wanted = norm(requested)
    exact = parent / requested
    if exact.exists():
        return exact
    matches = [p for p in parent.iterdir() if p.is_dir() and norm(p.name) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve '{requested}' under {parent}"
        )
    raise RuntimeError(f"Ambiguous directory '{requested}' under {parent}: {matches}")


def resolve_location(state: str, district: str, block: str):
    block_dir = resolve_child(resolve_child(BLOCK_ROOT, state), district)
    block_dir = resolve_child(block_dir, block)

    gp_dir = resolve_child(resolve_child(GP_ROOT, state), district)
    gp_dir = resolve_child(gp_dir, block)

    return block_dir, gp_dir


def duplicate_columns(df: pd.DataFrame) -> list[str]:
    return df.columns[df.columns.duplicated()].unique().tolist()


def likely_datetime_columns(columns):
    tokens = (
        "date", "datetime", "timestamp", "time",
        "issue", "target", "valid", "acquisition", "cycle"
    )
    return [
        c for c in columns
        if any(t in str(c).lower() for t in tokens)
    ]


def audit_csv(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)

    dup_cols = duplicate_columns(df)

    numeric = df.select_dtypes(include=[np.number])
    nonfinite = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if not numeric.empty else 0

    missing = df.isna().sum()
    missing_pct = (missing / max(len(df), 1) * 100).round(4)

    constants = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    duplicate_rows = int(df.duplicated(keep=False).sum())

    ranges = {}
    for c in numeric.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s[np.isfinite(s)]
        if len(valid):
            ranges[c] = {
                "min": float(valid.min()),
                "max": float(valid.max()),
            }

    missing_columns = [
        {
            "column": str(c),
            "missing_count": int(missing[c]),
            "missing_percent": float(missing_pct[c]),
        }
        for c in df.columns
        if missing[c] > 0
    ]

    return {
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "duplicate_column_labels": [str(c) for c in dup_cols],
        "exact_duplicate_rows": duplicate_rows,
        "nonfinite_numeric_values": nonfinite,
        "constant_columns": [str(c) for c in constants],
        "likely_datetime_columns": [str(c) for c in likely_datetime_columns(df.columns)],
        "missing_columns": missing_columns,
        "numeric_ranges": ranges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?")
    parser.add_argument("district", nargs="?")
    parser.add_argument("block", nargs="?")
    args = parser.parse_args()

    state = args.state or input("State: ").strip()
    district = args.district or input("District: ").strip()
    block = args.block or input("Block: ").strip()

    block_dir, gp_dir = resolve_location(state, district, block)

    files_to_audit = [
        block_dir / "processed" / "merge" / "block_observations_daily.csv",
        block_dir / "processed" / "merge" / "block_forecast_training_base.csv",
        block_dir / "processed" / "merge" / "block_current_forecast.csv",
        gp_dir / "processed" / "gp_merged_features.csv",
    ]

    audit_root = DATASETS_ROOT / "cleaned" / block_dir.parent.parent.name / block_dir.parent.name / block_dir.name / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    results = []
    for path in files_to_audit:
        if not path.exists():
            results.append({
                "file": str(path),
                "status": "MISSING",
            })
            continue

        print(f"[AUDIT] {path}")
        try:
            result = audit_csv(path)
            result["status"] = "OK"
            results.append(result)
        except Exception as exc:
            results.append({
                "file": str(path),
                "status": "ERROR",
                "error": repr(exc),
            })

    report = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "state": state,
        "district": district,
        "block": block,
        "source_block": str(block_dir),
        "source_gp": str(gp_dir),
        "non_destructive": True,
        "reports": results,
    }

    out = audit_root / "data_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("DATA AUDIT COMPLETE")
    print("=" * 72)
    print(f"Report: {out}")
    for r in results:
        print(f"  {r['status']}: {r['file']}")


if __name__ == "__main__":
    main()
