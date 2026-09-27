"""
SIH26 — TRUE FINAL training-data audit
READ-ONLY. Matches TRUE_FINAL_v4 manifest structure.
"""

from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = PROJECT_ROOT / "datasets" / "training"

TARGETS = [
    "actual_precipitation_sum",
    "actual_temperature_2m_mean",
    "actual_relative_humidity_2m_mean",
    "actual_wind_speed_10m_mean",
    "actual_surface_pressure_mean",
]

FORBIDDEN = (
    "forecast_lead_days_", "forecast_issued_date_",
    "forecast_issue_date_", "forecast_run", "forecast_retrieved",
    "forecast_rule_", "target_date", "target_datetime",
    "valid_time", "acquisition_date", "system_index",
    "source_dataset", "source_file", "provenance",
    "actual_is_reference_only", "synthetic_data_present",
    "interpolation_present", "satellite_temporal_validity",
)

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def resolve(root, state, district, block):
    hits = []
    for p in root.rglob("*"):
        if p.is_dir():
            parts = [norm(x) for x in p.parts]
            if norm(state) in parts and norm(district) in parts and norm(block) in parts:
                hits.append((p.name == block, len(p.parts), p))
    if not hits:
        return None
    hits.sort(key=lambda x: (-int(x[0]), x[1], str(x[2]).lower()))
    return hits[0][2]

def main():
    print("SIH26 TRUE FINAL audit — READ-ONLY")
    state = input("\nState: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    loc = resolve(TRAINING_ROOT, state, district, block)
    if loc is None:
        print("Could not resolve training location.")
        return 1

    mp = loc / "training_dataset_manifest.json"
    if not mp.exists():
        print("FAIL: manifest missing.")
        return 2
    manifest = json.loads(mp.read_text(encoding="utf-8"))

    failed = []

    for h in range(1, 8):
        print("\n" + "-" * 78)
        print(f"AUDITING D{h}")
        print("-" * 78)
        hd = loc / f"D{h}"

        xs = [pd.read_csv(hd / f"X_{s}.csv", low_memory=False)
              for s in ("train", "validation", "test")]
        ys = [pd.read_csv(hd / f"y_{s}.csv", low_memory=False)
              for s in ("train", "validation", "test")]

        X = pd.concat(xs, ignore_index=True)
        Y = pd.concat(ys, ignore_index=True)
        ok = True

        print(f"X={X.shape}, y={Y.shape}")

        forbidden = [c for c in X.columns if any(p in c.lower() for p in FORBIDDEN)]
        if forbidden:
            print("  FAIL: forbidden columns:", forbidden)
            ok = False
        else:
            print("  PASS: no forbidden metadata in X")

        target_in_x = [c for c in TARGETS if c in X.columns]
        if target_in_x:
            print("  FAIL: targets in X:", target_in_x)
            ok = False
        else:
            print("  PASS: targets absent from X")

        if X.select_dtypes(exclude=[np.number]).columns.tolist():
            print("  FAIL: nonnumeric X")
            ok = False
        else:
            print("  PASS: X numeric")

        if not np.isfinite(X.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all():
            print("  FAIL: non-finite X")
            ok = False
        else:
            print("  PASS: X finite")

        if any(X[c].notna().sum() == 0 for c in X.columns):
            print("  FAIL: all-missing X feature")
            ok = False
        else:
            print("  PASS: no all-missing X feature")

        if all(c in Y.columns for c in TARGETS):
            print("  PASS: all five targets in y")
        else:
            print("  FAIL: target schema incomplete")
            ok = False

        if len(X) == len(Y):
            print("  PASS: X/y aligned")
        else:
            print("  FAIL: X/y row mismatch")
            ok = False

        if list(xs[0].columns) == list(xs[1].columns) == list(xs[2].columns):
            print("  PASS: X schemas identical")
        else:
            print("  FAIL: X schemas differ")
            ok = False

        meta = manifest.get("horizons", {}).get(f"D{h}", {})
        split = meta.get("split", {})
        if set(split) != {"train", "validation", "test"}:
            print("  FAIL: manifest split metadata incomplete")
            ok = False
        else:
            print("  PASS: manifest split metadata complete")
            ranges = []
            for s in ("train", "validation", "test"):
                ranges.append((
                    pd.to_datetime(split[s]["start"]),
                    pd.to_datetime(split[s]["end"]),
                    int(split[s]["rows"])
                ))
            if ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]:
                print("  PASS: chronological split")
            else:
                print("  FAIL: split overlap")
                ok = False
            for s, r in zip(("train", "validation", "test"), ranges):
                actual = len(pd.read_csv(hd / f"X_{s}.csv", low_memory=False))
                if actual == r[2]:
                    print(f"  PASS: {s} rows={actual}")
                else:
                    print(f"  FAIL: {s} rows={actual}, manifest={r[2]}")
                    ok = False

        if not ok:
            failed.append(h)

    print("\n" + "=" * 78)
    print("FINAL AUDIT SUMMARY")
    print("=" * 78)
    if failed:
        print("FINAL STATUS: FAIL")
        print("Failed horizons:", ", ".join(f"D{x}" for x in failed))
        return 2
    print("FINAL STATUS: PASS")
    print("D1–D7 structural leakage audit passed.")
    print("ERA5/S1/S2/LST remain excluded pending temporal provenance proof.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
