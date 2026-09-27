"""
SIH26 — TRUE FINAL leakage-safe training dataset builder

This version is intentionally independent of the previously generated
training files. It ALWAYS rebuilds from the engineered SOURCE file and
therefore cannot accidentally reuse an old X matrix.

It uses the feature dictionary to identify forecast predictors, but applies
a hard exclusion list for issue/lead/run/provenance columns.

Source datasets are never modified.
"""

from pathlib import Path
import json
import re
import shutil
import tempfile
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENGINEERED_ROOT = PROJECT_ROOT / "datasets" / "engineered"
TRAINING_ROOT = PROJECT_ROOT / "datasets" / "training"

TARGETS = [
    "actual_precipitation_sum",
    "actual_temperature_2m_mean",
    "actual_relative_humidity_2m_mean",
    "actual_wind_speed_10m_mean",
    "actual_surface_pressure_mean",
]

HARD_EXCLUDE_EXACT = {
    "date", "latitude", "longitude",
    "actual_is_reference_only", "synthetic_data_present",
    "interpolation_present",
}

HARD_EXCLUDE_SUBSTRINGS = (
    "forecast_lead_days_",
    "forecast_issued_date_",
    "forecast_issue_date_",
    "forecast_run",
    "forecast_retrieved",
    "forecast_rule_",
    "forecast_source_",
    "forecast_model_",
    "target_date",
    "target_datetime",
    "valid_time",
    "acquisition_date",
    "system_index",
    "source_dataset",
    "source_file",
    "provenance",
)

SAFE_HISTORICAL_PREFIXES = (
    "actual_precipitation_sum_lag",
    "actual_temperature_2m_mean_lag",
    "actual_relative_humidity_2m_mean_lag",
    "actual_wind_speed_10m_mean_lag",
    "actual_surface_pressure_mean_lag",
    "actual_precipitation_sum_roll",
    "actual_temperature_2m_mean_roll",
    "actual_relative_humidity_2m_mean_roll",
    "actual_wind_speed_10m_mean_roll",
    "actual_surface_pressure_mean_roll",
    "rainfall_",
    "wet_day_",
    "calendar_",
    "historical_",
)

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

def resolve(root, state, district, block):
    hits = []
    ns, nd, nb = norm(state), norm(district), norm(block)
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        parts = [norm(x) for x in p.parts]
        if ns in parts and nd in parts and nb in parts:
            score = (p.name == block) * 10 + (p.parent.name == district) * 5
            hits.append((score, len(p.parts), p))
    if not hits:
        return None
    hits.sort(key=lambda x: (-x[0], x[1], str(x[2]).lower()))
    return hits[0][2]

def load_dictionary(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if "column" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        c = str(row["column"]).strip()
        result[c] = {
            "feature_group": str(row.get("feature_group", "")).strip().lower(),
            "final_role": str(row.get("final_role", "")).strip().lower(),
            "leakage_risk": str(row.get("leakage_risk", "")).strip().lower(),
        }
    return result

def hard_excluded(c):
    low = c.lower()
    if low in {x.lower() for x in HARD_EXCLUDE_EXACT}:
        return True
    if low in {x.lower() for x in TARGETS}:
        return True
    return any(x in low for x in HARD_EXCLUDE_SUBSTRINGS)

def forecast_predictor(c, info, h):
    if hard_excluded(c):
        return False
    low = c.lower()
    suffix = f"_d{h}"
    if not low.endswith(suffix):
        return False

    # Metadata names are excluded regardless of dictionary classification.
    if any(x in low for x in (
        "lead", "issued_date", "issue_date", "run", "retrieved",
        "rule", "source", "model"
    )):
        return False

    # Prefer the feature dictionary's explicit forecast classification.
    group = info.get("feature_group", "")
    role = info.get("final_role", "")
    if group == "weather_forecast" and role == "forecast_predictor":
        return True

    # Conservative fallback: suffix + meteorological value name.
    tokens = (
        "temperature", "temp", "humidity", "precip", "rain",
        "wind", "pressure", "cloud", "radiation", "snow"
    )
    return any(t in low for t in tokens)

def historical_predictor(c, info):
    if hard_excluded(c):
        return False
    low = c.lower()
    if low.startswith(SAFE_HISTORICAL_PREFIXES):
        return True
    return (
        info.get("feature_group") in {"historical_weather", "temporal_control"}
        and info.get("final_role") in {
            "historical_source", "temporal_alignment", "candidate_predictor"
        }
        and "high" not in info.get("leakage_risk", "")
        and "leak" not in info.get("leakage_risk", "")
    )

def build(source, dictionary, h):
    issue = f"forecast_issued_date_d{h}"
    lead = f"forecast_lead_days_d{h}"
    if issue not in source.columns or lead not in source.columns:
        raise RuntimeError(f"D{h}: missing issue/lead columns")

    dates = pd.to_datetime(source["date"], errors="coerce")
    issues = pd.to_datetime(source[issue], errors="coerce")
    leads = pd.to_numeric(source[lead], errors="coerce")

    if dates.isna().any() or issues.isna().any():
        raise RuntimeError(f"D{h}: invalid dates")
    if not ((dates - issues).dt.days == h).all():
        raise RuntimeError(f"D{h}: target-issue delta is not exactly {h}")
    if not (leads == h).all():
        raise RuntimeError(f"D{h}: lead is not exactly {h}")

    forecast = []
    historical = []

    for c in source.columns:
        info = dictionary.get(c, {})
        if forecast_predictor(c, info, h):
            forecast.append(c)
        elif historical_predictor(c, info):
            historical.append(c)

    # Remove any overlap and enforce hard exclusions one final time.
    selected = []
    for c in forecast + historical:
        if c not in selected and not hard_excluded(c):
            selected.append(c)

    if not selected:
        raise RuntimeError(f"D{h}: no safe predictors selected")

    X = source[selected].copy()
    y = source[TARGETS].copy()

    # Numeric conversion; +/- infinity becomes missing ONLY in the new X.
    inf_replaced = {}
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        mask = np.isinf(X[c].to_numpy(dtype=float))
        count = int(mask.sum())
        if count:
            inf_replaced[c] = count
            X.loc[mask, c] = np.nan

    # Hard final validation.
    forbidden = [
        c for c in X.columns
        if hard_excluded(c)
        or any(x in c.lower() for x in HARD_EXCLUDE_SUBSTRINGS)
        or "forecast_lead_days_" in c.lower()
    ]
    if forbidden:
        raise RuntimeError(f"D{h}: forbidden columns survived: {forbidden}")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise RuntimeError(f"D{h}: nonnumeric predictor survived")

    numeric = X.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"D{h}: non-finite value survived")

    all_missing = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_missing:
        raise RuntimeError(f"D{h}: all-missing predictors: {all_missing}")

    # Chronological 70/15/15.
    order = np.argsort(dates.to_numpy())
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)

    n = len(X)
    ntr = int(n * 0.70)
    nv = int(n * 0.15)
    cuts = {
        "train": (0, ntr),
        "validation": (ntr, ntr + nv),
        "test": (ntr + nv, n),
    }

    split = {}
    for name, (a, b) in cuts.items():
        split[name] = {
            "X": X.iloc[a:b].copy(),
            "y": y.iloc[a:b].copy(),
            "dates": dates.iloc[a:b].copy(),
        }

    return split, {
        "predictors": selected,
        "forecast_predictors": forecast,
        "historical_predictors": historical,
        "inf_to_nan": inf_replaced,
        "split": {
            k: {
                "rows": len(v["X"]),
                "start": str(v["dates"].min().date()),
                "end": str(v["dates"].max().date()),
            }
            for k, v in split.items()
        },
    }

def main():
    print("=" * 78)
    print("SIH26 — TRUE FINAL LEAKAGE-SAFE TRAINING DATA BUILDER")
    print("=" * 78)
    print("IMPORTANT: this run REBUILDS X/Y from engineered SOURCE.")
    print("It does NOT reuse the existing datasets/training X files.")

    state = input("\nState: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    engineered = resolve(ENGINEERED_ROOT, state, district, block)
    if engineered is None:
        raise RuntimeError("Could not resolve engineered location")

    source_path = engineered / "block_forecast_training_base_engineered.csv"
    if not source_path.exists():
        raise RuntimeError(f"Missing source: {source_path}")

    dictionary_path = (
        PROJECT_ROOT / "datasets" / "feature_dictionary" /
        state / district / block / "feature_dictionary.csv"
    )
    dictionary = load_dictionary(dictionary_path)
    source = pd.read_csv(source_path, low_memory=False)

    print(f"\nSource: {source_path}")
    print(f"Source shape: {source.shape}")
    print(f"Feature dictionary entries: {len(dictionary)}")

    final = TRAINING_ROOT / state / district / block
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix="TRUE_FINAL_", dir=str(TRAINING_ROOT)))
    temp = temp_parent / block
    temp.mkdir(parents=True)

    manifest = {
        "builder_version": "TRUE_FINAL_v4",
        "state": state,
        "district": district,
        "block": block,
        "source": str(source_path),
        "source_modified": False,
        "horizons": {},
        "policy": {
            "actual_weather_y_only": True,
            "forecast_metadata_never_x": True,
            "imputation": False,
            "interpolation": False,
            "synthetic": False,
            "unverified_era5_s1_s2_lst_excluded": True,
        },
    }

    try:
        for h in range(1, 8):
            print("\n" + "-" * 78)
            print(f"REBUILDING D{h}")
            print("-" * 78)
            split, meta = build(source, dictionary, h)

            hdir = temp / f"D{h}"
            hdir.mkdir()

            for name, d in split.items():
                d["X"].to_csv(hdir / f"X_{name}.csv", index=False)
                d["y"].to_csv(hdir / f"y_{name}.csv", index=False)

            pd.DataFrame({
                "date": pd.concat(
                    [d["dates"] for d in split.values()], ignore_index=True
                )
            }).to_csv(hdir / "sample_dates.csv", index=False)

            manifest["horizons"][f"D{h}"] = meta

            print(f"X predictors: {len(meta['predictors'])}")
            print("Forecast predictors:", meta["forecast_predictors"])
            print("Historical predictors:", len(meta["historical_predictors"]))
            print("inf -> NaN:", meta["inf_to_nan"])
            print("Split:", meta["split"])

        rows = []
        for hk, m in manifest["horizons"].items():
            for c in m["predictors"]:
                rows.append({
                    "horizon": hk,
                    "feature": c,
                    "role": (
                        "forecast" if c in m["forecast_predictors"]
                        else "historical_temporal"
                    ),
                })
        pd.DataFrame(rows).to_csv(temp / "feature_contract.csv", index=False)

        (temp / "training_dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Remove ONLY the previously generated training output.
        if final.exists():
            shutil.rmtree(final)
        temp.rename(final)
        shutil.rmtree(temp_parent, ignore_errors=True)

        print("\n" + "=" * 78)
        print("TRUE FINAL BUILD COMPLETED")
        print("=" * 78)
        print(f"Output: {final}")
        print("Source/cleaned/engineered data were NOT modified.")
        print("Now run the audit.")
        return 0

    except Exception:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise

if __name__ == "__main__":
    sys.exit(main())
