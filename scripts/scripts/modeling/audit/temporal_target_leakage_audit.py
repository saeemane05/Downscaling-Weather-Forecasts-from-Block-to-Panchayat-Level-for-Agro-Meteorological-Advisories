import argparse
from pathlib import Path
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# SIH26 TEMPORAL + TARGET LEAKAGE AUDIT
# ============================================================
# Purpose:
#   Verify the two issues left by the engineered-feature audit:
#   1) historical rolling features are genuinely past-only
#   2) actual_* columns in forecast training are targets/references,
#      not predictors containing future target information
#
# This script DOES NOT modify any dataset.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEANED_ROOT = PROJECT_ROOT / "datasets" / "cleaned"
ENGINEERED_ROOT = PROJECT_ROOT / "datasets" / "engineered"
DICTIONARY_ROOT = PROJECT_ROOT / "datasets" / "feature_dictionary"
REPORT_ROOT = PROJECT_ROOT / "datasets" / "leakage_audit"


def norm(value):
    return "".join(c.lower() for c in str(value) if c.isalnum())


def resolve_location(root, state, district, block):
    ws, wd, wb = norm(state), norm(district), norm(block)
    matches = []
    if not root.exists():
        raise FileNotFoundError(root)
    for path in root.rglob("*"):
        if not path.is_dir() or norm(path.name) != wb:
            continue
        d = path.parent
        s = d.parent
        if norm(d.name) == wd and norm(s.name) == ws:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve {state}/{district}/{block} under {root}"
        )
    return sorted(matches)[0]


def model_file(location_root, folder, stem):
    for name in (f"{stem}_model_features.csv", f"{stem}.csv"):
        p = location_root / folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing dataset: {location_root/folder/stem}")


def choose_date_column(df):
    preferred = [
        "target_date", "date", "valid_date", "forecast_date",
        "target_time", "valid_time", "datetime", "timestamp",
    ]
    lower = {str(c).lower(): c for c in df.columns}
    for x in preferred:
        if x in lower:
            return lower[x]

    candidates = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(c)
            continue
        if s.dtype == object:
            sample = s.dropna().astype(str).head(30)
            if len(sample):
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    candidates.append(c)
    return candidates[0] if candidates else None


def find_time_columns(df):
    names = {}
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ("run", "issue", "init", "forecast")) and any(
            k in n for k in ("date", "time", "timestamp")
        ):
            names.setdefault("run_or_issue", c)
        if any(k in n for k in ("target", "valid", "lead")) and any(
            k in n for k in ("date", "time", "timestamp")
        ):
            names.setdefault("target_or_valid", c)
    return names


def parse_datetime_series(s):
    return pd.to_datetime(s, errors="coerce")


def infer_lead_days(df):
    cols = {str(c).lower(): c for c in df.columns}
    lead = None
    for key in ("lead_time_days", "lead_days", "lead_time", "forecast_lead_days"):
        if key in cols:
            lead = pd.to_numeric(df[cols[key]], errors="coerce")
            break
    return lead


def actual_columns(df):
    return [c for c in df.columns if str(c).lower().startswith("actual_")]


def numeric_close(a, b, tol=1e-8):
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() == 0:
        return 0, 0, np.nan
    diff = np.abs(aa[mask] - bb[mask])
    return int(mask.sum()), int((diff <= tol).sum()), float(np.nanmax(diff))


def verify_rolling_features(clean_obs, engineered_obs):
    """
    Recompute every hist__* rolling feature from the clean source using
    exactly the intended construction: source.shift(1).rolling(...).
    Compare against engineered values by date.

    Returns per-feature diagnostics.
    """
    src = clean_obs.copy()
    eng = engineered_obs.copy()

    src_date = choose_date_column(src)
    eng_date = choose_date_column(eng)
    if not src_date or not eng_date:
        return [], ["Could not identify date column for rolling verification."]

    src[src_date] = pd.to_datetime(src[src_date], errors="coerce")
    eng[eng_date] = pd.to_datetime(eng[eng_date], errors="coerce")

    # Determine source column from hist__<source>__...
    hist_cols = [c for c in eng.columns if str(c).startswith("hist__")]
    results = []
    errors = []

    for feature in hist_cols:
        parts = str(feature).split("__")
        if len(parts) < 3:
            continue
        source_col = parts[1]
        operation = "__".join(parts[2:])

        if source_col not in src.columns:
            results.append({
                "feature": feature,
                "status": "SOURCE_COLUMN_MISSING",
                "max_abs_error": None,
            })
            continue

        values = pd.to_numeric(src[source_col], errors="coerce")
        past = values.shift(1)

        expected = None

        m = re.fullmatch(r"lag_(\d+)d", operation)
        if m:
            expected = values.shift(int(m.group(1)))

        m = re.fullmatch(r"rolling_(mean|std)_(\d+)d", operation)
        if m:
            kind, w = m.group(1), int(m.group(2))
            r = past.rolling(window=w, min_periods=w)
            expected = r.mean() if kind == "mean" else r.std()

        m = re.fullmatch(r"rolling_sum_(\d+)d", operation)
        if m:
            w = int(m.group(1))
            expected = past.rolling(window=w, min_periods=w).sum()

        m = re.fullmatch(r"wet_days_(\d+)d", operation)
        if m:
            w = int(m.group(1))
            expected = past.gt(0).rolling(window=w, min_periods=w).sum()

        if expected is None:
            continue

        check = pd.DataFrame({
            "_date": src[src_date],
            "_expected": expected,
        }).dropna(subset=["_date"])
        check = check.drop_duplicates("_date", keep="last")

        actual = eng[[eng_date, feature]].copy()
        actual["_date"] = actual[eng_date]
        actual = actual.drop_duplicates("_date", keep="last")
        merged = actual.merge(check, on="_date", how="inner")

        a = pd.to_numeric(merged[feature], errors="coerce")
        e = pd.to_numeric(merged["_expected"], errors="coerce")
        mask = a.notna() & e.notna()

        if mask.sum() == 0:
            status = "NO_OVERLAP_NON_NULL"
            max_err = None
            n = 0
        else:
            err = (a[mask] - e[mask]).abs()
            max_err = float(err.max())
            n = int(mask.sum())
            status = "PASS" if bool((err <= 1e-8).all()) else "MISMATCH"

        results.append({
            "feature": feature,
            "source_column": source_col,
            "operation": operation,
            "comparisons": n,
            "max_abs_error": max_err,
            "status": status,
        })

    return results, errors


def compare_actual_columns_to_observations(forecast, observations):
    """
    For each actual_* column, compare values against block observations
    at candidate dates. This helps distinguish target/reference columns
    from arbitrary predictors.

    We report:
      - target-date match if target_date exists
      - run/issue-date match if a run/issue date exists
    No classification is made solely from value matching.
    """
    f = forecast.copy()
    o = observations.copy()

    f_date = choose_date_column(f)
    o_date = choose_date_column(o)
    if not f_date or not o_date:
        return [], ["Could not identify forecast/observation date columns."]

    f[f_date] = pd.to_datetime(f[f_date], errors="coerce").dt.normalize()
    o[o_date] = pd.to_datetime(o[o_date], errors="coerce").dt.normalize()

    # Observation lookup for each actual column.
    actuals = actual_columns(f)
    obs_map = {c: c for c in o.columns}

    time_cols = find_time_columns(f)
    target_col = time_cols.get("target_or_valid")
    run_col = time_cols.get("run_or_issue")

    # Explicitly prefer common target names.
    lower = {str(c).lower(): c for c in f.columns}
    for candidate in ("target_date", "target_time", "valid_date", "valid_time"):
        if candidate in lower:
            target_col = lower[candidate]
            break

    rows = []
    for col in actuals:
        base = str(col)[7:]  # after actual_
        # Try likely observation name mappings.
        candidates = [base, col]
        candidates += [c for c in o.columns if str(c).lower() == base.lower()]
        obs_col = next((c for c in candidates if c in o.columns), None)

        rec = {
            "actual_column": col,
            "observation_column_match": obs_col,
            "target_date_exact_value_match_rate": None,
            "run_date_exact_value_match_rate": None,
            "target_date_comparisons": 0,
            "run_date_comparisons": 0,
        }

        if obs_col is None:
            rows.append(rec)
            continue

        lookup = o[[o_date, obs_col]].copy()
        lookup["_date"] = lookup[o_date]
        lookup[obs_col] = pd.to_numeric(lookup[obs_col], errors="coerce")
        lookup = lookup.drop_duplicates("_date", keep="last").set_index("_date")[obs_col]

        actual_vals = pd.to_numeric(f[col], errors="coerce")

        if target_col:
            dates = pd.to_datetime(f[target_col], errors="coerce").dt.normalize()
            expected = dates.map(lookup)
            mask = actual_vals.notna() & expected.notna()
            if mask.sum():
                rec["target_date_comparisons"] = int(mask.sum())
                rec["target_date_exact_value_match_rate"] = float(
                    np.isclose(actual_vals[mask], expected[mask], rtol=1e-8, atol=1e-8).mean()
                )

        if run_col:
            dates = pd.to_datetime(f[run_col], errors="coerce").dt.normalize()
            expected = dates.map(lookup)
            mask = actual_vals.notna() & expected.notna()
            if mask.sum():
                rec["run_date_comparisons"] = int(mask.sum())
                rec["run_date_exact_value_match_rate"] = float(
                    np.isclose(actual_vals[mask], expected[mask], rtol=1e-8, atol=1e-8).mean()
                )

        rows.append(rec)

    return rows, []


def inspect_forecast_temporal_structure(forecast):
    cols = {str(c).lower(): c for c in forecast.columns}
    target = None
    run = None

    for x in ("target_date", "target_time", "valid_date", "valid_time"):
        if x in cols:
            target = cols[x]
            break
    for x in ("run_date", "run_time", "issue_date", "issue_time",
              "forecast_run_date", "forecast_run_time", "init_date", "init_time"):
        if x in cols:
            run = cols[x]
            break

    result = {
        "target_column": target,
        "run_or_issue_column": run,
        "lead_column": None,
        "target_after_run_all_rows": None,
        "minimum_target_minus_run_days": None,
        "maximum_target_minus_run_days": None,
    }

    lead = infer_lead_days(forecast)
    if lead is not None:
        result["lead_column"] = str(
            next(c for c in forecast.columns
                 if str(c).lower() in (
                     "lead_time_days", "lead_days", "lead_time",
                     "forecast_lead_days"
                 ))
        )

    if target and run:
        t = parse_datetime_series(forecast[target])
        r = parse_datetime_series(forecast[run])
        delta = (t - r).dt.total_seconds() / 86400.0
        valid = delta.dropna()
        if len(valid):
            result["target_after_run_all_rows"] = bool((valid > 0).all())
            result["minimum_target_minus_run_days"] = float(valid.min())
            result["maximum_target_minus_run_days"] = float(valid.max())

    return result


def classify_actual_columns(forecast, observations):
    """
    Conservative classification based on column semantics and date/value
    relationships. This is a review aid, not a replacement for model X/y
    construction.
    """
    temporal = inspect_forecast_temporal_structure(forecast)
    matches, _ = compare_actual_columns_to_observations(forecast, observations)

    out = []
    for rec in matches:
        col = rec["actual_column"]
        n = col.lower()

        if n in ("actual_source", "actual_timezone", "actual_is_reference_only"):
            role = "metadata_or_reference"
            reason = "Metadata/provenance field; should not be a model predictor."
        elif "latitude" in n or "longitude" in n:
            role = "spatial_metadata"
            reason = "Location metadata; keep only if explicitly justified as a predictor."
        elif rec["target_date_exact_value_match_rate"] is not None and rec[
            "target_date_exact_value_match_rate"
        ] >= 0.999:
            role = "likely_target_or_target_reference"
            reason = "Values match block observations at target/valid date."
        elif rec["run_date_exact_value_match_rate"] is not None and rec[
            "run_date_exact_value_match_rate"
        ] >= 0.999:
            role = "likely_available_at_run_time"
            reason = "Values match block observations at forecast run/issue date."
        else:
            role = "REVIEW"
            reason = "Could not establish safe target/reference semantics automatically."

        out.append({**rec, "provisional_role": role, "reason": reason})

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="SIH26 temporal and target leakage audit")
    parser.add_argument("--state")
    parser.add_argument("--district")
    parser.add_argument("--block")
    args = parser.parse_args()
    if any((args.state, args.district, args.block)) and not all((args.state, args.district, args.block)):
        parser.error("--state, --district and --block must be supplied together")
    return args


def main():
    args = parse_args()
    print("=" * 72)
    print("SIH26 TEMPORAL + TARGET LEAKAGE AUDIT")
    print("=" * 72)
    print("No files will be modified.")

    state = args.state or input("\nState: ").strip()
    district = args.district or input("District: ").strip()
    block = args.block or input("Block: ").strip()

    if not all((state, district, block)):
        raise ValueError("State, District and Block are required.")

    print(f"\nState: {state}\nDistrict: {district}\nBlock: {block}")

    cleaned = resolve_location(CLEANED_ROOT, state, district, block)
    engineered = resolve_location(ENGINEERED_ROOT, state, district, block)

    print(f"\nCleaned location:\n  {cleaned}")
    print(f"\nEngineered location:\n  {engineered}")

    obs_path = model_file(cleaned, "block", "block_observations_daily")
    forecast_path = model_file(cleaned, "block", "block_forecast_training_base")
    eng_obs_path = engineered / "block" / "block_observations_daily_engineered.csv"
    eng_forecast_path = engineered / "block" / "block_forecast_training_base_engineered.csv"

    for p in (obs_path, forecast_path, eng_obs_path, eng_forecast_path):
        if not p.exists():
            raise FileNotFoundError(p)

    print("\nLoading datasets...")
    obs = pd.read_csv(obs_path)
    forecast = pd.read_csv(forecast_path)
    eng_obs = pd.read_csv(eng_obs_path)
    eng_forecast = pd.read_csv(eng_forecast_path)

    print(f"  observations source: {obs.shape}")
    print(f"  observations engineered: {eng_obs.shape}")
    print(f"  forecast source: {forecast.shape}")
    print(f"  forecast engineered: {eng_forecast.shape}")

    # --------------------------------------------------------
    # 1. ROLLING/PAST FEATURE RECOMPUTATION
    # --------------------------------------------------------
    rolling_results, rolling_errors = verify_rolling_features(obs, eng_obs)

    # --------------------------------------------------------
    # 2. FORECAST TEMPORAL STRUCTURE
    # --------------------------------------------------------
    temporal = inspect_forecast_temporal_structure(forecast)

    # --------------------------------------------------------
    # 3. ACTUAL COLUMN ANALYSIS
    # --------------------------------------------------------
    actual_analysis = classify_actual_columns(forecast, obs)

    # --------------------------------------------------------
    # 4. ENGINEERED FORECAST COLUMN INVENTORY
    # --------------------------------------------------------
    inherited_actual = [
        c for c in eng_forecast.columns
        if str(c).lower().startswith("actual_")
    ]
    target_candidates = [
        c for c in eng_forecast.columns
        if any(k in str(c).lower() for k in (
            "target", "observed", "actual", "y_", "label"
        ))
    ]

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------
    rolling_mismatches = [
        x for x in rolling_results
        if x.get("status") == "MISMATCH"
    ]
    rolling_missing = [
        x for x in rolling_results
        if x.get("status") == "SOURCE_COLUMN_MISSING"
    ]
    review_actuals = [
        x for x in actual_analysis
        if x["provisional_role"] == "REVIEW"
    ]

    target_after_run = temporal.get("target_after_run_all_rows")

    blockers = []
    if rolling_mismatches:
        blockers.append("Historical engineered features do not match past-only recomputation.")
    if rolling_missing:
        blockers.append("Some historical engineered features have missing source columns.")
    if target_after_run is False:
        blockers.append("Some forecast target timestamps are not after forecast run/issue timestamps.")
    if review_actuals:
        blockers.append(
            "Some actual_* columns could not be assigned safe target/reference semantics automatically."
        )

    status = "PASS" if not blockers else "REVIEW_REQUIRED"

    # --------------------------------------------------------
    # WRITE REPORT
    # --------------------------------------------------------
    out_dir = REPORT_ROOT / state / district / block
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "location": {"state": state, "district": district, "block": block},
        "status": status,
        "blockers": blockers,
        "dataset_shapes": {
            "block_observations_source": list(obs.shape),
            "block_observations_engineered": list(eng_obs.shape),
            "block_forecast_source": list(forecast.shape),
            "block_forecast_engineered": list(eng_forecast.shape),
        },
        "rolling_verification": rolling_results,
        "rolling_errors": rolling_errors,
        "forecast_temporal_structure": temporal,
        "actual_column_analysis": actual_analysis,
        "engineered_forecast_actual_columns": inherited_actual,
        "target_candidate_columns": target_candidates,
        "interpretation": {
            "rolling_features": "PASS only when recomputed from source.shift(1).rolling(...) exactly.",
            "actual_columns": "Actual observations may be legitimate targets/references, but must not enter model predictors X.",
            "training_gate": "Do not train until any REVIEW_REQUIRED items are resolved."
        }
    }

    json_path = out_dir / "temporal_target_leakage_audit.json"
    csv_path = out_dir / "actual_column_review.csv"
    roll_csv = out_dir / "rolling_feature_verification.csv"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(actual_analysis).to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rolling_results).to_csv(roll_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("TEMPORAL + TARGET LEAKAGE AUDIT COMPLETED")
    print("=" * 72)
    print(f"\nStatus: {status}")

    print("\nFORECAST TEMPORAL STRUCTURE:")
    for k, v in temporal.items():
        print(f"  {k}: {v}")

    print(f"\nRolling engineered features checked: {len(rolling_results)}")
    print(f"  Mismatches: {len(rolling_mismatches)}")
    print(f"  Missing source columns: {len(rolling_missing)}")

    print(f"\nactual_* columns reviewed: {len(actual_analysis)}")
    print(f"  Automatic REVIEW items: {len(review_actuals)}")

    print("\nREPORTS:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {roll_csv}")

    if blockers:
        print("\nBLOCKERS:")
        for b in blockers:
            print(f"  - {b}")
        print("\nDo NOT start model training yet.")
    else:
        print("\nAll automated temporal checks passed.")
        print("Next step: final model X/y construction audit before training.")


if __name__ == "__main__":
    main()
