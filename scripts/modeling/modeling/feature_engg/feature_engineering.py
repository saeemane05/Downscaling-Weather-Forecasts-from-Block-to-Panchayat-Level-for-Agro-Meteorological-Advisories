from pathlib import Path
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# SIH26 UNIVERSAL FEATURE ENGINEERING
# ============================================================
# Location is supplied by the user at runtime.
# No state/district/block is hardcoded.
#
# Inputs:
#   datasets/cleaned/<state>/<district>/<block>/block/*_model_features.csv
#   datasets/cleaned/<state>/<district>/<block>/gp/*_model_features.csv
#   datasets/feature_dictionary/<state>/<district>/<block>/feature_dictionary.csv
#
# Outputs:
#   datasets/engineered/<state>/<district>/<block>/block/
#   datasets/engineered/<state>/<district>/<block>/gp/
#
# Principles:
#   - no synthetic data
#   - no interpolation
#   - no blind imputation
#   - no future-target leakage
#   - historical weather features use only shifted/past values
#   - forecast variables remain aligned to their existing issue/lead structure
#   - static GP features remain spatial predictors
#   - source/model-feature datasets are never overwritten
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEANED_ROOT = PROJECT_ROOT / "datasets" / "cleaned"
DICTIONARY_ROOT = PROJECT_ROOT / "datasets" / "feature_dictionary"
ENGINEERED_ROOT = PROJECT_ROOT / "datasets" / "engineered"


# ============================================================
# LOCATION RESOLUTION
# ============================================================

def norm(value):
    return "".join(c.lower() for c in str(value) if c.isalnum())


def resolve_location(root, state, district, block):
    ws, wd, wb = norm(state), norm(district), norm(block)
    matches = []

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if norm(path.name) != wb:
            continue
        district_dir = path.parent
        state_dir = district_dir.parent
        if norm(district_dir.name) == wd and norm(state_dir.name) == ws:
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"Could not resolve location: {state} / {district} / {block}\n"
            f"Searched under: {root}"
        )

    return sorted(matches)[0]


# ============================================================
# FILE RESOLUTION
# ============================================================

def model_file(location_root, folder, stem):
    p = location_root / folder / f"{stem}_model_features.csv"
    if p.exists():
        return p

    p = location_root / folder / f"{stem}.csv"
    if p.exists():
        return p

    raise FileNotFoundError(f"Required dataset not found: {p}")


# ============================================================
# DICTIONARY
# ============================================================

def load_dictionary(location_root):
    """
    Load the feature dictionary using the physically resolved
    State/District/Block directory names.

    This prevents a user typo or spelling variation in the runtime
    input from creating a wrong output path or failing to find the
    already-generated dictionary.
    """

    state_dir = location_root.parent.parent
    district_dir = location_root.parent
    block_dir = location_root

    path = (
        DICTIONARY_ROOT
        / state_dir.name
        / district_dir.name
        / block_dir.name
        / "feature_dictionary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Feature dictionary not found for the resolved location. Run:\n"
            "python scripts\\modeling\\feature_engg\\build_feature_dictionary.py\n\n"
            f"Expected dictionary: {path}"
        )

    df = pd.read_csv(path)
    required = {"column", "source_dataset", "feature_group", "final_role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature dictionary missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# COLUMN HELPERS
# ============================================================

WEATHER_WORDS = (
    "rain", "rainfall", "precip", "temperature", "temp",
    "humidity", "relative_humidity", "wind", "pressure",
    "dewpoint", "dew_point", "radiation", "cloud", "evapotranspiration",
    "et0"
)


def is_weather_column(name):
    n = str(name).lower()
    return any(x in n for x in WEATHER_WORDS)


def is_rain_column(name):
    n = str(name).lower()
    return "rain" in n or "rainfall" in n or "precip" in n


def is_temperature_column(name):
    n = str(name).lower()
    return "temperature" in n or re.search(r"(^|_)temp($|_)", n) is not None


def is_humidity_column(name):
    n = str(name).lower()
    return "humidity" in n or "relative_humidity" in n


def is_wind_column(name):
    n = str(name).lower()
    return "wind" in n


def is_pressure_column(name):
    n = str(name).lower()
    return "pressure" in n


def is_dewpoint_column(name):
    n = str(name).lower()
    return "dewpoint" in n or "dew_point" in n


def is_date_like(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8
    return False


def choose_date_column(df):
    preferred = [
        "target_date", "date", "valid_date", "forecast_date",
        "target_time", "valid_time", "datetime", "timestamp"
    ]

    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    candidates = [c for c in df.columns if is_date_like(df[c])]
    if candidates:
        return candidates[0]

    return None


def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ============================================================
# PRIMARY HISTORICAL WEATHER SERIES
# ============================================================

def select_primary_weather_columns(df, dictionary):
    """
    Select a small scientifically meaningful set of observed weather
    variables for temporal feature engineering.

    We do NOT create lags for every column. This avoids uncontrolled
    feature explosion when min/mean/max/statistical variants coexist.
    """

    actual = dictionary[
        (dictionary["source_dataset"] == "block_observations_daily")
        & (dictionary["feature_group"] == "historical_weather")
        & (~dictionary["all_missing_100pct"].astype(bool))
    ]

    cols = [c for c in actual["column"].tolist() if c in df.columns]

    families = [
        ("rain", is_rain_column),
        ("temperature", is_temperature_column),
        ("humidity", is_humidity_column),
        ("wind", is_wind_column),
        ("pressure", is_pressure_column),
        ("dewpoint", is_dewpoint_column),
    ]

    selected = []

    for family, predicate in families:
        candidates = [c for c in cols if predicate(c)]
        if not candidates:
            continue

        def score(c):
            n = c.lower()
            score_value = 0
            if "mean" in n:
                score_value += 50
            if "avg" in n or "average" in n:
                score_value += 40
            if "2m" in n:
                score_value += 20
            if "sum" in n or "total" in n:
                score_value += 10
            if "max" in n or "min" in n:
                score_value -= 10
            if "std" in n or "median" in n:
                score_value -= 20
            return score_value

        selected.append(max(candidates, key=score))

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(selected))


# ============================================================
# BLOCK TEMPORAL FEATURE ENGINEERING
# ============================================================

def add_calendar_features(df, date_col):
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")

    if dt.notna().sum() == 0:
        return out

    # Cyclic calendar representation avoids treating December and January
    # as far apart in the feature space.
    doy = dt.dt.dayofyear.astype(float)
    month = dt.dt.month.astype(float)

    out["time__day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["time__day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["time__month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["time__month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return out


def add_weather_history_features(df, date_col, primary_columns):
    out = df.copy()

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col, kind="stable").reset_index(drop=True)

    created = []

    for col in primary_columns:
        if col not in out.columns:
            continue

        values = pd.to_numeric(out[col], errors="coerce")
        past = values.shift(1)

        # Explicit past lags. Shift happens before any rolling operation,
        # so the current target day's observation can never enter the feature.
        for lag in (1, 3, 7):
            name = f"hist__{col}__lag_{lag}d"
            out[name] = values.shift(lag)
            created.append(name)

        # Past-only rolling windows.
        for window in (3, 7, 14):
            mean_name = f"hist__{col}__rolling_mean_{window}d"
            std_name = f"hist__{col}__rolling_std_{window}d"

            out[mean_name] = past.rolling(
                window=window,
                min_periods=window,
            ).mean()

            out[std_name] = past.rolling(
                window=window,
                min_periods=window,
            ).std()

            created.extend([mean_name, std_name])

        # Rainfall gets physically meaningful accumulation and wet-day count.
        if is_rain_column(col):
            for window in (3, 7, 14):
                sum_name = f"hist__{col}__rolling_sum_{window}d"
                wet_name = f"hist__{col}__wet_days_{window}d"

                out[sum_name] = past.rolling(
                    window=window,
                    min_periods=window,
                ).sum()

                out[wet_name] = (
                    past.gt(0)
                    .rolling(window=window, min_periods=window)
                    .sum()
                )

                created.extend([sum_name, wet_name])

    return out, created


# ============================================================
# GP SPATIAL FEATURE ENGINEERING
# ============================================================

def add_gp_relative_features(df, dictionary):
    """
    Add conservative within-block spatial anomaly features.

    For continuous GP environmental/remote-sensing variables:
        GP anomaly = GP value - block median

    This is valid because all values are static/spatial observations
    belonging to the same block; it does not use a future weather target.
    """

    out = df.copy()
    created = []

    eligible_groups = {
        "remote_sensing",
        "static_spatial_environment",
    }

    eligible = dictionary[
        dictionary["feature_group"].isin(eligible_groups)
        & (dictionary["final_role"] == "gp_spatial_predictor")
        & (~dictionary["all_missing_100pct"].astype(bool))
    ]

    for col in eligible["column"].tolist():
        if col not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue

        values = pd.to_numeric(out[col], errors="coerce")

        if values.notna().sum() < 2:
            continue

        if values.nunique(dropna=True) <= 1:
            continue

        median = values.median(skipna=True)
        name = f"spatial__{col}__block_median_anomaly"
        out[name] = values - median
        created.append(name)

    return out, created


# ============================================================
# VALIDATION
# ============================================================

def validate_output(df, source_df, engineered_columns):
    problems = []

    if len(df) != len(source_df):
        problems.append(
            f"Row count changed: source={len(source_df)}, output={len(df)}"
        )

    if df.index.has_duplicates:
        problems.append("Output index contains duplicates.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        problems.append(
            f"Duplicate output columns: {duplicate_columns}"
        )

    if engineered_columns:
        for col in engineered_columns:
            if col not in df.columns:
                problems.append(
                    f"Expected engineered column missing: {col}"
                )

    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        problems.append("Output contains +/-inf.")

    return problems


# ============================================================
# WRITE ENGINEERING MANIFEST
# ============================================================

def write_manifest(output_dir, location, operations, inputs):
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "location": location,
        "inputs": inputs,
        "operations": operations,
        "principles": [
            "No synthetic data",
            "No interpolation",
            "No blind imputation",
            "Historical weather features use only past observations",
            "Forecast features retain issue/lead alignment",
            "Static GP features are spatial predictors",
            "Source datasets are never overwritten",
        ],
    }

    path = output_dir / "feature_engineering_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SIH26 FEATURE ENGINEERING")
    print("=" * 70)
    print("\nThis is the leakage-safe feature-engineering stage.")
    print("Source datasets will NOT be overwritten.")

    state = input("\nState: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()

    if not state or not district or not block:
        raise ValueError("State, District and Block are required.")

    cleaned_root = resolve_location(
        CLEANED_ROOT, state, district, block
    )

    print(f"\nResolved cleaned location:\n  {cleaned_root}")

    # Always use the actual physical directory names after resolution.
    # This keeps the pipeline universal and prevents input spelling
    # variations from creating incorrect output directories.
    physical_state = cleaned_root.parent.parent.name
    physical_district = cleaned_root.parent.name
    physical_block = cleaned_root.name

    dictionary = load_dictionary(cleaned_root)

    output_root = (
        ENGINEERED_ROOT
        / physical_state
        / physical_district
        / physical_block
    )
    block_out = output_root / "block"
    gp_out = output_root / "gp"
    block_out.mkdir(parents=True, exist_ok=True)
    gp_out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # BLOCK OBSERVATIONS
    # --------------------------------------------------------
    obs_path = model_file(
        cleaned_root, "block", "block_observations_daily"
    )
    obs = pd.read_csv(obs_path)

    date_col = choose_date_column(obs)
    if date_col is None:
        raise ValueError(
            "Could not identify a date column in block_observations_daily."
        )

    primary_weather = select_primary_weather_columns(
        obs, dictionary
    )

    print("\nPrimary historical weather variables selected:")
    for c in primary_weather:
        print(f"  {c}")

    obs_eng = add_calendar_features(obs, date_col)
    obs_eng, history_cols = add_weather_history_features(
        obs_eng, date_col, primary_weather
    )

    obs_output = block_out / "block_observations_daily_engineered.csv"
    obs_eng.to_csv(obs_output, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # BLOCK FORECAST TRAINING BASE
    # --------------------------------------------------------
    forecast_path = model_file(
        cleaned_root, "block", "block_forecast_training_base"
    )
    forecast = pd.read_csv(forecast_path)

    forecast_date_col = choose_date_column(forecast)
    if forecast_date_col is None:
        raise ValueError(
            "Could not identify a date column in block_forecast_training_base."
        )

    forecast_eng = add_calendar_features(
        forecast, forecast_date_col
    )

    forecast_output = (
        block_out / "block_forecast_training_base_engineered.csv"
    )
    forecast_eng.to_csv(
        forecast_output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # CURRENT FORECAST
    # --------------------------------------------------------
    current_path = model_file(
        cleaned_root, "block", "block_current_forecast"
    )
    current = pd.read_csv(current_path)

    current_date_col = choose_date_column(current)
    if current_date_col is not None:
        current_eng = add_calendar_features(
            current, current_date_col
        )
    else:
        current_eng = current.copy()

    current_output = (
        block_out / "block_current_forecast_engineered.csv"
    )
    current_eng.to_csv(
        current_output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # GP DATA
    # --------------------------------------------------------
    gp_path = model_file(
        cleaned_root, "gp", "gp_merged_features"
    )
    gp = pd.read_csv(gp_path)

    gp_eng, spatial_cols = add_gp_relative_features(
        gp, dictionary
    )

    gp_output = gp_out / "gp_merged_features_engineered.csv"
    gp_eng.to_csv(
        gp_output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------
    checks = {
        "block_observations_daily": validate_output(
            obs_eng, obs,
            [
                "time__day_of_year_sin",
                "time__day_of_year_cos",
            ] + history_cols,
        ),
        "block_forecast_training_base": validate_output(
            forecast_eng, forecast, []
        ),
        "block_current_forecast": validate_output(
            current_eng, current, []
        ),
        "gp_merged_features": validate_output(
            gp_eng, gp, spatial_cols
        ),
    }

    errors = {
        k: v for k, v in checks.items() if v
    }

    if errors:
        print("\nFEATURE ENGINEERING VALIDATION FAILED")
        for dataset, problems in errors.items():
            print(f"\n{dataset}:")
            for problem in problems:
                print(f"  - {problem}")
        raise RuntimeError(
            "Feature-engineered output failed validation."
        )

    operations = {
        "block_observations_daily": {
            "calendar_features": 4,
            "primary_weather_variables": primary_weather,
            "historical_feature_count": len(history_cols),
            "method": "past-only lags and rolling statistics",
        },
        "block_forecast_training_base": {
            "calendar_features": 4,
            "method": "cyclic calendar encoding only",
        },
        "block_current_forecast": {
            "calendar_features": 4 if current_date_col else 0,
            "method": "cyclic calendar encoding only when date available",
        },
        "gp_merged_features": {
            "spatial_anomaly_feature_count": len(spatial_cols),
            "method": "GP value minus block median",
        },
    }

    write_manifest(
        output_root,
        {
            "state": physical_state,
            "district": physical_district,
            "block": physical_block,
        },
        operations,
        {
            "block_observations_daily": str(obs_path),
            "block_forecast_training_base": str(forecast_path),
            "block_current_forecast": str(current_path),
            "gp_merged_features": str(gp_path),
        },
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"\nLocation: {physical_state} / {physical_district} / {physical_block}")

    print("\nOutputs:")
    print(f"  {obs_output}")
    print(f"  {forecast_output}")
    print(f"  {current_output}")
    print(f"  {gp_output}")
    print(f"  {output_root / 'feature_engineering_manifest.json'}")

    print("\nCreated:")
    print(f"  Historical weather features: {len(history_cols)}")
    print(f"  GP spatial anomaly features: {len(spatial_cols)}")
    print("  Calendar features: 4")

    print("\nValidation: PASS")
    print("No source dataset was overwritten.")


if __name__ == "__main__":
    main()
