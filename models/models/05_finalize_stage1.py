from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent

DATASETS = PROJECT_ROOT / "datasets"
TRAINING_ROOT = DATASETS / "training" / "Maharashtra" / "Nashik" / "Sinnar"
VALIDATION_ROOT = DATASETS / "validation" / "Maharashtra" / "Nashik" / "Sinnar"

OUTPUT_ROOT = VALIDATION_ROOT / "stage1_final"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FINAL_DATA_ROOT = DATASETS / "stage1_final"
FINAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = range(1, 8)

TARGETS = {
    "temperature": "actual_temperature_2m_mean",
    "humidity": "actual_relative_humidity_2m_mean",
    "precipitation": "actual_precipitation_sum",
    "wind": "actual_wind_speed_10m_mean",
    "pressure": "actual_surface_pressure_mean",
}

FORECAST_PATTERNS = {
    "temperature": "temp_mean_forecast_d{h}",
    "humidity": "rh_mean_forecast_d{h}",
    "precipitation": "precip_forecast_d{h}",
    "wind": "wind_speed_forecast_d{h}",
    "pressure": "pressure_forecast_d{h}",
}


# ============================================================
# UTILITIES
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def find_training_files(horizon):
    """
    Find X_train and y_train for a horizon.
    Handles small naming variations.
    """
    hroot = TRAINING_ROOT / f"D{horizon}"

    candidates_x = [
        hroot / "X_train.csv",
        hroot / f"X_train_D{horizon}.csv",
    ]

    candidates_y = [
        hroot / "y_train.csv",
        hroot / f"y_train_D{horizon}.csv",
    ]

    xfile = next((p for p in candidates_x if p.exists()), None)
    yfile = next((p for p in candidates_y if p.exists()), None)

    return xfile, yfile


def find_test_files(horizon):
    hroot = TRAINING_ROOT / f"D{horizon}"

    candidates_x = [
        hroot / "X_test.csv",
        hroot / f"X_test_D{horizon}.csv",
    ]

    candidates_y = [
        hroot / "y_test.csv",
        hroot / f"y_test_D{horizon}.csv",
    ]

    xfile = next((p for p in candidates_x if p.exists()), None)
    yfile = next((p for p in candidates_y if p.exists()), None)

    return xfile, yfile


def is_wind_column(c):
    c = str(c).lower()
    return (
        "wind" in c
        or "windspeed" in c
        or "wind_speed" in c
    )


def is_pressure_column(c):
    c = str(c).lower()
    return (
        "pressure" in c
        or "surface_pressure" in c
        or "pres" in c
    )


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 78)
print("STAGE 1 FINALIZATION")
print("=" * 78)

results = []
wind_diagnostics = []
pressure_inventory = []

models = {}

for h in HORIZONS:

    print(f"\n--- D{h} ---")

    x_train_file, y_train_file = find_training_files(h)

    if x_train_file is None or y_train_file is None:
        print("Training files not found.")
        continue

    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    forecast_cols = [
        c for c in X_train.columns
        if "forecast" in str(c).lower()
    ]

    wind_cols = [
        c for c in X_train.columns
        if is_wind_column(c)
    ]

    pressure_cols = [
        c for c in X_train.columns
        if is_pressure_column(c)
    ]

    pressure_inventory.append({
        "horizon": h,
        "pressure_columns": "|".join(map(str, pressure_cols)),
        "pressure_count": len(pressure_cols),
    })

    print("Wind columns:")
    print(wind_cols)

    print("Pressure columns:")
    print(pressure_cols)

    # ========================================================
    # 2. WIND UNIT DIAGNOSTIC
    # ========================================================

    wind_forecast_col = f"wind_speed_forecast_d{h}"
    wind_actual_col = TARGETS["wind"]

    if (
        wind_forecast_col in X_train.columns
        and wind_actual_col in y_train.columns
    ):

        raw_wind = pd.to_numeric(
            X_train[wind_forecast_col],
            errors="coerce"
        )

        actual_wind = pd.to_numeric(
            y_train[wind_actual_col],
            errors="coerce"
        )

        mask = raw_wind.notna() & actual_wind.notna()

        raw = raw_wind[mask]
        actual = actual_wind[mask]

        raw_mean = raw.mean()
        actual_mean = actual.mean()

        ratio = (
            actual_mean / raw_mean
            if raw_mean != 0
            else np.nan
        )

        # Check the characteristic m/s <-> km/h factor.
        factor_error = abs(ratio - 3.6)

        likely_unit_conversion = factor_error < 0.25

        wind_diagnostics.append({
            "horizon": h,
            "raw_mean": raw_mean,
            "actual_mean": actual_mean,
            "actual_raw_ratio": ratio,
            "difference_from_3_6": factor_error,
            "possible_ms_to_kmh_mismatch":
                bool(likely_unit_conversion),
            "raw_rmse_original_units":
                rmse(actual, raw),
            "raw_mae_original_units":
                mean_absolute_error(actual, raw),
            "raw_rmse_after_actual_to_ms":
                rmse(actual / 3.6, raw),
            "raw_mae_after_actual_to_ms":
                mean_absolute_error(actual / 3.6, raw),
        })

        print(
            f"Wind raw mean={raw_mean:.4f}, "
            f"actual mean={actual_mean:.4f}, "
            f"ratio={ratio:.4f}"
        )

        if likely_unit_conversion:
            print(
                "WARNING: wind values strongly suggest "
                "m/s vs km/h mismatch."
            )

    # ========================================================
    # 3. TRAIN ONLY VALIDATED STAGE-1 CORRECTIONS
    # ========================================================

    # Temperature:
    # Keep raw forecast.
    #
    # Humidity:
    # Keep raw forecast.
    #
    # Wind:
    # DO NOT train/use correction until units are resolved.
    #
    # Rain:
    # Specialized model retained.
    #
    # Pressure:
    # No forecast predictor currently available.

    # --------------------------------------------------------
    # Rainfall specialized correction
    # --------------------------------------------------------

    rain_forecast_col = f"precip_forecast_d{h}"
    rain_target_col = TARGETS["precipitation"]

    if (
        rain_forecast_col in X_train.columns
        and rain_target_col in y_train.columns
    ):

        rain_X = X_train[[rain_forecast_col]].copy()

        rain_y = pd.to_numeric(
            y_train[rain_target_col],
            errors="coerce"
        )

        mask = rain_y.notna()

        rain_X = rain_X.loc[mask]
        rain_y = rain_y.loc[mask]

        imputer = SimpleImputer(strategy="median")
        rain_X_imp = imputer.fit_transform(rain_X)

        # ----------------------------------------------------
        # Occurrence model
        # ----------------------------------------------------

        occurrence_y = (rain_y > 0.1).astype(int)

        occurrence_model = ExtraTreesRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        occurrence_model.fit(
            rain_X_imp,
            occurrence_y
        )

        # ----------------------------------------------------
        # Amount model
        # ----------------------------------------------------

        positive_mask = rain_y > 0.1

        amount_model = ExtraTreesRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        amount_model.fit(
            rain_X_imp[positive_mask.values],
            rain_y.loc[positive_mask]
        )

        models[h] = {
            "rain_imputer": imputer,
            "rain_occurrence": occurrence_model,
            "rain_amount": amount_model,
        }

        print("Rainfall correction model trained.")

    else:
        print("Rainfall columns unavailable.")

    # ========================================================
    # 4. FINAL STAGE-1 STATUS
    # ========================================================

    results.append({
        "horizon": h,
        "temperature": "RAW_FORECAST",
        "humidity": "RAW_FORECAST",
        "precipitation": "SPECIALIZED_CORRECTION",
        "wind": "RAW_FORECAST_PENDING_UNIT_VERIFICATION",
        "pressure": "UNAVAILABLE_PENDING_FORECAST_SOURCE",
    })


# ============================================================
# 5. SAVE DIAGNOSTICS
# ============================================================

wind_diag_df = pd.DataFrame(wind_diagnostics)
wind_diag_file = OUTPUT_ROOT / "wind_unit_diagnostic.csv"
wind_diag_df.to_csv(wind_diag_file, index=False)

pressure_df = pd.DataFrame(pressure_inventory)
pressure_file = OUTPUT_ROOT / "pressure_forecast_inventory.csv"
pressure_df.to_csv(pressure_file, index=False)

status_df = pd.DataFrame(results)
status_file = OUTPUT_ROOT / "stage1_final_status.csv"
status_df.to_csv(status_file, index=False)


# ============================================================
# 6. FINAL DECISION LOG
# ============================================================

wind_conversion_supported = False

if not wind_diag_df.empty:
    valid = wind_diag_df[
        wind_diag_df["possible_ms_to_kmh_mismatch"] == True
    ]

    if len(valid) >= 3:
        wind_conversion_supported = True


decision = {
    "temperature": {
        "decision": "RAW_FORECAST",
        "reason": (
            "Direct correction degraded test performance "
            "across all horizons."
        ),
    },

    "humidity": {
        "decision": "RAW_FORECAST",
        "reason": (
            "Direct correction degraded test performance "
            "across all horizons."
        ),
    },

    "precipitation": {
        "decision": "SPECIALIZED_CORRECTION",
        "reason": (
            "Specialized occurrence/amount correction "
            "improved D1-D5 but requires continued "
            "validation for D6-D7."
        ),
    },

    "wind": {
        "decision": (
            "UNIT_RECONCILIATION_REQUIRED"
            if wind_conversion_supported
            else "RAW_FORECAST_PENDING_VERIFICATION"
        ),
        "reason": (
            "Observed approximately 3.6 actual/raw ratio, "
            "consistent with a possible m/s-to-km/h mismatch. "
            "No wind correction should be accepted until "
            "source units are verified."
        ),
    },

    "pressure": {
        "decision": "SOURCE_REQUIRED",
        "reason": (
            "No pressure forecast predictor is present in "
            "the current modeling matrix."
        ),
    },
}


decision_file = OUTPUT_ROOT / "stage1_decision.json"

with open(decision_file, "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)


# ============================================================
# 7. REPORT
# ============================================================

print("\n" + "=" * 78)
print("STAGE 1 FINALIZATION RESULT")
print("=" * 78)

print("\nTemperature : RAW FORECAST")
print("Humidity    : RAW FORECAST")
print("Rainfall    : SPECIALIZED CORRECTION")
print(
    "Wind        : "
    + (
        "UNIT RECONCILIATION REQUIRED"
        if wind_conversion_supported
        else "RAW FORECAST PENDING VERIFICATION"
    )
)
print("Pressure    : FORECAST SOURCE REQUIRED")

print("\nFiles created:")
print(f"  {wind_diag_file}")
print(f"  {pressure_file}")
print(f"  {status_file}")
print(f"  {decision_file}")

print("\n" + "=" * 78)
print("IMPORTANT")
print("=" * 78)

print(
    """
Do NOT use the existing wind residual-correction model yet.

The current data strongly suggests a possible 3.6x unit mismatch:
    1 m/s = 3.6 km/h

This must be verified against the original forecast/observation
provenance before any wind model is accepted.

The Stage-2 GP downscaling model should consume:
    corrected/selected block forecast
    +
    GP spatial predictors

It must NOT use block actual weather as a GP target.
"""
)

print("\nStage 1 finalization complete.")