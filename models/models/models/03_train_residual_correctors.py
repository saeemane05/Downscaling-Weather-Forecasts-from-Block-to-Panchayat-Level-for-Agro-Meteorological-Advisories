# ================================================================
# SIH26 - RESIDUAL BLOCK FORECAST CORRECTORS
# File: models/03_train_residual_correctors.py
#
# Method:
#   Temperature : forecast + ML residual correction
#   Humidity    : forecast + ML residual correction
#   Wind        : forecast + ML residual correction
#   Rainfall    : occurrence model + conditional amount model
#   Pressure    : not modeled here
#
# IMPORTANT:
#   - Actual weather is used ONLY as training target/residual target.
#   - Actual weather is NEVER used as an input feature.
#   - Chronological train/validation/test splits are preserved.
#   - Raw forecast remains the starting prediction.
# ================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "training"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
    / "residual_corrector"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "models"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
    / "residual_corrector"
)

HORIZONS = range(1, 8)

RANDOM_STATE = 42
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3


# ================================================================
# VARIABLE DEFINITIONS
# ================================================================

VARIABLES = {
    "temperature": {
        "target": "actual_temperature_2m_mean",
        "forecast": "temp_mean_forecast_d{h}",
        "mode": "residual",
        "clip": None,
    },

    "humidity": {
        "target": "actual_relative_humidity_2m_mean",
        "forecast": "rh_mean_forecast_d{h}",
        "mode": "residual",
        "clip": (0.0, 100.0),
    },

    "precipitation": {
        "target": "actual_precipitation_sum",
        "forecast": "precip_forecast_d{h}",
        "mode": "rainfall",
        "clip": (0.0, None),
    },

    "wind": {
        "target": "actual_wind_speed_10m_mean",
        "forecast": "wind_speed_forecast_d{h}",
        "mode": "residual",
        "clip": (0.0, None),
    },
}


# ================================================================
# HELPERS
# ================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def find_file(directory, candidates):
    """
    Find the first existing file from a list of candidates.
    """
    for name in candidates:
        path = directory / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"\nCould not find any of:\n"
        + "\n".join(str(directory / x) for x in candidates)
    )


def load_split(horizon, split):
    """
    Load X/y split.

    Supports common naming conventions so the script does not depend
    on one exact filename.
    """

    hdir = DATA_ROOT / f"D{horizon}"

    if not hdir.exists():
        raise FileNotFoundError(f"Horizon directory not found: {hdir}")

    x_candidates = [
        f"X_{split}.csv",
        f"x_{split}.csv",
        f"X_{split}.parquet",
        f"x_{split}.parquet",
    ]

    y_candidates = [
        f"y_{split}.csv",
        f"Y_{split}.csv",
        f"y_{split}.parquet",
        f"Y_{split}.parquet",
    ]

    x_path = find_file(hdir, x_candidates)
    y_path = find_file(hdir, y_candidates)

    if x_path.suffix.lower() == ".parquet":
        X = pd.read_parquet(x_path)
    else:
        X = pd.read_csv(x_path)

    if y_path.suffix.lower() == ".parquet":
        y = pd.read_parquet(y_path)
    else:
        y = pd.read_csv(y_path)

    return X, y


def load_split_with_fallback(horizon, split):
    """
    Some pipelines store files directly under D1...D7 while others
    may use slightly different capitalization. This wrapper keeps
    loading centralized.
    """
    return load_split(horizon, split)


# ================================================================
# SAFETY CHECKS
# ================================================================

def check_no_actual_features(X):
    """
    Actual weather must never enter the model input.
    """

    forbidden = []

    for col in X.columns:
        c = str(col).lower()

        if c.startswith("actual_"):
            forbidden.append(col)

    if forbidden:
        raise RuntimeError(
            "\nLEAKAGE SAFETY CHECK FAILED.\n"
            "Actual-weather columns found in X:\n"
            + "\n".join(map(str, forbidden))
        )


def check_alignment(X, y):
    if len(X) != len(y):
        raise RuntimeError(
            f"X/y length mismatch: X={len(X)}, y={len(y)}"
        )

    if not X.index.equals(y.index):
        # Reset only after checking same length.
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)


# ================================================================
# MODEL CREATION
# ================================================================

def make_regressor():
    return ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_classifier():
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ================================================================
# FEATURE PREPARATION
# ================================================================

def prepare_features(X):
    """
    Keep numeric modeling features only.

    Remove obvious IDs / object columns where necessary.
    """

    X = X.copy()

    # Remove actual weather if present.
    actual_cols = [
        c for c in X.columns
        if str(c).lower().startswith("actual_")
    ]

    if actual_cols:
        X = X.drop(columns=actual_cols)

    # Keep numeric columns.
    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    X = X[numeric_cols].copy()

    # Replace infinities.
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def align_feature_columns(X_train, X_val, X_test):
    """
    Ensure all three splits have identical feature columns.
    """

    common = (
        set(X_train.columns)
        & set(X_val.columns)
        & set(X_test.columns)
    )

    common = sorted(common)

    if not common:
        raise RuntimeError("No common numeric feature columns.")

    X_train = X_train[common].copy()
    X_val = X_val[common].copy()
    X_test = X_test[common].copy()

    return X_train, X_val, X_test


# ================================================================
# RESIDUAL MODEL
# ================================================================

def train_residual_model(
    X_train,
    y_train,
    raw_train,
    X_val,
    y_val,
    raw_val,
    X_test,
    y_test,
    raw_test,
    variable,
    horizon,
):
    """
    Learn:

        residual = actual - raw_forecast

    Then:

        corrected = raw_forecast + predicted_residual
    """

    residual_train = y_train - raw_train

    # Safety checks
    if not np.isfinite(residual_train).all():
        raise RuntimeError(
            f"{variable} D{horizon}: non-finite residuals in training."
        )

    model = make_regressor()

    imputer = SimpleImputer(strategy="median")

    Xtr = imputer.fit_transform(X_train)
    Xv = imputer.transform(X_val)
    Xte = imputer.transform(X_test)

    model.fit(Xtr, residual_train)

    residual_val_pred = model.predict(Xv)
    residual_test_pred = model.predict(Xte)

    corrected_val = raw_val + residual_val_pred
    corrected_test = raw_test + residual_test_pred

    return {
        "model": model,
        "imputer": imputer,
        "val_prediction": corrected_val,
        "test_prediction": corrected_test,
        "val_residual_prediction": residual_val_pred,
        "test_residual_prediction": residual_test_pred,
    }


# ================================================================
# RAINFALL MODEL
# ================================================================

def train_rainfall_model(
    X_train,
    y_train,
    raw_train,
    X_val,
    y_val,
    raw_val,
    X_test,
    y_test,
    raw_test,
    horizon,
):
    """
    Two-stage rainfall model.

    Stage 1:
        P(rain > 0)

    Stage 2:
        rainfall amount conditional on rain

    The raw precipitation forecast is included as an input feature.

    Final prediction:
        occurrence probability controls whether rain is predicted,
        while the conditional model predicts amount.

    We also retain the raw forecast so that very small forecast
    values are not unnecessarily converted into large rainfall.
    """

    # ------------------------------------------------------------
    # OCCURRENCE TARGET
    # ------------------------------------------------------------

    rain_threshold = 0.1

    occurrence_train = (
        np.asarray(y_train) > rain_threshold
    ).astype(int)

    occurrence_val = (
        np.asarray(y_val) > rain_threshold
    ).astype(int)

    occurrence_test = (
        np.asarray(y_test) > rain_threshold
    ).astype(int)

    # ------------------------------------------------------------
    # Add raw forecast as explicit feature
    # ------------------------------------------------------------

    Xtr = X_train.copy()
    Xv = X_val.copy()
    Xte = X_test.copy()

    raw_feature = "__raw_precip_forecast"

    Xtr[raw_feature] = raw_train.values
    Xv[raw_feature] = raw_val.values
    Xte[raw_feature] = raw_test.values

    common = sorted(
        set(Xtr.columns)
        & set(Xv.columns)
        & set(Xte.columns)
    )

    Xtr = Xtr[common]
    Xv = Xv[common]
    Xte = Xte[common]

    imputer = SimpleImputer(strategy="median")

    Xtr_i = imputer.fit_transform(Xtr)
    Xv_i = imputer.transform(Xv)
    Xte_i = imputer.transform(Xte)

    # ------------------------------------------------------------
    # STAGE 1: OCCURRENCE
    # ------------------------------------------------------------

    occurrence_model = make_classifier()

    occurrence_model.fit(
        Xtr_i,
        occurrence_train,
    )

    occurrence_probability_val = (
        occurrence_model.predict_proba(Xv_i)[:, 1]
    )

    occurrence_probability_test = (
        occurrence_model.predict_proba(Xte_i)[:, 1]
    )

    # ------------------------------------------------------------
    # STAGE 2: AMOUNT
    # ------------------------------------------------------------

    wet_mask = y_train.values > rain_threshold

    amount_model = None

    if wet_mask.sum() >= 10:

        amount_model = make_regressor()

        # Log transform reduces rainfall skew.
        amount_target = np.log1p(
            y_train.values[wet_mask]
        )

        amount_model.fit(
            Xtr_i[wet_mask],
            amount_target,
        )

        amount_log_val = amount_model.predict(Xv_i)
        amount_log_test = amount_model.predict(Xte_i)

        conditional_amount_val = np.expm1(
            amount_log_val
        )

        conditional_amount_test = np.expm1(
            amount_log_test
        )

        conditional_amount_val = np.maximum(
            conditional_amount_val,
            0,
        )

        conditional_amount_test = np.maximum(
            conditional_amount_test,
            0,
        )

    else:
        # Fallback if the training period has insufficient wet cases.
        conditional_amount_val = np.full(
            len(X_val),
            max(float(y_train[y_train > 0].median()), 0.0)
            if (y_train > 0).any()
            else 0.0,
        )

        conditional_amount_test = np.full(
            len(X_test),
            max(float(y_train[y_train > 0].median()), 0.0)
            if (y_train > 0).any()
            else 0.0,
        )

    # ------------------------------------------------------------
    # FINAL RAINFALL
    # ------------------------------------------------------------

    # Probability-weighted expected amount.
    model_val = (
        occurrence_probability_val
        * conditional_amount_val
    )

    model_test = (
        occurrence_probability_test
        * conditional_amount_test
    )

    # Blend lightly with raw forecast.
    #
    # This prevents the ML model from completely discarding
    # information already contained in the numerical forecast.
    alpha = 0.70

    corrected_val = (
        alpha * model_val
        + (1.0 - alpha) * raw_val.values
    )

    corrected_test = (
        alpha * model_test
        + (1.0 - alpha) * raw_test.values
    )

    corrected_val = np.maximum(corrected_val, 0)
    corrected_test = np.maximum(corrected_test, 0)

    return {
        "occurrence_model": occurrence_model,
        "amount_model": amount_model,
        "imputer": imputer,
        "val_prediction": corrected_val,
        "test_prediction": corrected_test,
        "val_occurrence_probability": occurrence_probability_val,
        "test_occurrence_probability": occurrence_probability_test,
    }


# ================================================================
# SAVE MODEL
# ================================================================

def save_pickle(obj, path):
    import joblib

    ensure_dir(path.parent)
    joblib.dump(obj, path)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("SIH26 RESIDUAL BLOCK FORECAST CORRECTORS")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Model root   : {MODEL_ROOT}")

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(MODEL_ROOT)

    all_results = []
    all_prediction_records = []

    # ============================================================
    # HORIZONS
    # ============================================================

    for horizon in HORIZONS:

        print()
        print("#" * 70)
        print(f"HORIZON D{horizon}")
        print("#" * 70)

        # --------------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------------

        X_train, y_train = load_split_with_fallback(
            horizon,
            "train",
        )

        X_val, y_val = load_split_with_fallback(
            horizon,
            "validation",
        )

        X_test, y_test = load_split_with_fallback(
            horizon,
            "test",
        )

        check_alignment(X_train, y_train)
        check_alignment(X_val, y_val)
        check_alignment(X_test, y_test)

        # --------------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------------

        check_no_actual_features(X_train)
        check_no_actual_features(X_val)
        check_no_actual_features(X_test)

        # --------------------------------------------------------
        # FEATURE PREPARATION
        # --------------------------------------------------------

        X_train_model = prepare_features(X_train)
        X_val_model = prepare_features(X_val)
        X_test_model = prepare_features(X_test)

        (
            X_train_model,
            X_val_model,
            X_test_model,
        ) = align_feature_columns(
            X_train_model,
            X_val_model,
            X_test_model,
        )

        print(
            f"Rows: train={len(X_train_model)}, "
            f"validation={len(X_val_model)}, "
            f"test={len(X_test_model)}"
        )

        print(
            f"Features used: {X_train_model.shape[1]}"
        )

        # --------------------------------------------------------
        # VARIABLES
        # --------------------------------------------------------

        for variable, config in VARIABLES.items():

            target_col = config["target"]
            forecast_col = config["forecast"].format(
                h=horizon
            )

            if target_col not in y_train.columns:
                raise KeyError(
                    f"{target_col} missing from y_train D{horizon}"
                )

            if forecast_col not in X_train.columns:
                raise KeyError(
                    f"{forecast_col} missing from X_train D{horizon}"
                )

            if forecast_col not in X_val.columns:
                raise KeyError(
                    f"{forecast_col} missing from X_validation D{horizon}"
                )

            if forecast_col not in X_test.columns:
                raise KeyError(
                    f"{forecast_col} missing from X_test D{horizon}"
                )

            ytr = pd.to_numeric(
                y_train[target_col],
                errors="coerce",
            )

            yv = pd.to_numeric(
                y_val[target_col],
                errors="coerce",
            )

            yte = pd.to_numeric(
                y_test[target_col],
                errors="coerce",
            )

            raw_tr = pd.to_numeric(
                X_train[forecast_col],
                errors="coerce",
            )

            raw_v = pd.to_numeric(
                X_val[forecast_col],
                errors="coerce",
            )

            raw_te = pd.to_numeric(
                X_test[forecast_col],
                errors="coerce",
            )

            # ----------------------------------------------------
            # VALID ROW MASK
            # ----------------------------------------------------

            train_mask = (
                ytr.notna()
                & raw_tr.notna()
            )

            val_mask = (
                yv.notna()
                & raw_v.notna()
            )

            test_mask = (
                yte.notna()
                & raw_te.notna()
            )

            Xtr = X_train_model.loc[train_mask].copy()
            Xv = X_val_model.loc[val_mask].copy()
            Xte = X_test_model.loc[test_mask].copy()

            ytr_valid = ytr.loc[train_mask]
            yv_valid = yv.loc[val_mask]
            yte_valid = yte.loc[test_mask]

            raw_tr_valid = raw_tr.loc[train_mask]
            raw_v_valid = raw_v.loc[val_mask]
            raw_te_valid = raw_te.loc[test_mask]

            # ----------------------------------------------------
            # TRAIN MODEL
            # ----------------------------------------------------

            print()
            print(f"--- {variable.upper()} ---")

            if config["mode"] == "residual":

                print("Mode: residual correction")

                result = train_residual_model(
                    Xtr,
                    ytr_valid,
                    raw_tr_valid,
                    Xv,
                    yv_valid,
                    raw_v_valid,
                    Xte,
                    yte_valid,
                    raw_te_valid,
                    variable,
                    horizon,
                )

            elif config["mode"] == "rainfall":

                print(
                    "Mode: rainfall occurrence + "
                    "conditional amount"
                )

                result = train_rainfall_model(
                    Xtr,
                    ytr_valid,
                    raw_tr_valid,
                    Xv,
                    yv_valid,
                    raw_v_valid,
                    Xte,
                    yte_valid,
                    raw_te_valid,
                    horizon,
                )

            else:
                raise ValueError(
                    f"Unknown model mode: {config['mode']}"
                )

            # ----------------------------------------------------
            # PHYSICAL CONSTRAINTS
            # ----------------------------------------------------

            corrected_val = np.asarray(
                result["val_prediction"]
            )

            corrected_test = np.asarray(
                result["test_prediction"]
            )

            clip_range = config["clip"]

            if clip_range is not None:

                low, high = clip_range

                if low is not None:
                    corrected_val = np.maximum(
                        corrected_val,
                        low,
                    )
                    corrected_test = np.maximum(
                        corrected_test,
                        low,
                    )

                if high is not None:
                    corrected_val = np.minimum(
                        corrected_val,
                        high,
                    )
                    corrected_test = np.minimum(
                        corrected_test,
                        high,
                    )

            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            raw_val_mae = mean_absolute_error(
                yv_valid,
                raw_v_valid,
            )

            corrected_val_mae = mean_absolute_error(
                yv_valid,
                corrected_val,
            )

            raw_val_rmse = rmse(
                yv_valid,
                raw_v_valid,
            )

            corrected_val_rmse = rmse(
                yv_valid,
                corrected_val,
            )

            raw_test_mae = mean_absolute_error(
                yte_valid,
                raw_te_valid,
            )

            corrected_test_mae = mean_absolute_error(
                yte_valid,
                corrected_test,
            )

            raw_test_rmse = rmse(
                yte_valid,
                raw_te_valid,
            )

            corrected_test_rmse = rmse(
                yte_valid,
                corrected_test,
            )

            raw_val_bias = float(
                np.mean(
                    raw_v_valid.values
                    - yv_valid.values
                )
            )

            corrected_val_bias = float(
                np.mean(
                    corrected_val
                    - yv_valid.values
                )
            )

            raw_test_bias = float(
                np.mean(
                    raw_te_valid.values
                    - yte_valid.values
                )
            )

            corrected_test_bias = float(
                np.mean(
                    corrected_test
                    - yte_valid.values
                )
            )

            val_improvement = (
                (raw_val_mae - corrected_val_mae)
                / raw_val_mae
                * 100
                if raw_val_mae != 0
                else np.nan
            )

            test_improvement = (
                (raw_test_mae - corrected_test_mae)
                / raw_test_mae
                * 100
                if raw_test_mae != 0
                else np.nan
            )

            # ----------------------------------------------------
            # PRINT
            # ----------------------------------------------------

            val_status = (
                "IMPROVED"
                if val_improvement > 0
                else "WORSE"
            )

            test_status = (
                "IMPROVED"
                if test_improvement > 0
                else "WORSE"
            )

            print(
                f"Validation:"
                f" Raw MAE={raw_val_mae:.4f}"
                f" Corrected MAE={corrected_val_mae:.4f}"
                f" Improvement={val_improvement:.2f}%"
                f" {val_status}"
            )

            print(
                f"Test:"
                f" Raw MAE={raw_test_mae:.4f}"
                f" Corrected MAE={corrected_test_mae:.4f}"
                f" Improvement={test_improvement:.2f}%"
                f" {test_status}"
            )

            # ----------------------------------------------------
            # SAVE RESULT
            # ----------------------------------------------------

            all_results.append({
                "horizon": horizon,
                "variable": variable,

                "validation_raw_MAE":
                    raw_val_mae,

                "validation_corrected_MAE":
                    corrected_val_mae,

                "validation_raw_RMSE":
                    raw_val_rmse,

                "validation_corrected_RMSE":
                    corrected_val_rmse,

                "validation_raw_bias":
                    raw_val_bias,

                "validation_corrected_bias":
                    corrected_val_bias,

                "validation_MAE_improvement_pct":
                    val_improvement,

                "test_raw_MAE":
                    raw_test_mae,

                "test_corrected_MAE":
                    corrected_test_mae,

                "test_raw_RMSE":
                    raw_test_rmse,

                "test_corrected_RMSE":
                    corrected_test_rmse,

                "test_raw_bias":
                    raw_test_bias,

                "test_corrected_bias":
                    corrected_test_bias,

                "test_MAE_improvement_pct":
                    test_improvement,

                "training_rows":
                    len(Xtr),

                "validation_rows":
                    len(Xv),

                "test_rows":
                    len(Xte),
            })

            # ----------------------------------------------------
            # SAVE PREDICTIONS
            # ----------------------------------------------------

            val_predictions = pd.DataFrame({
                "raw_forecast": raw_v_valid.values,
                "actual": yv_valid.values,
                "corrected_forecast": corrected_val,
            })

            test_predictions = pd.DataFrame({
                "raw_forecast": raw_te_valid.values,
                "actual": yte_valid.values,
                "corrected_forecast": corrected_test,
            })

            pred_dir = (
                OUTPUT_ROOT
                / f"D{horizon}"
                / variable
            )

            ensure_dir(pred_dir)

            val_predictions.to_csv(
                pred_dir / "validation_predictions.csv",
                index=False,
            )

            test_predictions.to_csv(
                pred_dir / "test_predictions.csv",
                index=False,
            )

            # ----------------------------------------------------
            # SAVE MODELS
            # ----------------------------------------------------

            model_dir = (
                MODEL_ROOT
                / f"D{horizon}"
                / variable
            )

            ensure_dir(model_dir)

            if config["mode"] == "residual":

                save_pickle(
                    result["model"],
                    model_dir / "residual_model.pkl",
                )

                save_pickle(
                    result["imputer"],
                    model_dir / "imputer.pkl",
                )

            elif config["mode"] == "rainfall":

                save_pickle(
                    result["occurrence_model"],
                    model_dir / "occurrence_model.pkl",
                )

                if result["amount_model"] is not None:

                    save_pickle(
                        result["amount_model"],
                        model_dir / "amount_model.pkl",
                    )

                save_pickle(
                    result["imputer"],
                    model_dir / "imputer.pkl",
                )

    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    results_df = pd.DataFrame(all_results)

    summary_csv = (
        OUTPUT_ROOT
        / "residual_corrector_comparison.csv"
    )

    summary_json = (
        OUTPUT_ROOT
        / "residual_corrector_comparison.json"
    )

    results_df.to_csv(
        summary_csv,
        index=False,
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print()
    print("=" * 70)
    print("RESIDUAL CORRECTOR TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("Results:")
    print(summary_csv)

    print()
    print("Models:")
    print(MODEL_ROOT)

    print()
    print("SUMMARY")
    print("-" * 70)

    display_cols = [
        "horizon",
        "variable",
        "test_raw_MAE",
        "test_corrected_MAE",
        "test_MAE_improvement_pct",
    ]

    print(
        results_df[display_cols]
        .to_string(index=False)
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()