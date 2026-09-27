from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_ROOT = (
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
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "models"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = range(1, 8)

TARGETS = {
    "temperature": "actual_temperature_2m_mean",
    "humidity": "actual_relative_humidity_2m_mean",
    "precipitation": "actual_precipitation_sum",
    "pressure": "actual_surface_pressure_mean",
    "wind": "actual_wind_speed_10m_mean",
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, target_name):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "target": target_name,
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": float(r2_score(y_true, y_pred)),
    }

    # Extra rainfall metric
    if target_name == "precipitation":

        metrics["bias"] = float(
            np.mean(y_pred - y_true)
        )

        wet_mask = y_true > 0

        if wet_mask.any():
            metrics["wet_day_MAE"] = float(
                mean_absolute_error(
                    y_true[wet_mask],
                    y_pred[wet_mask]
                )
            )
        else:
            metrics["wet_day_MAE"] = None

    else:

        metrics["bias"] = float(
            np.mean(y_pred - y_true)
        )

    return metrics


# ============================================================
# CONSTRAINTS
# ============================================================

def apply_physical_constraints(pred, target_name):

    pred = np.asarray(pred, dtype=float)

    if target_name == "humidity":
        pred = np.clip(pred, 0.0, 100.0)

    elif target_name == "precipitation":
        pred = np.maximum(pred, 0.0)

    elif target_name == "wind":
        pred = np.maximum(pred, 0.0)

    elif target_name == "pressure":
        pred = np.maximum(pred, 0.0)

    # Temperature has no arbitrary clipping here.
    # We don't want to impose an unsupported physical range.

    return pred


# ============================================================
# LOAD DATA
# ============================================================

def load_horizon_data(horizon):

    horizon_dir = TRAIN_ROOT / f"D{horizon}"

    X_train = pd.read_csv(
        horizon_dir / "X_train.csv"
    )

    X_validation = pd.read_csv(
        horizon_dir / "X_validation.csv"
    )

    X_test = pd.read_csv(
        horizon_dir / "X_test.csv"
    )

    y_train = pd.read_csv(
        horizon_dir / "y_train.csv"
    )

    y_validation = pd.read_csv(
        horizon_dir / "y_validation.csv"
    )

    y_test = pd.read_csv(
        horizon_dir / "y_test.csv"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# SAFETY CHECKS
# ============================================================

def validate_predictors(X):

    forbidden = [
        c for c in X.columns
        if c.startswith("actual_")
    ]

    if forbidden:
        raise RuntimeError(
            "LEAKAGE BLOCKED: actual_* columns found in X: "
            + ", ".join(forbidden)
        )

    date_like = [
        c for c in X.columns
        if "date" in c.lower()
    ]

    # Dates should not silently enter the model.
    if date_like:
        print(
            "WARNING: date-like columns detected:",
            date_like
        )


def validate_shapes(X, y, name):

    if len(X) != len(y):
        raise RuntimeError(
            f"{name}: X/y row mismatch: "
            f"{len(X)} vs {len(y)}"
        )


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


# ============================================================
# TRAIN ONE TARGET
# ============================================================

def train_target(
    horizon,
    target_name,
    target_column,
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test,
):

    print()
    print("=" * 70)
    print(
        f"D{horizon} | {target_name.upper()}"
    )
    print("=" * 70)

    ytr = y_train[target_column].astype(float)
    yva = y_validation[target_column].astype(float)
    yte = y_test[target_column].astype(float)

    # Remove rows where the target itself is unavailable.
    train_mask = ytr.notna()
    val_mask = yva.notna()
    test_mask = yte.notna()

    Xtr = X_train.loc[train_mask].copy()
    Xva = X_validation.loc[val_mask].copy()
    Xte = X_test.loc[test_mask].copy()

    ytr = ytr.loc[train_mask]
    yva = yva.loc[val_mask]
    yte = yte.loc[test_mask]

    model = create_model()

    model.fit(Xtr, ytr)

    pred_train = apply_physical_constraints(
        model.predict(Xtr),
        target_name
    )

    pred_val = apply_physical_constraints(
        model.predict(Xva),
        target_name
    )

    pred_test = apply_physical_constraints(
        model.predict(Xte),
        target_name
    )

    train_metrics = calculate_metrics(
        ytr,
        pred_train,
        target_name
    )

    validation_metrics = calculate_metrics(
        yva,
        pred_val,
        target_name
    )

    test_metrics = calculate_metrics(
        yte,
        pred_test,
        target_name
    )

    print(
        f"Validation MAE : "
        f"{validation_metrics['MAE']:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{validation_metrics['RMSE']:.4f}"
    )

    print(
        f"Validation R²  : "
        f"{validation_metrics['R2']:.4f}"
    )

    print(
        f"Test MAE       : "
        f"{test_metrics['MAE']:.4f}"
    )

    print(
        f"Test RMSE      : "
        f"{test_metrics['RMSE']:.4f}"
    )

    print(
        f"Test R²        : "
        f"{test_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_dir = (
        MODEL_ROOT
        / f"D{horizon}"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir
        / f"{target_name}_corrector.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    validation_dir = (
        OUTPUT_ROOT
        / f"D{horizon}"
    )

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_predictions = pd.DataFrame(
        {
            "y_true": yva.values,
            "y_pred": pred_val,
        }
    )

    validation_predictions.to_csv(
        validation_dir
        / f"{target_name}_validation_predictions.csv",
        index=False
    )

    test_predictions = pd.DataFrame(
        {
            "y_true": yte.values,
            "y_pred": pred_test,
        }
    )

    test_predictions.to_csv(
        validation_dir
        / f"{target_name}_test_predictions.csv",
        index=False
    )

    return {
        "horizon": horizon,
        "target": target_name,
        "target_column": target_column,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIH26 BLOCK FORECAST CORRECTION")
    print("=" * 70)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Training root: {TRAIN_ROOT}"
    )

    all_results = []

    for horizon in HORIZONS:

        print()
        print("#" * 70)
        print(
            f"LOADING D{horizon}"
        )
        print("#" * 70)

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
        ) = load_horizon_data(horizon)

        validate_predictors(X_train)
        validate_predictors(X_validation)
        validate_predictors(X_test)

        validate_shapes(
            X_train,
            y_train,
            f"D{horizon} train"
        )

        validate_shapes(
            X_validation,
            y_validation,
            f"D{horizon} validation"
        )

        validate_shapes(
            X_test,
            y_test,
            f"D{horizon} test"
        )

        print(
            f"Train      : {X_train.shape}"
        )

        print(
            f"Validation : {X_validation.shape}"
        )

        print(
            f"Test       : {X_test.shape}"
        )

        for target_name, target_column in TARGETS.items():

            result = train_target(
                horizon,
                target_name,
                target_column,
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
            )

            all_results.append(result)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_ROOT
        / "block_corrector_results.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    summary_rows = []

    for result in all_results:

        row = {
            "horizon": result["horizon"],
            "target": result["target"],
            "train_MAE": result["train"]["MAE"],
            "validation_MAE": result["validation"]["MAE"],
            "validation_RMSE": result["validation"]["RMSE"],
            "validation_R2": result["validation"]["R2"],
            "test_MAE": result["test"]["MAE"],
            "test_RMSE": result["test"]["RMSE"],
            "test_R2": result["test"]["R2"],
            "test_bias": result["test"]["bias"],
        }

        summary_rows.append(row)

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT_ROOT
        / "block_corrector_results.csv",
        index=False
    )

    print()
    print("=" * 70)
    print("BLOCK CORRECTION TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Results: {summary_path}"
    )


if __name__ == "__main__":
    main()