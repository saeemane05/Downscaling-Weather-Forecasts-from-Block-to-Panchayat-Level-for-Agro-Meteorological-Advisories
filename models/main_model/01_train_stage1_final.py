from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "datasets"
TRAINING_ROOT = DATA_ROOT / "training"

MODEL_ROOT = PROJECT_ROOT / "models" / "main_model"
TRAINED_MODEL_ROOT = MODEL_ROOT / "trained_models"
PREDICTION_ROOT = MODEL_ROOT / "predictions"
METRICS_ROOT = MODEL_ROOT / "metrics"

TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONS = [f"D{i}" for i in range(1, 8)]

RANDOM_STATE = 42
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3

# Rainfall specialized correction:
# final = 70% specialized model + 30% raw forecast
RAIN_MODEL_WEIGHT = 0.70
RAIN_RAW_WEIGHT = 0.30

# Dynamic chronological split
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

assert abs(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
) < 1e-9


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mape(y_true, y_pred):
    """
    MAPE is unstable when rainfall contains many zeros.
    We therefore calculate it only for non-zero observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.abs(y_true) > 1e-8

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100.0
    )


def regression_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_nonzero": safe_mape(y_true, y_pred),
    }


def improvement_percent(raw_rmse, corrected_rmse):
    if raw_rmse == 0:
        return np.nan

    return float(
        (raw_rmse - corrected_rmse)
        / raw_rmse
        * 100.0
    )


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

def find_date_column(df):
    """
    Find the target date column.

    The final training X CSVs intentionally exclude ``date`` from the
    ML predictors, so callers must use the engineered master table when
    the date is not present in X.
    """

    preferred = [
        "target_date",
        "date",
        "forecast_date",
        "valid_date",
        "datetime",
        "target_datetime",
    ]

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for c in df.columns:
        name = str(c).strip().lower()
        if ("target" in name and "date" in name) or name.endswith("_date"):
            return c

    raise RuntimeError(
        "Could not identify the target/date column in this table. "
        f"Available columns: {list(df.columns)}"
    )


def find_engineered_training_master(block_dataset_root):
    """
    Locate the authoritative engineered block forecast training table.

    The horizon-specific X/y CSVs are ML-ready tables and deliberately do
    not contain the target date. The engineered master retains the date and
    the same forecast predictors/targets used to construct those tables.
    """

    project_root = Path(block_dataset_root).resolve().parents[4]
    # Prefer the exact project convention used for SIH26.
    candidates = [
        project_root
        / "datasets"
        / "engineered"
        / block_dataset_root.relative_to(project_root / "datasets" / "training")
        / "block"
        / "block_forecast_training_base_engineered.csv",
    ]

    # Explicit fallback built from the known Maharashtra/Nashik/Sinnar
    # directory hierarchy, while still remaining block-generic.
    rel_training = block_dataset_root.resolve().relative_to(
        project_root / "datasets" / "training"
    )
    candidates.append(
        project_root
        / "datasets"
        / "engineered"
        / rel_training
        / "block"
        / "block_forecast_training_base_engineered.csv"
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "The horizon X/y CSVs do not contain the target date, and the "
        "authoritative engineered training master could not be found.\n"
        "Expected: datasets/engineered/<state>/<district>/<block>/block/"
        "block_forecast_training_base_engineered.csv\n"
        f"Searched from block root: {block_dataset_root}"
    )


def find_rainfall_columns(X, y):
    """
    Identify rainfall forecast predictor and rainfall target.
    """

    forecast_candidates = [
        "precip_forecast",
        "precipitation_forecast",
        "rain_forecast",
        "rainfall_forecast",
        "precip_forecast_d1",
        "precipitation_forecast_d1",
    ]

    target_candidates = [
        "actual_precipitation_sum",
        "actual_precipitation",
        "precipitation_sum",
        "actual_rainfall",
        "rainfall",
        "precipitation",
        "rain",
    ]

    # ----------------------------
    # target
    # ----------------------------

    target_col = None

    lower_y = {
        str(c).lower(): c
        for c in y.columns
    }

    for candidate in target_candidates:
        if candidate.lower() in lower_y:
            target_col = lower_y[candidate.lower()]
            break

    if target_col is None:

        # Usually y contains one target column.
        if len(y.columns) == 1:
            target_col = y.columns[0]

        else:
            rainfall_like = [
                c for c in y.columns
                if any(
                    word in str(c).lower()
                    for word in [
                        "rain",
                        "precip"
                    ]
                )
            ]

            if len(rainfall_like) == 1:
                target_col = rainfall_like[0]

    if target_col is None:
        raise RuntimeError(
            "Could not identify rainfall target column."
        )

    # ----------------------------
    # forecast
    # ----------------------------

    forecast_col = None

    lower_x = {
        str(c).lower(): c
        for c in X.columns
    }

    for candidate in forecast_candidates:
        if candidate.lower() in lower_x:
            forecast_col = lower_x[candidate.lower()]
            break

    if forecast_col is None:

        rainfall_forecast_candidates = [
            c for c in X.columns
            if (
                (
                    "precip" in str(c).lower()
                    or "rain" in str(c).lower()
                )
                and
                "forecast" in str(c).lower()
            )
        ]

        if len(rainfall_forecast_candidates) == 1:
            forecast_col = rainfall_forecast_candidates[0]

    if forecast_col is None:
        raise RuntimeError(
            "Could not identify rainfall forecast predictor."
        )

    return forecast_col, target_col


# ============================================================
# DYNAMIC YEAR-BASED SPLIT
# ============================================================

def dynamic_year_split(df, date_col):
    """
    Chronological split based on complete calendar years.

    Available years are sorted.

    Approximately:
        60% earliest years -> TRAIN
        20% next years     -> VALIDATION
        20% latest years   -> TEST

    The split is NEVER random and never row-based.

    For 5 years:
        3 train / 1 validation / 1 test

    For 6 years:
        4 train / 1 validation / 1 test

    For other numbers, the function automatically chooses
    integer year counts while guaranteeing all three sets
    contain at least one complete year.
    """

    work = df.copy()

    work[date_col] = pd.to_datetime(
        work[date_col],
        errors="coerce"
    )

    work = work.dropna(
        subset=[date_col]
    ).copy()

    if work.empty:
        raise RuntimeError(
            "No valid dates available for chronological split."
        )

    work["_calendar_year"] = (
        work[date_col].dt.year
    )

    years = sorted(
        work["_calendar_year"]
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )

    n_years = len(years)

    if n_years < 3:
        raise RuntimeError(
            f"Need at least 3 calendar years for "
            f"60/20/20 chronological splitting. "
            f"Only {n_years} year(s) found: {years}"
        )

    # --------------------------------------------------------
    # Integer allocation.
    #
    # For the expected 5-year case this becomes:
    #   3 / 1 / 1
    #
    # We prioritize:
    #   - at least one year for validation
    #   - at least one year for test
    #   - approximately 60/20/20 overall
    # --------------------------------------------------------

    if n_years == 3:
        n_train = 1
        n_val = 1
        n_test = 1

    else:
        n_train = max(
            1,
            int(round(n_years * TRAIN_RATIO))
        )

        n_val = max(
            1,
            int(round(n_years * VALIDATION_RATIO))
        )

        # Whatever remains goes to test.
        n_test = (
            n_years
            - n_train
            - n_val
        )

        # Ensure test exists.
        if n_test < 1:
            n_test = 1
            n_val = n_years - n_train - n_test

        # If validation disappeared, rebalance.
        if n_val < 1:
            n_val = 1
            n_train = n_years - n_val - n_test

    train_years = years[:n_train]

    val_start = n_train
    val_end = n_train + n_val

    validation_years = years[
        val_start:val_end
    ]

    test_years = years[
        val_end:
    ]

    # Final safety check.
    if (
        not train_years
        or not validation_years
        or not test_years
    ):
        raise RuntimeError(
            "Dynamic year split failed to produce "
            "non-empty train/validation/test periods."
        )

    train = work[
        work["_calendar_year"].isin(train_years)
    ].copy()

    validation = work[
        work["_calendar_year"].isin(validation_years)
    ].copy()

    test = work[
        work["_calendar_year"].isin(test_years)
    ].copy()

    # Remove helper column.
    train.drop(
        columns=["_calendar_year"],
        inplace=True
    )

    validation.drop(
        columns=["_calendar_year"],
        inplace=True
    )

    test.drop(
        columns=["_calendar_year"],
        inplace=True
    )

    return (
        train,
        validation,
        test,
        train_years,
        validation_years,
        test_years,
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_no_actual_features(X):
    """
    actual_* columns must never be model predictors.
    """

    forbidden = [
        c for c in X.columns
        if str(c).lower().startswith("actual_")
    ]

    if forbidden:
        raise RuntimeError(
            "LEAKAGE BLOCKED: actual_* columns found in X:\n"
            + "\n".join(
                str(x)
                for x in forbidden
            )
        )


# ============================================================
# MODEL PREPARATION
# ============================================================

def prepare_features(
    X_train,
    X_validation,
    X_test,
):
    """
    Numeric-only predictors with median imputation fitted
    exclusively on training data.
    """

    check_no_actual_features(X_train)
    check_no_actual_features(X_validation)
    check_no_actual_features(X_test)

    # Keep only numeric columns.
    numeric_columns = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if not numeric_columns:
        raise RuntimeError(
            "No numeric predictors available."
        )

    Xtr = X_train[
        numeric_columns
    ].copy()

    Xv = X_validation[
        numeric_columns
    ].copy()

    Xte = X_test[
        numeric_columns
    ].copy()

    # Fit imputer ONLY on training.
    imputer = SimpleImputer(
        strategy="median"
    )

    Xtr_imp = imputer.fit_transform(Xtr)

    Xv_imp = imputer.transform(Xv)

    Xte_imp = imputer.transform(Xte)

    return (
        Xtr_imp,
        Xv_imp,
        Xte_imp,
        numeric_columns,
        imputer,
    )


# ============================================================
# RAINFALL SPECIALIZED MODEL
# ============================================================

def train_rainfall_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    raw_forecast_train,
    raw_forecast_validation,
    raw_forecast_test,
):
    """
    Two-stage rainfall correction:

    Stage A:
        classify occurrence (rain / no rain)

    Stage B:
        predict rainfall amount conditional on wet day

    Final specialized prediction:
        P(rain) * predicted_amount

    Final corrected forecast:
        70% specialized + 30% raw forecast
    """

    y_train = np.asarray(
        y_train,
        dtype=float
    )

    y_validation = np.asarray(
        y_validation,
        dtype=float
    )

    y_test = np.asarray(
        y_test,
        dtype=float
    )

    # Physical target safety.
    y_train = np.maximum(
        y_train,
        0.0
    )

    y_validation = np.maximum(
        y_validation,
        0.0
    )

    y_test = np.maximum(
        y_test,
        0.0
    )

    raw_forecast_train = np.maximum(
        np.asarray(
            raw_forecast_train,
            dtype=float
        ),
        0.0,
    )

    raw_forecast_validation = np.maximum(
        np.asarray(
            raw_forecast_validation,
            dtype=float
        ),
        0.0,
    )

    raw_forecast_test = np.maximum(
        np.asarray(
            raw_forecast_test,
            dtype=float
        ),
        0.0,
    )

    # --------------------------------------------------------
    # OCCURRENCE MODEL
    # --------------------------------------------------------

    y_occ_train = (
        y_train > 0.0
    ).astype(int)

    occurrence_model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    occurrence_model.fit(
        X_train,
        y_occ_train
    )

    occ_probability_validation = (
        occurrence_model
        .predict_proba(X_validation)[:, 1]
    )

    occ_probability_test = (
        occurrence_model
        .predict_proba(X_test)[:, 1]
    )

    occ_prediction_validation = (
        occ_probability_validation >= 0.5
    ).astype(int)

    occ_prediction_test = (
        occ_probability_test >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # CONDITIONAL AMOUNT MODEL
    # --------------------------------------------------------

    wet_mask = (
        y_train > 0.0
    )

    if wet_mask.sum() < 10:
        raise RuntimeError(
            "Too few wet training observations "
            "for conditional rainfall model."
        )

    amount_model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    amount_model.fit(
        X_train[wet_mask],
        y_train[wet_mask],
    )

    amount_prediction_validation = np.maximum(
        amount_model.predict(
            X_validation
        ),
        0.0,
    )

    amount_prediction_test = np.maximum(
        amount_model.predict(
            X_test
        ),
        0.0,
    )

    # --------------------------------------------------------
    # SPECIALIZED FORECAST
    # --------------------------------------------------------

    specialized_validation = (
        occ_probability_validation
        * amount_prediction_validation
    )

    specialized_test = (
        occ_probability_test
        * amount_prediction_test
    )

    # --------------------------------------------------------
    # FINAL BLEND
    # --------------------------------------------------------

    corrected_validation = (
        RAIN_MODEL_WEIGHT
        * specialized_validation
        +
        RAIN_RAW_WEIGHT
        * raw_forecast_validation
    )

    corrected_test = (
        RAIN_MODEL_WEIGHT
        * specialized_test
        +
        RAIN_RAW_WEIGHT
        * raw_forecast_test
    )

    corrected_validation = np.maximum(
        corrected_validation,
        0.0,
    )

    corrected_test = np.maximum(
        corrected_test,
        0.0,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    raw_val_metrics = regression_metrics(
        y_validation,
        raw_forecast_validation,
    )

    corrected_val_metrics = regression_metrics(
        y_validation,
        corrected_validation,
    )

    raw_test_metrics = regression_metrics(
        y_test,
        raw_forecast_test,
    )

    corrected_test_metrics = regression_metrics(
        y_test,
        corrected_test,
    )

    validation_occ_f1 = f1_score(
        y_validation > 0,
        occ_prediction_validation,
        zero_division=0,
    )

    test_occ_f1 = f1_score(
        y_test > 0,
        occ_prediction_test,
        zero_division=0,
    )

    metrics = {
        "validation": {
            "raw": raw_val_metrics,
            "corrected": corrected_val_metrics,
            "rmse_improvement_percent": improvement_percent(
                raw_val_metrics["rmse"],
                corrected_val_metrics["rmse"],
            ),
            "occurrence_f1": float(
                validation_occ_f1
            ),
            "wet_days": int(
                np.sum(y_validation > 0)
            ),
            "total_days": int(
                len(y_validation)
            ),
        },
        "test": {
            "raw": raw_test_metrics,
            "corrected": corrected_test_metrics,
            "rmse_improvement_percent": improvement_percent(
                raw_test_metrics["rmse"],
                corrected_test_metrics["rmse"],
            ),
            "occurrence_f1": float(
                test_occ_f1
            ),
            "wet_days": int(
                np.sum(y_test > 0)
            ),
            "total_days": int(
                len(y_test)
            ),
        },
    }

    return (
        occurrence_model,
        amount_model,
        corrected_validation,
        corrected_test,
        metrics,
    )


# ============================================================
# PROCESS ONE HORIZON
# ============================================================

def process_horizon(
    horizon,
    block_dataset_root,
):
    print()
    print("=" * 78)
    print(f"PROCESSING {horizon}")
    print("=" * 78)

    horizon_dir = (
        block_dataset_root / horizon
    )

    x_train_path = (
        horizon_dir / "X_train.csv"
    )

    x_validation_path = (
        horizon_dir / "X_validation.csv"
    )

    x_test_path = (
        horizon_dir / "X_test.csv"
    )

    y_train_path = (
        horizon_dir / "y_train.csv"
    )

    y_validation_path = (
        horizon_dir / "y_validation.csv"
    )

    y_test_path = (
        horizon_dir / "y_test.csv"
    )

    required = [
        x_train_path,
        x_validation_path,
        x_test_path,
        y_train_path,
        y_validation_path,
        y_test_path,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing training files:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Existing X_train/X_validation/X_test are NOT trusted
    # for the new split.
    #
    # We combine all three and recreate the chronological
    # year-based split from scratch.
    # --------------------------------------------------------

    X_parts = [
        pd.read_csv(x_train_path),
        pd.read_csv(x_validation_path),
        pd.read_csv(x_test_path),
    ]

    y_parts = [
        pd.read_csv(y_train_path),
        pd.read_csv(y_validation_path),
        pd.read_csv(y_test_path),
    ]

    X_all = pd.concat(
        X_parts,
        axis=0,
        ignore_index=True,
    )

    y_all = pd.concat(
        y_parts,
        axis=0,
        ignore_index=True,
    )

    if len(X_all) != len(y_all):
        raise RuntimeError(
            f"{horizon}: X/y row count mismatch: "
            f"{len(X_all)} vs {len(y_all)}"
        )

    # --------------------------------------------------------
    # IDENTIFY RAIN TARGET
    # --------------------------------------------------------

    rainfall_forecast_col, rainfall_target_col = (
        find_rainfall_columns(
            X_all,
            y_all,
        )
    )

    # --------------------------------------------------------
    # RECOVER THE TARGET DATE
    # --------------------------------------------------------
    #
    # ``date`` is intentionally excluded from the 13 ML predictor columns.
    # Therefore it is not present in X_train/X_validation/X_test. The date
    # and the complete 975-row chronology are retained in the engineered
    # block forecast training master. We use that authoritative table rather
    # than inventing dates or assuming row order.
    # --------------------------------------------------------

    try:
        date_col = find_date_column(X_all)

        data = X_all.copy()
        data["_target_value"] = pd.to_numeric(
            y_all[rainfall_target_col],
            errors="coerce",
        )

    except RuntimeError:
        master_path = find_engineered_training_master(
            block_dataset_root
        )

        print(
            "Date not present in ML X table; using authoritative engineered "
            f"training master:\n  {master_path}"
        )

        master = pd.read_csv(master_path)

        master_date_col = find_date_column(master)

        # The selected predictor columns in X are the exact ML feature
        # contract for this horizon. Rebuild the modeling table from the
        # authoritative master so date and predictors stay row-aligned.
        model_columns = list(X_all.columns)

        missing_predictors = [
            c for c in model_columns
            if c not in master.columns
        ]

        if missing_predictors:
            raise RuntimeError(
                f"{horizon}: authoritative engineered master is missing "
                f"ML predictor columns: {missing_predictors}"
            )

        if rainfall_target_col not in master.columns:
            raise RuntimeError(
                f"{horizon}: authoritative engineered master is missing "
                f"target column '{rainfall_target_col}'."
            )

        data = master[
            [master_date_col] + model_columns
        ].copy()

        data["_target_value"] = pd.to_numeric(
            master[rainfall_target_col],
            errors="coerce",
        )

        date_col = master_date_col

        if len(data) != len(X_all):
            raise RuntimeError(
                f"{horizon}: engineered master row count ({len(data)}) "
                f"does not match combined X row count ({len(X_all)}). "
                "Refusing to assume row alignment."
            )

        # The master is the authoritative chronological table. The ML-ready
        # X/y files are derived from this same 975-row source, but their
        # train/validation/test row order is not suitable for reconstructing
        # dates. We therefore deliberately do NOT attach dates by row order.
        print(
            "Authoritative master loaded: "
            f"{len(data)} rows, {len(model_columns)} ML predictors"
        )

    print(
        f"Date column: {date_col}"
    )

    print(
        f"Rainfall forecast: "
        f"{rainfall_forecast_col}"
    )

    print(
        f"Rainfall target: "
        f"{rainfall_target_col}"
    )

    print(
        f"Rows available for chronological split: {len(data)}"
    )

    # --------------------------------------------------------
    # DYNAMIC YEAR SPLIT
    # --------------------------------------------------------

    (
        split_data,
        validation_data,
        test_data,
        train_years,
        validation_years,
        test_years,
    ) = dynamic_year_split(
        data,
        date_col,
    )

    print()
    print("DYNAMIC CHRONOLOGICAL SPLIT")
    print("-" * 50)

    print(
        f"TRAIN      years: {train_years}"
    )

    print(
        f"VALIDATION years: {validation_years}"
    )

    print(
        f"TEST       years: {test_years}"
    )

    print(
        f"TRAIN rows:      {len(split_data)}"
    )

    print(
        f"VALIDATION rows: {len(validation_data)}"
    )

    print(
        f"TEST rows:       {len(test_data)}"
    )

    # --------------------------------------------------------
    # BUILD X/y
    # --------------------------------------------------------

    def extract_xy(frame):
        y = frame[
            "_target_value"
        ].to_numpy(
            dtype=float
        )

        X = frame.drop(
            columns=[
                "_target_value"
            ]
        ).copy()

        return X, y

    X_train, y_train = extract_xy(
        split_data
    )

    X_validation, y_validation = extract_xy(
        validation_data
    )

    X_test, y_test = extract_xy(
        test_data
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    check_no_actual_features(
        X_train
    )

    check_no_actual_features(
        X_validation
    )

    check_no_actual_features(
        X_test
    )

    # Remove date columns from ML predictors.
    date_like_columns = []

    for c in X_train.columns:

        lower = str(c).lower()

        if (
            c == date_col
            or "datetime" in lower
            or (
                lower.endswith("_date")
                and "forecast" not in lower
            )
        ):
            date_like_columns.append(c)

    X_train = X_train.drop(
        columns=date_like_columns,
        errors="ignore",
    )

    X_validation = X_validation.drop(
        columns=date_like_columns,
        errors="ignore",
    )

    X_test = X_test.drop(
        columns=date_like_columns,
        errors="ignore",
    )

    # --------------------------------------------------------
    # PREPARE NUMERIC FEATURES
    # --------------------------------------------------------

    (
        Xtr,
        Xv,
        Xte,
        feature_columns,
        imputer,
    ) = prepare_features(
        X_train,
        X_validation,
        X_test,
    )

    print(
        f"Numeric model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # RAW RAINFALL FORECAST
    # --------------------------------------------------------

    raw_train = pd.to_numeric(
        split_data[
            rainfall_forecast_col
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    raw_validation = pd.to_numeric(
        validation_data[
            rainfall_forecast_col
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    raw_test = pd.to_numeric(
        test_data[
            rainfall_forecast_col
        ],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    # --------------------------------------------------------
    # TRAIN RAINFALL MODEL
    # --------------------------------------------------------

    (
        occurrence_model,
        amount_model,
        corrected_validation,
        corrected_test,
        metrics,
    ) = train_rainfall_model(
        Xtr,
        y_train,
        Xv,
        y_validation,
        Xte,
        y_test,
        raw_train,
        raw_validation,
        raw_test,
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    horizon_model_dir = (
        TRAINED_MODEL_ROOT / horizon
    )

    horizon_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    import joblib

    joblib.dump(
        occurrence_model,
        horizon_model_dir
        / "rain_occurrence_model.pkl",
    )

    joblib.dump(
        amount_model,
        horizon_model_dir
        / "rain_amount_model.pkl",
    )

    joblib.dump(
        imputer,
        horizon_model_dir
        / "feature_imputer.pkl",
    )

    joblib.dump(
        feature_columns,
        horizon_model_dir
        / "feature_columns.pkl",
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    horizon_prediction_dir = (
        PREDICTION_ROOT / horizon
    )

    horizon_prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions = pd.DataFrame(
        {
            "date": validation_data[
                date_col
            ].values,

            "actual_precipitation": y_validation,

            "raw_precipitation_forecast":
                raw_validation,

            "corrected_precipitation_forecast":
                corrected_validation,
        }
    )

    test_predictions = pd.DataFrame(
        {
            "date": test_data[
                date_col
            ].values,

            "actual_precipitation": y_test,

            "raw_precipitation_forecast":
                raw_test,

            "corrected_precipitation_forecast":
                corrected_test,
        }
    )

    validation_predictions.to_csv(
        horizon_prediction_dir
        / "validation_predictions.csv",
        index=False,
    )

    test_predictions.to_csv(
        horizon_prediction_dir
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics["horizon"] = horizon

    metrics["split"] = {
        "method": "calendar_year_chronological",
        "train_ratio_target": TRAIN_RATIO,
        "validation_ratio_target":
            VALIDATION_RATIO,
        "test_ratio_target": TEST_RATIO,
        "train_years": train_years,
        "validation_years":
            validation_years,
        "test_years": test_years,
        "train_rows": len(split_data),
        "validation_rows":
            len(validation_data),
        "test_rows": len(test_data),
    }

    metrics["model"] = {
        "rain_model_weight":
            RAIN_MODEL_WEIGHT,
        "raw_forecast_weight":
            RAIN_RAW_WEIGHT,
        "n_estimators":
            N_ESTIMATORS,
        "min_samples_leaf":
            MIN_SAMPLES_LEAF,
        "random_state":
            RANDOM_STATE,
    }

    with open(
        METRICS_ROOT
        / f"{horizon}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("RESULTS")
    print("-" * 50)

    print(
        "Validation raw RMSE:      "
        f"{metrics['validation']['raw']['rmse']:.6f}"
    )

    print(
        "Validation corrected RMSE:"
        f" {metrics['validation']['corrected']['rmse']:.6f}"
    )

    print(
        "Validation improvement:   "
        f"{metrics['validation']['rmse_improvement_percent']:.2f}%"
    )

    print()

    print(
        "Test raw RMSE:            "
        f"{metrics['test']['raw']['rmse']:.6f}"
    )

    print(
        "Test corrected RMSE:      "
        f"{metrics['test']['corrected']['rmse']:.6f}"
    )

    print(
        "Test improvement:         "
        f"{metrics['test']['rmse_improvement_percent']:.2f}%"
    )

    print(
        "Test occurrence F1:       "
        f"{metrics['test']['occurrence_f1']:.4f}"
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("STAGE 1 — FINAL BLOCK FORECAST CORRECTION")
    print("=" * 78)

    print()
    print("Split policy:")
    print("  Dynamic calendar-year chronological split")
    print("  ~60% earliest years  -> TRAIN")
    print("  ~20% next years      -> VALIDATION")
    print("  ~20% latest years    -> TEST")

    # Discover block directories by their horizon structure rather than
    # assuming a fixed administrative depth. The project currently uses:
    #   training / Maharashtra / Nashik / Sinnar / D1 ... D7
    # so Sinnar is the block directory and D1-D7 are its children.
    required_files = (
        "X_train.csv",
        "X_validation.csv",
        "X_test.csv",
        "y_train.csv",
        "y_validation.csv",
        "y_test.csv",
    )

    block_dirs = []

    for candidate in TRAINING_ROOT.rglob("*"):
        if not candidate.is_dir():
            continue

        # A valid block directory must contain D1-D7, and every horizon
        # must contain all six expected training CSV files.
        if all(
            all(
                (candidate / horizon / filename).is_file()
                for filename in required_files
            )
            for horizon in HORIZONS
        ):
            block_dirs.append(candidate)

    # Deterministic order and protection against accidental duplicates.
    block_dirs = sorted(set(block_dirs), key=lambda p: str(p).lower())

    if not block_dirs:
        raise RuntimeError(
            "No block training datasets found."
        )

    print()
    print(
        f"Blocks discovered: {len(block_dirs)}"
    )

    all_results = []

    for block_dir in block_dirs:

        block_name = block_dir.name

        print()
        print("#" * 78)
        print(f"BLOCK: {block_name}")
        print("#" * 78)

        # Keep outputs separated by block.
        model_root = (
            MODEL_ROOT
            / "trained_models"
            / block_name
        )

        prediction_root = (
            MODEL_ROOT
            / "predictions"
            / block_name
        )

        metrics_root = (
            MODEL_ROOT
            / "metrics"
            / block_name
        )

        model_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Local copies for this block.
        global TRAINED_MODEL_ROOT
        global PREDICTION_ROOT
        global METRICS_ROOT

        old_model_root = TRAINED_MODEL_ROOT
        old_prediction_root = PREDICTION_ROOT
        old_metrics_root = METRICS_ROOT

        TRAINED_MODEL_ROOT = model_root
        PREDICTION_ROOT = prediction_root
        METRICS_ROOT = metrics_root

        try:

            for horizon in HORIZONS:

                result = process_horizon(
                    horizon,
                    block_dir,
                )

                result["block"] = block_name

                all_results.append(
                    result
                )

        finally:

            TRAINED_MODEL_ROOT = (
                old_model_root
            )

            PREDICTION_ROOT = (
                old_prediction_root
            )

            METRICS_ROOT = (
                old_metrics_root
            )

    summary_rows = []

    for result in all_results:

        summary_rows.append(
            {
                "block":
                    result["block"],

                "horizon":
                    result["horizon"],

                "validation_raw_rmse":
                    result["validation"]["raw"]["rmse"],

                "validation_corrected_rmse":
                    result["validation"]["corrected"]["rmse"],

                "validation_improvement_percent":
                    result["validation"][
                        "rmse_improvement_percent"
                    ],

                "test_raw_rmse":
                    result["test"]["raw"]["rmse"],

                "test_corrected_rmse":
                    result["test"]["corrected"]["rmse"],

                "test_improvement_percent":
                    result["test"][
                        "rmse_improvement_percent"
                    ],

                "test_occurrence_f1":
                    result["test"][
                        "occurrence_f1"
                    ],
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        MODEL_ROOT
        / "stage1_dynamic_year_split_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("STAGE 1 COMPLETED")
    print("=" * 78)

    print(
        f"Summary saved to:\n{summary_path}"
    )


if __name__ == "__main__":
    main()