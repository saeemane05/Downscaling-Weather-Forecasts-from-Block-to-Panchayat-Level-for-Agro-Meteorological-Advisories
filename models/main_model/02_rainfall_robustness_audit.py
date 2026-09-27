from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# SIH26
# FINAL RAINFALL ROBUSTNESS / DISTRIBUTION AUDIT
#
# PURPOSE
# -------
# Determine whether the large validation/test RMSE differences are
# caused by rainfall distribution shift and/or extreme rainfall.
#
# IMPORTANT
# ---------
# This is a DIAGNOSTIC script only.
#
# It does NOT:
#   - retrain models
#   - modify trained models
#   - select a model using test performance
#   - create synthetic data
#   - interpolate targets
#   - modify X/y datasets
#
# MODEL-SELECTION RULE
# --------------------
# Validation performance is used for model-selection interpretation.
# Test performance is reported as final evaluation evidence only.
# ================================================================


# ================================================================
# 1. PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "training"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "main_model"
)

PREDICTION_ROOT = (
    MODEL_ROOT
    / "predictions"
)

OUTPUT_ROOT = (
    MODEL_ROOT
    / "rainfall_robustness_audit"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# 2. CONFIGURATION
# ================================================================

HORIZONS = range(1, 8)

TARGET_COLUMN = "actual_precipitation_sum"

RAW_COLUMN_PREFIX = "precip_forecast_d"

CORRECTED_COLUMN = "final_corrected_prediction"

RAW_COLUMN = "raw_precipitation_forecast"


# Rainfall categories.
#
# These are used only for diagnostic stratification.
#
# 0 mm:
#   dry
#
# >0 to <=2.5:
#   light
#
# >2.5 to <=10:
#   moderate
#
# >10 to <=25:
#   heavy
#
# >25:
#   extreme
#
RAIN_BINS = [
    -np.inf,
    0.0,
    2.5,
    10.0,
    25.0,
    np.inf,
]

RAIN_LABELS = [
    "dry",
    "light",
    "moderate",
    "heavy",
    "extreme",
]


# ================================================================
# 3. HELPERS
# ================================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


def rmse(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mae(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def bias(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.mean(
            y_pred - y_true
        )
    )


def improvement_pct(
    raw_error,
    corrected_error,
):

    if raw_error == 0:
        return np.nan

    return float(
        (
            raw_error
            - corrected_error
        )
        / raw_error
        * 100.0
    )


def percentile(
    series,
    q,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            q,
        )
    )


def distribution_statistics(
    series,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "wet_pct": np.nan,
        }

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(values.max()),
        "wet_pct": float(
            (values > 0).mean() * 100
        ),
    }


def safe_ratio(
    numerator,
    denominator,
):

    if denominator == 0 or pd.isna(denominator):

        return np.nan

    return float(
        numerator / denominator
    )


def category_metrics(
    y_true,
    raw_pred,
    corrected_pred,
):

    df = pd.DataFrame(
        {
            "actual": pd.to_numeric(
                y_true,
                errors="coerce",
            ),
            "raw": pd.to_numeric(
                raw_pred,
                errors="coerce",
            ),
            "corrected": pd.to_numeric(
                corrected_pred,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(df) == 0:

        return pd.DataFrame()

    df["category"] = pd.cut(
        df["actual"],
        bins=RAIN_BINS,
        labels=RAIN_LABELS,
        right=True,
        include_lowest=True,
    )

    rows = []

    for category in RAIN_LABELS:

        subset = df[
            df["category"] == category
        ]

        if len(subset) == 0:

            rows.append(
                {
                    "category": category,
                    "n": 0,
                    "actual_mean": np.nan,
                    "raw_RMSE": np.nan,
                    "corrected_RMSE": np.nan,
                    "raw_MAE": np.nan,
                    "corrected_MAE": np.nan,
                    "raw_bias": np.nan,
                    "corrected_bias": np.nan,
                    "RMSE_improvement_pct": np.nan,
                }
            )

            continue

        raw_rmse = rmse(
            subset["actual"],
            subset["raw"],
        )

        corrected_rmse = rmse(
            subset["actual"],
            subset["corrected"],
        )

        rows.append(
            {
                "category": category,
                "n": int(len(subset)),
                "actual_mean": float(
                    subset["actual"].mean()
                ),
                "raw_RMSE": raw_rmse,
                "corrected_RMSE": corrected_rmse,
                "raw_MAE": mae(
                    subset["actual"],
                    subset["raw"],
                ),
                "corrected_MAE": mae(
                    subset["actual"],
                    subset["corrected"],
                ),
                "raw_bias": bias(
                    subset["actual"],
                    subset["raw"],
                ),
                "corrected_bias": bias(
                    subset["actual"],
                    subset["corrected"],
                ),
                "RMSE_improvement_pct":
                    improvement_pct(
                        raw_rmse,
                        corrected_rmse,
                    ),
            }
        )

    return pd.DataFrame(rows)


def load_predictions(
    horizon,
    split,
):

    path = (
        PREDICTION_ROOT
        / f"D{horizon}_{split}_predictions.csv"
    )

    if not path.exists():

        return None

    return load_csv(path)


# ================================================================
# 4. START
# ================================================================

print("\n" + "=" * 80)
print("SIH26 RAINFALL ROBUSTNESS AUDIT")
print("=" * 80)

print(
    f"\nProject root:\n{PROJECT_ROOT}"
)

print(
    f"\nTraining root:\n{TRAINING_ROOT}"
)

print(
    f"\nPrediction root:\n{PREDICTION_ROOT}"
)

print(
    f"\nOutput root:\n{OUTPUT_ROOT}"
)


# ================================================================
# 5. DISTRIBUTION AUDIT
# ================================================================

print("\n" + "=" * 80)
print("1. TRAIN / VALIDATION / TEST DISTRIBUTION AUDIT")
print("=" * 80)


distribution_rows = []

distribution_json = {}


for horizon in HORIZONS:

    print(
        f"\nD{horizon}"
    )

    horizon_dir = (
        TRAINING_ROOT
        / f"D{horizon}"
    )

    split_files = {
        "train": horizon_dir / "y_train.csv",
        "validation": horizon_dir / "y_validation.csv",
        "test": horizon_dir / "y_test.csv",
    }

    distribution_json[f"D{horizon}"] = {}

    for split, path in split_files.items():

        data = load_csv(path)

        if TARGET_COLUMN not in data.columns:

            raise RuntimeError(
                f"{path} does not contain "
                f"{TARGET_COLUMN}"
            )

        stats = distribution_statistics(
            data[TARGET_COLUMN]
        )

        distribution_json[
            f"D{horizon}"
        ][split] = stats

        distribution_rows.append(
            {
                "horizon": horizon,
                "split": split,
                **stats,
            }
        )

        print(
            f"  {split:12s}"
            f" n={stats['n']:4d}"
            f" mean={stats['mean']:.4f}"
            f" median={stats['median']:.4f}"
            f" p95={stats['p95']:.4f}"
            f" p99={stats['p99']:.4f}"
            f" max={stats['max']:.4f}"
            f" wet={stats['wet_pct']:.2f}%"
        )


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    OUTPUT_ROOT
    / "rainfall_distribution_summary.csv",
    index=False,
)

with open(
    OUTPUT_ROOT
    / "rainfall_distribution_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        distribution_json,
        f,
        indent=2,
    )


# ================================================================
# 6. DISTRIBUTION SHIFT
# ================================================================

print("\n" + "=" * 80)
print("2. VALIDATION / TEST DISTRIBUTION SHIFT")
print("=" * 80)


shift_rows = []


for horizon in HORIZONS:

    horizon_data = distribution_json[
        f"D{horizon}"
    ]

    train = horizon_data["train"]
    val = horizon_data["validation"]
    test = horizon_data["test"]

    row = {
        "horizon": horizon,

        "train_val_mean_ratio":
            safe_ratio(
                val["mean"],
                train["mean"],
            ),

        "train_test_mean_ratio":
            safe_ratio(
                test["mean"],
                train["mean"],
            ),

        "val_test_mean_ratio":
            safe_ratio(
                test["mean"],
                val["mean"],
            ),

        "train_val_p95_ratio":
            safe_ratio(
                val["p95"],
                train["p95"],
            ),

        "train_test_p95_ratio":
            safe_ratio(
                test["p95"],
                train["p95"],
            ),

        "train_val_p99_ratio":
            safe_ratio(
                val["p99"],
                train["p99"],
            ),

        "train_test_p99_ratio":
            safe_ratio(
                test["p99"],
                train["p99"],
            ),

        "train_test_max_ratio":
            safe_ratio(
                test["max"],
                train["max"],
            ),

        "val_test_max_ratio":
            safe_ratio(
                test["max"],
                val["max"],
            ),

        "validation_wet_pct":
            val["wet_pct"],

        "test_wet_pct":
            test["wet_pct"],
    }

    shift_rows.append(row)

    print(
        f"\nD{horizon}"
    )

    print(
        f"  Test / Train mean : "
        f"{row['train_test_mean_ratio']:.3f}"
    )

    print(
        f"  Test / Train P95  : "
        f"{row['train_test_p95_ratio']:.3f}"
    )

    print(
        f"  Test / Train P99  : "
        f"{row['train_test_p99_ratio']:.3f}"
    )

    print(
        f"  Test / Train MAX  : "
        f"{row['train_test_max_ratio']:.3f}"
    )


shift_df = pd.DataFrame(
    shift_rows
)

shift_df.to_csv(
    OUTPUT_ROOT
    / "rainfall_distribution_shift.csv",
    index=False,
)


# ================================================================
# 7. EXTREME EVENT COUNTS
# ================================================================

print("\n" + "=" * 80)
print("3. EXTREME RAINFALL EVENT AUDIT")
print("=" * 80)


extreme_rows = []


thresholds = [
    0.0,
    2.5,
    10.0,
    25.0,
    50.0,
]


for horizon in HORIZONS:

    horizon_dir = (
        TRAINING_ROOT
        / f"D{horizon}"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        path = (
            horizon_dir
            / f"y_{split}.csv"
        )

        data = load_csv(path)

        rainfall = pd.to_numeric(
            data[TARGET_COLUMN],
            errors="coerce",
        ).dropna()

        row = {
            "horizon": horizon,
            "split": split,
            "n": len(rainfall),
        }

        for threshold in thresholds:

            row[
                f"count_gt_{str(threshold).replace('.', '_')}"
            ] = int(
                (rainfall > threshold).sum()
            )

        row["max"] = float(
            rainfall.max()
        )

        extreme_rows.append(row)


extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_df.to_csv(
    OUTPUT_ROOT
    / "rainfall_extreme_event_counts.csv",
    index=False,
)

print(
    extreme_df.to_string(
        index=False
    )
)


# ================================================================
# 8. ERROR ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("4. RAW VS CORRECTED ERROR ANALYSIS")
print("=" * 80)


overall_error_rows = []

category_error_frames = []


for horizon in HORIZONS:

    for split in [
        "validation",
        "test",
    ]:

        predictions = load_predictions(
            horizon,
            split,
        )

        if predictions is None:

            print(
                f"Missing predictions: "
                f"D{horizon} {split}"
            )

            continue

        required = [
            "actual_precipitation_sum",
            "raw_precipitation_forecast",
            "final_corrected_prediction",
        ]

        missing = [
            c
            for c in required
            if c not in predictions.columns
        ]

        if missing:

            raise RuntimeError(
                f"D{horizon} {split}: "
                f"missing prediction columns "
                f"{missing}"
            )

        data = predictions[
            required
        ].copy()

        data = data.dropna()

        y = data[
            "actual_precipitation_sum"
        ]

        raw = data[
            "raw_precipitation_forecast"
        ]

        corrected = data[
            "final_corrected_prediction"
        ]

        raw_rmse = rmse(
            y,
            raw,
        )

        corrected_rmse = rmse(
            y,
            corrected,
        )

        raw_mae = mae(
            y,
            raw,
        )

        corrected_mae = mae(
            y,
            corrected,
        )

        row = {
            "horizon": horizon,
            "split": split,
            "n": len(data),

            "raw_RMSE": raw_rmse,
            "corrected_RMSE": corrected_rmse,

            "raw_MAE": raw_mae,
            "corrected_MAE": corrected_mae,

            "raw_bias": bias(
                y,
                raw,
            ),

            "corrected_bias": bias(
                y,
                corrected,
            ),

            "RMSE_improvement_pct":
                improvement_pct(
                    raw_rmse,
                    corrected_rmse,
                ),

            "MAE_improvement_pct":
                improvement_pct(
                    raw_mae,
                    corrected_mae,
                ),
        }

        overall_error_rows.append(
            row
        )

        category_df = category_metrics(
            y,
            raw,
            corrected,
        )

        if not category_df.empty:

            category_df.insert(
                0,
                "split",
                split,
            )

            category_df.insert(
                0,
                "horizon",
                horizon,
            )

            category_error_frames.append(
                category_df
            )

        print(
            f"\nD{horizon} {split}"
        )

        print(
            f"  Raw RMSE       : "
            f"{raw_rmse:.6f}"
        )

        print(
            f"  Corrected RMSE : "
            f"{corrected_rmse:.6f}"
        )

        print(
            f"  Improvement    : "
            f"{row['RMSE_improvement_pct']:.2f}%"
        )


overall_error_df = pd.DataFrame(
    overall_error_rows
)

overall_error_df.to_csv(
    OUTPUT_ROOT
    / "raw_vs_corrected_error_summary.csv",
    index=False,
)


if category_error_frames:

    category_error_df = pd.concat(
        category_error_frames,
        ignore_index=True,
    )

else:

    category_error_df = pd.DataFrame()


category_error_df.to_csv(
    OUTPUT_ROOT
    / "rainfall_category_error_analysis.csv",
    index=False,
)


# ================================================================
# 9. EXTREME EVENT CONTRIBUTION TO RMSE
# ================================================================

print("\n" + "=" * 80)
print("5. EXTREME EVENT CONTRIBUTION ANALYSIS")
print("=" * 80)


extreme_contribution_rows = []


for horizon in HORIZONS:

    for split in [
        "validation",
        "test",
    ]:

        predictions = load_predictions(
            horizon,
            split,
        )

        if predictions is None:
            continue

        data = predictions[
            [
                "actual_precipitation_sum",
                "raw_precipitation_forecast",
                "final_corrected_prediction",
            ]
        ].dropna()

        if len(data) == 0:
            continue

        actual = data[
            "actual_precipitation_sum"
        ]

        raw_error_squared = (
            data[
                "raw_precipitation_forecast"
            ]
            - actual
        ) ** 2

        corrected_error_squared = (
            data[
                "final_corrected_prediction"
            ]
            - actual
        ) ** 2

        # Define extreme events using the
        # 95th percentile of the SAME split.
        #
        # This is diagnostic only.
        p95 = actual.quantile(0.95)

        extreme_mask = (
            actual >= p95
        )

        if extreme_mask.sum() == 0:
            continue

        raw_total_error = (
            raw_error_squared.sum()
        )

        corrected_total_error = (
            corrected_error_squared.sum()
        )

        raw_extreme_error = (
            raw_error_squared[
                extreme_mask
            ].sum()
        )

        corrected_extreme_error = (
            corrected_error_squared[
                extreme_mask
            ].sum()
        )

        row = {

            "horizon": horizon,

            "split": split,

            "n": len(data),

            "p95_threshold": float(p95),

            "extreme_event_count":
                int(extreme_mask.sum()),

            "extreme_event_pct":
                float(
                    extreme_mask.mean()
                    * 100
                ),

            "raw_total_squared_error":
                float(raw_total_error),

            "raw_extreme_squared_error":
                float(raw_extreme_error),

            "raw_extreme_error_contribution_pct":
                float(
                    raw_extreme_error
                    / raw_total_error
                    * 100
                )
                if raw_total_error > 0
                else np.nan,

            "corrected_total_squared_error":
                float(corrected_total_error),

            "corrected_extreme_squared_error":
                float(
                    corrected_extreme_error
                ),

            "corrected_extreme_error_contribution_pct":
                float(
                    corrected_extreme_error
                    / corrected_total_error
                    * 100
                )
                if corrected_total_error > 0
                else np.nan,
        }

        extreme_contribution_rows.append(
            row
        )

        print(
            f"\nD{horizon} {split}"
        )

        print(
            f"  P95 threshold: "
            f"{p95:.4f}"
        )

        print(
            f"  Extreme events: "
            f"{extreme_mask.sum()}"
        )

        print(
            f"  Raw RMSE contribution: "
            f"{row['raw_extreme_error_contribution_pct']:.2f}%"
        )

        print(
            f"  Corrected RMSE contribution: "
            f"{row['corrected_extreme_error_contribution_pct']:.2f}%"
        )


extreme_contribution_df = pd.DataFrame(
    extreme_contribution_rows
)

extreme_contribution_df.to_csv(
    OUTPUT_ROOT
    / "extreme_event_error_contribution.csv",
    index=False,
)


# ================================================================
# 10. MODEL-SELECTION INTERPRETATION
# ================================================================

print("\n" + "=" * 80)
print("6. VALIDATION-BASED MODEL SELECTION INTERPRETATION")
print("=" * 80)


selection_rows = []


for horizon in HORIZONS:

    validation_rows = overall_error_df[
        (overall_error_df["horizon"] == horizon)
        &
        (overall_error_df["split"] == "validation")
    ]

    test_rows = overall_error_df[
        (overall_error_df["horizon"] == horizon)
        &
        (overall_error_df["split"] == "test")
    ]

    if validation_rows.empty:

        continue

    validation = validation_rows.iloc[0]

    test = (
        test_rows.iloc[0]
        if not test_rows.empty
        else None
    )

    validation_improvement = (
        validation[
            "RMSE_improvement_pct"
        ]
    )

    if validation_improvement > 0:

        validation_decision = (
            "SPECIALIZED_CORRECTION"
        )

    else:

        validation_decision = (
            "RAW_FORECAST"
        )

    selection_rows.append(
        {
            "horizon": horizon,

            "validation_RMSE_improvement_pct":
                validation_improvement,

            "test_RMSE_improvement_pct":
                (
                    test[
                        "RMSE_improvement_pct"
                    ]
                    if test is not None
                    else np.nan
                ),

            "validation_selected_method":
                validation_decision,

            "selection_basis":
                "VALIDATION_ONLY",
        }
    )

    print(
        f"D{horizon}: "
        f"{validation_decision}"
        f" "
        f"(validation improvement "
        f"{validation_improvement:.2f}%)"
    )


selection_df = pd.DataFrame(
    selection_rows
)

selection_df.to_csv(
    OUTPUT_ROOT
    / "validation_based_stage1_selection.csv",
    index=False,
)


# ================================================================
# 11. BORDERLINE HORIZONS
# ================================================================

print("\n" + "=" * 80)
print("7. BORDERLINE HORIZON IDENTIFICATION")
print("=" * 80)


borderline_rows = []


for _, row in selection_df.iterrows():

    validation_improvement = (
        row[
            "validation_RMSE_improvement_pct"
        ]
    )

    test_improvement = (
        row[
            "test_RMSE_improvement_pct"
        ]
    )

    reasons = []

    if (
        pd.notna(validation_improvement)
        and abs(validation_improvement) < 5
    ):

        reasons.append(
            "validation improvement < 5%"
        )

    if (
        pd.notna(test_improvement)
        and abs(test_improvement) < 5
    ):

        reasons.append(
            "test improvement < 5%"
        )

    if (
        pd.notna(validation_improvement)
        and pd.notna(test_improvement)
        and np.sign(validation_improvement)
        != np.sign(test_improvement)
    ):

        reasons.append(
            "validation/test improvement signs disagree"
        )

    status = (
        "BORDERLINE"
        if reasons
        else "CLEAR"
    )

    borderline_rows.append(
        {
            "horizon": int(
                row["horizon"]
            ),

            "status": status,

            "reasons":
                "; ".join(reasons)
                if reasons
                else "",
        }
    )

    if reasons:

        print(
            f"D{int(row['horizon'])}: "
            f"BORDERLINE"
        )

        for reason in reasons:
            print(
                f"  - {reason}"
            )


borderline_df = pd.DataFrame(
    borderline_rows
)

borderline_df.to_csv(
    OUTPUT_ROOT
    / "borderline_horizons.csv",
    index=False,
)


# ================================================================
# 12. FINAL AUDIT JSON
# ================================================================

final_audit = {

    "project": "SIH26",

    "location": {
        "state": "Maharashtra",
        "district": "Nashik",
        "taluka": "Sinnar",
    },

    "purpose":
        "Rainfall distribution and robustness audit",

    "model_selection_rule":
        "Validation performance only",

    "test_usage":
        "Final evaluation evidence only",

    "horizons": list(HORIZONS),

    "validation_based_selection":
        selection_rows,

    "borderline_horizons":
        borderline_rows,

    "outputs": {
        "distribution":
            "rainfall_distribution_summary.csv",

        "distribution_shift":
            "rainfall_distribution_shift.csv",

        "extreme_events":
            "rainfall_extreme_event_counts.csv",

        "error_summary":
            "raw_vs_corrected_error_summary.csv",

        "category_error":
            "rainfall_category_error_analysis.csv",

        "extreme_error_contribution":
            "extreme_event_error_contribution.csv",

        "validation_selection":
            "validation_based_stage1_selection.csv",

        "borderline":
            "borderline_horizons.csv",
    },

    "policy": {
        "automatic_model_reselection":
            False,

        "test_based_model_selection":
            False,

        "synthetic_data":
            False,

        "interpolation":
            False,
    },
}


with open(
    OUTPUT_ROOT
    / "rainfall_robustness_audit.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_audit,
        f,
        indent=2,
        default=str,
    )


# ================================================================
# 13. FINAL TERMINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("RAINFALL ROBUSTNESS AUDIT COMPLETE")
print("=" * 80)

print(
    "\nValidation-based Stage-1 selection:"
)

for _, row in selection_df.iterrows():

    print(
        f"  D{int(row['horizon'])}: "
        f"{row['validation_selected_method']}"
        f" "
        f"({row['validation_RMSE_improvement_pct']:.2f}%)"
    )


print(
    "\nBorderline horizons:"
)

borderline_found = False

for _, row in borderline_df.iterrows():

    if row["status"] == "BORDERLINE":

        borderline_found = True

        print(
            f"  D{int(row['horizon'])}: "
            f"{row['reasons']}"
        )


if not borderline_found:

    print(
        "  None"
    )


print(
    "\nImportant:"
)

print(
    "  ✓ No models were retrained."
)

print(
    "  ✓ No datasets were modified."
)

print(
    "  ✓ Test data was NOT used for model selection."
)

print(
    "  ✓ No synthetic/interpolated rainfall was created."
)

print(
    "\nAudit output:"
)

print(
    OUTPUT_ROOT
)

print("=" * 80)