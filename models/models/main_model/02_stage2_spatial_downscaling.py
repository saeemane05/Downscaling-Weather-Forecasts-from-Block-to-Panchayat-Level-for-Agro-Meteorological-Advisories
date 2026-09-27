# ================================================================
# SIH26 — STAGE 2: GP-LEVEL SPATIAL DOWNSCALING
# ================================================================
#
# BLOCK:
#   Sinnar, Nashik, Maharashtra
#
# PURPOSE:
#   Prepare a scientifically controlled Stage-2 spatial
#   downscaling framework that combines:
#
#       1. Block-level GFS forecast
#       2. GP-level spatial predictors
#
#   The current script DOES NOT train a supervised GP weather
#   model because genuine historical GP weather targets are not
#   available.
#
#   It therefore:
#
#       - validates the GP spatial dataset
#       - selects spatial predictors
#       - excludes temporally unsafe features
#       - loads current block GFS forecast
#       - creates GP × forecast combinations
#       - constructs normalized spatial allocation weights
#       - checks weight conservation
#       - saves all Stage-2 preparation outputs
#
# IMPORTANT:
#   The spatial weights generated here are an intermediate
#   representation only. They are NOT claimed as validated
#   weather predictions.
#
# ================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ================================================================
# 1. PROJECT CONFIGURATION
# ================================================================

ROOT = Path(r"E:\SIH26_Downscaling")

BLOCK_ROOT = (
    ROOT
    / "block"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "main_model"
)

CLEANED_ROOT = (
    ROOT
    / "datasets"
    / "cleaned"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

GP_ROOT = (
    CLEANED_ROOT
    / "gp"
)

OUTPUT_ROOT = (
    MODEL_ROOT
    / "stage2_outputs"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 2. UTILITY FUNCTIONS
# ================================================================

def print_header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def clean_column_name(column):
    """
    Standardize column names while preserving their meaning.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_file(root, filename):
    """
    Recursively search for a filename.
    """

    matches = list(root.rglob(filename))

    if not matches:
        return None

    matches.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return matches[0]


# ================================================================
# 3. FIND GP DATASET
# ================================================================

def find_gp_dataset():

    exact_path = (
        GP_ROOT
        / "gp_merged_features.csv"
    )

    if exact_path.exists():
        return exact_path

    candidates = [
        "gp_merged_features.csv",
        "gp_features.csv",
        "gp_spatial_features.csv",
        "gp_model_features.csv",
        "gp_features_cleaned.csv",
    ]

    for name in candidates:

        path = find_file(
            ROOT,
            name
        )

        if path is not None:
            return path

    # Final fallback
    matches = []

    for path in ROOT.rglob("*.csv"):

        name = path.name.lower()

        if (
            "gp" in name
            and (
                "feature" in name
                or "merge" in name
                or "spatial" in name
            )
        ):
            matches.append(path)

    if matches:

        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return matches[0]

    return None


# ================================================================
# 4. FIND CURRENT BLOCK FORECAST
# ================================================================

def find_current_forecast():

    exact_path = (
        BLOCK_ROOT
        / "processed"
        / "merge"
        / "block_current_forecast.csv"
    )

    if exact_path.exists():
        return exact_path

    return find_file(
        BLOCK_ROOT,
        "block_current_forecast.csv"
    )


# ================================================================
# 5. FIND STAGE-1 SUMMARY
# ================================================================

def find_stage1_summary():

    candidates = [

        MODEL_ROOT
        / "stage1_dynamic_year_split_summary.csv",

        MODEL_ROOT
        / "stage1_final_summary.csv",
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


# ================================================================
# 6. IDENTIFY GP SPATIAL PREDICTORS
# ================================================================

def identify_spatial_predictors(df):

    # ------------------------------------------------------------
    # Explicitly excluded variables
    # ------------------------------------------------------------

    exclude_keywords = [

        # --------------------------------------------------------
        # GP identifiers
        # --------------------------------------------------------

        "gp_id",
        "gp_code",
        "gpcode",
        "lgdcode",
        "lgd_code",

        # --------------------------------------------------------
        # Geographic metadata
        # --------------------------------------------------------

        "centroid_lat",
        "centroid_lon",
        "latitude",
        "longitude",

        # --------------------------------------------------------
        # Administrative metadata
        # --------------------------------------------------------

        "state",
        "district",
        "block",
        "taluka",
        "village",
        "name",

        # --------------------------------------------------------
        # Geometry / source metadata
        # --------------------------------------------------------

        "geometry",
        "source_layer",
        "source_snapshot",
        "source_feature_count",
        "geometry_status",

        # --------------------------------------------------------
        # Actual weather
        # --------------------------------------------------------

        "actual_",
        "observed_",
        "observation_",

        # --------------------------------------------------------
        # Forecast weather
        # --------------------------------------------------------

        "forecast_",

        "temp_mean_forecast",
        "rh_mean_forecast",
        "precip_forecast",
        "wind_speed_forecast",
        "wind_direction_forecast",
        "wind_gust_forecast",
        "cloud_cover_forecast",
        "vpd_forecast",

        # --------------------------------------------------------
        # Target-like fields
        # --------------------------------------------------------

        "target",
        "label",
        "y_",

        # --------------------------------------------------------
        # Temporal metadata
        # --------------------------------------------------------

        "date",
        "datetime",
        "timestamp",
        "year",
        "month",

        # --------------------------------------------------------
        # Temporally unsafe LST
        #
        # Current GP table does not carry sufficient per-feature
        # acquisition cutoff metadata for forecast issue dates.
        # --------------------------------------------------------

        "lst__",
        "lst_",
        "land_surface_temperature",

    ]

    numeric_columns = (
        df
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    selected = []

    for column in numeric_columns:

        lc = clean_column_name(column)

        excluded = any(
            keyword in lc
            for keyword in exclude_keywords
        )

        if not excluded:
            selected.append(column)

    return selected


# ================================================================
# 7. REMOVE CONSTANT FEATURES
# ================================================================

def remove_constant_features(
    df,
    features
):

    valid = []

    for column in features:

        unique_count = (
            df[column]
            .nunique(
                dropna=True
            )
        )

        if unique_count <= 1:
            continue

        valid.append(column)

    return valid


# ================================================================
# 8. MAIN
# ================================================================

print_header(
    "STAGE 2 — GP SPATIAL DOWNSCALING"
)

print(
    f"Project root : {ROOT}"
)

print(
    "Block        : Sinnar, Nashik, Maharashtra"
)

print(
    f"Output       : {OUTPUT_ROOT}"
)


# ================================================================
# 9. LOAD GP DATASET
# ================================================================

gp_path = find_gp_dataset()

if gp_path is None:

    raise FileNotFoundError(
        "\nCould not find GP spatial feature dataset."
    )

print("\nGP dataset:")
print(gp_path)


gp = pd.read_csv(
    gp_path
)

print(
    "\nGP dataset shape:"
)

print(
    gp.shape
)


# ================================================================
# 10. STANDARDIZE COLUMN NAMES
# ================================================================

gp.columns = [
    clean_column_name(c)
    for c in gp.columns
]

print(
    "\nNumber of columns:",
    len(gp.columns)
)


# ================================================================
# 11. GP DATASET VALIDATION
# ================================================================

print_header(
    "GP DATASET VALIDATION"
)

gp_count = len(gp)

print(
    "Rows:",
    gp_count
)

if gp_count != 113:

    print(
        "WARNING:",
        f"Expected 113 GPs but found {gp_count}."
    )

# ---------------------------------------------------------------
# Geographic columns
# ---------------------------------------------------------------

lat_candidates = [

    "landcover__centroid_lat",

    "latitude",

    "lat",

    "gp_latitude",

    "centroid_latitude",
]

lon_candidates = [

    "landcover__centroid_lon",

    "longitude",

    "lon",

    "gp_longitude",

    "centroid_longitude",
]


lat_col = next(
    (
        c
        for c in lat_candidates
        if c in gp.columns
    ),
    None
)

lon_col = next(
    (
        c
        for c in lon_candidates
        if c in gp.columns
    ),
    None
)


print(
    "Latitude column :",
    lat_col
)

print(
    "Longitude column:",
    lon_col
)


if lat_col is None:

    raise ValueError(
        "Could not identify GP latitude column."
    )


if lon_col is None:

    raise ValueError(
        "Could not identify GP longitude column."
    )


# ================================================================
# 12. CHECK GP COORDINATES
# ================================================================

lat_missing = gp[lat_col].isna().sum()
lon_missing = gp[lon_col].isna().sum()

print(
    "\nMissing latitude:",
    lat_missing
)

print(
    "Missing longitude:",
    lon_missing
)


if lat_missing > 0 or lon_missing > 0:

    raise ValueError(
        "GP coordinates contain missing values."
    )


# ================================================================
# 13. IDENTIFY SPATIAL PREDICTORS
# ================================================================

print_header(
    "IDENTIFYING GP SPATIAL PREDICTORS"
)


spatial_features = (
    identify_spatial_predictors(gp)
)


spatial_features = (
    remove_constant_features(
        gp,
        spatial_features
    )
)


print(
    "Candidate spatial predictors:",
    len(spatial_features)
)


for column in spatial_features:

    print(
        "  ",
        column
    )


# ================================================================
# 14. MISSINGNESS AUDIT
# ================================================================

print_header(
    "SPATIAL FEATURE MISSINGNESS AUDIT"
)


missing_fraction = (
    gp[spatial_features]
    .isna()
    .mean()
)


MAX_MISSING_FRACTION = 0.20


valid_features = [

    column

    for column in spatial_features

    if missing_fraction[column]
    <= MAX_MISSING_FRACTION

]


removed_missing = sorted(
    set(spatial_features)
    -
    set(valid_features)
)


if removed_missing:

    print(
        "\nRemoved because missingness > 20%:"
    )

    for column in removed_missing:

        print(
            f"  {column}: "
            f"{missing_fraction[column] * 100:.2f}%"
        )


spatial_features = valid_features


print(
    "\nFinal spatial predictor count:",
    len(spatial_features)
)


# ================================================================
# 15. VERIFY LST EXCLUSION
# ================================================================

lst_features_present = [

    c

    for c in spatial_features

    if "lst" in c.lower()

]


if lst_features_present:

    raise RuntimeError(
        "LST features were incorrectly included in Stage 2."
    )


print(
    "Temporally unsafe LST features: EXCLUDED"
)


# ================================================================
# 16. VERIFY GP IDENTIFIER EXCLUSION
# ================================================================

identifier_leaks = [

    c

    for c in spatial_features

    if (
        c == "gp_id"
        or "centroid_lat" in c
        or "centroid_lon" in c
    )

]


if identifier_leaks:

    raise RuntimeError(
        "GP identifiers/geographic metadata incorrectly "
        "included as environmental predictors: "
        + str(identifier_leaks)
    )


print(
    "GP ID / centroid metadata: EXCLUDED"
)


# ================================================================
# 17. CREATE SPATIAL FEATURE MATRIX
# ================================================================

X_gp = (
    gp[spatial_features]
    .copy()
)


# ---------------------------------------------------------------
# Median imputation
# ---------------------------------------------------------------

feature_medians = (
    X_gp
    .median()
)


X_gp = (
    X_gp
    .fillna(feature_medians)
)


# ---------------------------------------------------------------
# Verify no missing values remain
# ---------------------------------------------------------------

remaining_missing = (
    X_gp
    .isna()
    .sum()
    .sum()
)


if remaining_missing > 0:

    raise RuntimeError(
        "Missing values remain after imputation."
    )


# ================================================================
# 18. STANDARDIZE SPATIAL FEATURES
# ================================================================

scaler = StandardScaler()


X_scaled_array = (
    scaler
    .fit_transform(X_gp)
)


X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=spatial_features,
    index=gp.index
)


print(
    "\nStandardized spatial matrix:",
    X_scaled.shape
)


# ================================================================
# 19. BUILD SPATIAL REPRESENTATION
# ================================================================
#
# This is NOT a weather prediction.
#
# It represents how different each GP is from the mean GP in
# standardized spatial-feature space.
#
# This is deliberately kept as an intermediate representation.
# ================================================================

print_header(
    "BUILDING SPATIAL REPRESENTATION"
)


spatial_distance = np.sqrt(
    np.mean(
        np.square(
            X_scaled.to_numpy()
        ),
        axis=1
    )
)


gp["spatial_distance_from_mean"] = (
    spatial_distance
)


distance_mean = (
    np.mean(
        spatial_distance
    )
)


distance_std = (
    np.std(
        spatial_distance
    )
)


if (
    not np.isfinite(distance_std)
    or distance_std == 0
):

    distance_std = 1.0


spatial_anomaly_score = (
    spatial_distance
    -
    distance_mean
) / distance_std


gp["spatial_anomaly_score"] = (
    spatial_anomaly_score
)


# ================================================================
# 20. LOAD CURRENT BLOCK FORECAST
# ================================================================

print_header(
    "LOADING CURRENT BLOCK FORECAST"
)


current_forecast_path = (
    find_current_forecast()
)


if current_forecast_path is None:

    raise FileNotFoundError(
        "Could not find block_current_forecast.csv"
    )


print(
    current_forecast_path
)


block_forecast = pd.read_csv(
    current_forecast_path
)


block_forecast.columns = [

    clean_column_name(c)

    for c in block_forecast.columns

]


print(
    "Current forecast shape:",
    block_forecast.shape
)


# ================================================================
# 21. VERIFY CURRENT FORECAST SCHEMA
# ================================================================

EXPECTED_FORECAST_COLUMNS = [

    "target_date",

    "gfs_temp_mean_c",

    "gfs_temp_max_c",

    "gfs_temp_min_c",

    "gfs_rh_mean_pct",

    "gfs_precipitation_sum_mm",

    "gfs_cloud_cover_mean_pct",

    "gfs_wind_speed_mean_ms",

    "gfs_wind_gust_max_ms",

    "gfs_surface_pressure_mean_hpa",

    "gfs_vpd_mean_kpa",

]


missing_forecast_columns = [

    c

    for c in EXPECTED_FORECAST_COLUMNS

    if c not in block_forecast.columns

]


if missing_forecast_columns:

    raise ValueError(
        "Current forecast is missing expected columns:\n"
        + "\n".join(
            missing_forecast_columns
        )
    )


# ================================================================
# 22. SELECT ACTUAL FORECAST VARIABLES
# ================================================================

forecast_candidates = [

    "gfs_temp_mean_c",

    "gfs_temp_max_c",

    "gfs_temp_min_c",

    "gfs_rh_mean_pct",

    "gfs_precipitation_sum_mm",

    "gfs_cloud_cover_mean_pct",

    "gfs_wind_speed_mean_ms",

    "gfs_wind_gust_max_ms",

    "gfs_surface_pressure_mean_hpa",

    "gfs_vpd_mean_kpa",

]


forecast_candidates = [

    c

    for c in forecast_candidates

    if c in block_forecast.columns

]


print(
    "\nForecast variables detected:"
)


for column in forecast_candidates:

    print(
        "  ",
        column
    )


# ================================================================
# 23. FORECAST VALIDITY CHECK
# ================================================================

print_header(
    "CURRENT FORECAST VALIDATION"
)


if len(block_forecast) != 7:

    print(
        "WARNING:",
        "Expected 7 forecast horizons, found",
        len(block_forecast)
    )


if "synthetic" in block_forecast.columns:

    synthetic_values = (
        block_forecast["synthetic"]
        .astype(str)
        .str.lower()
    )

    if synthetic_values.isin(
        ["true", "1", "yes"]
    ).any():

        raise RuntimeError(
            "Current forecast contains synthetic rows."
        )


if "fallback_used" in block_forecast.columns:

    fallback_values = (
        block_forecast["fallback_used"]
        .astype(str)
        .str.lower()
    )

    if fallback_values.isin(
        ["true", "1", "yes"]
    ).any():

        print(
            "WARNING: fallback forecast rows detected."
        )


if (
    "actual_weather_used_as_forecast_input"
    in block_forecast.columns
):

    actual_as_forecast = (
        block_forecast[
            "actual_weather_used_as_forecast_input"
        ]
        .astype(str)
        .str.lower()
    )

    if actual_as_forecast.isin(
        ["true", "1", "yes"]
    ).any():

        raise RuntimeError(
            "Actual weather has been used as forecast input."
        )


print(
    "Synthetic forecast input: PASS"
)

print(
    "Actual weather as forecast input: PASS"
)


# ================================================================
# 24. STAGE-1 SUMMARY
# ================================================================

stage1_summary_path = (
    find_stage1_summary()
)


if stage1_summary_path is not None:

    print_header(
        "STAGE 1 OUTPUT FOUND"
    )

    print(
        stage1_summary_path
    )

    stage1_summary = pd.read_csv(
        stage1_summary_path
    )

    print(
        "\nStage-1 summary shape:",
        stage1_summary.shape
    )

else:

    stage1_summary = None

    print(
        "\nStage-1 summary not found."
    )


# ================================================================
# 25. IMPORTANT STAGE-1 / CURRENT FORECAST NOTE
# ================================================================
#
# The Stage-1 historical summary contains evaluation metrics.
#
# The current GFS forecast file contains current raw GFS forecast
# values.
#
# This script does NOT assume that the historical Stage-1 RMSE
# table can simply be applied as a correction factor to today's
# forecast.
#
# That correction must eventually come from the saved Stage-1
# trained models.
#
# Therefore, current forecast variables remain explicitly marked
# as raw block forecasts here.
# ================================================================


print(
    "\nStage-1 metric summary is informational only."
)

print(
    "No historical metric is directly applied to current forecast."
)


# ================================================================
# 26. CREATE GP × CURRENT BLOCK FORECAST MATRIX
# ================================================================

print_header(
    "CREATING GP-LEVEL FORECAST FRAME"
)


rows = []


for forecast_idx, forecast_row in (
    block_forecast.iterrows()
):

    target_date = (
        forecast_row["target_date"]
    )


    for gp_idx, gp_row in (
        gp.iterrows()
    ):

        row = {

            "forecast_row":
                forecast_idx,

            "target_date":
                target_date,

            "gp_index":
                gp_idx,

            "gp_id":
                (
                    gp_row["gp_id"]
                    if "gp_id" in gp.columns
                    else gp_idx
                ),

            "gp_latitude":
                gp_row[lat_col],

            "gp_longitude":
                gp_row[lon_col],

            "spatial_distance_from_mean":
                gp_row[
                    "spatial_distance_from_mean"
                ],

            "spatial_anomaly_score":
                gp_row[
                    "spatial_anomaly_score"
                ],
        }


        # --------------------------------------------------------
        # Attach block forecast variables
        # --------------------------------------------------------

        for column in forecast_candidates:

            row[column] = (
                forecast_row[column]
            )


        rows.append(
            row
        )


stage2 = pd.DataFrame(
    rows
)


print(
    "\nStage-2 GP forecast frame:",
    stage2.shape
)


expected_rows = (
    len(block_forecast)
    *
    len(gp)
)


if len(stage2) != expected_rows:

    raise RuntimeError(
        "Unexpected GP × forecast row count."
    )


# ================================================================
# 27. CONSTRUCT SPATIAL ALLOCATION WEIGHTS
# ================================================================
#
# The weights are normalized independently for every target date.
#
# Therefore:
#
#   sum(weight across 113 GPs)
#       =
#   1.0
#
# for every forecast horizon.
#
# ================================================================

print_header(
    "SPATIAL WEIGHT CONSTRUCTION"
)


def normalize_weights(group):

    group = group.copy()


    score = (
        group[
            "spatial_anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if not np.isfinite(score).all():

        raise ValueError(
            "Non-finite spatial anomaly score detected."
        )


    # ------------------------------------------------------------
    # Convert spatial score into positive relative weights.
    #
    # This is an INTERMEDIATE spatial allocation representation,
    # not a validated weather model.
    # ------------------------------------------------------------

    weights = np.exp(
        np.clip(
            score,
            -5,
            5
        )
    )


    weight_sum = (
        weights.sum()
    )


    if (
        weight_sum <= 0
        or
        not np.isfinite(weight_sum)
    ):

        raise ValueError(
            "Invalid spatial weight sum."
        )


    group[
        "spatial_weight"
    ] = (
        weights
        /
        weight_sum
    )


    return group


stage2 = (

    stage2

    .groupby(
        "forecast_row",
        group_keys=False
    )

    .apply(
        normalize_weights
    )

    .reset_index(
        drop=True
    )

)


# ================================================================
# 28. WEIGHT CONSERVATION CHECK
# ================================================================

weight_sum = (

    stage2

    .groupby(
        "forecast_row"
    )[

        "spatial_weight"

    ]

    .sum()

)


max_weight_error = np.max(

    np.abs(
        weight_sum.to_numpy()
        -
        1.0
    )

)


print(
    "Forecast groups:",
    len(weight_sum)
)

print(
    "Expected GP count per forecast:",
    len(gp)
)

print(
    "Maximum GP weight-sum error:",
    f"{max_weight_error:.12f}"
)


if max_weight_error > 1e-10:

    raise RuntimeError(
        "Spatial weights failed conservation check."
    )


# ================================================================
# 29. GP COUNT PER FORECAST CHECK
# ================================================================

gp_counts = (

    stage2

    .groupby(
        "forecast_row"
    )

    .size()

)


if not (
    gp_counts
    ==
    len(gp)
).all():

    raise RuntimeError(
        "Every forecast horizon must contain exactly "
        f"{len(gp)} GP records."
    )


print(
    "GP count consistency: PASS"
)


# ================================================================
# 30. SPATIAL WEIGHT VALIDITY CHECK
# ================================================================

if (
    stage2["spatial_weight"]
    <= 0
).any():

    raise RuntimeError(
        "Non-positive spatial weight detected."
    )


if (
    ~np.isfinite(
        stage2[
            "spatial_weight"
        ]
    )
).all():

    raise RuntimeError(
        "Invalid spatial weights detected."
    )


print(
    "Spatial weight positivity: PASS"
)


# ================================================================
# 31. SAVE GP SPATIAL FEATURE DATASET
# ================================================================

spatial_output = (

    OUTPUT_ROOT
    /
    "stage2_gp_spatial_features.csv"

)


gp.to_csv(
    spatial_output,
    index=False
)


print(
    "\nSaved GP spatial features:"
)

print(
    spatial_output
)


# ================================================================
# 32. SAVE STANDARDIZED SPATIAL FEATURES
# ================================================================

scaled_output = (

    OUTPUT_ROOT
    /
    "stage2_gp_spatial_features_scaled.csv"

)


scaled_export = (
    X_scaled
    .copy()
)


scaled_export.insert(
    0,
    "gp_index",
    gp.index
)


if "gp_id" in gp.columns:

    scaled_export.insert(
        1,
        "gp_id",
        gp["gp_id"]
    )


scaled_export.to_csv(
    scaled_output,
    index=False
)


print(
    "\nSaved standardized spatial features:"
)

print(
    scaled_output
)


# ================================================================
# 33. SAVE GP × BLOCK FORECAST MATRIX
# ================================================================

forecast_output = (

    OUTPUT_ROOT
    /
    "stage2_gp_block_forecast_matrix.csv"

)


stage2.to_csv(
    forecast_output,
    index=False
)


print(
    "\nSaved GP × block forecast matrix:"
)

print(
    forecast_output
)


# ================================================================
# 34. SAVE SPATIAL FEATURE MEDIANS
# ================================================================

median_output = (

    OUTPUT_ROOT
    /
    "stage2_spatial_feature_medians.csv"

)


feature_medians.to_csv(
    median_output,
    header=["median"]
)


print(
    "\nSaved spatial feature medians:"
)

print(
    median_output
)


# ================================================================
# 35. SAVE FEATURE MANIFEST
# ================================================================

manifest = {

    "project":
        "SIH26",

    "stage":
        2,

    "state":
        "Maharashtra",

    "district":
        "Nashik",

    "block":
        "Sinnar",

    "gp_count":
        int(len(gp)),

    "forecast_horizon_count":
        int(len(block_forecast)),

    "gp_forecast_rows":
        int(len(stage2)),

    "candidate_spatial_features":
        int(len(spatial_features)),

    "spatial_features":
        spatial_features,

    "temporally_unsafe_lsts_excluded":
        True,

    "gp_identifiers_excluded":
        True,

    "gp_coordinates_excluded_as_predictors":
        True,

    "actual_weather_used_as_gp_target":
        False,

    "synthetic_gp_target_used":
        False,

    "interpolated_gp_target_used":
        False,

    "supervised_gp_weather_target_available":
        False,

    "spatial_weight_conservation":
        True,

    "maximum_weight_sum_error":
        float(max_weight_error),

    "weight_definition":
        (
            "Positive spatial allocation weights derived "
            "from standardized GP spatial heterogeneity. "
            "Weights are normalized independently for every "
            "forecast horizon."
        ),

    "scientific_status":
        (
            "Stage-2 spatial preparation only. "
            "The spatial weights are not validated GP weather "
            "predictions and must not be reported as accuracy."
        ),

    "excluded_feature_policy":
        {
            "actual_weather":
                "excluded",

            "forecast_weather":
                "excluded_from_spatial_predictors",

            "lst_without_issue_time_cutoff":
                "excluded",

            "gp_identifier":
                "excluded",

            "coordinates":
                "metadata_only",
        },

}


manifest_path = (

    OUTPUT_ROOT
    /
    "stage2_manifest.json"

)


with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ================================================================
# 36. SAVE WEIGHT CONSERVATION AUDIT
# ================================================================

weight_audit = (

    stage2

    .groupby(
        "forecast_row"
    )

    .agg(

        target_date=(
            "target_date",
            "first"
        ),

        gp_count=(
            "gp_index",
            "count"
        ),

        weight_sum=(
            "spatial_weight",
            "sum"
        ),

        minimum_weight=(
            "spatial_weight",
            "min"
        ),

        maximum_weight=(
            "spatial_weight",
            "max"
        ),

    )

    .reset_index()

)


weight_audit[
    "weight_sum_error"
] = (

    weight_audit[
        "weight_sum"
    ]
    -
    1.0

)


weight_audit_path = (

    OUTPUT_ROOT
    /
    "stage2_weight_conservation_audit.csv"

)


weight_audit.to_csv(
    weight_audit_path,
    index=False
)


# ================================================================
# 37. FINAL REPORT
# ================================================================

print_header(
    "STAGE 2 PREPARATION COMPLETE"
)


print(
    f"""
Block                         : Sinnar, Nashik
GPs processed                 : {len(gp)}
Forecast horizons             : {len(block_forecast)}
GP × forecast rows            : {len(stage2)}

Final spatial predictors      : {len(spatial_features)}

LST without cutoff            : EXCLUDED
GP identifiers as predictors  : EXCLUDED
Coordinates as predictors     : EXCLUDED

Synthetic GP target           : NO
Interpolated GP target        : NO
Actual weather as GP target   : NO

Forecast schema               : PASS
GP coordinates                : PASS
GP count consistency          : PASS
Spatial weight positivity     : PASS
Spatial weight conservation   : PASS

Maximum weight-sum error      :
    {max_weight_error:.12e}

Outputs:

  {spatial_output}

  {scaled_output}

  {forecast_output}

  {median_output}

  {weight_audit_path}

  {manifest_path}


SCIENTIFIC STATUS
-----------------
This script has prepared the Stage-2 spatial input framework.

The spatial weights are NOT yet a validated GP-level weather
prediction model.

No fabricated or interpolated GP weather targets have been used.

The next Stage-2 step is to define and implement the actual
spatial downscaling formulation using legitimate physical
constraints and the validated GP predictors.
"""
)

print_header(
    "END"
)