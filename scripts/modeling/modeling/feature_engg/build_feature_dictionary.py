from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime


# ============================================================
# PROJECT ROOT
# scripts/modeling/build_feature_dictionary.py
# parents[2] = SIH26_Downscaling
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_ROOT = PROJECT_ROOT / "datasets" / "cleaned"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "feature_dictionary"


# ============================================================
# UNIVERSAL LOCATION HELPERS
# ============================================================

def norm(value):
    """Normalize names for filesystem matching."""
    return "".join(
        c.lower()
        for c in str(value)
        if c.isalnum()
    )


def resolve_location_root(state, district, block):
    """
    Resolve:
    datasets/cleaned/<state>/<district>/<block>

    without hardcoding spelling, spaces, underscores, etc.
    """

    wanted_state = norm(state)
    wanted_district = norm(district)
    wanted_block = norm(block)

    if not CLEANED_ROOT.exists():
        raise FileNotFoundError(
            f"Cleaned dataset root does not exist:\n{CLEANED_ROOT}"
        )

    matches = []

    for path in CLEANED_ROOT.rglob("*"):

        if not path.is_dir():
            continue

        if norm(path.name) != wanted_block:
            continue

        district_dir = path.parent
        state_dir = district_dir.parent

        if norm(district_dir.name) != wanted_district:
            continue

        if norm(state_dir.name) != wanted_state:
            continue

        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "Could not find cleaned dataset location.\n\n"
            f"State   : {state}\n"
            f"District: {district}\n"
            f"Block   : {block}\n\n"
            f"Searched under:\n{CLEANED_ROOT}"
        )

    if len(matches) > 1:
        print("\nWARNING: Multiple matching cleaned locations found:")
        for m in matches:
            print(f"  {m}")

        print(f"\nUsing:\n{matches[0]}")

    return matches[0]


# ============================================================
# DATASET DEFINITIONS
# ============================================================

DATASET_DEFINITIONS = [
    {
        "name": "block_observations_daily",
        "level": "block",
        "folder": "block",
        "normal_file": "block_observations_daily.csv",
        "model_file": "block_observations_daily_model_features.csv",
    },
    {
        "name": "block_forecast_training_base",
        "level": "block",
        "folder": "block",
        "normal_file": "block_forecast_training_base.csv",
        "model_file": "block_forecast_training_base_model_features.csv",
    },
    {
        "name": "block_current_forecast",
        "level": "block",
        "folder": "block",
        "normal_file": "block_current_forecast.csv",
        "model_file": "block_current_forecast_model_features.csv",
    },
    {
        "name": "gp_merged_features",
        "level": "gp",
        "folder": "gp",
        "normal_file": "gp_merged_features.csv",
        "model_file": "gp_merged_features_model_features.csv",
    },
]


# ============================================================
# FILE RESOLUTION
# ============================================================

def resolve_dataset_file(location_root, definition):
    """
    Prefer the final MODEL FEATURES file.

    If unavailable, fall back to the normal cleaned file.
    """

    folder = location_root / definition["folder"]

    model_path = folder / definition["model_file"]

    if model_path.exists():
        return model_path, "model_features"

    normal_path = folder / definition["normal_file"]

    if normal_path.exists():
        return normal_path, "cleaned"

    return None, None


# ============================================================
# MANIFEST
# ============================================================

def load_unavailable_manifest(location_root):
    """
    Read the 100%-unavailable feature removal manifest.

    This allows removed features to remain documented in the
    feature dictionary even though they are absent from the
    final model-feature CSV.
    """

    manifest_path = (
        location_root /
        "100pct_unavailable_feature_manifest.json"
    )

    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        print(
            f"WARNING: Could not read manifest:\n"
            f"{manifest_path}\n"
            f"{exc}"
        )
        return {}


# ============================================================
# COLUMN TYPE HELPERS
# ============================================================

def is_identifier_column(name):
    n = name.lower()

    exact = {
        "id",
        "gp_id",
        "block_id",
        "district_id",
        "state_id",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "geometry",
        "geom",
        "fid",
        "objectid",
    }

    if n in exact:
        return True

    patterns = [
        r"^gp_?id$",
        r"^block_?id$",
        r"^district_?id$",
        r"^state_?id$",
        r"_id$",
    ]

    return any(re.search(p, n) for p in patterns)


def is_temporal_column(name):
    n = name.lower()

    keywords = [
        "date",
        "datetime",
        "timestamp",
        "time",
        "issue_time",
        "issue_date",
        "target_time",
        "target_date",
        "valid_time",
        "lead_days",
        "lead_hours",
        "forecast_run",
        "run_timestamp",
    ]

    return any(k in n for k in keywords)


def is_provenance_column(name):
    n = name.lower()

    keywords = [
        "source",
        "provider",
        "dataset",
        "collection",
        "product",
        "version",
        "retrieved",
        "retrieval",
        "url",
        "api",
        "provenance",
        "synthetic",
        "interpolat",
        "fallback",
        "method",
        "processing",
        "crs",
        "epsg",
    ]

    return any(k in n for k in keywords)


def is_availability_column(name):
    n = name.lower()

    keywords = [
        "availability",
        "available",
        "acquisition_count",
        "valid_pixel",
        "valid_count",
        "valid_fraction",
        "coverage",
        "cloud_percentage",
        "cloud_pct",
        "missing_count",
        "observation_count",
        "image_count",
        "scene_count",
    ]

    return any(k in n for k in keywords)


# ============================================================
# WEATHER VARIABLE IDENTIFICATION
# ============================================================

WEATHER_VARIABLE_KEYWORDS = [
    "temperature",
    "temp",
    "relative_humidity",
    "humidity",
    "precipitation",
    "precip",
    "rain",
    "rainfall",
    "wind",
    "pressure",
    "dewpoint",
    "dew_point",
    "cloud",
    "radiation",
    "shortwave",
    "longwave",
    "evapotranspiration",
    "et0",
]


def is_weather_variable(name):
    n = name.lower()

    return any(
        k in n
        for k in WEATHER_VARIABLE_KEYWORDS
    )


def is_actual_weather(name):
    n = name.lower()

    if not is_weather_variable(n):
        return False

    return (
        n.startswith("actual_")
        or "_actual_" in n
        or n.startswith("obs_")
        or n.startswith("observed_")
    )


def is_forecast_feature(name):
    n = name.lower()

    forecast_tokens = [
        "forecast",
        "previous_day",
        "lead",
        "_d1",
        "_d2",
        "_d3",
        "_d4",
        "_d5",
        "_d6",
        "_d7",
        "gfs",
    ]

    return any(
        token in n
        for token in forecast_tokens
    )


def is_reanalysis_feature(name):
    n = name.lower()

    return (
        "era5" in n
        or "reanalysis" in n
    )


# ============================================================
# SATELLITE / SPATIAL FEATURE IDENTIFICATION
# ============================================================

def is_satellite_feature(name):
    n = name.lower()

    satellite_tokens = [
        "sentinel1",
        "sentinel_1",
        "s1_",
        "sentinel2",
        "sentinel_2",
        "s2_",
        "landsat",
        "lst",
        "ndvi",
        "evi",
        "savi",
        "ndmi",
        "nbr",
        "vh",
        "vv",
        "incidence_angle",
        "orbit",
    ]

    return any(
        token in n
        for token in satellite_tokens
    )


def is_static_spatial_feature(name):
    n = name.lower()

    spatial_tokens = [
        "elevation",
        "dem",
        "slope",
        "aspect",
        "terrain",
        "landcover",
        "land_cover",
        "soil",
        "texture",
        "sand",
        "clay",
        "silt",
        "organic_carbon",
        "bulk_density",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]

    return any(
        token in n
        for token in spatial_tokens
    )


# ============================================================
# FEATURE CLASSIFICATION
# ============================================================

def classify_feature(
    column,
    dataset_name,
    level,
    series,
    missing_count,
    missing_percent,
    all_missing,
    constant_non_null,
):
    """
    Scientific classification of a feature.

    This does NOT automatically perform feature engineering.
    It creates the feature dictionary / contract for later
    modeling.
    """

    n = column.lower()

    # --------------------------------------------------------
    # 1. Completely unavailable
    # --------------------------------------------------------

    if all_missing:
        return {
            "feature_group": "unavailable",
            "final_role": "exclude",
            "engineering_action": "exclude_100pct_unavailable",
            "availability_rule": "unavailable_in_current_dataset",
            "leakage_risk": "none",
            "scientific_reason": (
                "Feature has no observed values in the available "
                "dataset and therefore cannot provide empirical "
                "predictive information for this location."
            ),
        }

    # --------------------------------------------------------
    # 2. Identifier / spatial identity
    # --------------------------------------------------------

    if is_identifier_column(column):

        if n in {"latitude", "longitude", "lat", "lon"}:
            return {
                "feature_group": "spatial_location",
                "final_role": "candidate_predictor",
                "engineering_action": "retain_for_spatial_modeling",
                "availability_rule": "static",
                "leakage_risk": "low",
                "scientific_reason": (
                    "Geographical coordinates describe the spatial "
                    "position of the observation or GP."
                ),
            }

        return {
            "feature_group": "identifier",
            "final_role": "metadata",
            "engineering_action": "retain_for_joining_only",
            "availability_rule": "static",
            "leakage_risk": "none",
            "scientific_reason": (
                "Identifier used for entity tracking, joins or "
                "provenance rather than as a physical predictor."
            ),
        }

    # --------------------------------------------------------
    # 3. Temporal columns
    # --------------------------------------------------------

    if is_temporal_column(column):

        return {
            "feature_group": "temporal_control",
            "final_role": "temporal_alignment",
            "engineering_action": "retain_for_temporal_alignment",
            "availability_rule": (
                "must_not expose information unavailable at "
                "prediction_issue_time"
            ),
            "leakage_risk": "medium",
            "scientific_reason": (
                "Temporal information is required to align forecasts, "
                "observations, lead times and prediction targets."
            ),
        }

    # --------------------------------------------------------
    # 4. Provenance
    # --------------------------------------------------------

    if is_provenance_column(column):

        return {
            "feature_group": "provenance",
            "final_role": "metadata",
            "engineering_action": "retain_for_audit_provenance",
            "availability_rule": "static_or_source_defined",
            "leakage_risk": "none",
            "scientific_reason": (
                "Documents data origin, processing or collection "
                "history and should not be treated as a physical "
                "weather predictor."
            ),
        }

    # --------------------------------------------------------
    # 5. Availability indicators
    # --------------------------------------------------------

    if is_availability_column(column):

        return {
            "feature_group": "data_availability",
            "final_role": "candidate_predictor",
            "engineering_action": "retain_as_availability_feature",
            "availability_rule": (
                "must represent information available by "
                "prediction_issue_time"
            ),
            "leakage_risk": "medium",
            "scientific_reason": (
                "Observation availability and quality can contain "
                "useful information about the reliability of remote "
                "sensing or weather observations."
            ),
        }

    # --------------------------------------------------------
    # 6. Actual observed weather
    # --------------------------------------------------------

    if is_actual_weather(column):

        return {
            "feature_group": "historical_weather",
            "final_role": "historical_source",
            "engineering_action": (
                "use_only_as_past_lag_rolling_feature_or_target"
            ),
            "availability_rule": (
                "only observations available before the "
                "prediction_issue_time"
            ),
            "leakage_risk": "high_if_same_target",
            "scientific_reason": (
                "Observed weather variables are valuable for "
                "historical context and targets, but using the "
                "same-target observation as an input would create "
                "temporal leakage."
            ),
        }

    # --------------------------------------------------------
    # 7. Forecast features
    # --------------------------------------------------------

    if is_forecast_feature(column):

        return {
            "feature_group": "weather_forecast",
            "final_role": "forecast_predictor",
            "engineering_action": (
                "retain_with_issue_time_and_lead_alignment"
            ),
            "availability_rule": (
                "forecast must have been issued and available "
                "at prediction_issue_time"
            ),
            "leakage_risk": "controlled",
            "scientific_reason": (
                "Numerical weather forecast information is the "
                "primary block-level forecast input and must be "
                "aligned using forecast issue time and lead time."
            ),
        }

    # --------------------------------------------------------
    # 8. ERA5 / reanalysis
    # --------------------------------------------------------

    if is_reanalysis_feature(column):

        return {
            "feature_group": "reanalysis",
            "final_role": "candidate_predictor",
            "engineering_action": (
                "retain_with_temporal_availability_control"
            ),
            "availability_rule": (
                "use only when the reanalysis value would have "
                "been available at prediction_issue_time"
            ),
            "leakage_risk": "medium",
            "scientific_reason": (
                "Reanalysis variables describe atmospheric or soil "
                "conditions and can provide useful environmental "
                "context, subject to real-time availability constraints."
            ),
        }

    # --------------------------------------------------------
    # 9. Satellite / remote sensing
    # --------------------------------------------------------

    if is_satellite_feature(column):

        return {
            "feature_group": "remote_sensing",
            "final_role": "gp_spatial_predictor",
            "engineering_action": (
                "retain_with_acquisition_date_and_availability_control"
            ),
            "availability_rule": (
                "satellite observation must be available on or "
                "before prediction_issue_time"
            ),
            "leakage_risk": "medium",
            "scientific_reason": (
                "Satellite observations provide spatially resolved "
                "surface, vegetation, radar or land-surface information "
                "useful for block-to-GP downscaling."
            ),
        }

    # --------------------------------------------------------
    # 10. Static spatial/environmental features
    # --------------------------------------------------------

    if is_static_spatial_feature(column):

        return {
            "feature_group": "static_spatial_environment",
            "final_role": "gp_spatial_predictor",
            "engineering_action": "retain_as_static_spatial_predictor",
            "availability_rule": "static",
            "leakage_risk": "low",
            "scientific_reason": (
                "Static terrain, soil, land-cover and location "
                "characteristics describe persistent spatial "
                "heterogeneity between GPs."
            ),
        }

    # --------------------------------------------------------
    # 11. Constant physical variables
    # --------------------------------------------------------

    if constant_non_null:

        return {
            "feature_group": "constant_physical_or_metadata",
            "final_role": "review",
            "engineering_action": (
                "retain_for_multi_block_review_before_removal"
            ),
            "availability_rule": "dataset_defined",
            "leakage_risk": "low",
            "scientific_reason": (
                "The feature is constant within this dataset/block. "
                "It may still contain useful information when a model "
                "is trained across multiple blocks, so it should not "
                "be automatically deleted."
            ),
        }

    # --------------------------------------------------------
    # 12. Generic numeric/categorical feature
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        return {
            "feature_group": "other_numeric",
            "final_role": "candidate_predictor",
            "engineering_action": "retain_for_scientific_review",
            "availability_rule": (
                "must be available by prediction_issue_time "
                "if used for forecasting"
            ),
            "leakage_risk": "review",
            "scientific_reason": (
                "Numeric feature not matched to a predefined source "
                "category; retain until its physical meaning and "
                "temporal availability are documented."
            ),
        }

    return {
        "feature_group": "other_categorical",
        "final_role": "metadata_or_candidate",
        "engineering_action": "retain_for_scientific_review",
        "availability_rule": "dataset_defined",
        "leakage_risk": "review",
        "scientific_reason": (
            "Categorical feature whose scientific role requires "
            "dataset-specific review before modeling."
        ),
    }


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def inspect_dataset(
    file_path,
    dataset_name,
    level,
    source_variant,
):
    print(f"\nReading:")
    print(f"  {file_path}")

    df = pd.read_csv(file_path)

    print(
        f"  Rows={len(df):,} | "
        f"Columns={len(df.columns):,}"
    )

    records = []

    for column in df.columns:

        series = df[column]

        row_count = len(df)

        missing_count = int(
            series.isna().sum()
        )

        missing_percent = (
            missing_count / row_count * 100
            if row_count > 0
            else 100.0
        )

        all_missing = (
            row_count > 0
            and missing_count == row_count
        )

        non_null = series.dropna()

        constant_non_null = (
            len(non_null) > 0
            and non_null.nunique(dropna=True) <= 1
        )

        classification = classify_feature(
            column=column,
            dataset_name=dataset_name,
            level=level,
            series=series,
            missing_count=missing_count,
            missing_percent=missing_percent,
            all_missing=all_missing,
            constant_non_null=constant_non_null,
        )

        record = {
            "column": column,
            "source_dataset": dataset_name,
            "source_variant": source_variant,
            "level": level,
            "dtype": str(series.dtype),
            "row_count": row_count,
            "missing_count": missing_count,
            "missing_percent": round(
                missing_percent,
                4,
            ),
            "all_missing_100pct": bool(
                all_missing
            ),
            "constant_non_null": bool(
                constant_non_null
            ),
            **classification,
        }

        records.append(record)

    return records


# ============================================================
# REMOVED 100% UNAVAILABLE FEATURES
# ============================================================

def extract_removed_features(
    manifest,
    existing_columns_by_dataset,
):
    """
    Add features removed by the 100%-unavailable rule.

    The manifest may store removed columns either as:
        ["feature_a", "feature_b"]

    or as structured records such as:
        [
            {"column": "feature_a", ...},
            {"column": "feature_b", ...}
        ]

    This function supports both formats.
    """

    records = []

    if not manifest:
        return records

    datasets_section = manifest.get(
        "datasets",
        {}
    )

    if not isinstance(datasets_section, dict):
        return records

    for dataset_name, info in datasets_section.items():

        if not isinstance(info, dict):
            continue

        removed_columns = info.get(
            "removed_columns",
            []
        )

        if not removed_columns:
            continue

        # ----------------------------------------------------
        # Determine dataset level
        # ----------------------------------------------------

        if dataset_name == "gp_merged_features":
            level = "gp"
        else:
            level = "block"

        # ----------------------------------------------------
        # Existing columns in current model-feature file
        # ----------------------------------------------------

        existing = existing_columns_by_dataset.get(
            dataset_name,
            set()
        )

        # Make absolutely sure this is a set of strings.
        existing = {
            str(x)
            for x in existing
        }

        # ----------------------------------------------------
        # Normalize manifest records
        # ----------------------------------------------------

        normalized_removed = []

        for item in removed_columns:

            # Case 1:
            # Plain string
            if isinstance(item, str):

                normalized_removed.append(
                    {
                        "column": item
                    }
                )

            # Case 2:
            # Structured dictionary
            elif isinstance(item, dict):

                column_name = (
                    item.get("column")
                    or item.get("name")
                    or item.get("feature")
                    or item.get("feature_name")
                )

                if column_name is None:
                    print(
                        "\nWARNING: Could not determine "
                        "removed feature name from manifest record:"
                    )
                    print(item)
                    continue

                normalized_removed.append(
                    {
                        "column": str(column_name),
                        "manifest_record": item,
                    }
                )

            else:

                print(
                    "\nWARNING: Ignoring unsupported "
                    "removed-column manifest entry:"
                )
                print(item)

        # ----------------------------------------------------
        # Create dictionary records
        # ----------------------------------------------------

        for item in normalized_removed:

            column = item["column"]

            # If the feature is somehow still present in the
            # current model-feature dataset, don't duplicate it.
            if column in existing:
                continue

            records.append(
                {
                    "column": column,
                    "source_dataset": dataset_name,
                    "source_variant": (
                        "removed_100pct_unavailable"
                    ),
                    "level": level,
                    "dtype": (
                        "not_present_in_model_file"
                    ),
                    "row_count": None,
                    "missing_count": None,
                    "missing_percent": 100.0,
                    "all_missing_100pct": True,
                    "constant_non_null": False,
                    "feature_group": "unavailable",
                    "final_role": "exclude",
                    "engineering_action": (
                        "exclude_100pct_unavailable"
                    ),
                    "availability_rule": (
                        "unavailable_in_current_dataset"
                    ),
                    "leakage_risk": "none",
                    "scientific_reason": (
                        "Feature was removed from the "
                        "model-feature dataset because it "
                        "was completely unavailable for "
                        "the selected location."
                    ),
                }
            )

    return records

# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    records,
    state,
    district,
    block,
    location_root,
):

    df = pd.DataFrame(records)

    summary = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "project_root": str(PROJECT_ROOT),
        "location": {
            "state": state,
            "district": district,
            "block": block,
        },
        "location_root": str(location_root),
        "total_dictionary_records": int(
            len(df)
        ),
        "datasets": {},
        "feature_groups": {},
        "final_roles": {},
        "engineering_actions": {},
        "unavailable_features": [],
    }

    if df.empty:
        return summary

    # Dataset counts
    for dataset_name, group in df.groupby(
        "source_dataset"
    ):
        summary["datasets"][dataset_name] = {
            "feature_count": int(len(group)),
            "100pct_missing": int(
                group["all_missing_100pct"].sum()
            ),
        }

    # Feature groups
    summary["feature_groups"] = {
        str(k): int(v)
        for k, v in df[
            "feature_group"
        ].value_counts().items()
    }

    # Final roles
    summary["final_roles"] = {
        str(k): int(v)
        for k, v in df[
            "final_role"
        ].value_counts().items()
    }

    # Engineering actions
    summary["engineering_actions"] = {
        str(k): int(v)
        for k, v in df[
            "engineering_action"
        ].value_counts().items()
    }

    # 100% unavailable
    unavailable = df[
        df["all_missing_100pct"] == True
    ]

    summary["unavailable_features"] = (
        unavailable[
            [
                "source_dataset",
                "column",
            ]
        ]
        .to_dict(orient="records")
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIH26 FEATURE DICTIONARY GENERATOR")
    print("=" * 70)

    print(
        "\nThis script regenerates the feature dictionary from the "
        "latest cleaned/model-feature datasets."
    )

    print(
        "\nNo raw, processed or cleaned source dataset will be modified."
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    state = input(
        "\nState: "
    ).strip()

    district = input(
        "District: "
    ).strip()

    block = input(
        "Block: "
    ).strip()

    if not state or not district or not block:
        raise ValueError(
            "State, District and Block are required."
        )

    # --------------------------------------------------------
    # LOCATION ROOT
    # --------------------------------------------------------

    location_root = resolve_location_root(
        state,
        district,
        block,
    )

    print("\nResolved cleaned location:")
    print(f"  {location_root}")

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / state
        / district
        / block
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    all_records = []

    existing_columns_by_dataset = {}

    loaded_datasets = 0

    for definition in DATASET_DEFINITIONS:

        file_path, source_variant = (
            resolve_dataset_file(
                location_root,
                definition,
            )
        )

        if file_path is None:

            print(
                f"\nWARNING: Dataset not found:"
                f" {definition['name']}"
            )

            continue

        try:

            records = inspect_dataset(
                file_path=file_path,
                dataset_name=definition["name"],
                level=definition["level"],
                source_variant=source_variant,
            )

            all_records.extend(records)

            existing_columns_by_dataset[
                definition["name"]
            ] = {
                r["column"]
                for r in records
            }

            loaded_datasets += 1

        except Exception as exc:

            print(
                f"\nERROR while reading "
                f"{definition['name']}:\n"
                f"{exc}"
            )

    if loaded_datasets == 0:
        raise RuntimeError(
            "No usable cleaned/model-feature datasets were found."
        )

    # --------------------------------------------------------
    # LOAD REMOVAL MANIFEST
    # --------------------------------------------------------

    manifest = load_unavailable_manifest(
        location_root
    )

    removed_records = extract_removed_features(
        manifest=manifest,
        existing_columns_by_dataset=(
            existing_columns_by_dataset
        ),
    )

    if removed_records:

        print(
            f"\nDocumenting "
            f"{len(removed_records)} "
            f"100%-unavailable removed features."
        )

        all_records.extend(
            removed_records
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    dictionary_df = pd.DataFrame(
        all_records
    )

    if dictionary_df.empty:
        raise RuntimeError(
            "Feature dictionary is empty."
        )

    # --------------------------------------------------------
    # SORTING
    # --------------------------------------------------------

    dataset_order = {
        "block_observations_daily": 1,
        "block_forecast_training_base": 2,
        "block_current_forecast": 3,
        "gp_merged_features": 4,
    }

    dictionary_df["_dataset_order"] = (
        dictionary_df["source_dataset"]
        .map(dataset_order)
        .fillna(99)
    )

    dictionary_df = (
        dictionary_df
        .sort_values(
            [
                "_dataset_order",
                "feature_group",
                "column",
            ]
        )
        .drop(
            columns=["_dataset_order"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # OUTPUT CSV
    # --------------------------------------------------------

    csv_path = (
        output_dir /
        "feature_dictionary.csv"
    )

    dictionary_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # OUTPUT JSON
    # --------------------------------------------------------

    json_path = (
        output_dir /
        "feature_dictionary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dictionary_df.to_dict(
                orient="records"
            ),
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = build_summary(
        records=all_records,
        state=state,
        district=district,
        block=block,
        location_root=location_root,
    )

    summary_path = (
        output_dir /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE DICTIONARY GENERATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nLocation:"
        f" {state} / {district} / {block}"
    )

    print(
        f"\nDatasets loaded: "
        f"{loaded_datasets}"
    )

    print(
        f"Total dictionary records: "
        f"{len(dictionary_df):,}"
    )

    print("\nOutput files:")

    print(
        f"  CSV    : {csv_path}"
    )

    print(
        f"  JSON   : {json_path}"
    )

    print(
        f"  Summary: {summary_path}"
    )

    print("\nFeature groups:")

    for key, value in summary[
        "feature_groups"
    ].items():

        print(
            f"  {key:<35} {value}"
        )

    print("\nFinal roles:")

    for key, value in summary[
        "final_roles"
    ].items():

        print(
            f"  {key:<35} {value}"
        )

    unavailable_count = len(
        summary["unavailable_features"]
    )

    print(
        f"\n100% unavailable/excluded features: "
        f"{unavailable_count}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()