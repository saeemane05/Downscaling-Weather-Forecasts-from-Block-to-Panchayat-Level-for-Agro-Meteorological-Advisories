from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

# ================================================================
# SIH26 DOWNscaling
# FINAL PRE-TRAINING ANALYSIS / GATEKEEPER
#
# Purpose:
#   Perform the final scientific audit before Stage-1/Stage-2 training.
#
# IMPORTANT:
#   This script NEVER assumes wind units and NEVER performs automatic
#   unit conversion.
#
# Decision outputs:
#   READY_FOR_TRAINING
#   WIND_REQUIRES_SOURCE_FIX
#   DO_NOT_TRAIN_YET
# ================================================================


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLOCK_ROOT = (
    PROJECT_ROOT
    / "block"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

RAW_FORECAST = (
    BLOCK_ROOT
    / "raw"
    / "historical_forecasts"
    / "open_meteo_previous_runs_d1.csv"
)

ACTUAL_FILE = (
    BLOCK_ROOT
    / "processed"
    / "open_meteo_actual_daily.csv"
)

TRAINING_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "training"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

VALIDATION_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "Maharashtra"
    / "Nashik"
    / "Sinnar"
)

OUTPUT_DIR = VALIDATION_ROOT / "final_pretraining_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def safe_read_csv(path):
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Could not read {path}")
        print(e)
        return None


def normalize_text(value):
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip().lower()


def search_project_text(patterns):
    """
    Search Python / TXT / JSON / YAML / CSV metadata files for
    Open-Meteo wind configuration.

    This is intentionally evidence-based.
    It does NOT infer units from numerical values.
    """

    matches = []

    extensions = {
        ".py",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".csv",
    }

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        # Skip massive / irrelevant generated directories
        path_string = str(path).lower()

        if any(
            x in path_string
            for x in [
                "\\.git\\",
                "\\__pycache__\\",
                "\\node_modules\\",
                "\\datasets\\validation\\final_pretraining_analysis\\",
            ]
        ):
            continue

        try:
            if path.stat().st_size > 20 * 1024 * 1024:
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception:
            continue

        lower = text.lower()

        found = []

        for pattern in patterns:
            if pattern.lower() in lower:
                found.append(pattern)

        if found:
            matches.append(
                {
                    "file": str(path),
                    "patterns": found,
                }
            )

    return matches


def find_unit_mentions(text):
    """
    Extract explicit unit mentions only.
    """

    units = []

    patterns = [
        r"wind_speed_unit\s*=\s*['\"]([^'\"]+)['\"]",
        r"wind_speed_unit\s*:\s*['\"]([^'\"]+)['\"]",
        r"windspeed_unit\s*=\s*['\"]([^'\"]+)['\"]",
        r"windspeed_unit\s*:\s*['\"]([^'\"]+)['\"]",
        r"wind_gusts_unit\s*=\s*['\"]([^'\"]+)['\"]",
        r"wind_gusts_unit\s*:\s*['\"]([^'\"]+)['\"]",
        r"wind_speed_unit\s*=\s*([A-Za-z]+)",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.I):
            units.append(str(match))

    return sorted(set(units))


def find_height_mentions(text):
    heights = []

    patterns = [
        r"wind_speed_10m",
        r"wind_gusts_10m",
        r"wind_speed.*10m",
        r"windspeed.*10m",
    ]

    for pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            heights.append("10m")

    return sorted(set(heights))


# ================================================================
# 3. LOAD SOURCES
# ================================================================

section("1. LOADING FINAL SOURCE DATA")

forecast = safe_read_csv(RAW_FORECAST)
actual = safe_read_csv(ACTUAL_FILE)

if forecast is None or actual is None:
    print("\nFINAL DECISION: DO_NOT_TRAIN_YET")
    raise SystemExit(1)


print(f"Forecast shape : {forecast.shape}")
print(f"Actual shape   : {actual.shape}")


# ================================================================
# 4. REQUIRED COLUMN CHECK
# ================================================================

section("2. REQUIRED COLUMN AUDIT")

forecast_required = [
    "date",
    "forecast_issued_date",
    "forecast_lead_days",
    "wind_speed_forecast",
    "wind_gust_forecast",
]

actual_required = [
    "date",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

forecast_missing = [
    c for c in forecast_required
    if c not in forecast.columns
]

actual_missing = [
    c for c in actual_required
    if c not in actual.columns
]

print("Forecast missing columns:", forecast_missing)
print("Actual missing columns:", actual_missing)

required_columns_pass = (
    len(forecast_missing) == 0
    and len(actual_missing) == 0
)

print(
    "Required columns:",
    pass_fail(required_columns_pass)
)


# ================================================================
# 5. DATE / LEAD-TIME AUDIT
# ================================================================

section("3. FORECAST TEMPORAL VALIDITY AUDIT")

forecast["date"] = pd.to_datetime(
    forecast["date"],
    errors="coerce"
)

forecast["forecast_issued_date"] = pd.to_datetime(
    forecast["forecast_issued_date"],
    errors="coerce"
)

actual["date"] = pd.to_datetime(
    actual["date"],
    errors="coerce"
)

forecast["calculated_lead"] = (
    forecast["date"]
    - forecast["forecast_issued_date"]
).dt.days

lead_match = (
    forecast["calculated_lead"]
    == forecast["forecast_lead_days"]
)

print(
    "Rows:",
    len(forecast)
)

print(
    "Invalid forecast dates:",
    forecast["date"].isna().sum()
)

print(
    "Invalid issue dates:",
    forecast["forecast_issued_date"].isna().sum()
)

print(
    "Lead-time mismatches:",
    (~lead_match).sum()
)

temporal_pass = (
    forecast["date"].notna().all()
    and forecast["forecast_issued_date"].notna().all()
    and lead_match.all()
)

print(
    "Temporal forecast integrity:",
    pass_fail(temporal_pass)
)


# ================================================================
# 6. WIND NUMERICAL QUALITY
# ================================================================

section("4. WIND NUMERICAL QUALITY AUDIT")

wind_columns = [
    "wind_speed_forecast",
    "wind_gust_forecast",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

for column in wind_columns:

    if column in forecast.columns:
        series = forecast[column]

    elif column in actual.columns:
        series = actual[column]

    else:
        continue

    print(f"\n{column}")
    print("  missing:", series.isna().sum())
    print("  negative:", (series < 0).sum())
    print("  min:", series.min())
    print("  mean:", series.mean())
    print("  max:", series.max())

forecast_wind_valid = (
    forecast["wind_speed_forecast"].notna().all()
    and (forecast["wind_speed_forecast"] >= 0).all()
)

actual_wind_valid = (
    actual["wind_speed_10m_mean"].notna().all()
    and (actual["wind_speed_10m_mean"] >= 0).all()
)

numerical_wind_pass = (
    forecast_wind_valid
    and actual_wind_valid
)

print(
    "\nNumerical wind validity:",
    pass_fail(numerical_wind_pass)
)


# ================================================================
# 7. EXPLICIT UNIT / SOURCE SEARCH
# ================================================================

section("5. ORIGINAL OPEN-METEO SOURCE / UNIT PROVENANCE SEARCH")

patterns = [
    "wind_speed_unit",
    "windspeed_unit",
    "wind_gusts_unit",
    "wind_speed_10m",
    "wind_speed",
    "wind_gust",
    "open-meteo.com",
    "open_meteo",
    "hourly=",
    "daily=",
    "forecast_days",
    "past_days",
]

source_matches = search_project_text(patterns)

print(
    "Files containing relevant wind/source configuration:",
    len(source_matches)
)

for item in source_matches:

    print("\nFILE:")
    print(item["file"])

    print("MATCHES:")
    for p in item["patterns"]:
        print("  -", p)


# ================================================================
# 8. EXPLICIT UNIT EXTRACTION
# ================================================================

section("6. EXPLICIT WIND UNIT EXTRACTION")

explicit_unit_evidence = []

for item in source_matches:

    path = Path(item["file"])

    try:
        text_content = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        continue

    units = find_unit_mentions(text_content)

    if units:

        explicit_unit_evidence.append(
            {
                "file": str(path),
                "units": units,
            }
        )

if explicit_unit_evidence:

    for item in explicit_unit_evidence:

        print("\nFILE:")
        print(item["file"])

        print(
            "Explicit units:",
            item["units"]
        )

else:

    print(
        "NO EXPLICIT WIND UNIT CONFIGURATION FOUND."
    )


# ================================================================
# 9. HEIGHT / VARIABLE SEMANTICS
# ================================================================

section("7. VARIABLE SEMANTICS")

print("\nFORECAST VARIABLES:")
print("  wind_speed_forecast")
print("  wind_gust_forecast")

print("\nACTUAL VARIABLES:")
print("  wind_speed_10m_mean")
print("  wind_speed_10m_max")
print("  wind_gusts_10m_max")

print(
    "\nActual wind speed explicitly indicates 10 m."
)

print(
    "Forecast wind speed does NOT explicitly indicate "
    "height in the processed CSV."
)

height_comparable = False

print(
    "\nHeight comparability:",
    "NOT PROVEN"
)


# ================================================================
# 10. AGGREGATION ANALYSIS
# ================================================================

section("8. AGGREGATION COMPATIBILITY")

print(
    """
Forecast:
    wind_speed_forecast

Actual:
    wind_speed_10m_mean

The processed forecast file does not contain enough metadata
to prove how wind_speed_forecast was aggregated.

The actual file explicitly uses a daily mean variable.

Therefore aggregation equivalence cannot be established
from the processed CSV alone.
"""
)

aggregation_proven = False

print(
    "Aggregation compatibility:",
    "NOT PROVEN"
)


# ================================================================
# 11. UNIT COMPATIBILITY
# ================================================================

section("9. UNIT COMPATIBILITY")

if explicit_unit_evidence:

    print(
        "Explicit unit evidence was found in project files."
    )

    for item in explicit_unit_evidence:
        print(
            item["file"],
            "=>",
            item["units"]
        )

    unit_status = "EXPLICIT_EVIDENCE_FOUND"

else:

    print(
        "No explicit unit configuration was found."
    )

    print(
        "Numerical magnitude will NOT be used to infer units."
    )

    unit_status = "UNKNOWN"

print(
    "\nUnit status:",
    unit_status
)


# ================================================================
# 12. DO NOT PERFORM AUTOMATIC CONVERSION
# ================================================================

section("10. UNIT CONVERSION SAFETY CHECK")

print(
    """
NO automatic conversion is performed.

The script deliberately does NOT apply:
    ×3.6
    ÷3.6
    ×5
    ÷5
    or any other conversion.

A numerical ratio between forecast and actual values
is NOT accepted as evidence of a unit conversion.
"""
)


# ================================================================
# 13. WIND DECISION
# ================================================================

section("11. FINAL WIND DECISION")

wind_ready = (
    numerical_wind_pass
    and temporal_pass
    and unit_status == "EXPLICIT_EVIDENCE_FOUND"
    and aggregation_proven
)

if wind_ready:

    wind_decision = "READY_FOR_TRAINING"

else:

    wind_decision = "WIND_REQUIRES_SOURCE_FIX"

print(
    "WIND DECISION:",
    wind_decision
)


# ================================================================
# 14. OTHER STAGE-1 DECISIONS
# ================================================================

section("12. STAGE-1 VARIABLE DECISIONS")

stage1_decisions = {
    "temperature": {
        "decision": "RAW_FORECAST",
        "reason": "Existing diagnostics showed correction degraded performance."
    },

    "humidity": {
        "decision": "RAW_FORECAST",
        "reason": "Existing diagnostics showed correction degraded performance."
    },

    "precipitation": {
        "decision": "SPECIALIZED_CORRECTION",
        "reason": "Existing diagnostics showed improvement across most horizons; retain for further training."
    },

    "wind": {
        "decision": wind_decision,
        "reason": (
            "Wind source compatibility is not established."
            if not wind_ready
            else "Wind source compatibility established."
        ),
    },

    "pressure": {
        "decision": "UNAVAILABLE_PENDING_FORECAST_SOURCE",
        "reason": "No verified pressure forecast predictor is currently available."
    },
}

for variable, information in stage1_decisions.items():

    print(
        f"{variable.upper():15s}: "
        f"{information['decision']}"
    )


# ================================================================
# 15. CHECK TRAINING DATASET EXISTENCE
# ================================================================

section("13. TRAINING DATASET INVENTORY")

training_inventory = {}

for horizon in range(1, 8):

    horizon_dir = TRAINING_ROOT / f"D{horizon}"

    exists = horizon_dir.exists()

    training_inventory[f"D{horizon}"] = exists

    print(
        f"D{horizon}:",
        "FOUND" if exists else "MISSING"
    )


training_datasets_exist = all(
    training_inventory.values()
)


# ================================================================
# 16. CHECK FOR ACTUAL WEATHER IN X
# ================================================================

section("14. TARGET LEAKAGE CHECK")

leakage_columns = []

for horizon in range(1, 8):

    horizon_dir = TRAINING_ROOT / f"D{horizon}"

    x_train_file = horizon_dir / "X_train.csv"

    if not x_train_file.exists():
        continue

    try:
        X = pd.read_csv(x_train_file)

    except Exception:
        continue

    bad_columns = [
        c
        for c in X.columns
        if c.lower().startswith("actual_")
    ]

    if bad_columns:

        leakage_columns.extend(
            [
                f"D{horizon}:{c}"
                for c in bad_columns
            ]
        )


target_leakage_pass = (
    len(leakage_columns) == 0
)

print(
    "Actual-weather columns inside X:",
    leakage_columns
)

print(
    "Target leakage check:",
    pass_fail(target_leakage_pass)
)


# ================================================================
# 17. SYNTHETIC / INTERPOLATED DATA CHECK
# ================================================================

section("15. SYNTHETIC / INTERPOLATION CHECK")

synthetic_keywords = [
    "synthetic",
    "interpolated",
    "interpolation",
    "fake_target",
    "pseudo_target",
]

synthetic_files = []

for path in PROJECT_ROOT.rglob("*"):

    if not path.is_file():
        continue

    name = path.name.lower()

    if any(
        keyword in name
        for keyword in synthetic_keywords
    ):

        synthetic_files.append(str(path))


print(
    "Files with synthetic/interpolation naming:",
    len(synthetic_files)
)

for item in synthetic_files[:30]:
    print("  ", item)

print(
    """
IMPORTANT:
File naming alone does not prove synthetic data exists.
The current project policy remains:

    synthetic_data_allowed = FALSE
    interpolation_allowed = FALSE
"""
)


# ================================================================
# 18. GP TARGET CHECK
# ================================================================

section("16. GP TARGET POLICY CHECK")

print(
    """
Current SIH26 methodology:

Stage 1:
    Block forecast
        ↓
    Block actual observations
        ↓
    Block forecast correction

Stage 2:
    Corrected block forecast
        +
    GP spatial predictors
        ↓
    GP-level downscaled forecast

There is NO requirement for fabricated GP-level historical
weather targets.

Therefore block actual weather must NOT be copied/repeated
as GP-level training targets.
"""
)


# ================================================================
# 19. FINAL GLOBAL GATE
# ================================================================

section("17. FINAL PRE-TRAINING GATE")

global_checks = {

    "required_source_columns": required_columns_pass,

    "forecast_temporal_integrity": temporal_pass,

    "wind_numerical_validity": numerical_wind_pass,

    "target_leakage_absent": target_leakage_pass,

    "training_datasets_exist": training_datasets_exist,

    "wind_source_compatibility": wind_ready,
}


for check, status in global_checks.items():

    print(
        f"{check:40s}: "
        f"{'PASS' if status else 'FAIL'}"
    )


# ================================================================
# 20. FINAL DECISION
# ================================================================

if not required_columns_pass:

    final_decision = "DO_NOT_TRAIN_YET"

elif not temporal_pass:

    final_decision = "DO_NOT_TRAIN_YET"

elif not target_leakage_pass:

    final_decision = "DO_NOT_TRAIN_YET"

elif not training_datasets_exist:

    final_decision = "DO_NOT_TRAIN_YET"

elif not wind_ready:

    final_decision = "WIND_REQUIRES_SOURCE_FIX"

else:

    final_decision = "READY_FOR_TRAINING"


# ================================================================
# 21. SAVE RESULTS
# ================================================================

decision = {

    "timestamp": datetime.now().isoformat(),

    "project": "SIH26",

    "location": {
        "state": "Maharashtra",
        "district": "Nashik",
        "taluka": "Sinnar",
    },

    "final_decision": final_decision,

    "wind_decision": wind_decision,

    "unit_status": unit_status,

    "aggregation_proven": aggregation_proven,

    "height_comparable": height_comparable,

    "global_checks": global_checks,

    "stage1_decisions": stage1_decisions,

    "training_inventory": training_inventory,

    "explicit_unit_evidence": explicit_unit_evidence,

    "policy": {
        "automatic_wind_conversion": False,
        "synthetic_data_allowed": False,
        "interpolation_allowed": False,
        "actual_weather_as_gp_target": False,
        "actual_weather_as_X": False,
    },

    "next_action": (
        "Fix/verify wind source metadata before training."
        if final_decision == "WIND_REQUIRES_SOURCE_FIX"
        else
        "Proceed to training."
        if final_decision == "READY_FOR_TRAINING"
        else
        "Resolve failed audit checks before training."
    ),
}


json_path = OUTPUT_DIR / "final_pretraining_decision.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        decision,
        f,
        indent=2,
        default=str
    )


# ================================================================
# 22. SAVE HUMAN-READABLE REPORT
# ================================================================

report_path = OUTPUT_DIR / "final_pretraining_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SIH26 FINAL PRE-TRAINING ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Final Decision: {final_decision}\n"
    )

    f.write(
        f"Wind Decision: {wind_decision}\n"
    )

    f.write(
        f"Wind Unit Status: {unit_status}\n"
    )

    f.write(
        f"Aggregation Proven: {aggregation_proven}\n"
    )

    f.write(
        f"Height Comparable: {height_comparable}\n\n"
    )

    f.write(
        "STAGE-1 DECISIONS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for variable, information in stage1_decisions.items():

        f.write(
            f"{variable}: "
            f"{information['decision']}\n"
        )

        f.write(
            f"  Reason: "
            f"{information['reason']}\n"
        )

    f.write("\nGLOBAL CHECKS\n")
    f.write("-" * 80 + "\n")

    for check, status in global_checks.items():

        f.write(
            f"{check}: "
            f"{'PASS' if status else 'FAIL'}\n"
        )

    f.write("\nPOLICY\n")
    f.write("-" * 80 + "\n")

    for key, value in decision["policy"].items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write("\nFINAL ACTION\n")
    f.write("-" * 80 + "\n")

    f.write(
        decision["next_action"] + "\n"
    )


# ================================================================
# 23. TERMINAL SUMMARY
# ================================================================

section("FINAL RESULT")

print(
    f"""
FINAL DECISION
--------------
{final_decision}

WIND
----
{wind_decision}

UNIT STATUS
-----------
{unit_status}

AGGREGATION PROVEN
-------------------
{aggregation_proven}

HEIGHT COMPARABILITY
--------------------
{height_comparable}

TEMPERATURE
-----------
RAW_FORECAST

HUMIDITY
--------
RAW_FORECAST

PRECIPITATION
-------------
SPECIALIZED_CORRECTION

WIND
----
{wind_decision}

PRESSURE
--------
UNAVAILABLE_PENDING_FORECAST_SOURCE

OUTPUTS
-------
{json_path}
{report_path}
"""
)

print("=" * 80)
print("FINAL PRE-TRAINING ANALYSIS COMPLETE")
print("=" * 80)