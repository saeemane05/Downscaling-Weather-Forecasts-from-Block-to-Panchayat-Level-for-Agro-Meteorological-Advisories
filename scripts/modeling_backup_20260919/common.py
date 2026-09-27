from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
BLOCK_ROOT = PROJECT_ROOT / "block"
GP_ROOT = PROJECT_ROOT / "gp"
CLEANED_ROOT = DATASETS_ROOT / "cleaned"
ENGINEERED_ROOT = DATASETS_ROOT / "engineered"
DICTIONARY_ROOT = DATASETS_ROOT / "feature_dictionary"
TRAINING_ROOT = DATASETS_ROOT / "training"
TARGET_ROOT = DATASETS_ROOT / "targets"
MODELS_ROOT = PROJECT_ROOT / "models"

TARGET_COLUMNS = [
    "actual_precipitation_sum",
    "actual_temperature_2m_mean",
    "actual_relative_humidity_2m_mean",
    "actual_wind_speed_10m_mean",
    "actual_surface_pressure_mean",
]


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_child(parent: Path, requested: str) -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {parent}")
    exact = parent / requested
    if exact.is_dir():
        return exact
    wanted = norm(requested)
    matches = [p for p in parent.iterdir() if p.is_dir() and norm(p.name) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not resolve '{requested}' under {parent}")
    raise RuntimeError(f"Ambiguous location '{requested}' under {parent}: {matches}")


def resolve_physical_location(root: Path, state: str, district: str, block: str) -> Path:
    return resolve_child(resolve_child(resolve_child(root, state), district), block)


def physical_location_names(location: Path) -> tuple[str, str, str]:
    return location.parent.parent.name, location.parent.name, location.name


def resolve_all_locations(state: str, district: str, block: str):
    block_dir = resolve_physical_location(BLOCK_ROOT, state, district, block)
    gp_dir = resolve_physical_location(GP_ROOT, state, district, block)
    return block_dir, gp_dir


def cleaned_location(state: str, district: str, block: str) -> Path:
    return resolve_physical_location(CLEANED_ROOT, state, district, block)


def engineered_location(state: str, district: str, block: str) -> Path:
    return resolve_physical_location(ENGINEERED_ROOT, state, district, block)


def dictionary_path_for(cleaned_or_physical_location: Path) -> Path:
    state, district, block = physical_location_names(cleaned_or_physical_location)
    return DICTIONARY_ROOT / state / district / block / "feature_dictionary.csv"


def training_location(state: str, district: str, block: str) -> Path:
    # Resolve the cleaned/engineered physical spelling when possible so that
    # North Goa vs North_Goa cannot create parallel outputs.
    try:
        loc = engineered_location(state, district, block)
        state, district, block = physical_location_names(loc)
    except Exception:
        pass
    return TRAINING_ROOT / state / district / block


def targets_location(state: str, district: str, block: str) -> Path:
    # Target files are user/data-provider supplied and never synthesized here.
    # Prefer physical spelling from the GP directory when available.
    try:
        loc = resolve_physical_location(GP_ROOT, state, district, block)
        state, district, block = physical_location_names(loc)
    except Exception:
        pass
    return TARGET_ROOT / state / district / block


def discover_target_source(state: str, district: str, block: str) -> Path | None:
    candidates = []
    loc = targets_location(state, district, block)
    candidates.append(loc / "gp_weather_targets.csv")
    candidates.append(loc / "gp_weather_targets_model_features.csv")

    try:
        cleaned = cleaned_location(state, district, block)
        candidates.extend([
            cleaned / "gp" / "gp_weather_targets.csv",
            cleaned / "gp" / "gp_weather_targets_model_features.csv",
        ])
    except Exception:
        pass

    return next((p for p in candidates if p.exists()), None)


def detect_date_column(df: pd.DataFrame, preferred=None) -> str | None:
    preferred = preferred or [
        "date", "target_date", "valid_date", "forecast_date",
        "datetime", "target_datetime", "timestamp", "time"
    ]
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    for c in df.columns:
        if any(tok in str(c).lower() for tok in ("date", "datetime", "timestamp")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return c
    return None


def detect_gp_id_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "gp_id", "gp_code", "gram_panchayat_id", "gram_panchayat_code",
        "panchayat_id", "panchayat_code", "village_panchayat_id",
        "lgd_gp_code", "lgd_code", "unique_gp_id"
    ]
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p in lower:
            return lower[p]
    for c in df.columns:
        n = norm(c)
        if "gpid" in n or "gpcode" in n or "panchayatid" in n or "panchayatcode" in n:
            return c
    return None


def assert_unique_key(df: pd.DataFrame, keys: list[str], label: str):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        examples = df.loc[dup, keys].head(10).to_dict("records")
        raise RuntimeError(f"{label} has duplicate key {keys}. Examples: {examples}")


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if df.columns.duplicated().any():
        raise RuntimeError(f"Duplicate column labels in {path}: {df.columns[df.columns.duplicated()].tolist()}")
    return df
