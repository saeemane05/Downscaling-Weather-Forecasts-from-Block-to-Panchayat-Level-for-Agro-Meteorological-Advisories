#!/usr/bin/env python3
"""Incremental, location-locked orchestrator for the existing SIH26 pipeline.

This module intentionally contains no collection, feature, or model logic.  It
only checks the output contracts of existing scripts and invokes a missing
stage with the requested State/District/Block.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def existing_child(parent: Path, requested: str) -> Path | None:
    if not parent.is_dir():
        return None
    matches = [p for p in parent.iterdir() if p.is_dir() and key(p.name) == key(requested)]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous location {requested!r} under {parent}: {matches}")
    return matches[0] if matches else None


def resolve_location(state: str, district: str, block: str) -> tuple[str, str, str]:
    """Use one existing physical spelling, without creating directories."""
    for base in (ROOT / "block", ROOT / "gp", ROOT / "datasets" / "cleaned"):
        state_dir = existing_child(base, state)
        district_dir = existing_child(state_dir, district) if state_dir else None
        block_dir = existing_child(district_dir, block) if district_dir else None
        if block_dir:
            return block_dir.parent.parent.name, block_dir.parent.name, block_dir.name
    return state, district, block


def location_path(base: Path, loc: tuple[str, str, str]) -> Path:
    return base.joinpath(*loc)


def valid_csv(path: Path, columns: Sequence[str] = ()) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not header or not set(columns).issubset(header):
                return False
            return next(reader, None) is not None
    except (OSError, UnicodeError, csv.Error):
        return False


def valid_json(path: Path, loc: tuple[str, str, str] | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if loc is None:
        return True
    candidate = payload.get("location") or payload.get("Location")
    if isinstance(candidate, list) and len(candidate) == 3:
        return all(key(a) == key(b) for a, b in zip(candidate, loc))
    if isinstance(candidate, dict):
        return all(key(candidate.get(name, "")) == key(value) for name, value in zip(("state", "district", "block"), loc))
    # Stage-2's established manifest writes the location as top-level fields.
    if all(name in payload for name in ("state", "district", "block")):
        return all(key(payload[name]) == key(value) for name, value in zip(("state", "district", "block"), loc))
    return False


def all_files(paths: Sequence[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def run(command: list[str], location: tuple[str, str, str], *, stdin_location: bool = False) -> None:
    print("      Command:", subprocess.list2cmdline(command))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            input="\n".join(location) + "\n" if stdin_location else None,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start command: {exc}") from exc
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{subprocess.list2cmdline(command)}"
        )


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    complete: Callable[[tuple[str, str, str]], bool]
    execute: Callable[[tuple[str, str, str]], None]
    deferred_reason: Callable[[tuple[str, str, str]], str | None] = lambda _loc: None


def block_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "block", loc) / "processed" / "merge"
    return all(valid_csv(root / name) for name in (
        "block_observations_daily.csv", "block_forecast_training_base.csv", "block_current_forecast.csv"))


def gp_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "gp", loc) / "processed"
    return valid_csv(root / "gp_master.csv", ("gp_id",)) and valid_csv(root / "gp_merged_features.csv", ("gp_id",))


def cleaned_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "datasets" / "cleaned", loc)
    return all(valid_csv(root / folder / name) for folder, name in (
        ("block", "block_observations_daily.csv"),
        ("block", "block_forecast_training_base.csv"),
        ("block", "block_current_forecast.csv"),
        ("gp", "gp_merged_features.csv"),
    )) and valid_json(root / "100pct_unavailable_feature_manifest.json", loc)


def dictionary_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "datasets" / "feature_dictionary", loc)
    return valid_csv(root / "feature_dictionary.csv", ("column",)) and valid_json(root / "summary.json", loc)


def engineered_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "datasets" / "engineered", loc)
    return all(valid_csv(root / folder / name) for folder, name in (
        ("block", "block_observations_daily_engineered.csv"),
        ("block", "block_forecast_training_base_engineered.csv"),
        ("block", "block_current_forecast_engineered.csv"),
        ("gp", "gp_merged_features_engineered.csv"),
    )) and valid_json(root / "feature_engineering_manifest.json", loc)


def audits_complete(loc: tuple[str, str, str]) -> bool:
    datasets = ROOT / "datasets"
    expected = (
        datasets / "engineered_audit" / Path(*loc) / "engineered_feature_audit.json",
        datasets / "leakage_audit" / Path(*loc) / "temporal_target_leakage_audit.json",
        datasets / "information_availability_audit" / Path(*loc) / "final_provenance_information_availability_audit.json",
    )
    return all(valid_json(path) for path in expected)


def training_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "datasets" / "training", loc)
    if not valid_json(root / "training_dataset_manifest.json", loc):
        return False
    return all(
        all(valid_csv(root / f"D{h}" / f"{kind}_{split}.csv") for kind in ("X", "y") for split in ("train", "validation", "test"))
        for h in range(1, 8)
    )


def genuine_gp_target_available(loc: tuple[str, str, str]) -> bool:
    """Check for a supplied genuine GP target without manufacturing one."""
    required = (
        "gp_id",
        "date",
        "actual_precipitation_sum",
        "actual_temperature_2m_mean",
        "actual_relative_humidity_2m_mean",
        "actual_wind_speed_10m_mean",
        "actual_surface_pressure_mean",
    )
    candidates = (
        location_path(ROOT / "datasets" / "targets", loc) / "gp_weather_targets.csv",
        location_path(ROOT / "datasets" / "cleaned", loc) / "gp" / "gp_weather_targets.csv",
        location_path(ROOT / "datasets" / "cleaned", loc) / "gp" / "gp_weather_targets_model_features.csv",
    )
    return any(valid_csv(path, required) for path in candidates)


def target_deferred_reason(loc: tuple[str, str, str]) -> str | None:
    if genuine_gp_target_available(loc):
        return None
    return (
        "A genuine GP target is unavailable. This target-dependent Stage-1 "
        "path is deferred; no target will be fabricated. The independent "
        "direct Stage-2 ERA5-Land workflow continues."
    )


def stage1_complete(loc: tuple[str, str, str]) -> bool:
    root = ROOT / "models" / "main_model" / "trained_models" / Path(*loc)
    return all_files([root / f"D{h}" / name for h in range(1, 8) for name in (
        "rain_occurrence_model.pkl", "rain_amount_model.pkl", "feature_imputer.pkl", "feature_columns.pkl")])


def stage2_complete(loc: tuple[str, str, str]) -> bool:
    root = location_path(ROOT / "models" / "main_model" / "stage2_direct_gp_training", loc)
    return (
        valid_json(root / "stage2_manifest.json", loc)
        and valid_csv(root / "gp_era5_land_reference_targets.csv")
        and valid_csv(root / "stage2_training_frame.csv")
        and valid_csv(root / "stage2_current_gp_forecast.csv", ("gp_id",))
    )


def script(path: str, loc: tuple[str, str, str], *, stdin: bool = False, flags: Sequence[str] = ()) -> None:
    # Unbuffered child output makes failures from existing collectors visible
    # immediately to the orchestrator and its caller.
    run([PYTHON, "-u", str(ROOT / path), *flags], loc, stdin_location=stdin)


STAGES = (
    Stage("block", "Block data", block_complete, lambda loc: script("scripts/block/run_block_pipeline.py", loc, stdin=True)),
    Stage("gp", "GP spatial data", gp_complete, lambda loc: script("scripts/gp/08_gp_runner.py", loc, stdin=True)),
    Stage("cleaning", "Cleaning", cleaned_complete, lambda loc: script("scripts/modeling/cleaning/run_cleaning.py", loc, flags=loc)),
    Stage("dictionary", "Feature dictionary", dictionary_complete, lambda loc: script("scripts/modeling/feature_engg/build_feature_dictionary.py", loc, stdin=True)),
    Stage("engineering", "Feature engineering", engineered_complete, lambda loc: script("scripts/modeling/feature_engg/feature_engineering.py", loc, flags=loc)),
    Stage("audits", "Existing audits", audits_complete, lambda loc: [script(path, loc, flags=("--state", loc[0], "--district", loc[1], "--block", loc[2])) for path in (
        "scripts/modeling/audit/audit_engineered_features.py",
        "scripts/modeling/audit/temporal_target_leakage_audit.py",
        "scripts/modeling/audit/information_availability_audit.py",
    )]),
    Stage("stage2", "Stage-2 GP downscaling and current forecast", stage2_complete, lambda loc: script("models/main_model/02_stage2_direct_gp_training.py", loc, flags=("--state", loc[0], "--district", loc[1], "--block", loc[2]))),
    # Direct Stage-2 has its own legitimate ERA5-Land GP reference extractor.
    # It does not depend on the separate, supplied-GP-target Stage-1 path.
    Stage("training", "Training dataset", training_complete, lambda loc: (script("scripts/modeling/training_data/00_target_preflight.py", loc, flags=loc), script("scripts/modeling/training_data/01_build_leakage_safe_training_dataset.py", loc, flags=loc), script("scripts/modeling/training_data/02_audit_training_dataset.py", loc, flags=loc)), target_deferred_reason),
    Stage("stage1", "Stage-1 final correction", stage1_complete, lambda loc: script("models/main_model/01_train_stage1_final.py", loc, flags=("--state", loc[0], "--district", loc[1], "--block", loc[2])), target_deferred_reason),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the existing SIH26 pipeline incrementally for one block.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--district", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--from-stage", choices=[stage.key for stage in STAGES])
    parser.add_argument("--to-stage", choices=[stage.key for stage in STAGES])
    parser.add_argument("--force", action="store_true", help="Rerun each stage in the selected range without deleting other stages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loc = resolve_location(args.state, args.district, args.block)
    first = next((i for i, stage in enumerate(STAGES) if stage.key == args.from_stage), 0)
    last = next((i for i, stage in enumerate(STAGES) if stage.key == args.to_stage), len(STAGES) - 1)
    if first > last:
        raise SystemExit("--from-stage must not be after --to-stage")

    print("=" * 68)
    print("SIH26 HYPERLOCAL WEATHER DOWNSCALING")
    print("=" * 68)
    print(f"State    : {loc[0]}\nDistrict : {loc[1]}\nBlock    : {loc[2]}")
    print("=" * 68)
    print("PIPELINE STATUS")

    selected = STAGES[first:last + 1]
    deferred = []
    for number, stage in enumerate(selected, 1):
        # Child runners may resolve an existing physical spelling after an
        # upstream collector creates it, so refresh before every stage.
        loc = resolve_location(*loc)
        available = stage.complete(loc)
        print(f"\n[{number}/{len(selected)}] {stage.label}")
        print(f"      Status : {'AVAILABLE' if available else 'MISSING OR INCOMPLETE'}")
        reason = stage.deferred_reason(loc)
        if not available and reason:
            print("      Action : DEFERRED")
            print(f"      Reason : {reason}")
            deferred.append(stage.label)
            continue
        if available and not args.force:
            print("      Action : REUSING")
            continue
        print("      Action : RUNNING")
        try:
            stage.execute(loc)
        except Exception as exc:
            print(
                f"\nPIPELINE STOPPED\n"
                f"Stage: {stage.label}\n"
                f"Error: {exc}\n"
                "Required input: the failing stage's required external input or service.\n"
                "Suggested next action: resolve the reported dependency, then rerun the same command.",
                file=sys.stderr,
            )
            return 1
        loc = resolve_location(*loc)
        if not stage.complete(loc):
            print(f"\nPIPELINE STOPPED at stage: {stage.key}\nError: expected output contract was not produced.", file=sys.stderr)
            return 1

    print("\n" + "=" * 68)
    print("PIPELINE COMPLETE")
    print("=" * 68)
    print("Location:", " / ".join(loc))
    print("Stage-2 output:", location_path(ROOT / "models" / "main_model" / "stage2_direct_gp_training", loc))
    if deferred:
        print("Deferred target-dependent stages:", ", ".join(deferred))
        print("Next action: supply a genuine GP target through the existing target contract to enable them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
