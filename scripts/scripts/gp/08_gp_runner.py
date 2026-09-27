#!/usr/bin/env python3
"""
SIH26 — UNIVERSAL GP DATA PIPELINE RUNNER
=========================================

Single entry point for the GP-level downscaling data pipeline.

User enters:
    State
    District
    Block

The runner then:
    1. Resolves the EXISTING physical block directory if one already exists.
    2. Creates the canonical directory only when the block is new.
    3. Locks every downstream stage to that one block.
    4. Reuses valid static datasets.
    5. Lets incremental collectors inspect existing raw data and fetch only
       genuinely new/missing periods.
    6. Never deletes raw data.
    7. Never creates synthetic/imputed/interpolated observations.
    8. Never scans the project to choose a data target.
    9. Keeps the GP merger as a separate child script.
   10. Refuses to execute an unsafe multi-block child collector automatically.

IMPORTANT:
    The runner does NOT contain any State/District/Block-specific mapping.
    Filesystem normalization is generic only:
        spaces -> underscores
        unsafe filesystem characters -> underscores

Run from the project root:
    python scripts/gp/08_gp_runner.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ============================================================================
# PROJECT PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GP_ROOT = PROJECT_ROOT / "gp"


# ============================================================================
# STAGE DEFINITIONS
# ============================================================================

@dataclass(frozen=True)
class Stage:
    key: str
    script_name: str
    description: str
    kind: str                    # boundary / static / incremental / merge
    required_outputs: tuple[str, ...]
    raw_outputs: tuple[str, ...] = ()
    requires_target_only: bool = False
    cli_location: bool = False
    cli_block_only: bool = False


STAGES: tuple[Stage, ...] = (
    Stage(
        key="boundary",
        script_name="01_gp_boundary.py",
        description="GP boundary and authoritative GP master",
        kind="boundary",
        required_outputs=(
            "processed/gp_master.geojson",
            "processed/gp_master.csv",
        ),
    ),
    Stage(
        key="terrain",
        script_name="02_gp_terrain.py",
        description="GP terrain features",
        kind="static",
        required_outputs=(
            "processed/gp_terrain_features.csv",
            "processed/gp_terrain_features.geojson",
        ),
    ),
    Stage(
        key="landcover",
        script_name="03_gp_landcover.py",
        description="Dynamic World land-cover features",
        kind="static",
        required_outputs=(
            "processed/gp_landcover_features.csv",
            "processed/gp_landcover_features.geojson",
        ),
        # Current land-cover implementation accepts --block.
        cli_block_only=True,
    ),
    Stage(
        key="sentinel2",
        script_name="04_sentinel2.py",
        description="Sentinel-2 optical observations/features",
        kind="incremental",
        required_outputs=(
            "processed/gp_sentinel2_features.csv",
            "processed/gp_sentinel2_features.geojson",
        ),
        raw_outputs=(
            "raw/sentinel2_observations.csv",
        ),
        # V7 explicitly supports the runner target contract.
        requires_target_only=True,
    ),
    Stage(
        key="sentinel1",
        script_name="04_sentinel1.py",
        description="Sentinel-1 SAR observations/features",
        kind="incremental",
        required_outputs=(
            "processed/gp_sentinel1_features.csv",
            "processed/gp_sentinel1_features.geojson",
        ),
        raw_outputs=(
            "raw/sentinel1_observations.csv",
        ),
        # Current V4 discovers ALL GP masters and is therefore unsafe here.
        requires_target_only=True,
    ),
    Stage(
        key="soil",
        script_name="05_soil.py",
        description="SoilGrids root-zone soil features",
        kind="static",
        required_outputs=(
            "processed/gp_soil_features.csv",
            "processed/gp_soil_features.geojson",
        ),
        cli_location=True,
    ),
    Stage(
        key="lst",
        script_name="06_lst.py",
        description="Landsat land-surface temperature features",
        kind="static",
        required_outputs=(
            "processed/gp_lst_features.csv",
            "processed/gp_lst_features.geojson",
        ),
        cli_location=True,
    ),
    Stage(
        key="merge",
        script_name="07_gp_merge.py",
        description="Final GP feature merge",
        kind="merge",
        required_outputs=(
            "processed/gp_merged_features.csv",
            "processed/gp_merged_features.geojson",
            "processed/gp_merge_metadata.json",
        ),
    ),
)


# ============================================================================
# CONSOLE
# ============================================================================

def line(char: str = "=", n: int = 72) -> None:
    print(char * n)


def title(text: str) -> None:
    print()
    line()
    print(text)
    line()


def info(text: str) -> None:
    print(f"[INFO] {text}")


def success(text: str) -> None:
    print(f"[SUCCESS] {text}")


def warning(text: str) -> None:
    print(f"[WARNING] {text}")


def error(text: str) -> None:
    print(f"[ERROR] {text}")


# ============================================================================
# LOCATION / DIRECTORY RESOLUTION
# ============================================================================

def clean_location_name(value: str) -> str:
    """
    Generic filesystem normalization.

    This is NOT a geographic lookup and contains no location-specific mapping.
    """
    value = str(value).strip()
    if not value:
        return ""

    value = re.sub(r"\s+", "_", value)
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = value.strip(" ._")

    return value


def location_identity(value: str) -> str:
    """
    Generic comparison key for filesystem names.

    North Goa / North_Goa / north_goa -> northgoa
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def resolve_existing_child(
    parent: Path,
    requested_name: str,
) -> Optional[Path]:
    """
    Resolve an existing directory without assuming its exact capitalization
    or whether spaces were represented as underscores.
    """
    if not parent.exists():
        return None

    target = location_identity(requested_name)

    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir()
        and location_identity(child.name) == target
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple directories match the requested location component:\n"
            + "\n".join(f"  {p}" for p in matches)
        )

    return None


def resolve_block_directory(
    state: str,
    district: str,
    block: str,
) -> Path:
    """
    Reuse the exact existing physical directory whenever possible.

    If no matching directory exists, create:
        gp/<state>/<district>/<block>
    with spaces represented by underscores.
    """
    GP_ROOT.mkdir(parents=True, exist_ok=True)

    state_dir = resolve_existing_child(GP_ROOT, state)
    if state_dir is None:
        state_dir = GP_ROOT / clean_location_name(state)
        state_dir.mkdir(parents=True, exist_ok=True)

    district_dir = resolve_existing_child(state_dir, district)
    if district_dir is None:
        district_dir = state_dir / clean_location_name(district)
        district_dir.mkdir(parents=True, exist_ok=True)

    block_dir = resolve_existing_child(district_dir, block)
    if block_dir is None:
        block_dir = district_dir / clean_location_name(block)
        block_dir.mkdir(parents=True, exist_ok=True)

    return block_dir.resolve()


# ============================================================================
# GP MASTER
# ============================================================================

def find_gp_master(block_dir: Path) -> Optional[Path]:
    """
    The authoritative GP master may only be selected from the locked block.
    """
    candidates = (
        block_dir / "processed" / "gp_master.geojson",
        block_dir / "gp_master.geojson",
    )

    for candidate in candidates:
        if candidate.is_file():
            try:
                if candidate.stat().st_size > 0:
                    return candidate.resolve()
            except OSError:
                pass

    return None


def gp_master_feature_count(master_path: Path) -> Optional[int]:
    """
    Lightweight validation of the locked GP master.
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(master_path)

        if gdf.empty or "gp_id" not in gdf.columns:
            return None

        ids = gdf["gp_id"].astype(str).str.strip()

        if ids.eq("").any() or ids.duplicated().any():
            return None

        if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
            return None

        return int(len(gdf))

    except Exception:
        return None


# ============================================================================
# OUTPUT CHECKS
# ============================================================================

def output_exists(
    block_dir: Path,
    relative_path: str,
) -> bool:
    path = block_dir / relative_path

    if not path.is_file():
        return False

    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def stage_outputs_exist(
    stage: Stage,
    block_dir: Path,
) -> bool:
    return bool(stage.required_outputs) and all(
        output_exists(block_dir, rel)
        for rel in stage.required_outputs
    )


def raw_outputs_exist(
    stage: Stage,
    block_dir: Path,
) -> bool:
    return bool(stage.raw_outputs) and all(
        output_exists(block_dir, rel)
        for rel in stage.raw_outputs
    )


def print_output_status(
    stage: Stage,
    block_dir: Path,
) -> bool:
    print(
        f"\nExisting-data check: "
        f"{stage.key} - {stage.description}"
    )

    all_ok = True

    for rel in stage.raw_outputs:
        ok = output_exists(block_dir, rel)
        print(
            f"  [{'AVAILABLE' if ok else 'MISSING  '}] {rel}"
        )
        all_ok &= ok

    for rel in stage.required_outputs:
        ok = output_exists(block_dir, rel)
        print(
            f"  [{'AVAILABLE' if ok else 'MISSING  '}] {rel}"
        )
        all_ok &= ok

    if all_ok:
        print("[PASS] All expected outputs are available.")
    else:
        print("[INFO] One or more expected outputs are missing.")

    return all_ok


# ============================================================================
# SAFETY FOR LEGACY CHILDREN
# ============================================================================

def script_text(script: Path) -> str:
    try:
        return script.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""


def script_supports_target_contract(script: Path) -> bool:
    """
    Conservative source-level check for target-aware collectors.
    """
    text = script_text(script)

    if "SIH26_RUNNER_MODE" not in text:
        return False

    return any(
        token in text
        for token in (
            "SIH26_TARGET_GP_MASTER",
            "SIH26_TARGET_BLOCK_DIR",
        )
    )


def matching_gp_masters_by_block_name(
    block: str,
) -> list[Path]:
    """
    Used for legacy children such as the current land-cover script, which
    accepts --block but not an exact GP-master path.

    The runner refuses to use such a child if the block name is ambiguous
    project-wide.
    """
    target = location_identity(block)
    matches: list[Path] = []

    if not GP_ROOT.exists():
        return matches

    for master in GP_ROOT.rglob(
        "processed/gp_master.geojson"
    ):
        if not master.is_file():
            continue

        block_dir_name = master.parent.parent.name

        if location_identity(block_dir_name) == target:
            matches.append(master.resolve())

    return sorted(set(matches))


def assert_block_only_child_safe(
    stage: Stage,
    block: str,
    locked_master: Optional[Path],
) -> None:
    if not stage.cli_block_only:
        return

    matches = matching_gp_masters_by_block_name(block)

    if locked_master is not None:
        locked_master = locked_master.resolve()

    if len(matches) == 1 and locked_master is not None:
        if matches[0] == locked_master:
            return

    if len(matches) == 0:
        raise RuntimeError(
            f"{stage.key}: no GP master matches block name "
            f"'{block}'."
        )

    raise RuntimeError(
        f"{stage.key}: the legacy child accepts only '--block', "
        f"and '{block}' matches {len(matches)} GP masters.\n"
        "The universal runner will not allow it to process multiple "
        "locations.\n"
        "Matches:\n"
        + "\n".join(
            f"  {p}" for p in matches
        )
    )


def assert_safe_target_stage(
    stage: Stage,
    script: Path,
) -> None:
    """
    Never launch a known multi-block collector as though it were target locked.
    """
    if not stage.requires_target_only:
        return

    if not script_supports_target_contract(script):
        raise RuntimeError(
            f"{stage.key} cannot be safely launched by the "
            "universal runner.\n"
            f"Script: {script}\n\n"
            "The child collector does not advertise the SIH26 "
            "runner target contract "
            "(SIH26_RUNNER_MODE + target path variables).\n"
            "Running it automatically could process other blocks.\n"
            "No data was changed by the runner."
        )


# ============================================================================
# CHILD PROCESS ENVIRONMENT / COMMAND
# ============================================================================

def build_environment(
    state: str,
    district: str,
    block: str,
    block_dir: Path,
    gp_master: Optional[Path],
) -> dict[str, str]:
    env = os.environ.copy()

    env["SIH26_RUNNER_MODE"] = "1"
    env["SIH26_NO_INTERACTIVE"] = "1"
    env["SIH26_TARGET_ONLY"] = "1"

    env["SIH26_STATE"] = state
    env["SIH26_DISTRICT"] = district
    env["SIH26_BLOCK"] = block

    env["SIH26_TARGET_BLOCK_DIR"] = str(block_dir)

    if gp_master is not None:
        env["SIH26_TARGET_GP_MASTER"] = str(
            gp_master
        )
    else:
        env.pop("SIH26_TARGET_GP_MASTER", None)

    env["PYTHONUNBUFFERED"] = "1"

    return env


def location_stdin(
    state: str,
    district: str,
    block: str,
) -> str:
    """
    Backward compatibility for older scripts that still call input().
    """
    return f"{state}\n{district}\n{block}\n"


def build_command(
    stage: Stage,
    script: Path,
    state: str,
    district: str,
    block: str,
) -> list[str]:
    command = [
        sys.executable,
        str(script),
    ]

    if stage.cli_location:
        command += [
            "--state", state,
            "--district", district,
            "--block", block,
        ]

    elif stage.cli_block_only:
        command += [
            "--block",
            block,
        ]

    return command


# ============================================================================
# PRE-CHECKS
# ============================================================================

def precheck_stage(
    stage: Stage,
    block_dir: Path,
) -> bool:
    """
    Static stages:
        reuse complete-looking outputs.

    Incremental stages:
        ALWAYS run the collector so it can inspect raw data for new/missing
        periods. Existing outputs alone never cause an incremental stage
        to be skipped.

    Merge:
        always evaluated after upstream stages.
    """
    if stage.key == "boundary":
        title("PRE-CHECK - GP BOUNDARY")

        master = find_gp_master(block_dir)

        if master is None:
            info("No existing GP master found.")
            info("Boundary stage will create it.")
            return False

        count = gp_master_feature_count(master)

        if count is None:
            warning(
                "Existing GP master failed lightweight validation."
            )
            warning(
                "Boundary stage will be rerun."
            )
            return False

        success(
            f"GP master available: {count} features"
        )
        success(
            "Existing GP master will be reused."
        )
        return True

    title(
        f"PRE-CHECK - STAGE {stage.key.upper()}"
    )

    outputs_ok = print_output_status(
        stage,
        block_dir,
    )

    if stage.kind == "static":
        if outputs_ok:
            success(
                f"{stage.description}: existing valid-looking "
                "data found. REUSING it."
            )
            return True

        info(
            f"{stage.description}: outputs incomplete. "
            "Stage will run."
        )
        return False

    if stage.kind == "incremental":
        if outputs_ok:
            info(
                f"{stage.description}: existing outputs found."
            )
            info(
                "The incremental collector will inspect existing "
                "raw observations and determine whether new/missing "
                "data must be collected."
            )
            info(
                "Existing raw observations must not be deleted or "
                "unnecessarily re-downloaded."
            )
        else:
            info(
                f"{stage.description}: first run or incomplete outputs."
            )

        return False

    if stage.kind == "merge":
        info(
            "Merge is evaluated after upstream stages; existing merged "
            "outputs are not used as inputs."
        )
        return False

    return False


# ============================================================================
# RUN ONE STAGE
# ============================================================================

def run_stage(
    stage: Stage,
    state: str,
    district: str,
    block: str,
    block_dir: Path,
) -> bool:
    script = SCRIPT_DIR / stage.script_name

    title(
        f"STAGE: {stage.key.upper()} | "
        f"{stage.description}"
    )

    print(f"Script : {script}")
    print(f"Target : {block_dir}")

    if not script.is_file():
        error(
            "Required stage script does not exist:\n"
            f"  {script}"
        )
        return False

    try:
        assert_safe_target_stage(
            stage,
            script,
        )
    except Exception as exc:
        error(str(exc))
        return False

    gp_master = find_gp_master(block_dir)

    try:
        assert_block_only_child_safe(
            stage,
            block,
            gp_master,
        )
    except Exception as exc:
        error(str(exc))
        return False

    if stage.key != "boundary" and gp_master is None:
        error(
            "Authoritative gp_master.geojson is missing from the "
            "locked block before this stage."
        )
        return False

    env = build_environment(
        state=state,
        district=district,
        block=block,
        block_dir=block_dir,
        gp_master=gp_master,
    )

    command = build_command(
        stage=stage,
        script=script,
        state=state,
        district=district,
        block=block,
    )

    print()
    info(
        "Launching child script in runner mode."
    )
    info(
        "Child location prompts are suppressed where supported."
    )
    print(
        f"Command: {' '.join(command)}"
    )

    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            input=location_stdin(
                state,
                district,
                block,
            ),
            text=True,
            check=False,
        )

    except KeyboardInterrupt:
        warning(
            f"{stage.key} interrupted by user."
        )
        return False

    except Exception as exc:
        error(
            f"Could not execute {stage.key}: {exc}"
        )
        return False

    elapsed = time.time() - start

    print()
    info(
        f"Stage runtime: {elapsed:.1f} seconds"
    )

    if result.returncode != 0:
        error(
            f"{stage.key} failed with exit code "
            f"{result.returncode}."
        )
        return False

    # Boundary must create/retain the authoritative master.
    if stage.key == "boundary":
        new_master = find_gp_master(
            block_dir
        )

        if new_master is None:
            error(
                "Boundary returned exit code 0, but the locked "
                "block does not contain processed/gp_master.geojson."
            )
            return False

        count = gp_master_feature_count(
            new_master
        )

        if count is None:
            error(
                "Boundary returned exit code 0, but the resulting "
                "GP master failed lightweight validation."
            )
            return False

        success(
            "Boundary completed and authoritative GP master is available:"
        )
        print(f"  {new_master}")
        print(f"  GP count: {count}")

        return True

    # All non-merge stages must produce declared outputs.
    if not stage_outputs_exist(
        stage,
        block_dir,
    ):
        error(
            f"{stage.key} returned exit code 0, but one or more "
            "declared outputs are missing."
        )

        for rel in stage.required_outputs:
            status = (
                "OK"
                if output_exists(
                    block_dir,
                    rel,
                )
                else "MISSING"
            )
            print(
                f"  [{status}] "
                f"{block_dir / rel}"
            )

        return False

    success(
        f"{stage.key} completed successfully."
    )
    return True


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    title(
        "SIH26 — UNIVERSAL GP DATA PIPELINE RUNNER"
    )

    print(
        "Enter the target location once.\n"
        "The runner will lock all downstream processing "
        "to this block."
    )

    print()

    state = input(
        "State   : "
    ).strip()

    district = input(
        "District: "
    ).strip()

    block = input(
        "Block   : "
    ).strip()

    if not state or not district or not block:
        error(
            "State, District and Block are all required."
        )
        return 1

    # ------------------------------------------------------------------
    # Resolve the EXISTING physical directory first.
    #
    # This fixes:
    #   North Goa
    #       vs
    #   North_Goa
    #
    # The runner never creates a second block merely because the spelling
    # differs.
    # ------------------------------------------------------------------
    block_dir = resolve_block_directory(
        state,
        district,
        block,
    )

    title("TARGET LOCK")

    print(f"State   : {state}")
    print(f"District: {district}")
    print(f"Block   : {block}")
    print("Directory:")
    print(f"  {block_dir}")

    print()

    info(
        "Only this physical block directory may be used by this run."
    )

    gp_master = find_gp_master(
        block_dir
    )

    if gp_master is not None:
        count = gp_master_feature_count(
            gp_master
        )

        print()

        success(
            "Existing authoritative GP master detected."
        )
        print(f"  {gp_master}")

        if count is not None:
            print(f"  GP count: {count}")

    else:
        print()
        info(
            "No existing GP master found."
        )
        info(
            "Boundary stage will create it."
        )

    # ------------------------------------------------------------------
    # Stage 01 — Boundary
    # ------------------------------------------------------------------
    boundary = next(
        s for s in STAGES
        if s.key == "boundary"
    )

    if precheck_stage(
        boundary,
        block_dir,
    ):
        gp_master = find_gp_master(
            block_dir
        )

        if gp_master is None:
            error(
                "Boundary pre-check claimed success but GP master "
                "cannot be resolved."
            )
            return 1

    else:
        if not run_stage(
            boundary,
            state,
            district,
            block,
            block_dir,
        ):
            error(
                "Pipeline stopped at GP boundary stage."
            )
            return 1

        gp_master = find_gp_master(
            block_dir
        )

        if gp_master is None:
            error(
                "GP master is missing after successful boundary stage."
            )
            return 1

    # ------------------------------------------------------------------
    # Stages 02..07 except merge
    # ------------------------------------------------------------------
    for stage in STAGES:

        if stage.key in {
            "boundary",
            "merge",
        }:
            continue

        if precheck_stage(
            stage,
            block_dir,
        ):
            continue

        # --------------------------------------------------------------
        # Sentinel-1 special safety gate
        # --------------------------------------------------------------
        if stage.key == "sentinel1":
            title(
                "STAGE 05 SAFETY CHECK - SENTINEL-1"
            )

            print(
                "The current Sentinel-1 V4 collector is project-wide:"
            )
            print(
                "it discovers every gp_master.geojson below gp/."
            )
            print()

            print(
                "It is therefore NOT allowed to be launched by the "
                "target-locked universal runner until the child script "
                "itself supports SIH26_TARGET_GP_MASTER."
            )
            print()

            if (
                raw_outputs_exist(
                    stage,
                    block_dir,
                )
                and stage_outputs_exist(
                    stage,
                    block_dir,
                )
            ):
                success(
                    "Existing Sentinel-1 outputs are present for "
                    "the locked block."
                )
                info(
                    "They are preserved. Sentinel-1 will NOT be rerun."
                )
                continue

            error(
                "Sentinel-1 is required for this block, but its current "
                "collector is not target-aware."
            )
            error(
                "Pipeline stopped before Sentinel-1 could accidentally "
                "process other blocks."
            )
            return 1

        if not run_stage(
            stage,
            state,
            district,
            block,
            block_dir,
        ):
            error(
                f"Pipeline stopped because Stage "
                f"{stage.key} failed."
            )
            print()
            print(
                "Existing raw/processed data in the locked block "
                "has not been deleted by the runner."
            )
            return 1

    # ------------------------------------------------------------------
    # Final merge — separate child script
    # ------------------------------------------------------------------
    merge = next(
        s for s in STAGES
        if s.key == "merge"
    )

    title(
        "FINAL GP MERGE"
    )

    if not run_stage(
        merge,
        state,
        district,
        block,
        block_dir,
    ):
        error(
            "Pipeline stopped at the final GP merge."
        )
        return 1

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    title(
        "SIH26 GP PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(f"State   : {state}")
    print(f"District: {district}")
    print(f"Block   : {block}")
    print()

    print("Locked directory:")
    print(f"  {block_dir}")

    print()
    print("Authoritative GP master:")
    print(
        f"  {find_gp_master(block_dir)}"
    )

    print()
    print("Final merged dataset:")
    print(
        f"  {block_dir / 'processed' / 'gp_merged_features.csv'}"
    )
    print(
        f"  {block_dir / 'processed' / 'gp_merged_features.geojson'}"
    )

    print()
    print(
        "No State/District/Block was hard-coded into the runner."
    )
    print(
        "No synthetic, interpolated, or imputed data was created "
        "by the runner."
    )
    print(
        "Raw observations were preserved."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
