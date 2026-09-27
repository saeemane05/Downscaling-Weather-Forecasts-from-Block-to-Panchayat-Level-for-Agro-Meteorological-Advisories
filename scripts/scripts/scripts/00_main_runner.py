#!/usr/bin/env python3
"""
SIH26 — FINAL UNIVERSAL MAIN RUNNER
===================================

Single entry point for the complete SIH26 downscaling data pipeline.

User enters State / District / Block ONCE.

Modes:
    1 = Block + GP complete pipeline
    2 = Block pipeline only
    3 = GP pipeline only
    4 = Exit

Design:
    - No State/District/Block-specific hardcoding.
    - Uses the existing physical block directory when it already exists.
    - Tolerates generic spelling differences such as North Goa / North_Goa.
    - Passes the resolved physical location to both child runners.
    - Preserves raw data.
    - Does not create synthetic/imputed/interpolated observations.
    - Does not modify the working block or GP child scripts.
    - Block child retains its ERA5 compatibility wrapper.
    - GP child retains its target-locking and safety checks.

Expected project layout:
    PROJECT_ROOT/
        scripts/
            00_main_runner.py
            block/
                run_block_pipeline.py
            gp/
                08_gp_runner.py
"""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCK_RUNNER = PROJECT_ROOT / "scripts" / "block" / "run_block_pipeline.py"
GP_RUNNER = PROJECT_ROOT / "scripts" / "gp" / "08_gp_runner.py"


# ============================================================================
# CONSOLE
# ============================================================================

def line(char: str = "=", n: int = 80) -> None:
    print(char * n)


def title(text: str) -> None:
    print()
    line()
    print(text)
    line()


def error(text: str) -> None:
    print(f"[ERROR] {text}")


def info(text: str) -> None:
    print(f"[INFO] {text}")


def success(text: str) -> None:
    print(f"[SUCCESS] {text}")


# ============================================================================
# GENERIC LOCATION / PHYSICAL DIRECTORY RESOLUTION
# ============================================================================

def location_key(value: str) -> str:
    """
    Generic comparison key only.

    North Goa / North_Goa / north-goa -> northgoa
    This is not geographic mapping.
    """
    return "".join(
        ch.lower()
        for ch in str(value).strip()
        if ch.isalnum()
    )


def clean_location_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    return value.strip(" ._")


def resolve_existing_child(parent: Path, requested: str) -> Path | None:
    if not parent.is_dir():
        return None

    target = location_key(requested)

    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir() and location_key(child.name) == target
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            "Ambiguous physical directory resolution.\n"
            f"Requested: {requested!r}\n"
            f"Parent: {parent}\n"
            f"Matches:\n"
            + "\n".join(f"  {p}" for p in matches)
        )

    return None


def resolve_physical_block_dir(
    state: str,
    district: str,
    block: str,
) -> Path:
    """
    Resolve an existing physical BLOCK directory.

    If the block is new, create the canonical spelling using underscores.
    Existing directories are NEVER renamed.
    """
    block_root = PROJECT_ROOT / "block"
    block_root.mkdir(parents=True, exist_ok=True)

    state_dir = resolve_existing_child(block_root, state)
    if state_dir is None:
        state_dir = block_root / clean_location_name(state)
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


def physical_location_input(block_dir: Path) -> str:
    """
    Child runners reconstruct paths from State/District/Block.

    Therefore send the ACTUAL physical directory spelling, not the user's
    original spelling. This prevents North Goa vs North_Goa mismatches.
    """
    return (
        f"{block_dir.parents[1].name}\n"
        f"{block_dir.parents[0].name}\n"
        f"{block_dir.name}\n"
    )


# ============================================================================
# MODE
# ============================================================================

def choose_mode() -> str:
    title("SIH26 — FINAL UNIVERSAL MAIN RUNNER")

    print("Select pipeline mode:")
    print()
    print("  1. BLOCK + GP — complete downscaling data pipeline")
    print("  2. BLOCK ONLY  — block-level data pipeline")
    print("  3. GP ONLY     — Gram Panchayat-level data pipeline")
    print("  4. EXIT")
    print()

    mode = input("Mode: ").strip()

    if mode not in {"1", "2", "3", "4"}:
        raise ValueError("Invalid mode. Enter 1, 2, 3, or 4.")

    return mode


# ============================================================================
# CHILD RUNNER
# ============================================================================

def run_child(
    label: str,
    script: Path,
    location_input: str,
) -> bool:
    title(label)

    if not script.is_file():
        error(f"Required runner does not exist:\n  {script}")
        return False

    print(f"Runner: {script}")
    print()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(script),
            ],
            cwd=str(PROJECT_ROOT),
            input=location_input,
            text=True,
            check=False,
        )
    except KeyboardInterrupt:
        error(f"{label} interrupted by user.")
        return False
    except Exception as exc:
        error(f"Could not execute {label}: {exc}")
        return False

    if result.returncode != 0:
        error(
            f"{label} failed with exit code "
            f"{result.returncode}."
        )
        return False

    success(f"{label} completed successfully.")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    title("SIH26 — FINAL UNIVERSAL MAIN RUNNER")

    print(f"Project root:")
    print(f"  {PROJECT_ROOT}")
    print()

    if not BLOCK_RUNNER.is_file():
        error(f"Block runner missing:\n  {BLOCK_RUNNER}")
        return 1

    if not GP_RUNNER.is_file():
        error(f"GP runner missing:\n  {GP_RUNNER}")
        return 1

    try:
        mode = choose_mode()
    except ValueError as exc:
        error(str(exc))
        return 1

    if mode == "4":
        print("Exiting.")
        return 0

    # ------------------------------------------------------------------------
    # ONE LOCATION ENTRY
    # ------------------------------------------------------------------------

    title("TARGET LOCATION")

    state = input("State   : ").strip()
    district = input("District: ").strip()
    block = input("Block   : ").strip()

    if not state or not district or not block:
        error("State, District and Block are all required.")
        return 1

    # Resolve the real physical block directory once.
    try:
        block_dir = resolve_physical_block_dir(
            state,
            district,
            block,
        )
    except Exception as exc:
        error(f"Could not resolve target block directory:\n{exc}")
        return 1

    # Use the actual physical spelling for BOTH child runners.
    child_location = physical_location_input(block_dir)

    title("TARGET LOCK")

    print(f"User input:")
    print(f"  State   : {state}")
    print(f"  District: {district}")
    print(f"  Block   : {block}")
    print()
    print(f"Physical block directory:")
    print(f"  {block_dir}")
    print()
    print("Child runners will receive the physical directory spelling:")
    print(
        f"  {block_dir.parents[1].name} / "
        f"{block_dir.parents[0].name} / "
        f"{block_dir.name}"
    )

    # ------------------------------------------------------------------------
    # BLOCK
    # ------------------------------------------------------------------------

    if mode in {"1", "2"}:
        if not run_child(
            "BLOCK PIPELINE — 7 STAGES",
            BLOCK_RUNNER,
            child_location,
        ):
            error("Main pipeline stopped at the BLOCK pipeline.")
            return 1

    # ------------------------------------------------------------------------
    # GP
    # ------------------------------------------------------------------------

    if mode in {"1", "3"}:
        if not run_child(
            "GP PIPELINE — TARGET-LOCKED",
            GP_RUNNER,
            child_location,
        ):
            error("Main pipeline stopped at the GP pipeline.")
            return 1

    # ------------------------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------------------------

    title("SIH26 MAIN PIPELINE COMPLETED SUCCESSFULLY")

    print(f"State   : {state}")
    print(f"District: {district}")
    print(f"Block   : {block}")
    print()
    print("Locked physical block directory:")
    print(f"  {block_dir}")
    print()
    print("Completed mode:")
    if mode == "1":
        print("  BLOCK + GP")
    elif mode == "2":
        print("  BLOCK ONLY")
    else:
        print("  GP ONLY")
    print()
    print("No State/District/Block-specific mapping is hard-coded.")
    print("Existing physical directories are reused.")
    print("No synthetic/imputed/interpolated observations are created by this runner.")
    print("The working BLOCK and GP child runners were not modified.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
