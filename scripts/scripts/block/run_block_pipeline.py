
"""
SIH26 — TOTAL BLOCK PIPELINE RUNNER
CORRECTED ERA5-LAND COMPATIBILITY VERSION

The seven existing stages and retry behavior are preserved.

Critical ERA5 fix:
A CDS ZIP archive may contain complementary NetCDF members, e.g.
    data_0.nc -> swvl1
    data_1.nc -> d2m + sp

An individual member is therefore NOT required to contain all variables.
Each member is normalized first, all members are merged, and ONLY THEN
is the complete required variable set validated.

04_era5_land.py is not modified.
Raw ERA5 files are not modified.
No synthetic data are created.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "block"


# ============================================================================
# PIPELINE STAGES
# ============================================================================

STAGES = [
    ("01 — Block boundary", SCRIPT_DIR / "01_block_boundary.py"),
    ("02 — Open-Meteo actual weather", SCRIPT_DIR / "02_open_meteo_actual.py"),
    ("03 — Historical forecasts", SCRIPT_DIR / "03_historical_forecasts.py"),
    ("04 — ERA5-Land", SCRIPT_DIR / "04_era5_land.py"),
    ("05 — GFS forecast", SCRIPT_DIR / "05_gfs.py"),
    ("06 — Block satellite features", SCRIPT_DIR / "06_block_satellite.py"),
    ("07 — Block merge", SCRIPT_DIR / "07_block_merge.py"),
]


# ============================================================================
# RETRY CONFIGURATION
# ============================================================================

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = [10, 20, 40]


# ============================================================================
# PHYSICAL BLOCK DIRECTORY RESOLUTION
# ============================================================================

def _path_key(value: str) -> str:
    """Compare path components while ignoring spaces, underscores and hyphens."""
    return "".join(
        ch.lower()
        for ch in str(value).strip()
        if ch.isalnum()
    )


def _resolve_component(parent: Path, requested: str) -> Path:
    """
    Resolve one directory component against the actual filesystem spelling.

    Example:
        requested: 'North Goa'
        existing:  'North_Goa'

    The existing directory is reused; nothing is renamed or created.
    """
    exact = parent / requested
    if exact.is_dir():
        return exact

    requested_key = _path_key(requested)
    if not requested_key:
        raise FileNotFoundError(
            f"Empty path component cannot be resolved under: {parent}"
        )

    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir() and _path_key(child.name) == requested_key
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            "Ambiguous directory resolution.\n"
            f"Requested : {requested!r}\n"
            f"Parent    : {parent}\n"
            f"Matches   : {[m.name for m in matches]}"
        )

    raise FileNotFoundError(
        "Could not resolve physical directory.\n"
        f"Requested : {requested!r}\n"
        f"Parent    : {parent}"
    )


def resolve_existing_block_dir(
    project_root: Path,
    state: str,
    district: str,
    block: str,
) -> Path:
    """
    Resolve the real block directory already present on disk.

    This deliberately does NOT rename or create directories.
    It only tolerates differences such as:
        'North Goa' == 'North_Goa'
    """
    block_root = project_root / "block"

    state_dir = _resolve_component(block_root, state)
    district_dir = _resolve_component(state_dir, district)
    block_dir = _resolve_component(district_dir, block)

    return block_dir


def make_location_input_from_physical_dir(block_dir: Path) -> str:
    """
    Build child-stage stdin from the physical directory names.

    This is critical for existing collectors that reconstruct paths directly
    from State/District/Block input.
    """
    return (
        f"{block_dir.parents[1].name}\n"
        f"{block_dir.parents[0].name}\n"
        f"{block_dir.name}\n"
    )


# ============================================================================
# ERA5 RUNNER-ONLY COMPATIBILITY WRAPPER
# ============================================================================

ERA5_WRAPPER_CODE = r"""
from __future__ import annotations

import io
import runpy
import sys
import zipfile
from pathlib import Path

import xarray as xr


_ORIGINAL_OPEN_DATASET = xr.open_dataset


REQUIRED_VARIABLES = {
    "2m_dewpoint_temperature",
    "surface_pressure",
    "volumetric_soil_water_layer_1",
}


VARIABLE_ALIASES = {
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "swvl1": "volumetric_soil_water_layer_1",
}


def normalize_dataset(ds, source_name, require_all=False):
    # Normalize one ERA5 dataset.
    # require_all=False is used for individual ZIP members because
    # complementary CDS members may contain different variables.
    # require_all=True is used only after the complete ZIP is merged.

    # ------------------------------------------------------------------
    # Time normalization
    # ------------------------------------------------------------------

    if "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
        print(
            "  [ERA5 compatibility] "
            f"{source_name}: valid_time -> time"
        )

    elif (
        "time" not in ds.coords
        and "valid_time" in ds.coords
        and "valid_time" not in ds.dims
    ):
        ds = ds.rename({"valid_time": "time"})
        print(
            "  [ERA5 compatibility] "
            f"{source_name}: valid_time -> time"
        )

    if "time" not in ds.dims and "time" not in ds.coords:
        raise RuntimeError(
            "ERA5 dataset contains neither time nor valid_time: "
            f"{source_name}"
        )

    # ------------------------------------------------------------------
    # ERA5 variable-name normalization
    # ------------------------------------------------------------------

    rename_vars = {
        short_name: long_name
        for short_name, long_name in VARIABLE_ALIASES.items()
        if short_name in ds.data_vars
        and long_name not in ds.data_vars
    }

    if rename_vars:
        ds = ds.rename(rename_vars)
        print(
            "  [ERA5 compatibility] "
            f"{source_name}: variables normalized: {rename_vars}"
        )

    # ------------------------------------------------------------------
    # IMPORTANT: only the FINAL merged dataset is required to contain
    # all variables.
    # ------------------------------------------------------------------

    if require_all:
        missing = REQUIRED_VARIABLES - set(ds.data_vars)

        if missing:
            raise RuntimeError(
                "ERA5 COMPLETE dataset is missing requested variables: "
                f"{sorted(missing)}; "
                f"available: {sorted(ds.data_vars)}; "
                f"source: {source_name}"
            )

    return ds


def open_genuine_era5(path, *args, **kwargs):
    # Open and normalize a normal NetCDF file.

    ds = _ORIGINAL_OPEN_DATASET(path, *args, **kwargs)

    try:
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")

        ds = normalize_dataset(
            ds,
            Path(path).name,
            require_all=True,
        )

        ds.load()

        print(
            "  [ERA5 compatibility] Genuine NetCDF normalized: "
            f"{Path(path).name}"
        )

        return ds

    except Exception:
        try:
            ds.close()
        except Exception:
            pass
        raise


def open_zip_era5(path_obj):
    # Open a ZIP-disguised .nc archive.
    # Every member is normalized independently; members may be partial.
    # All members are merged, then the merged dataset is validated.

    print(
        "  [ERA5 compatibility] ZIP-disguised NetCDF detected: "
        f"{path_obj.name}"
    )

    datasets = []

    try:
        with zipfile.ZipFile(path_obj, "r") as zf:

            nc_members = [
                name
                for name in zf.namelist()
                if name.lower().endswith(".nc")
                and not name.endswith("/")
            ]

            if not nc_members:
                raise RuntimeError(
                    "ERA5 ZIP contains no NetCDF member: "
                    f"{path_obj}"
                )

            print(
                "  [ERA5 compatibility] NetCDF members: "
                f"{[Path(x).name for x in nc_members]}"
            )

            for member in nc_members:

                member_name = Path(member).name
                source_name = f"{path_obj.name}:{member_name}"

                raw = zf.read(member)

                print(
                    "  [ERA5 compatibility] Opening "
                    f"{member_name} ({len(raw):,} bytes)"
                )

                ds = _ORIGINAL_OPEN_DATASET(
                    io.BytesIO(raw)
                )

                if "expver" in ds.coords:
                    ds = ds.drop_vars("expver")

                # ======================================================
                # CRITICAL FIX:
                # Do NOT require d2m/sp/swvl1 here.
                # ======================================================
                ds = normalize_dataset(
                    ds,
                    source_name,
                    require_all=False,
                )

                member_variables = set(ds.data_vars)
                useful = member_variables & REQUIRED_VARIABLES

                if not useful:
                    ds.close()
                    raise RuntimeError(
                        "ERA5 ZIP member contains none of the requested "
                        f"variables: {source_name}; "
                        f"available: {sorted(member_variables)}"
                    )

                # Load before the ZIP stream disappears.
                ds.load()

                print(
                    "  [ERA5 compatibility] Member accepted: "
                    f"{member_name} -> {sorted(useful)}"
                )

                datasets.append(ds)

        # ------------------------------------------------------------------
        # Merge ALL members only after ALL have been read.
        # ------------------------------------------------------------------

        if len(datasets) == 1:

            merged = datasets[0]

        else:

            print(
                "  [ERA5 compatibility] Merging "
                f"{len(datasets)} complementary NetCDF members..."
            )

            try:
                merged = xr.merge(
                    datasets,
                    compat="no_conflicts",
                    join="outer",
                    combine_attrs="override",
                )
            except TypeError:
                # Compatibility with older xarray versions.
                merged = xr.merge(
                    datasets,
                    compat="no_conflicts",
                    join="outer",
                )

        # ------------------------------------------------------------------
        # FINAL normalization + FINAL validation.
        # ------------------------------------------------------------------

        merged = normalize_dataset(
            merged,
            path_obj.name,
            require_all=True,
        )

        merged.load()

        print(
            "  [ERA5 compatibility] COMPLETE merged dataset ready:"
        )
        print(
            f"     dimensions={dict(merged.sizes)}"
        )
        print(
            f"     variables={sorted(merged.data_vars)}"
        )

        # Do not close the dataset that is being returned.
        for ds in datasets:
            if ds is not merged:
                try:
                    ds.close()
                except Exception:
                    pass

        return merged

    except Exception:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        raise


def open_era5(path, *args, **kwargs):

    path_obj = Path(path)

    try:
        is_zip = zipfile.is_zipfile(path_obj)
    except (OSError, ValueError, TypeError):
        is_zip = False

    if is_zip:
        return open_zip_era5(path_obj)

    return open_genuine_era5(
        path,
        *args,
        **kwargs,
    )


# Patch ONLY this child process.
xr.open_dataset = open_era5


# Run the original collector unchanged.
target_script = Path(sys.argv[1]).resolve()
sys.argv = [str(target_script)]

runpy.run_path(
    str(target_script),
    run_name="__main__",
)
"""


# ============================================================================
# TEMPORARY STAGE-04 LAUNCHER
# ============================================================================

class Era5TemporaryCompatibility:

    def __init__(self, block_dir: Path):
        self.block_dir = block_dir
        self.temp_dir = None
        self.wrapper_path = None

    def create(self) -> Path:

        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="sih26_era5_runner_"
        )

        self.wrapper_path = (
            Path(self.temp_dir.name)
            / "run_era5_compat.py"
        )

        self.wrapper_path.write_text(
            ERA5_WRAPPER_CODE,
            encoding="utf-8",
        )

        return self.wrapper_path

    def close(self):

        if self.temp_dir is not None:

            try:
                self.temp_dir.cleanup()

            except Exception as exc:

                print(
                    "[WARNING] Could not remove temporary "
                    f"ERA5 runner directory: {exc}"
                )

            self.temp_dir = None
            self.wrapper_path = None


# ============================================================================
# ERA5 RAW ARCHIVE AUDIT
# ============================================================================

def audit_era5_raw_archives(block_dir: Path):

    raw_dir = (
        block_dir
        / "raw"
        / "era5_land"
    )

    if not raw_dir.exists():

        print(
            "[ERA5 compatibility] "
            "Raw ERA5 directory does not exist yet."
        )

        return

    nc_files = sorted(
        raw_dir.glob("*.nc")
    )

    print()
    print(
        "[ERA5 compatibility] "
        f"Auditing {len(nc_files)} ERA5 raw .nc file(s)."
    )

    zip_count = 0
    genuine_count = 0

    for path in nc_files:

        try:

            if zipfile.is_zipfile(path):

                zip_count += 1

                with zipfile.ZipFile(path, "r") as zf:

                    members = [
                        Path(name).name
                        for name in zf.namelist()
                        if name.lower().endswith(".nc")
                        and not name.endswith("/")
                    ]

                print(
                    f"  • ZIP    {path.name}: {members}"
                )

            else:

                genuine_count += 1

                print(
                    f"  • NETCDF {path.name}"
                )

        except Exception as exc:

            print(
                f"  • ERROR  {path.name}: {exc}"
            )

    print()
    print(
        "[ERA5 compatibility] "
        f"ZIP-disguised: {zip_count}"
    )

    print(
        "[ERA5 compatibility] "
        f"Genuine NetCDF: {genuine_count}"
    )

    print(
        "[ERA5 compatibility] Raw archives will NOT be modified."
    )


# ============================================================================
# RUN ONE STAGE
# ============================================================================

def run_stage(
    stage_name: str,
    script_path: Path,
    user_input: str,
    block_dir: Path | None = None,
) -> bool:

    print()
    print("=" * 80)
    print(stage_name)
    print("=" * 80)

    if not script_path.exists():

        print("[ERROR] Script not found:")
        print(script_path)

        return False

    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    compatibility = None

    if (
        stage_name == "04 — ERA5-Land"
        and block_dir is not None
    ):

        print()
        print(
            "[ERA5 compatibility] "
            "Runner-only compatibility mode enabled."
        )

        audit_era5_raw_archives(
            block_dir
        )

        compatibility = Era5TemporaryCompatibility(
            block_dir
        )

        try:

            wrapper = compatibility.create()

            command = [
                sys.executable,
                "-u",
                str(wrapper),
                str(script_path),
            ]

        except Exception as exc:

            print()
            print(
                "[ERROR] Could not create temporary "
                "ERA5 compatibility wrapper:"
            )

            print(exc)

            compatibility.close()

            return False

    try:

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            input=user_input,
            text=True,
            check=False,
        )

        if result.returncode == 0:

            print()
            print(
                f"✓ {stage_name} completed successfully."
            )

            return True

        print()
        print(
            f"✗ {stage_name} failed "
            f"(exit code {result.returncode})."
        )

        return False

    except KeyboardInterrupt:

        print()
        print(
            f"[INTERRUPTED] {stage_name} was interrupted by the user."
        )

        return False

    except Exception as exc:

        print()
        print(
            f"[ERROR] Could not execute {stage_name}:"
        )

        print(exc)

        return False

    finally:

        if compatibility is not None:

            compatibility.close()

            print(
                "[ERA5 compatibility] "
                "Temporary wrapper removed."
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("SIH26 — BLOCK WEATHER DATA TOTAL PIPELINE")
    print("=" * 80)

    print()
    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print()
    print("Pipeline stages:")

    for index, (
        name,
        _,
    ) in enumerate(
        STAGES,
        start=1,
    ):

        print(
            f"  {index}. {name}"
        )

    # =========================================================================
    # LOCATION
    # =========================================================================

    print()
    print("-" * 80)
    print("LOCATION")
    print("-" * 80)

    state = input(
        "State: "
    ).strip()

    district = input(
        "District: "
    ).strip()

    block = input(
        "Block: "
    ).strip()

    if not state:
        raise ValueError(
            "State cannot be empty."
        )

    if not district:
        raise ValueError(
            "District cannot be empty."
        )

    if not block:
        raise ValueError(
            "Block cannot be empty."
        )

    # Initial path uses the user's exact input. After Stage 1 succeeds,
    # the physical directory is resolved from disk so later stages receive
    # the spelling that actually exists (e.g. North_Goa, not North Goa).
    block_dir = (
        PROJECT_ROOT
        / "block"
        / state
        / district
        / block
    )

    user_input = (
        f"{state}\n"
        f"{district}\n"
        f"{block}\n"
    )

    # =========================================================================
    # RUN ALL STAGES
    # =========================================================================

    total_stages = len(STAGES)

    for stage_number, (
        stage_name,
        script_path,
    ) in enumerate(
        STAGES,
        start=1,
    ):

        print()
        print()
        print("#" * 80)

        print(
            f"STAGE {stage_number}/{total_stages}: "
            f"{stage_name}"
        )

        print("#" * 80)

        stage_success = False

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            print()
            print(
                f"Attempt {attempt}/{MAX_RETRIES}"
            )

            stage_success = run_stage(
                stage_name,
                script_path,
                user_input,
                block_dir,
            )

            if stage_success:
                # Stage 1 may create/use a directory whose physical spelling
                # differs from the user's input (e.g. North_Goa vs North Goa).
                # Resolve that real directory immediately and use it for every
                # subsequent stage. Never rename or create a second directory.
                if stage_number == 1:
                    try:
                        block_dir = resolve_existing_block_dir(
                            PROJECT_ROOT,
                            state,
                            district,
                            block,
                        )

                        user_input = make_location_input_from_physical_dir(
                            block_dir
                        )

                        print()
                        print(
                            "[PATH RESOLUTION] Physical block directory:"
                        )
                        print(f"  {block_dir}")
                        print(
                            "[PATH RESOLUTION] Later stages will receive:"
                        )
                        print(
                            f"  State   : {block_dir.parents[1].name}"
                        )
                        print(
                            f"  District: {block_dir.parents[0].name}"
                        )
                        print(
                            f"  Block   : {block_dir.name}"
                        )

                    except Exception as exc:
                        print()
                        print(
                            "[ERROR] Stage 1 succeeded, but the resulting "
                            "physical block directory could not be resolved."
                        )
                        print(exc)
                        sys.exit(1)

                break

            if attempt >= MAX_RETRIES:

                print()
                print("=" * 80)
                print("PIPELINE STOPPED")
                print("=" * 80)

                print(
                    f"Failed stage : {stage_name}"
                )

                print(
                    f"Attempts made: {attempt}"
                )

                print(
                    "No further automatic retries."
                )

                print("=" * 80)

                sys.exit(1)

            wait_seconds = RETRY_WAIT_SECONDS[
                min(
                    attempt - 1,
                    len(RETRY_WAIT_SECONDS) - 1,
                )
            ]

            print()
            print(
                f"Retrying {stage_name} "
                f"in {wait_seconds} seconds..."
            )

            try:

                time.sleep(
                    wait_seconds
                )

            except KeyboardInterrupt:

                print()
                print(
                    "[INTERRUPTED] Retry wait cancelled by user."
                )

                sys.exit(1)

        print()
        print(
            f"✓ Stage {stage_number}/{total_stages} complete."
        )

    # =========================================================================
    # FINAL SUCCESS
    # =========================================================================

    print()
    print()
    print("=" * 80)
    print("SIH26 BLOCK PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(
        f"State   : {state}"
    )

    print(
        f"District: {district}"
    )

    print(
        f"Block   : {block}"
    )

    print()
    print(
        "All stages completed:"
    )

    for index, (
        stage_name,
        _,
    ) in enumerate(
        STAGES,
        start=1,
    ):

        print(
            f"  ✓ {index}. {stage_name}"
        )

    print()
    print(
        "Original raw datasets were preserved."
    )

    print(
        "04_era5_land.py was NOT modified."
    )

    print(
        "ERA5 ZIP compatibility was applied only "
        "inside the Stage-04 child process."
    )

    print(
        "ZIP members were validated only after "
        "complementary members were merged."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
