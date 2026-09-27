# SIH26 — FULLY FIXED MODELING PIPELINE

This folder replaces the previous `scripts/modeling` package.

## What was fixed

1. **Correct engineered paths**
   - Engineered block files live under `engineered/<state>/<district>/<block>/block/`.
   - The old training builder incorrectly looked one directory too high.

2. **Physical location spelling**
   - State/District/Block are resolved once and the physical directory spelling is reused for all outputs.

3. **Cleaning dependency**
   - The 100%-missing-column model copies are now made from the already-cleaned datasets, not directly from the original processed files.

4. **Historical feature connection**
   - Past-only lag/rolling weather features are now actually merged into the forecast training base by target date.

5. **GP-level training contract**
   - One training sample is `GP × target date × forecast horizon`.
   - The builder does NOT replicate a block actual observation to every GP.

6. **Real GP target requirement**
   - A genuine Panchayat/GP target dataset is required before final model training.
   - No synthetic target is created by this package.

7. **Static vs dynamic GP features**
   - Static terrain/environmental GP features can be used historically.
   - The current GP satellite/LST snapshot is not silently treated as historical information.
   - Date-indexed dynamic satellite features require a separate historical table and issue-time cutoff verification.

8. **Chronological splitting**
   - Train/validation/test are split by unique target dates, so GPs from one date cannot be split across datasets.

9. **Training audit**
   - Checks forbidden metadata, target leakage, nonnumeric/nonfinite X, missing targets, duplicate GP/date keys and date overlap between splits.

10. **Executable baseline/model/evaluation scripts**
   - Baseline compares the raw block forecast replicated to each GP against genuine GP targets.
   - Training uses `HistGradientBoostingRegressor` as a reproducible candidate baseline model.
   - Evaluation reports test MAE/RMSE/bias.

## Required GP target contract

Create:

`datasets/targets/<State>/<District>/<Block>/gp_weather_targets.csv`

with:

```text
gp_id,date,actual_precipitation_sum,actual_temperature_2m_mean,actual_relative_humidity_2m_mean,actual_wind_speed_10m_mean,actual_surface_pressure_mean
```

The values must come from a real/reference source. This modeling package will refuse to fabricate Panchayat targets.

## Main commands

From the project root:

```powershell
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through clean
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through features
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through audits
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through training
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through baseline
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through train
python scripts\modeling\00_modeling_runner.py Maharashtra Nashik Sinnar --through evaluate
```

Do not run `--through training` until `00_target_preflight.py` passes.
Do not train until `02_audit_training_dataset.py` reports `FINAL STATUS: PASS`.

## Important scientific boundary

The current block/GP pipeline does not itself create a genuine GP-level weather target. Therefore the target file is an explicit dependency. This is intentional: using the block actual value as the target for every Panchayat would not be genuine Panchayat-level downscaling and would create a scientifically misleading training set.
