# Leakage-safe GP training data

## Scientific contract

One sample is:

`one Panchayat × one target date × one forecast horizon`

The builder requires a **genuine Panchayat/GP-level target dataset**. It will not copy a block-level actual observation to every GP.

Required target file:

`datasets/targets/<State>/<District>/<Block>/gp_weather_targets.csv`

Required columns:

- `gp_id` (or a recognized equivalent)
- `date` (or a recognized equivalent)
- `actual_precipitation_sum`
- `actual_temperature_2m_mean`
- `actual_relative_humidity_2m_mean`
- `actual_wind_speed_10m_mean`
- `actual_surface_pressure_mean`

The target must be a real/reference dataset. Do not generate synthetic GP targets.

## Dynamic satellite rule

The current `gp_merged_features_engineered.csv` is treated as a static GP context table. Its current Sentinel/LST snapshot is **not** used as a historical predictor. Dynamic satellite features can be added only after a date-indexed GP time series exists and passes the forecast issue-time cutoff audit.

## Splitting

Train/validation/test are split by unique target dates, not by rows. Therefore all Panchayats belonging to one date remain in the same split.
