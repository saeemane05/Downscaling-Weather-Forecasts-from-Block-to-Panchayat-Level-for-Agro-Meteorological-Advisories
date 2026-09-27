# SIH26 TRUE FINAL

The repeated failure was caused by running the audit against a training folder
created by the older builder. The audit cannot remove columns from existing X
files.

This package therefore uses a builder that ALWAYS rebuilds from:

`datasets/engineered/<State>/<District>/<Block>/block_forecast_training_base_engineered.csv`

It never reads existing `datasets/training/.../D*/X_*.csv` as input.

Run BOTH steps:

```powershell
python scripts\modeling\training_data\01_build_leakage_safe_training_dataset.py
python scripts\modeling\training_data\02_audit_training_dataset.py
```

The builder explicitly excludes `forecast_lead_days_d1..d7` and all forecast
issue/run metadata.

It also writes the exact manifest structure expected by the audit.

Do not train until the audit reports FINAL STATUS: PASS.
