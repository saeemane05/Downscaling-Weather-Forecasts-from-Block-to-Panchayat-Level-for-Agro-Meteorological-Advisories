# SIH26 Modeling Data Cleaning

## Folder

Place these files at:

```text
SIH26_Downscaling/
└── scripts/
    └── modeling/
        └── cleaning/
            ├── 00_data_audit.py
            ├── 01_clean_block.py
            ├── 02_clean_gp.py
            └── run_cleaning.py
```

## Run

From the project root:

```powershell
python scripts\modeling\cleaning\run_cleaning.py
```

Then enter:

```text
State   : Maharashtra
District: Nashik
Block   : Sinnar
```

Or pass them directly:

```powershell
python scripts\modeling\cleaning\run_cleaning.py Maharashtra Nashik Sinnar
```

## Output

The original `block/` and `gp/` data are never overwritten.

Cleaned copies are created under:

```text
datasets/
└── cleaned/
    └── <State>/
        └── <District>/
            └── <Block>/
                ├── audit/
                │   └── data_audit.json
                ├── block/
                │   ├── block_observations_daily.csv
                │   ├── block_forecast_training_base.csv
                │   ├── block_current_forecast.csv
                │   └── block_cleaning_report.json
                └── gp/
                    ├── gp_merged_features.csv
                    └── gp_cleaning_report.json
```

## What this first cleaning stage does

- audits schemas and missingness
- detects duplicate column labels
- safely collapses identical duplicate columns
- stops if duplicate column labels contain conflicting values
- trims whitespace from string cells
- converts infinite numeric values to NaN
- removes exact duplicate rows from the cleaned copy
- preserves legitimate missing values
- preserves source data and provenance

## What it deliberately does NOT do yet

- no imputation
- no interpolation
- no synthetic values
- no outlier removal
- no normalization/scaling
- no feature selection
- no feature engineering
- no train/validation/test split
- no model training

Those are separate scientific stages and should be decided after the audit.
