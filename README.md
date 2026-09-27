# SIH26 Downscaling

Universal pipeline for downscaling Block-level weather forecasts to
Gram-Panchayat level.

## Structure

- `block/` — Block-level raw, processed and metadata data
- `gp/` — GP-level spatial predictors
- `datasets/` — training, validation and testing datasets
- `models/` — trained models
- `scripts/common/geo.py` — shared GIS utilities
- `scripts/block/` — Block data collectors
- `scripts/gp/` — GP data collectors
- `scripts/modeling/` — dataset construction, baseline, training and evaluation

The pipeline must never insert fabricated weather/spatial values when a
real source fails.
