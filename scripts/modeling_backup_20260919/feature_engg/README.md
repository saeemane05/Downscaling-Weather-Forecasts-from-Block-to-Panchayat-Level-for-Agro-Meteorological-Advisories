# Feature engineering

Outputs are written to:

`datasets/engineered/<physical State>/<physical District>/<physical Block>/`

with `block/` and `gp/` subdirectories.

Important fixes:
- project root is resolved with `parents[3]`;
- physical directory spelling is used after location resolution;
- past-only historical weather features are now actually merged into the engineered forecast training base by target date;
- the current GP snapshot is treated as static spatial context only;
- dynamic satellite/LST features are **not** silently used as historical predictors unless a date-indexed GP dynamic table exists and passes the issue-time cutoff audit.
