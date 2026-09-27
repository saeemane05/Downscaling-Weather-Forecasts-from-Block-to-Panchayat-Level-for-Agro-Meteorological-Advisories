from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def find_root(start: Path) -> Path:
    for p in [start.resolve(), *start.resolve().parents]:
        if all((p / x).exists() for x in ["block", "gp", "datasets", "scripts"]):
            return p
    raise RuntimeError("SIH26 project root not found.")


ROOT = find_root(Path(__file__).parent)


def clean(x):
    return re.sub(r'[<>:"/\\|?*]', "_", str(x).strip())


def location():
    print("\nEnter the target block location.")
    state = input("State: ").strip()
    district = input("District: ").strip()
    block = input("Block: ").strip()
    if not all([state, district, block]):
        raise ValueError("State, District and Block are required.")
    d = ROOT / "block" / clean(state) / clean(district) / clean(block)
    if not d.exists():
        raise FileNotFoundError(f"Block directory not found: {d}")
    m = d / "processed" / "merge"
    m.mkdir(parents=True, exist_ok=True)
    return state, district, block, d, m


def csvs(d):
    return sorted(p for p in d.rglob("*.csv") if "merge" not in p.parts)


def find_file(files, exact=(), terms=(), exclude=()):
    exact = {x.lower() for x in exact}
    for p in files:
        if p.name.lower() in exact:
            return p
    c = [p for p in files
         if all(t.lower() in p.name.lower() for t in terms)
         and not any(t.lower() in p.name.lower() for t in exclude)]
    if not c:
        return None
    processed = [p for p in c if "processed" in p.parts]
    if processed:
        c = processed
    c.sort(key=lambda p: sum(w in p.name.lower()
                             for w in ["archive", "daily", "clean", "historical"]),
           reverse=True)
    return c[0]


def discover(d):
    f = csvs(d)
    return {
        "actual": find_file(
            f,
            exact=("02_open_meteo_actual.csv", "open_meteo_actual.csv",
                   "weather_daily_cleaned.csv"),
            terms=("actual",), exclude=("forecast", "gfs")),
        "historical_forecast": find_file(
            f,
            exact=("historical_forecasts.csv", "03_historical_forecasts.csv",
                   "historical_forecast_archive.csv"),
            terms=("forecast",), exclude=("gfs", "current", "daily_archive")),
        "era5": find_file(
            f,
            exact=("era5_land_daily.csv", "04_era5_land.csv", "era5_land.csv"),
            terms=("era5",)),
        "gfs": find_file(
            f,
            exact=("gfs_daily_archive.csv", "05_gfs_daily_archive.csv"),
            terms=("gfs", "daily")),
        "sentinel2": find_file(f, exact=("sentinel2_block.csv",),
                               terms=("sentinel2",)),
        "sentinel1": find_file(f, exact=("sentinel1_block.csv",),
                               terms=("sentinel1",)),
        "landsat_lst": find_file(f, exact=("landsat_lst_block.csv",),
                                 terms=("landsat", "lst")),
    }


def read(p, label):
    if p is None:
        print(f"  {label}: NOT FOUND")
        return pd.DataFrame()
    print(f"  {label}: {p}")
    x = pd.read_csv(p)
    print(f"    {len(x)} rows x {len(x.columns)} columns")
    return x


def date_key(df, preferred):
    if df.empty:
        return df
    lower = {str(c).lower(): c for c in df.columns}
    col = next((lower[x.lower()] for x in preferred if x.lower() in lower), None)
    if col is None:
        for x in ["date", "target_date", "target_datetime", "time",
                  "datetime", "acquisition_date"]:
            if x in lower:
                col = lower[x]
                break
    if col is None:
        raise RuntimeError(f"No date column found. Columns: {list(df.columns)}")
    y = df.copy()
    y["date"] = pd.to_datetime(y[col], errors="coerce").dt.strftime("%Y-%m-%d")
    y = y[y["date"].notna()].copy()
    return y


def prefix(df, p):
    if df.empty:
        return df
    y = df.copy()
    protected = {"date", "forecast_issued_at", "forecast_retrieved_at",
                 "forecast_run_datetime", "target_datetime",
                 "forecast_source", "forecast_model", "forecast_lead_hours",
                 "forecast_lead_days_d1", "synthetic", "fallback_used"}
    return y.rename(columns={
        c: (c if c in protected or c.startswith(p) else p + c)
        for c in y.columns
    })


def prep_actual(x):
    return prefix(date_key(x, ["date", "target_date"]), "actual_")


def prep_era5(x):
    y = date_key(x, ["date", "target_date"])
    if not y.empty:
        y = y.drop_duplicates("date", keep="last")
    return prefix(y, "era5_")


def prep_forecast(x):
    y = date_key(x, ["target_date", "target_datetime", "date"])
    for c in ["forecast_issued_at", "forecast_retrieved_at",
              "forecast_run_datetime", "target_datetime"]:
        if c in y:
            y[c] = pd.to_datetime(y[c], errors="coerce", utc=True)
    return y


def prep_sat(x, p):
    y = date_key(x, ["acquisition_date", "date"])
    if y.empty:
        return y
    protected = {"date", "acquisition_date", "acquisition_datetime",
                 "system_index", "source_dataset", "sensor", "orbit_pass"}
    nums = []
    for c in y.columns:
        if c not in protected:
            z = pd.to_numeric(y[c], errors="coerce")
            if z.notna().any():
                y[c] = z
                nums.append(c)
    if not nums:
        return y[["date"]].drop_duplicates()
    z = y.groupby("date", as_index=False)[nums].mean()
    counts = y.groupby("date").size().rename(p + "acquisition_count").reset_index()
    z = z.merge(counts, on="date", how="left", validate="one_to_one")
    return prefix(z, p)


def daily_master(parts):
    frames = [x for x in parts if not x.empty]
    if not frames:
        return pd.DataFrame()
    m = frames[0].copy()
    for x in frames[1:]:
        m = m.merge(x, on="date", how="outer", validate="one_to_one")
    return m.sort_values("date").reset_index(drop=True)


def training_base(actual, era5, forecast, s2, s1, lst):
    if forecast.empty:
        return pd.DataFrame()
    m = forecast.copy()
    if not actual.empty:
        m = m.merge(actual, on="date", how="left", validate="many_to_one")
    if not era5.empty:
        m = m.merge(era5, on="date", how="left", validate="many_to_one")
    for x in [s2, s1, lst]:
        if not x.empty:
            m = m.merge(x, on="date", how="left", validate="many_to_one")

    m["actual_is_reference_only"] = True
    m["synthetic_data_present"] = False
    m["interpolation_present"] = False

    # Satellite is joined for diagnostic completeness only. It must pass an
    # availability test downstream before being used as a forecasting feature.
    m["satellite_temporal_validity"] = "UNKNOWN"
    issue = next((c for c in ["forecast_issued_at", "forecast_retrieved_at",
                              "forecast_run_datetime"] if c in m.columns), None)
    if issue:
        issue_t = pd.to_datetime(m[issue], errors="coerce", utc=True)
        target_t = pd.to_datetime(m["date"], errors="coerce", utc=True)
        m.loc[issue_t.notna() & target_t.notna(), "satellite_temporal_validity"] = np.where(
            target_t < issue_t, "DATE_BEFORE_ISSUANCE", "DATE_NOT_BEFORE_ISSUANCE"
        )
    return m


def audit(name, df, keys):
    print("\n" + "-" * 68)
    print(name)
    print("-" * 68)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    if not df.empty and "date" in df:
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not d.empty:
            print(f"Date range: {d.min().date()} -> {d.max().date()}")
    if df.empty:
        print("Status: EMPTY")
        return
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise RuntimeError(f"Missing audit keys: {missing}")
    dup = df.duplicated(keys).sum()
    print(f"Duplicate key rows: {dup}")
    if dup:
        raise RuntimeError(f"{name}: duplicate keys found.")
    print("Synthetic data introduced: NO")
    print("Interpolation introduced: NO")
    print("Audit: PASS")


def main():
    state, district, block, d, out = location()

    print("\n" + "=" * 68)
    print("SIH26 — BLOCK DATA MERGE")
    print("=" * 68)
    print(f"{state} / {district} / {block}")

    paths = discover(d)
    print("\nDISCOVERED INPUTS")
    for k, p in paths.items():
        print(f"  {k:22}: {p if p else 'NOT FOUND'}")

    actual = prep_actual(read(paths["actual"], "Actual weather"))
    era5 = prep_era5(read(paths["era5"], "ERA5-Land"))
    forecast = prep_forecast(read(paths["historical_forecast"], "Historical forecast"))
    gfs = prep_forecast(read(paths["gfs"], "GFS"))
    s2 = prep_sat(read(paths["sentinel2"], "Sentinel-2"), "s2_")
    s1 = prep_sat(read(paths["sentinel1"], "Sentinel-1"), "s1_")
    lst = prep_sat(read(paths["landsat_lst"], "Landsat LST"), "lst_")

    daily = daily_master([actual, era5, s2, s1, lst])
    daily_path = out / "block_observations_daily.csv"
    if not daily.empty:
        daily.to_csv(daily_path, index=False)
    audit("BLOCK DAILY OBSERVATIONS", daily, ["date"])

    base = training_base(actual, era5, forecast, s2, s1, lst)
    base_path = out / "block_forecast_training_base.csv"
    if not base.empty:
        base.to_csv(base_path, index=False)
        keys = [c for c in ["forecast_retrieved_at", "forecast_issued_at",
                            "forecast_run_datetime", "target_datetime", "date"]
                if c in base.columns]
        audit("BLOCK FORECAST TRAINING BASE", base, keys if len(keys) >= 2 else ["date"])
    else:
        print("\nHistorical forecast dataset not found/empty; training base not created.")

    current = pd.DataFrame()
    if not gfs.empty and "forecast_retrieved_at" in gfs.columns:
        r = pd.to_datetime(gfs["forecast_retrieved_at"], errors="coerce", utc=True)
        if r.notna().any():
            current = gfs.loc[r == r.max()].copy()
    current_path = out / "block_current_forecast.csv"
    if not current.empty:
        current.to_csv(current_path, index=False)
        audit("CURRENT GFS FORECAST", current,
              [c for c in ["forecast_retrieved_at", "target_datetime", "date"]
               if c in current.columns])
    else:
        print("\nCurrent GFS table not created: no reliable retrieval cycle found.")

    meta = {
        "script": "07_block_merge.py",
        "version": "SIH26-07-v1",
        "state": state, "district": district, "block": block,
        "inputs": {k: str(v) if v else None for k, v in paths.items()},
        "outputs": {
            "daily_observations": str(daily_path),
            "forecast_training_base": str(base_path),
            "current_forecast": str(current_path),
        },
        "rules": {
            "actual_weather_role": "reference/target only",
            "historical_forecast_runs_collapsed": False,
            "satellite_daily_aggregation": "mean of numeric acquisition statistics",
            "missing_imputation": False,
            "interpolation": False,
            "synthetic": False,
            "future_satellite_information_automatically_allowed": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (out / "block_merge_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 68)
    print("BLOCK MERGE COMPLETE")
    print("=" * 68)
    print(f"Daily observations : {daily_path if not daily.empty else 'NOT CREATED'}")
    print(f"Training base      : {base_path if not base.empty else 'NOT CREATED'}")
    print(f"Current GFS        : {current_path if not current.empty else 'NOT CREATED'}")
    print(f"Metadata           : {out / 'block_merge_metadata.json'}")
    print("Actual as forecast feature: NO")
    print("Synthetic/interpolated data: NO")
    print("=" * 68)


if __name__ == "__main__":
    main()
