from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))
from common import DATASETS_ROOT, BLOCK_ROOT, GP_ROOT, resolve_all_locations


def audit_csv(path: Path):
    df = pd.read_csv(path, low_memory=False)
    numeric = df.select_dtypes(include=[np.number])
    nonfinite = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if not numeric.empty else 0
    missing = df.isna().sum()
    return {
        "file": str(path), "rows": len(df), "columns": len(df.columns),
        "duplicate_columns": df.columns[df.columns.duplicated()].unique().tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "nonfinite_numeric": nonfinite,
        "all_missing_columns": [c for c in df.columns if df[c].notna().sum() == 0],
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "missing_columns": {str(c): int(v) for c, v in missing.items() if v},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?"); ap.add_argument("district", nargs="?"); ap.add_argument("block", nargs="?")
    a = ap.parse_args()
    state=a.state or input("State: ").strip(); district=a.district or input("District: ").strip(); block=a.block or input("Block: ").strip()
    bd, gd = resolve_all_locations(state,district,block)
    files=[
        bd/"processed"/"merge"/"block_observations_daily.csv",
        bd/"processed"/"merge"/"block_forecast_training_base.csv",
        bd/"processed"/"merge"/"block_current_forecast.csv",
        gd/"processed"/"gp_merged_features.csv",
    ]
    outroot=DATASETS_ROOT/"modeling_audits"/bd.parent.parent.name/bd.parent.name/bd.name
    outroot.mkdir(parents=True,exist_ok=True)
    reports=[]
    for f in files:
        if f.exists():
            try: reports.append({"status":"OK",**audit_csv(f)})
            except Exception as e: reports.append({"status":"ERROR","file":str(f),"error":repr(e)})
        else: reports.append({"status":"MISSING","file":str(f)})
    out=outroot/"data_audit.json"
    out.write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"state":state,"district":district,"block":block,"reports":reports},indent=2,default=str),encoding="utf-8")
    print(f"Audit report: {out}")
    for r in reports: print(r["status"],r["file"])

if __name__=="__main__": main()
