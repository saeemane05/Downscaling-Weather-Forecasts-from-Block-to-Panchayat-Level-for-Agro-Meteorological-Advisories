import argparse
from pathlib import Path
import json
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEANED_ROOT = PROJECT_ROOT / "datasets" / "cleaned"
ENGINEERED_ROOT = PROJECT_ROOT / "datasets" / "engineered"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
REPORT_ROOT = PROJECT_ROOT / "datasets" / "information_availability_audit"

def norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def resolve_location(root, state, district, block):
    if not root.exists():
        raise FileNotFoundError(root)
    target = [norm(state), norm(district), norm(block)]
    candidates = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        parts = [norm(x) for x in p.parts]
        if all(t in parts for t in target):
            score = 0
            if norm(p.name) == target[2]: score += 10
            if norm(p.parent.name) == target[1]: score += 5
            candidates.append((score, len(p.parts), p))
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve {state}/{district}/{block} under {root}"
        )
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]

def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def extract_context(text, pattern, radius=220):
    hits = []
    for m in re.finditer(pattern, text, flags=re.I):
        a=max(0,m.start()-radius); b=min(len(text),m.end()+radius)
        snippet=text[a:b].replace("\n"," ")
        hits.append(snippet)
    return hits[:8]

def inspect_scripts():
    targets = {
        "era5": [
            SCRIPTS_ROOT/"block"/"04_era5_land.py",
            SCRIPTS_ROOT/"block"/"04_era5_land.py.txt",
        ],
        "satellite": [
            SCRIPTS_ROOT/"block"/"06_block_satellite.py",
            SCRIPTS_ROOT/"block"/"06_block_satellite.py.txt",
        ],
        "merge": [
            SCRIPTS_ROOT/"block"/"07_block_merge.py",
            SCRIPTS_ROOT/"block"/"07_block_merge.py.txt",
        ],
    }
    result={}
    for family, paths in targets.items():
        found=next((p for p in paths if p.exists()), None)
        if found is None:
            # fallback filename search
            pats={
                "era5":["*era5*"],
                "satellite":["*satellite*"],
                "merge":["*merge*"],
            }[family]
            matches=[]
            for pat in pats:
                matches.extend((SCRIPTS_ROOT/"block").glob(pat))
            found=matches[0] if matches else None
        if not found:
            result[family]={
                "script_found":False,
                "status":"UNVERIFIED",
                "reason":"Expected collector/merge script was not found in the local project."
            }
            continue
        text=read_text(found)
        patterns=[
            r"start_date", r"end_date", r"window", r"days", r"lookback",
            r"filterDate", r"filter_date", r"acquisition", r"date",
            r"target_date", r"forecast_issued", r"valid_time", r"time_start",
            r"time_end", r"advance", r"subtract", r"timedelta", r"rolling"
        ]
        contexts=[]
        for p in patterns:
            contexts += extract_context(text,p)
        # de-duplicate while preserving order
        contexts=list(dict.fromkeys(contexts))
        result[family]={
            "script_found":True,
            "path":str(found),
            "line_count":len(text.splitlines()),
            "status":"REVIEW_REQUIRED",
            "relevant_code_contexts":contexts[:40],
            "note":"This report records collector logic for human verification; it does not infer a safe cutoff from column names."
        }
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="SIH26 information-availability audit")
    parser.add_argument("--state")
    parser.add_argument("--district")
    parser.add_argument("--block")
    args = parser.parse_args()
    if any((args.state, args.district, args.block)) and not all((args.state, args.district, args.block)):
        parser.error("--state, --district and --block must be supplied together")
    return args


def main():
    args = parse_args()
    print("="*72)
    print("SIH26 FINAL PROVENANCE + INFORMATION-AVAILABILITY AUDIT")
    print("="*72)
    print("READ-ONLY. No datasets or source scripts will be modified.\n")

    state=args.state or input("State: ").strip()
    district=args.district or input("District: ").strip()
    block=args.block or input("Block: ").strip()

    if not all((state, district, block)):
        raise ValueError("State, District and Block are required.")

    print(f"State: {state}\nDistrict: {district}\nBlock: {block}")

    cleaned=resolve_location(CLEANED_ROOT,state,district,block)
    engineered=resolve_location(ENGINEERED_ROOT,state,district,block)
    print("\nCleaned location:\n ",cleaned)
    print("\nEngineered location:\n ",engineered)

    src_path=cleaned/"block"/"block_forecast_training_base.csv"
    eng_path=engineered/"block"/"block_forecast_training_base_engineered.csv"
    src=pd.read_csv(src_path,nrows=None)
    eng=pd.read_csv(eng_path,nrows=None)
    print(f"\nForecast source: {src.shape}")
    print(f"Forecast engineered: {eng.shape}")

    report={
        "location":{"state":state,"district":district,"block":block},
        "source_shape":list(src.shape),
        "engineered_shape":list(eng.shape),
        "forecast_temporal_checks":[],
        "ground_truth_controls":{},
        "collector_provenance":{},
        "feature_family_status":{},
        "status":"REVIEW_REQUIRED",
        "blockers":[],
    }

    # Forecast cutoff
    target=pd.to_datetime(src["date"],errors="coerce") if "date" in src.columns else pd.Series(dtype="datetime64[ns]")
    for d in range(1,8):
        ic=f"forecast_issued_date_d{d}"
        lc=f"forecast_lead_days_d{d}"
        if ic in src.columns:
            issue=pd.to_datetime(src[ic],errors="coerce")
            valid=target.notna()&issue.notna()
            delta=(target[valid]-issue[valid]).dt.total_seconds()/86400
            bad=int((delta<=0).sum())
            report["forecast_temporal_checks"].append({
                "day":d,"issue_column":ic,"rows_checked":int(valid.sum()),
                "bad_rows":bad,
                "min_target_minus_issue_days":float(delta.min()) if len(delta) else None,
                "max_target_minus_issue_days":float(delta.max()) if len(delta) else None,
                "status":"PASS" if bad==0 else "FAIL"
            })
            print(f"D{d}: target > issue: {'PASS' if bad==0 else 'FAIL'}; range={delta.min() if len(delta) else None}..{delta.max() if len(delta) else None} days")
            if bad: report["blockers"].append(f"D{d} has target dates at/before issue dates.")
        if lc in src.columns:
            lead=pd.to_numeric(src[lc],errors="coerce")
            valid=lead.notna()
            bad=int((lead[valid]!=d).sum())
            if bad: report["blockers"].append(f"D{d} lead-day values do not equal {d}.")

    # Ground truth controls
    controls={"actual_is_reference_only":True,"synthetic_data_present":False,"interpolation_present":False}
    print("\nGround-truth controls:")
    for c,expected in controls.items():
        if c not in src.columns:
            report["ground_truth_controls"][c]={"status":"UNVERIFIED","reason":"Missing"}
            report["blockers"].append(f"Missing control column: {c}")
            print(f"  {c}: UNVERIFIED")
            continue
        vals=src[c].dropna().astype(str).str.lower().str.strip().unique().tolist()
        expected_s=str(expected).lower()
        ok=all(v==expected_s for v in vals)
        report["ground_truth_controls"][c]={"status":"PASS" if ok else "FAIL","values":vals}
        print(f"  {c}: {'PASS' if ok else 'FAIL'}")
        if not ok: report["blockers"].append(f"Unexpected values in {c}.")

    # Feature families: counts only; dates must come from collector logic.
    families={
        "ERA5":[c for c in src.columns if c.lower().startswith("era5_")],
        "Sentinel-1":[c for c in src.columns if c.lower().startswith("s1_")],
        "Sentinel-2":[c for c in src.columns if c.lower().startswith("s2_")],
        "LST":[c for c in src.columns if c.lower().startswith("lst_")],
    }
    print("\nMerged-table feature families:")
    for k,v in families.items():
        report["feature_family_status"][k]={
            "feature_count":len(v),
            "acquisition_count_columns":[c for c in v if "acquisition_count" in c.lower()],
            "status":"REQUIRES_COLLECTOR_PROVENANCE"
        }
        print(f"  {k}: {len(v)} features")
        if any("acquisition_count" in c.lower() for c in v):
            print("    acquisition_count is treated as COUNT, NOT DATE.")

    # Inspect actual collector code
    print("\n" + "="*72)
    print("COLLECTOR / MERGE PROVENANCE")
    print("="*72)
    provenance=inspect_scripts()
    report["collector_provenance"]=provenance
    for family,info in provenance.items():
        print(f"\n{family.upper()}:")
        print(f"  script_found: {info.get('script_found')}")
        if info.get("script_found"):
            print(f"  path: {info['path']}")
            print(f"  line_count: {info['line_count']}")
            print("  relevant code excerpts:")
            for s in info["relevant_code_contexts"][:12]:
                print("   ",s)
        else:
            print("  status: UNVERIFIED")
            report["blockers"].append(f"{family} collector/merge script could not be located.")

    # Explicitly require collector evidence for these families.
    for family in ["era5","satellite","merge"]:
        if not provenance[family].get("script_found"):
            continue
        report["blockers"].append(
            f"{family} timing still requires verification from the displayed collector/merge logic and its source-date fields."
        )

    out=REPORT_ROOT/state/district/block
    out.mkdir(parents=True,exist_ok=True)
    json_path=out/"final_provenance_information_availability_audit.json"
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(report,f,indent=2,default=str)

    print("\n" + "="*72)
    print("FINAL PROVENANCE AUDIT COMPLETED")
    print("="*72)
    print(f"Status: {report['status']}")
    print(f"JSON: {json_path}")
    print("\nNo files were modified.")
    print("\nIMPORTANT:")
    print("The audit does NOT declare ERA5/S1/S2/LST safe merely from merged feature names.")
    print("It also does NOT treat acquisition_count as a timestamp.")
    print("Review the printed collector excerpts before training.")

if __name__=="__main__":
    main()
