from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import TRAINING_ROOT,training_location,TARGET_COLUMNS
FORBIDDEN=('forecast_lead_days_','forecast_issued_date_','forecast_issue_date_','forecast_run','forecast_retrieved','forecast_rule_','forecast_source_','forecast_model_','target_date','target_datetime','valid_time','acquisition_date','provenance')

def audit_h(hdir,h,manifest):
    issues=[]
    for split in ('train','validation','test'):
        xp=hdir/f'X_{split}.csv';yp=hdir/f'y_{split}.csv';ip=hdir/f'sample_index_{split}.csv'
        if not xp.exists() or not yp.exists() or not ip.exists():issues.append(f'missing files for {split}');continue
        X=pd.read_csv(xp);y=pd.read_csv(yp);idx=pd.read_csv(ip)
        bad=[c for c in X.columns if any(t in c.lower() for t in FORBIDDEN) or c.lower() in {'actual_is_reference_only','synthetic_data_present','interpolation_present'}]
        if bad:issues.append(f'{split}: forbidden X columns {bad}')
        if any(c in X.columns for c in TARGET_COLUMNS):issues.append(f'{split}: target in X')
        if len(X)!=len(y) or len(X)!=len(idx):issues.append(f'{split}: row alignment failure')
        if X.select_dtypes(exclude=[np.number]).shape[1]:issues.append(f'{split}: nonnumeric X')
        if not np.isfinite(X.to_numpy(dtype=float)).all():issues.append(f'{split}: nonfinite X')
        if any(X[c].notna().sum()==0 for c in X.columns):issues.append(f'{split}: all-missing X feature')
        if not all(c in y.columns for c in TARGET_COLUMNS):issues.append(f'{split}: missing target columns')
        dates=pd.to_datetime(idx['date'],errors='coerce')
        if dates.isna().any():issues.append(f'{split}: invalid dates')
        if idx['gp_id'].isna().any():issues.append(f'{split}: missing gp_id')
        if idx.duplicated(['date','gp_id']).any():issues.append(f'{split}: duplicate date/gp_id rows')
    # date separation
    sets=[]
    for s in ('train','validation','test'):
        p=hdir/f'sample_index_{s}.csv'
        if p.exists(): sets.append(set(pd.to_datetime(pd.read_csv(p)['date']).dt.normalize()))
    if len(sets)==3 and (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]):issues.append('date overlap between splits')
    m=manifest.get('horizons',{}).get(f'D{h}',{})
    if not m.get('split') or not all(k in m['split'] for k in ('train','validation','test')):issues.append('manifest split metadata incomplete')
    return issues

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip();root=training_location(s,d,b);mp=root/'training_dataset_manifest.json'
    print('SIH26 GP training-data audit — READ-ONLY');print('Location:',root)
    if not mp.exists():raise RuntimeError(f'Missing manifest: {mp}')
    manifest=json.loads(mp.read_text(encoding='utf-8'));failed=[]
    for h in range(1,8):
        hdir=root/f'D{h}';issues=audit_h(hdir,h,manifest);print(f'\nD{h}:', 'PASS' if not issues else 'FAIL')
        for x in issues:print('  FAIL:',x)
        if issues:failed.append(h)
    print('\nFINAL STATUS:', 'FAIL' if failed else 'PASS')
    if failed:print('Failed horizons:',','.join('D'+str(x) for x in failed));raise SystemExit(2)
if __name__=='__main__':main()
