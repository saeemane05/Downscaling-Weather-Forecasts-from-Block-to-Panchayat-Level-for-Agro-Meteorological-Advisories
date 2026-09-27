from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import pandas as pd
import numpy as np
MODEL_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(MODEL_ROOT))
from common import training_location,TARGET_COLUMNS

def family(t):
    n=t.lower()
    if 'precip' in n or 'rain' in n:return ('precip','rain')
    if 'temperature' in n or 'temp' in n:return ('temperature','temp')
    if 'humidity' in n:return ('humidity','humidity')
    if 'wind' in n:return ('wind','wind')
    if 'pressure' in n:return ('pressure','pressure')
    return ('other','')

def find_forecast_col(cols,target,h):
    fam,_=family(target);cands=[c for c in cols if c.lower().endswith(f'_d{h}') and any(tok in c.lower() for tok in ([fam] if fam!='temperature' else ['temperature','temp']))]
    return cands[0] if cands else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args();s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip();root=training_location(s,d,b)
    manifest=json.loads((root/'training_dataset_manifest.json').read_text())
    rows=[]
    for h in range(1,8):
        m=manifest['horizons'][f'D{h}'];preds=m['forecast_predictors']
        y=pd.read_csv(root/f'D{h}'/'y_test.csv');idx=pd.read_csv(root/f'D{h}'/'sample_index_test.csv')
        Xsrc=pd.read_csv(root/f'D{h}'/'X_test.csv')
        for t in TARGET_COLUMNS:
            c=find_forecast_col(preds,t,h)
            if not c:
                rows.append({'horizon':h,'target':t,'forecast_column':None,'mae':None,'rmse':None,'n':0,'status':'UNAVAILABLE'});continue
            yv=pd.to_numeric(y[t],errors='coerce');pv=pd.to_numeric(Xsrc[c],errors='coerce');mask=yv.notna()&pv.notna();
            if not mask.any():rows.append({'horizon':h,'target':t,'forecast_column':c,'mae':None,'rmse':None,'n':0,'status':'NO_OVERLAP'});continue
            err=pv[mask].to_numpy()-yv[mask].to_numpy();rows.append({'horizon':h,'target':t,'forecast_column':c,'mae':float(np.mean(np.abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'bias':float(np.mean(err)),'n':int(mask.sum()),'status':'PASS'})
    out=root/'baseline_metrics.json';out.write_text(json.dumps(rows,indent=2),encoding='utf-8');pd.DataFrame(rows).to_csv(root/'baseline_metrics.csv',index=False);print('Baseline metrics:',out)
if __name__=='__main__':main()
