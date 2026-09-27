from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(MODEL_ROOT))
from common import DATASETS_ROOT, resolve_physical_location, BLOCK_ROOT

def clean(df, source):
    if df.columns.duplicated().any():
        for c in df.columns[df.columns.duplicated()].unique():
            pos=[i for i,x in enumerate(df.columns) if x==c]
            if not all(df.iloc[:,pos[0]].equals(df.iloc[:,j]) for j in pos[1:]):
                raise RuntimeError(f"Conflicting duplicate column '{c}' in {source}")
        df=df.loc[:,~df.columns.duplicated(keep='first')]
    for c in df.select_dtypes(include=['object','string']).columns:
        df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x)
    inf=0
    for c in df.select_dtypes(include=[np.number]).columns:
        v=pd.to_numeric(df[c],errors='coerce'); m=np.isinf(v.to_numpy(dtype=float,copy=False)); inf+=int(m.sum())
        if m.any(): df.loc[m,c]=np.nan
    dup=int(df.duplicated().sum()); df=df.drop_duplicates().reset_index(drop=True)
    return df,{"source":str(source),"rows":len(df),"exact_duplicate_rows_removed":dup,"infinite_to_nan":inf,"imputation":False,"interpolation":False,"outlier_removal":False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('state',nargs='?'); ap.add_argument('district',nargs='?'); ap.add_argument('block',nargs='?'); a=ap.parse_args()
    s=a.state or input('State: ').strip(); d=a.district or input('District: ').strip(); b=a.block or input('Block: ').strip()
    bd=resolve_physical_location(BLOCK_ROOT,s,d,b); ps,pd_,pb=bd.parent.parent.name,bd.parent.name,bd.name
    out=DATASETS_ROOT/'cleaned'/ps/pd_/pb/'block'; out.mkdir(parents=True,exist_ok=True)
    sources={
      'block_observations_daily':bd/'processed'/'merge'/'block_observations_daily.csv',
      'block_forecast_training_base':bd/'processed'/'merge'/'block_forecast_training_base.csv',
      'block_current_forecast':bd/'processed'/'merge'/'block_current_forecast.csv'}
    reports=[]
    for name,src in sources.items():
        if not src.exists(): print('[SKIP]',src); continue
        df=pd.read_csv(src,low_memory=False); cl,r=clean(df,src); dst=out/f'{name}.csv'; cl.to_csv(dst,index=False); r['output']=str(dst); reports.append(r); print('[OK]',dst,cl.shape)
    (out/'block_cleaning_report.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'location':[ps,pd_,pb],'reports':reports},indent=2),encoding='utf-8')
if __name__=='__main__': main()
