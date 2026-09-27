from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import DATASETS_ROOT,resolve_physical_location,GP_ROOT

def clean(df,source):
    if df.columns.duplicated().any():
        for c in df.columns[df.columns.duplicated()].unique():
            pos=[i for i,x in enumerate(df.columns) if x==c]
            if not all(df.iloc[:,pos[0]].equals(df.iloc[:,j]) for j in pos[1:]): raise RuntimeError(f"Conflicting duplicate column '{c}' in {source}")
        df=df.loc[:,~df.columns.duplicated(keep='first')]
    for c in df.select_dtypes(include=['object','string']).columns: df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x)
    inf=0
    for c in df.select_dtypes(include=[np.number]).columns:
        v=pd.to_numeric(df[c],errors='coerce');m=np.isinf(v.to_numpy(dtype=float,copy=False));inf+=int(m.sum());
        if m.any():df.loc[m,c]=np.nan
    dup=int(df.duplicated().sum());df=df.drop_duplicates().reset_index(drop=True)
    return df,{"source":str(source),"rows":len(df),"exact_duplicate_rows_removed":dup,"infinite_to_nan":inf,"imputation":False,"interpolation":False,"outlier_removal":False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip()
    gd=resolve_physical_location(GP_ROOT,s,d,b);ps,pd_,pb=gd.parent.parent.name,gd.parent.name,gd.name
    src=gd/'processed'/'gp_merged_features.csv'
    if not src.exists():raise FileNotFoundError(src)
    out=DATASETS_ROOT/'cleaned'/ps/pd_/pb/'gp';out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(src,low_memory=False);cl,r=clean(df,src);dst=out/'gp_merged_features.csv';cl.to_csv(dst,index=False);r['output']=str(dst)
    (out/'gp_cleaning_report.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'location':[ps,pd_,pb],'report':r},indent=2),encoding='utf-8')
    print('[OK]',dst,cl.shape)
if __name__=='__main__':main()
