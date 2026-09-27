from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
MODEL_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(MODEL_ROOT))
from common import training_location,MODELS_ROOT,TARGET_COLUMNS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args();s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip();root=training_location(s,d,b);rel=root.relative_to(root.parents[2]);mdir=MODELS_ROOT/rel
    if not mdir.exists():raise RuntimeError(f'Model directory not found: {mdir}')
    rows=[]
    for h in range(1,8):
        hd=root/f'D{h}';X=pd.read_csv(hd/'X_test.csv');y=pd.read_csv(hd/'y_test.csv')
        for target in TARGET_COLUMNS:
            mp=mdir/f'D{h}__{target}.joblib'
            if not mp.exists(): rows.append({'horizon':h,'target':target,'status':'MODEL_MISSING'});continue
            bundle=joblib.load(mp);pred=bundle['model'].predict(X[bundle['features']]);err=pred-y[target].to_numpy();rows.append({'horizon':h,'target':target,'mae':float(mean_absolute_error(y[target],pred)),'rmse':float(mean_squared_error(y[target],pred)**0.5),'bias':float(np.mean(err)),'n':len(y),'status':'PASS'})
    pd.DataFrame(rows).to_csv(mdir/'test_metrics.csv',index=False);(mdir/'test_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('Evaluation:',mdir/'test_metrics.csv')
if __name__=='__main__':main()
