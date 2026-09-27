from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
MODEL_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(MODEL_ROOT))
from common import training_location,MODELS_ROOT,TARGET_COLUMNS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args();s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip();root=training_location(s,d,b);manifest_path=root/'training_dataset_manifest.json'
    if not manifest_path.exists():raise RuntimeError('Training dataset manifest not found. Build and audit training data first.')
    out=MODELS_ROOT/root.relative_to(root.anchor) if False else MODELS_ROOT/root.relative_to(root.parents[2])
    # root = datasets/training/state/district/block -> relative part from datasets/training
    rel=root.relative_to(root.parents[2]);out=MODELS_ROOT/rel;out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for h in range(1,8):
        hd=root/f'D{h}';Xtr=pd.read_csv(hd/'X_train.csv');Xv=pd.read_csv(hd/'X_validation.csv');ytr=pd.read_csv(hd/'y_train.csv');yv=pd.read_csv(hd/'y_validation.csv')
        for target in TARGET_COLUMNS:
            model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)
            model.fit(Xtr,ytr[target])
            pred=model.predict(Xv);mae=mean_absolute_error(yv[target],pred);rmse=mean_squared_error(yv[target],pred)**0.5
            path=out/f'D{h}__{target}.joblib';joblib.dump({'model':model,'features':Xtr.columns.tolist(),'target':target,'horizon':h},path)
            summary.append({'horizon':h,'target':target,'validation_mae':float(mae),'validation_rmse':float(rmse),'model':str(path)})
            print(f'D{h} {target}: validation MAE={mae:.4f} RMSE={rmse:.4f}')
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('Models:',out)
if __name__=='__main__':main()
