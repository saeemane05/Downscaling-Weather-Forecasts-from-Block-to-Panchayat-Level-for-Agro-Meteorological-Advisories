from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import DATASETS_ROOT,resolve_all_locations,physical_location_names


def remove_100pct(df):
    removed=[c for c in df.columns if len(df)>0 and df[c].notna().sum()==0]
    return df.drop(columns=removed),removed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip()
    # Run child cleaners so all model-feature copies start from cleaned data.
    import subprocess
    for script in [MODEL_ROOT/'cleaning'/'00_data_audit.py',MODEL_ROOT/'cleaning'/'01_clean_block.py',MODEL_ROOT/'cleaning'/'02_clean_gp.py']:
        subprocess.run([sys.executable,str(script),s,d,b],check=True)
    bd,gd=resolve_all_locations(s,d,b);ps,pd_,pb=physical_location_names(bd)
    root=DATASETS_ROOT/'cleaned'/ps/pd_/pb
    block_sources=['block_observations_daily','block_forecast_training_base','block_current_forecast']
    specs=[]
    for name in block_sources:
        src=root/'block'/f'{name}.csv'
        if src.exists(): specs.append(('block',name,src))
    gp_src=root/'gp'/'gp_merged_features.csv'
    if gp_src.exists():specs.append(('gp','gp_merged_features',gp_src))
    manifest={'location':[ps,pd_,pb],'rule':'Remove only 100%-missing columns from already-cleaned copies.','datasets':{}}
    for typ,name,src in specs:
        df=pd.read_csv(src,low_memory=False);cl,removed=remove_100pct(df)
        dst=root/typ/f'{name}_model_features.csv';cl.to_csv(dst,index=False)
        manifest['datasets'][name]={'input':str(src),'output':str(dst),'rows':len(cl),'columns_before':len(df.columns),'columns_after':len(cl.columns),'removed_100pct_columns':removed,'rows_removed':0,'imputation':False,'interpolation':False}
        print(f'[MODEL COPY] {name}: {len(df.columns)} -> {len(cl.columns)} columns; removed {len(removed)}')
    (root/'100pct_unavailable_feature_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('CLEANING PIPELINE COMPLETE')
if __name__=='__main__':main()
