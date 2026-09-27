from __future__ import annotations
import argparse,json,re,shutil,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import *

FORECAST_META_TOKENS=('lead','issued_date','issue_date','run','retrieved','rule','source','model')
HIST_PREFIXES=('hist__','rainfall_','wet_day_','calendar_','time__','historical_')


def hard_excluded(c):
    low=str(c).lower()
    if low in {x.lower() for x in TARGET_COLUMNS}: return True
    if low in {'date','latitude','longitude','actual_is_reference_only','synthetic_data_present','interpolation_present'}: return True
    if any(t in low for t in ('forecast_lead_days_','forecast_issued_date_','forecast_issue_date_','forecast_run','forecast_retrieved','forecast_rule_','forecast_source_','forecast_model_','target_date','target_datetime','valid_time','acquisition_date','system_index','source_dataset','source_file','provenance')): return True
    return False

def forecast_predictor(c,h,dictionary):
    if hard_excluded(c): return False
    low=c.lower(); suf=f'_d{h}'
    if not low.endswith(suf): return False
    if any(t in low for t in FORECAST_META_TOKENS): return False
    info=dictionary.get(c,{})
    if info.get('feature_group','')=='weather_forecast' and info.get('final_role','')=='forecast_predictor': return True
    return any(t in low for t in ('temperature','temp','humidity','precip','rain','wind','pressure','cloud','radiation'))

def historical_predictor(c,dictionary):
    if hard_excluded(c): return False
    low=c.lower()
    if low.startswith(HIST_PREFIXES): return True
    info=dictionary.get(c,{})
    return info.get('feature_group','') in {'historical_weather','temporal_control'} and info.get('final_role','') in {'historical_source','candidate_predictor','temporal_alignment'}

def load_dict(path):
    if not path.exists(): return {}
    d=pd.read_csv(path);return {str(r['column']):{'feature_group':str(r.get('feature_group','')).lower(),'final_role':str(r.get('final_role','')).lower()} for _,r in d.iterrows() if 'column' in d.columns}

def static_gp_columns(gp,dictionary,gp_id):
    selected=[gp_id]
    for c in gp.columns:
        if c==gp_id or hard_excluded(c): continue
        if not pd.api.types.is_numeric_dtype(gp[c]): continue
        if gp[c].notna().sum()==0: continue
        info=dictionary.get(c,{})
        group=info.get('feature_group',''); role=info.get('final_role','')
        # Only static spatial context. Current remote-sensing/LST snapshots are not
        # assumed historically available and therefore stay out of X.
        if group in {'static_spatial_environment','spatial_location'} and role in {'gp_spatial_predictor','candidate_predictor','metadata_or_candidate'}:
            selected.append(c)
        elif c.lower().startswith('spatial__'):
            selected.append(c)
    return list(dict.fromkeys(selected))

def load_target(state,district,block):
    p=discover_target_source(state,district,block)
    if p is None: raise RuntimeError('No genuine GP target source found. Run training_data\\00_target_preflight.py first.')
    t=read_csv_checked(p);gid=detect_gp_id_column(t);dt=detect_date_column(t)
    missing=[x for x in TARGET_COLUMNS if x not in t.columns]
    if gid is None or dt is None or missing: raise RuntimeError(f'Invalid GP target contract: gp_id={gid}, date={dt}, missing={missing}')
    t=t.rename(columns={gid:'__gp_id_target',dt:'__target_date'}).copy();t['__target_date']=pd.to_datetime(t['__target_date'],errors='coerce')
    if t['__target_date'].isna().any(): raise RuntimeError('Target contains invalid dates.')
    t['__gp_id_target']=t['__gp_id_target'].astype(str).str.strip();
    if t.duplicated(['__gp_id_target','__target_date']).any(): raise RuntimeError('Duplicate GP/date target rows.')
    for c in TARGET_COLUMNS:t[c]=pd.to_numeric(t[c],errors='coerce')
    return t,p

def load_gp(gp_path,dictionary):
    g=read_csv_checked(gp_path);gid=detect_gp_id_column(g)
    if gid is None: raise RuntimeError('Could not identify GP ID in gp_merged_features_engineered.csv')
    g=g.rename(columns={gid:'__gp_id'}).copy();g['__gp_id']=g['__gp_id'].astype(str).str.strip()
    if g['__gp_id'].duplicated().any(): raise RuntimeError('GP feature table has duplicate GP IDs.')
    cols=static_gp_columns(g,dictionary,'__gp_id')
    if len(cols)<=1: raise RuntimeError('No valid static GP predictors were found.')
    return g[cols], '__gp_id', cols

def build_one(fc,h,gp,dictionary,target):
    issue=f'forecast_issued_date_d{h}';lead=f'forecast_lead_days_d{h}'
    if issue not in fc or lead not in fc: raise RuntimeError(f'D{h}: missing issue/lead columns')
    target_date=pd.to_datetime(fc['date'],errors='coerce') if 'date' in fc else pd.to_datetime(fc[detect_date_column(fc)],errors='coerce')
    issues=pd.to_datetime(fc[issue],errors='coerce');leads=pd.to_numeric(fc[lead],errors='coerce')
    if target_date.isna().any() or issues.isna().any(): raise RuntimeError(f'D{h}: invalid date values')
    if not ((target_date-issues).dt.days==h).all(): raise RuntimeError(f'D{h}: target-issue chronology failed')
    if not (leads==h).all(): raise RuntimeError(f'D{h}: lead validation failed')
    selected=[c for c in fc.columns if forecast_predictor(c,h,dictionary) or historical_predictor(c,dictionary)]
    selected=list(dict.fromkeys(selected))
    if not selected: raise RuntimeError(f'D{h}: no safe block predictors selected')
    base=fc[[('date' if 'date' in fc.columns else detect_date_column(fc))]+selected].copy();base=base.rename(columns={base.columns[0]:'__target_date'});base['__target_date']=target_date.values
    # Cross product of each forecast date and each GP's static spatial context.
    base['__key']=1; g=gp.copy();g['__key']=1; samples=base.merge(g,on='__key',how='inner').drop(columns='__key')
    samples['__target_date']=pd.to_datetime(samples['__target_date'])
    samples['__issue_date']=samples['__target_date']-pd.to_timedelta(h,unit='D')
    samples=samples.merge(target,left_on=['__gp_id','__target_date'],right_on=['__gp_id_target','__target_date'],how='inner',validate='one_to_one').drop(columns=['__gp_id_target'])
    if samples.empty: raise RuntimeError(f'D{h}: no GP/date target matches.')
    y=samples[TARGET_COLUMNS].copy()
    xcols=[c for c in samples.columns if c not in {'__target_date','__issue_date','__gp_id',*TARGET_COLUMNS}]
    X=samples[xcols].copy()
    for c in X.columns:X[c]=pd.to_numeric(X[c],errors='coerce')
    X=X.replace([np.inf,-np.inf],np.nan)
    forbidden=[c for c in X.columns if hard_excluded(c) or any(t in c.lower() for t in ('forecast_lead_days_','forecast_issued_date_','forecast_run','forecast_rule_'))]
    if forbidden: raise RuntimeError(f'D{h}: forbidden columns survived: {forbidden}')
    if X.select_dtypes(exclude=[np.number]).shape[1]: raise RuntimeError(f'D{h}: nonnumeric predictors survived')
    if any(X[c].notna().sum()==0 for c in X.columns): raise RuntimeError(f'D{h}: all-missing predictor survived')
    # Remove rows with missing target only; never impute target.
    valid_y=y.notna().all(axis=1);X=X.loc[valid_y].reset_index(drop=True);y=y.loc[valid_y].reset_index(drop=True);dates=samples.loc[valid_y,'__target_date'].reset_index(drop=True);gids=samples.loc[valid_y,'__gp_id'].reset_index(drop=True)
    # Chronological split by UNIQUE DATE. Every GP on a date remains in one split.
    unique_dates=sorted(dates.dt.normalize().unique());n=len(unique_dates);a=int(n*.70);b=int(n*.85);train_dates=set(unique_dates[:a]);val_dates=set(unique_dates[a:b]);test_dates=set(unique_dates[b:])
    split={}
    for name,ds in [('train',train_dates),('validation',val_dates),('test',test_dates)]:
        mask=dates.dt.normalize().isin(ds);split[name]={'X':X.loc[mask].reset_index(drop=True),'y':y.loc[mask].reset_index(drop=True),'dates':dates.loc[mask].reset_index(drop=True),'gp_ids':gids.loc[mask].reset_index(drop=True)}
    return split,{'horizon':h,'rows':len(X),'unique_dates':len(unique_dates),'gps':int(gids.nunique()),'predictors':X.columns.tolist(),'forecast_predictors':[c for c in X.columns if forecast_predictor(c,h,dictionary)],'historical_predictors':[c for c in X.columns if historical_predictor(c,dictionary)],'static_gp_predictors':[c for c in gp.columns if c!='__gp_id'],'split':{k:{'rows':len(v['X']),'unique_dates':int(v['dates'].dt.normalize().nunique()),'start':str(v['dates'].min().date()) if len(v['dates']) else None,'end':str(v['dates'].max().date()) if len(v['dates']) else None} for k,v in split.items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip()
    print('='*78);print('SIH26 — LEAKAGE-SAFE GP TRAINING DATA BUILDER');print('='*78)
    eng=resolve_physical_location(ENGINEERED_ROOT,s,d,b);ps,pd_,pb=physical_location_names(eng)
    fc_path=eng/'block'/'block_forecast_training_base_engineered.csv';gp_path=eng/'gp'/'gp_merged_features_engineered.csv'
    if not fc_path.exists():raise RuntimeError(f'Missing engineered source: {fc_path}')
    if not gp_path.exists():raise RuntimeError(f'Missing engineered GP source: {gp_path}')
    target,target_path=load_target(s,d,b);dictionary=load_dict(DICTIONARY_ROOT/ps/pd_/pb/'feature_dictionary.csv')
    fc=read_csv_checked(fc_path);gp,gpid,gpcols=load_gp(gp_path,dictionary)
    print('Block source:',fc_path,fc.shape);print('GP source:',gp_path,gp.shape);print('Target source:',target_path,target.shape);print('Static GP predictors:',len(gpcols))
    final=TRAINING_ROOT/ps/pd_/pb;tmp_parent=Path(tempfile.mkdtemp(prefix='gp_training_',dir=str(TRAINING_ROOT)));tmp=tmp_parent/'location';tmp.mkdir(parents=True)
    manifest={'builder_version':'GP_DOWNSCALING_FIXED_v1','location':[ps,pd_,pb],'source':{'forecast':str(fc_path),'gp_features':str(gp_path),'targets':str(target_path)},'policy':{'no_synthetic_targets':True,'no_target_imputation':True,'no_dynamic_satellite_without_time_series':True,'date_based_split':True,'forecast_metadata_never_x':True},'horizons':{}}
    try:
        for h in range(1,8):
            print('\n--- D%d ---'%h);split,meta=build_one(fc,h,gp,dictionary,target);hd=tmp/f'D{h}';hd.mkdir()
            for n,v in split.items():
                v['X'].to_csv(hd/f'X_{n}.csv',index=False);v['y'].to_csv(hd/f'y_{n}.csv',index=False);pd.DataFrame({'date':v['dates'],'gp_id':v['gp_ids']}).to_csv(hd/f'sample_index_{n}.csv',index=False)
            manifest['horizons'][f'D{h}']=meta;print('Rows:',meta['rows'],'GPs:',meta['gps'],'predictors:',len(meta['predictors']));print('Forecast predictors:',meta['forecast_predictors']);print('Historical predictors:',len(meta['historical_predictors']));print('Split:',meta['split'])
        (tmp/'training_dataset_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
        rows=[]
        for hk,m in manifest['horizons'].items():
            for c in m['predictors']:rows.append({'horizon':hk,'feature':c,'role':'forecast' if c in m['forecast_predictors'] else ('historical' if c in m['historical_predictors'] else 'static_gp')})
        pd.DataFrame(rows).to_csv(tmp/'feature_contract.csv',index=False)
        if final.exists():shutil.rmtree(final)
        tmp.rename(final);shutil.rmtree(tmp_parent,ignore_errors=True);print('\nBUILD COMPLETE:',final)
    except Exception:shutil.rmtree(tmp_parent,ignore_errors=True);raise
if __name__=='__main__':main()
