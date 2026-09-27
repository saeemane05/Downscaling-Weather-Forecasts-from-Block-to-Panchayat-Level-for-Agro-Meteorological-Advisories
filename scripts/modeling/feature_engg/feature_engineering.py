from __future__ import annotations
import argparse,json,sys,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import CLEANED_ROOT,DICTIONARY_ROOT,ENGINEERED_ROOT,resolve_physical_location,physical_location_names

WEATHER_WORDS=("rain","rainfall","precip","temperature","temp","humidity","relative_humidity","wind","pressure","dewpoint","dew_point")
TARGETS={"actual_precipitation_sum","actual_temperature_2m_mean","actual_relative_humidity_2m_mean","actual_wind_speed_10m_mean","actual_surface_pressure_mean"}

def choose_date(df):
    for n in ["date","target_date","valid_date","forecast_date","datetime","target_datetime","timestamp","time"]:
        for c in df.columns:
            if str(c).lower()==n:
                return c
    for c in df.columns:
        if any(x in str(c).lower() for x in ("date","datetime","timestamp")):
            if pd.to_datetime(df[c],errors='coerce').notna().mean()>=.8:return c
    return None

def model_file(loc,folder,stem):
    for p in [loc/folder/f'{stem}_model_features.csv',loc/folder/f'{stem}.csv']:
        if p.exists():return p
    raise FileNotFoundError(f'Missing dataset: {loc/folder}/{stem}')

def load_dictionary(loc):
    p=DICTIONARY_ROOT/loc.parent.parent.name/loc.parent.name/loc.name/'feature_dictionary.csv'
    if not p.exists():raise FileNotFoundError(f'Missing feature dictionary: {p}')
    return pd.read_csv(p)

def select_weather(dictionary,obs):
    if 'feature_group' not in dictionary.columns:return []
    q=dictionary[(dictionary.source_dataset=='block_observations_daily') & (dictionary.feature_group=='historical_weather')]
    candidates=[c for c in q.column.astype(str) if c in obs.columns and c not in TARGETS]
    selected=[]
    families=[('rain',lambda x:'rain' in x.lower() or 'precip' in x.lower()),('temperature',lambda x:'temp' in x.lower() or 'temperature' in x.lower()),('humidity',lambda x:'humidity' in x.lower()),('wind',lambda x:'wind' in x.lower()),('pressure',lambda x:'pressure' in x.lower()),('dewpoint',lambda x:'dewpoint' in x.lower() or 'dew_point' in x.lower())]
    for _,pred in families:
        cs=[c for c in candidates if pred(c)]
        if cs:
            cs.sort(key=lambda c:(('mean' in c.lower())*50+('2m' in c.lower())*20-('std' in c.lower())*20-('max' in c.lower())*10-('min' in c.lower())*10),reverse=True)
            selected.append(cs[0])
    return list(dict.fromkeys(selected))

def add_history(obs,date_col,primary):
    out=obs.copy();out[date_col]=pd.to_datetime(out[date_col],errors='coerce');out=out.sort_values(date_col).reset_index(drop=True);created=[]
    for col in primary:
        v=pd.to_numeric(out[col],errors='coerce')
        for lag in (1,3,7):
            n=f'hist__{col}__lag_{lag}d';out[n]=v.shift(lag);created.append(n)
        past=v.shift(1)
        for w in (3,7,14):
            n=f'hist__{col}__rolling_mean_{w}d';out[n]=past.rolling(w,min_periods=w).mean();created.append(n)
            n=f'hist__{col}__rolling_std_{w}d';out[n]=past.rolling(w,min_periods=w).std();created.append(n)
        if 'rain' in col.lower() or 'precip' in col.lower():
            for w in (3,7,14):
                n=f'hist__{col}__rolling_sum_{w}d';out[n]=past.rolling(w,min_periods=w).sum();created.append(n)
                n=f'hist__{col}__wet_days_{w}d';out[n]=(past.gt(0).rolling(w,min_periods=w).sum());created.append(n)
    return out,created

def add_calendar(df,date_col):
    out=df.copy();dt=pd.to_datetime(out[date_col],errors='coerce');doy=dt.dt.dayofyear.astype(float);m=dt.dt.month.astype(float)
    out['time__day_of_year_sin']=np.sin(2*np.pi*doy/365.25);out['time__day_of_year_cos']=np.cos(2*np.pi*doy/365.25)
    out['time__month_sin']=np.sin(2*np.pi*m/12);out['time__month_cos']=np.cos(2*np.pi*m/12)
    return out

def add_gp_anomalies(gp,dictionary):
    out=gp.copy();created=[]
    if 'feature_group' not in dictionary.columns:return out,created
    q=dictionary[(dictionary.source_dataset=='gp_merged_features') & (dictionary.feature_group.isin(['static_spatial_environment','spatial_location'])) & (dictionary.final_role.isin(['gp_spatial_predictor','candidate_predictor','metadata_or_candidate']))]
    for c in q.column.astype(str):
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):continue
        v=pd.to_numeric(out[c],errors='coerce')
        if v.notna().sum()<2 or v.nunique(dropna=True)<=1:continue
        n=f'spatial__{c}__block_median_anomaly';out[n]=v-v.median();created.append(n)
    return out,created

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip()
    loc=resolve_physical_location(CLEANED_ROOT,s,d,b);ps,pd_,pb=physical_location_names(loc)
    dictionary=load_dictionary(loc);outroot=ENGINEERED_ROOT/ps/pd_/pb;bo=outroot/'block';go=outroot/'gp';bo.mkdir(parents=True,exist_ok=True);go.mkdir(parents=True,exist_ok=True)
    obs_path=model_file(loc,'block','block_observations_daily');fc_path=model_file(loc,'block','block_forecast_training_base');cur_path=model_file(loc,'block','block_current_forecast');gp_path=model_file(loc,'gp','gp_merged_features')
    obs=pd.read_csv(obs_path,low_memory=False);fc=pd.read_csv(fc_path,low_memory=False);cur=pd.read_csv(cur_path,low_memory=False);gp=pd.read_csv(gp_path,low_memory=False)
    od=choose_date(obs);fd=choose_date(fc);cd=choose_date(cur)
    if od is None or fd is None:raise RuntimeError('Could not identify date in observations or forecast training base')
    primary=select_weather(dictionary,obs);obs_eng,hist=add_history(obs,od,primary);obs_eng=add_calendar(obs_eng,od)
    obs_eng.to_csv(bo/'block_observations_daily_engineered.csv',index=False)
    fc_eng=add_calendar(fc,fd)
    # IMPORTANT FIX: merge only past-only engineered history onto the forecast table by target date.
    hist_frame=obs_eng[[od]+hist].copy();hist_frame[od]=pd.to_datetime(hist_frame[od]);hist_frame=hist_frame.rename(columns={od:'__join_date'})
    fc_eng['__join_date']=pd.to_datetime(fc_eng[fd],errors='coerce')
    if hist_frame['__join_date'].duplicated().any():raise RuntimeError('Historical engineered observations contain duplicate dates; refusing unsafe merge.')
    fc_eng=fc_eng.merge(hist_frame,on='__join_date',how='left',validate='many_to_one').drop(columns='__join_date')
    fc_eng.to_csv(bo/'block_forecast_training_base_engineered.csv',index=False)
    cur_eng=add_calendar(cur,cd) if cd else cur.copy();cur_eng.to_csv(bo/'block_current_forecast_engineered.csv',index=False)
    gp_eng,gpan=add_gp_anomalies(gp,dictionary);gp_eng.to_csv(go/'gp_merged_features_engineered.csv',index=False)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'location':[ps,pd_,pb],'inputs':{'observations':str(obs_path),'forecast':str(fc_path),'current':str(cur_path),'gp':str(gp_path)},'history_columns':hist,'gp_static_anomaly_columns':gpan,'dynamic_gp_satellite_policy':'excluded from historical training unless a date-indexed GP dynamic feature table exists and passes issue-time cutoff audit','source_overwrite':False}
    (outroot/'feature_engineering_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print('FEATURE ENGINEERING COMPLETE');print('Historical features:',len(hist));print('Forecast shape:',fc_eng.shape);print('GP shape:',gp_eng.shape);print('Output:',outroot)
if __name__=='__main__':main()
