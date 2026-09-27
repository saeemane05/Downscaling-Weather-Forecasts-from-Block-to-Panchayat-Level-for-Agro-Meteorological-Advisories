from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
MODEL_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(MODEL_ROOT))
from common import discover_target_source,read_csv_checked,detect_date_column,detect_gp_id_column,TARGET_COLUMNS,resolve_physical_location,GP_ROOT,ENGINEERED_ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('state',nargs='?');ap.add_argument('district',nargs='?');ap.add_argument('block',nargs='?');a=ap.parse_args()
    s=a.state or input('State: ').strip();d=a.district or input('District: ').strip();b=a.block or input('Block: ').strip()
    print('='*78);print('SIH26 GP TARGET PREFLIGHT');print('='*78)
    p=discover_target_source(s,d,b)
    if p is None:
        print('STATUS: BLOCKED')
        print('No genuine Panchayat/GP weather target file was found.')
        print('Expected: datasets/targets/<State>/<District>/<Block>/gp_weather_targets.csv')
        print('Required columns: gp_id, date, and the five target variables:')
        for x in TARGET_COLUMNS: print('  -',x)
        print('This tool will NOT use block actual weather as a fake GP target.')
        raise SystemExit(2)
    df=read_csv_checked(p);gp=detect_gp_id_column(df);dt=detect_date_column(df)
    missing=[x for x in TARGET_COLUMNS if x not in df.columns]
    if gp is None or dt is None or missing:
        raise RuntimeError(f'Invalid target contract. gp_id={gp}, date={dt}, missing_targets={missing}')
    if df[[gp,dt]].isna().any().any(): raise RuntimeError('Target contains missing GP IDs or dates.')
    dup=df.duplicated([gp,dt],keep=False)
    if dup.any(): raise RuntimeError('Target has duplicate GP/date rows.')
    print('Target source:',p);print('Rows:',len(df));print('GPs:',df[gp].nunique());print('Date range:',pd.to_datetime(df[dt]).min().date(),'to',pd.to_datetime(df[dt]).max().date());print('STATUS: PASS')
if __name__=='__main__':main()
