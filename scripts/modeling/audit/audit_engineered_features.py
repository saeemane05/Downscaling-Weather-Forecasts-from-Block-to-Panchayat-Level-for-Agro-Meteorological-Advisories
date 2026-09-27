import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENGINEERED_ROOT = PROJECT_ROOT / 'datasets' / 'engineered'
AUDIT_ROOT = PROJECT_ROOT / 'datasets' / 'engineered_audit'

DATASETS = [
    ('block_observations_daily_engineered', 'block', 'block', 'block_observations_daily_engineered.csv'),
    ('block_forecast_training_base_engineered', 'block', 'block', 'block_forecast_training_base_engineered.csv'),
    ('block_current_forecast_engineered', 'block', 'block', 'block_current_forecast_engineered.csv'),
    ('gp_merged_features_engineered', 'gp', 'gp', 'gp_merged_features_engineered.csv'),
]


def norm(x):
    return ''.join(c.lower() for c in str(x) if c.isalnum())


def resolve_location(state, district, block):
    matches = []
    for p in ENGINEERED_ROOT.rglob(block):
        if not p.is_dir():
            continue
        if norm(p.name) != norm(block):
            continue
        d = p.parent
        s = d.parent
        if norm(d.name) == norm(district) and norm(s.name) == norm(state):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f'Engineered location not found under {ENGINEERED_ROOT}')
    return matches[0]


def date_columns(df):
    return [c for c in df.columns if any(k in c.lower() for k in ['date', 'datetime', 'timestamp', 'valid_time', 'issue_time'])]


def duplicate_columns(df):
    return [str(c) for c in df.columns[df.columns.duplicated()].tolist()]


def nonfinite_numeric(df):
    cols = df.select_dtypes(include=[np.number]).columns
    return int(np.isinf(df[cols].to_numpy()).sum()) if len(cols) else 0


def leakage_flags(df, dataset_name):
    flags = []
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    # Same-target observed weather in forecast-training data is a critical flag.
    if 'forecast_training' in dataset_name:
        for c, n in lower.items():
            if n.startswith('actual_') and not any(x in n for x in ['lag', 'rolling', 'roll_', 'shift', 'previous']):
                flags.append({'column': c, 'risk': 'REVIEW', 'reason': 'Raw actual-weather feature appears in forecast training dataset; verify it is not the same target timestamp.'})

    # Future-looking engineered names.
    future_terms = ['future', 'lead_target', 'target_value', 'next_day', 'nextday']
    for c, n in lower.items():
        if any(t in n for t in future_terms):
            flags.append({'column': c, 'risk': 'REVIEW', 'reason': 'Name suggests future/target information; verify temporal construction.'})

    # Rolling/lag features should be explicitly past-looking.
    for c, n in lower.items():
        if any(t in n for t in ['rolling', 'roll_', 'lag_', '_lag', 'rainfall_sum', 'wet_day_count']):
            if not any(t in n for t in ['past', 'lag', 'previous', 'shift']):
                flags.append({'column': c, 'risk': 'REVIEW', 'reason': 'Temporal aggregation feature found; verify implementation uses shift(1) before rolling.'})

    return flags


def inspect_file(path, dataset_name):
    df = pd.read_csv(path)
    result = {
        'dataset': dataset_name,
        'path': str(path),
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'duplicate_column_labels': duplicate_columns(df),
        'duplicate_rows': int(df.duplicated().sum()),
        'nonfinite_numeric_values': nonfinite_numeric(df),
        'all_missing_columns': [],
        'constant_columns': [],
        'date_columns': date_columns(df),
        'min_dates': {},
        'max_dates': {},
        'leakage_flags': leakage_flags(df, dataset_name),
    }

    for c in df.columns:
        s = df[c]
        if s.isna().all():
            result['all_missing_columns'].append(c)
        if s.notna().any() and s.nunique(dropna=True) <= 1:
            result['constant_columns'].append(c)

        if c in result['date_columns']:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().any():
                result['min_dates'][c] = str(parsed.min())
                result['max_dates'][c] = str(parsed.max())

    return df, result


def parse_args():
    parser = argparse.ArgumentParser(description='SIH26 engineered feature audit')
    parser.add_argument('--state')
    parser.add_argument('--district')
    parser.add_argument('--block')
    args = parser.parse_args()
    if any((args.state, args.district, args.block)) and not all((args.state, args.district, args.block)):
        parser.error('--state, --district and --block must be supplied together')
    return args


def main():
    args = parse_args()
    print('=' * 70)
    print('SIH26 ENGINEERED FEATURE AUDIT')
    print('=' * 70)
    state = args.state or input('\nState: ').strip()
    district = args.district or input('District: ').strip()
    block = args.block or input('Block: ').strip()
    if not all([state, district, block]):
        raise ValueError('State, District and Block are required.')

    print(f'\nState: {state}\nDistrict: {district}\nBlock: {block}')

    location = resolve_location(state, district, block)
    print(f'\nResolved engineered location:\n  {location}')

    out_dir = AUDIT_ROOT / location.parent.parent.name / location.parent.name / location.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    frames = {}
    missing_files = []

    for dataset_name, level, folder, filename in DATASETS:
        path = location / folder / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        print(f'\nAuditing: {path}')
        df, result = inspect_file(path, dataset_name)
        frames[dataset_name] = df
        results.append(result)
        print(f"  Rows={result['rows']:,} | Columns={result['columns']:,}")
        print(f"  Duplicate rows={result['duplicate_rows']} | Nonfinite={result['nonfinite_numeric_values']}")
        print(f"  All-missing columns={len(result['all_missing_columns'])}")
        print(f"  Leakage flags={len(result['leakage_flags'])}")

    if missing_files:
        print('\nWARNING: Missing expected files:')
        for p in missing_files:
            print(' ', p)

    # Cross-dataset schema summary.
    schema = []
    for r in results:
        for c in r['date_columns']:
            schema.append({'dataset': r['dataset'], 'date_column': c, 'min': r['min_dates'].get(c), 'max': r['max_dates'].get(c)})

    all_flags = [f for r in results for f in r['leakage_flags']]
    report = {
        'location': {'state': state, 'district': district, 'block': block},
        'engineered_root': str(location),
        'datasets': results,
        'missing_expected_files': missing_files,
        'date_ranges': schema,
        'leakage_flags': all_flags,
        'status': 'PASS' if not missing_files and not any(r['duplicate_column_labels'] or r['nonfinite_numeric_values'] or r['all_missing_columns'] for r in results) and not all_flags else 'REVIEW_REQUIRED',
    }

    json_path = out_dir / 'engineered_feature_audit.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_rows = []
    for r in results:
        summary_rows.append({
            'dataset': r['dataset'],
            'rows': r['rows'],
            'columns': r['columns'],
            'duplicate_rows': r['duplicate_rows'],
            'nonfinite_numeric_values': r['nonfinite_numeric_values'],
            'all_missing_columns': len(r['all_missing_columns']),
            'constant_columns': len(r['constant_columns']),
            'leakage_flags': len(r['leakage_flags']),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / 'engineered_feature_audit_summary.csv', index=False)

    print('\n' + '=' * 70)
    print('ENGINEERED FEATURE AUDIT COMPLETED')
    print('=' * 70)
    print(f"\nStatus: {report['status']}")
    print(f"JSON: {json_path}")
    print(f"CSV : {out_dir / 'engineered_feature_audit_summary.csv'}")

    if all_flags:
        print('\nLEAKAGE/TEMPORAL REVIEW FLAGS:')
        for f in all_flags:
            print(f"  [{f['risk']}] {f['column']} — {f['reason']}")
    else:
        print('\nNo automated leakage-review flags were found.')

    if report['status'] != 'PASS':
        print('\nDo NOT start model training yet. Review the report first.')
    else:
        print('\nAutomated audit passed. Proceed to final training-dataset construction.')


if __name__ == '__main__':
    main()
