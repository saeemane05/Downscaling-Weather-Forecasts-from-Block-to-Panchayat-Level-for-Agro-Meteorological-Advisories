"""Read-only development API over the repository's existing model outputs."""
from __future__ import annotations
import csv, json, mimetypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'main_model' / 'stage2_direct_gp_training'
GP_ROOT = ROOT / 'gp'
DIST = Path(__file__).resolve().parent / 'dist'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def number(value):
    try:
        x=float(value)
        return x if x == x and abs(x) != float('inf') else None
    except (TypeError, ValueError): return None
def block_path(state,district,block): return MODEL/state/district/block
def discover():
    result=[]
    if not MODEL.exists(): return result
    for f in MODEL.glob('*/*/*/stage2_current_gp_forecast.csv'):
        state,district,block=f.parent.relative_to(MODEL).parts
        try:
            rs=rows(f); dates=sorted({r['target_date'] for r in rs}); n=len({r['gp_id'] for r in rs})
            result.append({'state':state,'district':district,'block':block,'gpCount':n,'forecastDates':dates})
        except (OSError, KeyError): continue
    return sorted(result,key=lambda x:(x['state'],x['district'],x['block']))
def simplify_geometry(fc):
    # Keep only the selected block's geometry and reduce boundary payload size.
    try:
        from shapely.geometry import shape, mapping
        features=[]
        for feature in fc.get('features',[]):
            geom=shape(feature['geometry']).simplify(0.00015,preserve_topology=True)
            features.append({'type':'Feature','properties':feature.get('properties',{}),'geometry':mapping(geom)})
        return {'type':'FeatureCollection','features':features}
    except Exception:
        return fc
def block_data(state,district,block):
    folder=block_path(state,district,block)
    forecast_file=folder/'stage2_current_gp_forecast.csv'
    if not forecast_file.is_file(): return None
    frows=rows(forecast_file)
    # Pick the latest seven-day run represented by the existing output.
    dates=sorted({r['target_date'] for r in frows})
    if not dates:return None
    date_set=set(dates[-7:]); grouped={}
    for r in frows:
        if r['target_date'] not in date_set:continue
        day=int(r['forecast_horizon_day']); key=str(r['gp_id'])
        item=grouped.setdefault(key,{'id':key,'name':key,'forecast':[]})
        item['forecast'].append({'date':r['target_date'],'day':day,
            'temperature':number(r.get('gp_temperature_corrected')),
            'humidity':number(r.get('gp_humidity_corrected')),
            'rainfall':number(r.get('gp_precipitation_corrected')),
            'wind':number(r.get('gp_wind_corrected'))})
    gp_dir=GP_ROOT/state/district/block/'processed'
    master_path=gp_dir/'gp_master.geojson'
    master=json.loads(master_path.read_text(encoding='utf-8-sig')) if master_path.exists() else {'type':'FeatureCollection','features':[]}
    by_id={}
    for feature in master.get('features',[]):
        p=feature.get('properties',{}); key=str(p.get('gp_id',''))
        if not key:continue
        by_id[key]=p
        if key in grouped:
            grouped[key]['name']=p.get('gp_name') or grouped[key]['name']
            grouped[key]['lat']=number(p.get('centroid_lat')); grouped[key]['lon']=number(p.get('centroid_lon'))
            grouped[key]['area']=number(p.get('area_km2'))
    for filename, field, col in [('gp_terrain_features.csv','elevation','elevation_mean'),('gp_terrain_features.csv','slope','slope_mean'),('gp_soil_features.csv','soilClay','soil_clay_rootzone_0_30cm_pct'),('gp_landcover_features.csv','landCover','dominant_landcover')]:
        p=gp_dir/filename
        if not p.exists():continue
        for row in rows(p):
            item=grouped.get(str(row.get('gp_id')))
            if item is not None:item[field]=number(row.get(col)) if field!='landCover' else (row.get(col) or None)
    gp_fc={'type':'FeatureCollection','features':[]}
    for feature in master.get('features',[]):
        if str(feature.get('properties',{}).get('gp_id','')) in grouped:gp_fc['features'].append(feature)
    gp_fc=simplify_geometry(gp_fc)
    block_fc=None
    candidates=list((GP_ROOT/state/district/block/'raw'/'boundaries').glob('gram_manchitra_block*.geojson'))
    if candidates:
        candidate=max(candidates,key=lambda p:p.stat().st_mtime)
        try:block_fc=simplify_geometry(json.loads(candidate.read_text(encoding='utf-8-sig')))
        except (OSError,json.JSONDecodeError):pass
    for item in grouped.values():item['forecast'].sort(key=lambda d:d['day'])
    location=next((x for x in discover() if (x['state'],x['district'],x['block'])==(state,district,block)),None)
    if not location:return None
    meta_path=folder/'stage2_manifest.json'; meta={}
    if meta_path.exists():
        try:meta=json.loads(meta_path.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:pass
    updated=meta.get('forecast_retrieved_at')
    if not updated:
        source=ROOT/'block'/state/district/block/'processed'/'merge'/'block_current_forecast.csv'
        if source.exists():
            try:
                source_rows=rows(source)
                updated=next((r.get('forecast_retrieved_at') for r in source_rows if r.get('forecast_retrieved_at')) ,None)
            except (OSError,KeyError):pass
    return {'location':location,'gps':sorted(grouped.values(),key=lambda x:x['name'].casefold()),'gpGeometry':gp_fc,'blockGeometry':block_fc,'updatedAt':updated}

class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        data=json.dumps(value,separators=(',',':')).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Access-Control-Allow-Origin','*'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/api/locations':return self.send_json(discover())
        if u.path=='/api/gp-location':
            gp_id=parse_qs(u.query).get('id',[''])[0]
            if not gp_id:return self.send_json({'error':'id is required'},400)
            for location in discover():
                f=block_path(location['state'],location['district'],location['block'])/'stage2_current_gp_forecast.csv'
                try:
                    if any(str(row.get('gp_id'))==gp_id for row in rows(f)):return self.send_json(location)
                except (OSError,KeyError):continue
            return self.send_json({'error':'No GP forecast found for this id'},404)
        if u.path=='/api/block':
            q=parse_qs(u.query); keys=[q.get(k,[''])[0] for k in ('state','district','block')]
            if not all(keys):return self.send_json({'error':'state, district and block are required'},400)
            try:data=block_data(*keys)
            except (OSError,ValueError,KeyError) as exc:return self.send_json({'error':str(exc)},500)
            return self.send_json(data,200) if data else self.send_json({'error':'No GP forecast available for this block'},404)
        # In production, serve the Vite build from this same process.
        rel='index.html' if u.path=='/' or u.path.startswith(('/dashboard','/map','/forecast','/advisory','/insights','/about','/gp/','/block/')) else u.path.lstrip('/')
        path=(DIST/rel).resolve()
        if not str(path).startswith(str(DIST.resolve())) or not path.is_file():return self.send_error(404)
        content=path.read_bytes(); self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream'); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
    def log_message(self,fmt,*args):print('%s - %s'%(self.address_string(),fmt%args))

if __name__=='__main__':
    print(f'HyperWeather API at http://127.0.0.1:8000 (repository: {ROOT})')
    ThreadingHTTPServer(('127.0.0.1',8000),Handler).serve_forever()
