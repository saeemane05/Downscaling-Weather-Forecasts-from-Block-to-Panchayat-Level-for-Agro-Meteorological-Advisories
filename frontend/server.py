"""Read-only development API over the repository's existing model outputs."""
from __future__ import annotations
import csv, json, mimetypes, os, re, subprocess, shutil, socket, sys, threading, time, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'main_model' / 'stage2_direct_gp_training'
GP_ROOT = ROOT / 'gp'
DIST = Path(__file__).resolve().parent / 'dist'
OFFICIAL_BLOCK_REGISTRY = ROOT / 'block' / 'metadata' / 'official_registry' / 'gram_manchitra_block_registry.geojson'

# Display names required by the statewide UI.  Every entry is resolved from
# the official Gram Manchitra registry below; this is not a block/GP mapping.
MAHARASHTRA_DISTRICTS = (
    'Ahilyanagar', 'Akola', 'Amravati', 'Beed', 'Bhandara', 'Buldhana',
    'Chhatrapati Sambhajinagar', 'Chandrapur', 'Dhule', 'Dharashiv',
    'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur',
    'Latur', 'Mumbai City', 'Mumbai Suburban', 'Nagpur', 'Nanded',
    'Nandurbar', 'Nashik', 'Palghar', 'Parbhani', 'Pune', 'Raigad',
    'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane',
    'Wardha', 'Washim', 'Yavatmal',
)
OFFICIAL_DISTRICT_ALIASES = {'Ahilyanagar': 'Ahmednagar', 'Mumbai City': 'Mumbai'}
OFFICIAL_TO_DISPLAY_DISTRICT = {value: key for key, value in OFFICIAL_DISTRICT_ALIASES.items()}
_maharashtra_registry = None
_locations_cache = None
_locations_cache_expires = 0.0
_block_data_cache = {}
_data_cache_lock = threading.Lock()
_processing_jobs = {}
_processing_jobs_by_location = {}
_processing_jobs_lock = threading.Lock()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def number(value):
    try:
        x=float(value)
        return x if x == x and abs(x) != float('inf') else None
    except (TypeError, ValueError): return None
def block_path(state,district,block):
    direct=MODEL/state/district/block
    if direct.exists() or state!='Maharashtra':return direct
    return MODEL/state/OFFICIAL_DISTRICT_ALIASES.get(district,district)/block
def location_key(state,district,block):
    display_district=OFFICIAL_TO_DISPLAY_DISTRICT.get(district,district)
    return tuple(re.sub(r'[^a-z0-9]+','',str(value).casefold()) for value in (state,display_district,block))
def find_location(state,district,block):
    key=location_key(state,district,block)
    return next((item for item in locations() if location_key(item['state'],item['district'],item['block'])==key),None)
def valid_forecast_output(location):
    """Validate cached Stage-2 data against location, LGD, GP, and model identity."""
    folder=block_path(location['state'],location['district'],location['block'])
    manifest_path=folder/'stage2_manifest.json'
    forecast_path=folder/'stage2_current_gp_forecast.csv'
    master_path=GP_ROOT/location['state']/folder.parent.name/location['block']/'processed'/'gp_master.geojson'
    try:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if manifest.get('stage')!='Stage 2' or int(manifest.get('model_count',0))<1:
            return False
        if any(manifest.get(flag) is not False for flag in ('synthetic_target','interpolated_target','block_actual_copied_to_gp')):
            return False
        if location_key(manifest.get('state',''),manifest.get('district',''),manifest.get('block','')) != location_key(location['state'],location['district'],location['block']):
            return False
        forecast_rows=rows(forecast_path)
        if not forecast_rows or not {'gp_id','target_date','forecast_horizon_day','gp_temperature_corrected','gp_humidity_corrected','gp_precipitation_corrected','gp_wind_corrected'}.issubset(forecast_rows[0]):
            return False
        forecast_ids={str(row['gp_id']).strip() for row in forecast_rows if row.get('gp_id')}
        if not forecast_ids or len(forecast_ids)!=int(manifest.get('gp_count',-1)):
            return False
        master=json.loads(master_path.read_text(encoding='utf-8-sig'))
        features=master.get('features',[])
        master_ids=set()
        expected_lgd=str(location.get('blockLgdCode') or '').strip()
        for feature in features:
            props=feature.get('properties',{})
            if location_key(props.get('state',''),props.get('district',''),props.get('block','')) != location_key(location['state'],location['district'],location['block']):
                return False
            lgd=str(props.get('block_lgd_code') or props.get('block_lgd') or '').strip()
            if expected_lgd and lgd != expected_lgd:
                return False
            if props.get('gp_id') is not None:
                master_ids.add(str(props['gp_id']).strip())
        return bool(master_ids) and forecast_ids.issubset(master_ids)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
        return False
def forecast_is_current(location):
    """A displayed forecast must include today or a later India target date."""
    folder=block_path(location['state'],location['district'],location['block'])
    try:
        target_dates={datetime.strptime(row['target_date'],'%Y-%m-%d').date() for row in rows(folder/'stage2_current_gp_forecast.csv') if row.get('target_date')}
        return bool(target_dates) and max(target_dates)>=datetime.now(ZoneInfo('Asia/Kolkata')).date()
    except (OSError,KeyError,TypeError,ValueError):
        return False
def configure_cdsapi_credentials():
    """Materialize Render's secret environment variables for cdsapi if needed.

    A developer's local %USERPROFILE%\\.cdsapirc is never copied into the
    repository. Hosted deployments can provide CDSAPI_URL and CDSAPI_KEY as
    private service environment variables instead.
    """
    config_path=Path.home()/'.cdsapirc'
    if config_path.is_file():
        return
    url=os.environ.get('CDSAPI_URL','').strip()
    key=os.environ.get('CDSAPI_KEY','').strip()
    if not url or not key or any(char in url+key for char in '\r\n'):
        return
    payload=f'url: {url}\nkey: {key}\n'
    try:
        fd=os.open(config_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        # Another request may have initialized the shared service config.
        return
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as config:
        config.write(payload)
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
def advisory_is_healthy(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=0.6) as response:
            payload=json.loads(response.read().decode('utf-8'))
            return response.status==200 and payload.get('success') is True and payload.get('service')=='AgriWeather Advisory API'
    except Exception:return False
def port_is_available(port):
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(('127.0.0.1',port))==0:return False
            probe.bind(('127.0.0.1',port))
        return True
    except OSError:return False
def unused_local_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1',0))
        return probe.getsockname()[1]
def discover():
    result=[]
    if not MODEL.exists(): return result
    for f in MODEL.glob('*/*/*/stage2_current_gp_forecast.csv'):
        state,district,block=f.parent.relative_to(MODEL).parts
        try:
            rs=rows(f); dates=sorted({r['target_date'] for r in rs}); n=len({r['gp_id'] for r in rs})
            location={'state':state,'district':district,'block':block}
            if valid_forecast_output(location):
                result.append({'state':state,'district':district,'block':block,'gpCount':n,'forecastDates':dates,'available':True})
        except (OSError, KeyError): continue
    return sorted(result,key=lambda x:(x['state'],x['district'],x['block']))
def maharashtra_registry():
    """Return compact district/block records derived from the official registry."""
    global _maharashtra_registry
    if _maharashtra_registry is not None:return _maharashtra_registry
    if not OFFICIAL_BLOCK_REGISTRY.is_file():
        _maharashtra_registry=[]
        return _maharashtra_registry
    try:
        features=json.loads(OFFICIAL_BLOCK_REGISTRY.read_text(encoding='utf-8')).get('features',[])
    except (OSError,json.JSONDecodeError):
        _maharashtra_registry=[]
        return _maharashtra_registry
    records=[]
    for feature in features:
        props=feature.get('properties',{})
        if str(props.get('state','')).strip().casefold()!='maharashtra':continue
        official_district=str(props.get('district','')).strip()
        district=OFFICIAL_TO_DISPLAY_DISTRICT.get(official_district,official_district)
        block=str(props.get('block_name','')).strip()
        if district in MAHARASHTRA_DISTRICTS and block:
            records.append({'state':'Maharashtra','district':district,'block':block,'blockLgdCode':str(props.get('block_lgd') or props.get('blkcode11') or '')})
    _maharashtra_registry=sorted(records,key=lambda x:(x['district'],x['block'].casefold()))
    return _maharashtra_registry
def _build_locations():
    """Merge official statewide choices with existing calculated model outputs."""
    calculated=discover()
    def location_key(item):
        return (item['state'],item['district'].casefold(),item['block'].casefold())
    by_key={location_key(x):x for x in calculated}
    result={}
    for item in calculated:
        district=OFFICIAL_TO_DISPLAY_DISTRICT.get(item['district'],item['district']) if item['state']=='Maharashtra' else item['district']
        normalized={**item,'district':district}
        result[location_key(normalized)]=normalized
    for official in maharashtra_registry():
        key=location_key(official)
        # Older outputs retain the official historic district spelling.
        legacy_district=OFFICIAL_DISTRICT_ALIASES.get(official['district'],official['district'])
        legacy_key=(official['state'],legacy_district.casefold(),official['block'].casefold())
        computed=result.get(key) or by_key.get(legacy_key)
        result[key]={**official,**(computed or {}),'blockLgdCode':official['blockLgdCode'],'gpCount':computed['gpCount'] if computed else 0,'forecastDates':computed['forecastDates'] if computed else [],'available':computed is not None}
    return sorted(result.values(),key=lambda x:(x['state'],x['district'],x['block'].casefold()))
def locations():
    """Cache the filesystem-backed registry scan for quick dashboard navigation."""
    global _locations_cache, _locations_cache_expires
    now=time.monotonic()
    with _data_cache_lock:
        if _locations_cache is not None and now < _locations_cache_expires:
            return _locations_cache
    result=_build_locations()
    with _data_cache_lock:
        _locations_cache=result
        _locations_cache_expires=time.monotonic()+60
    return result
def processing_status_for_line(line,active_stage=None,active_chunk=None):
    lower=line.casefold()
    batch=re.search(r'\bgp batch\s+(\d+)\s*/\s*(\d+)',line,re.IGNORECASE)
    if batch and active_stage:
        chunk=f' ({active_chunk})' if active_chunk else ''
        return 'processing_geospatial_data',f'Processing {active_stage}{chunk}: GP batch {batch.group(1)}/{batch.group(2)}...'
    chunk_match=re.search(r'\bdate chunk:\s*(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})',line,re.IGNORECASE)
    if chunk_match and active_stage:
        chunk=f'{chunk_match.group(1)} to {chunk_match.group(2)}'
        return 'processing_geospatial_data',f'Processing {active_stage}: {chunk}...'
    if 'soil qc failed' in lower:
        return 'processing_geospatial_data','Validating SoilGrids coverage; missing source values are not imputed...'
    if 'extraction returned' in lower and 'soil' in (active_stage or '').casefold():
        return 'processing_geospatial_data','Validating returned SoilGrids values for every GP...'
    if any(token in lower for token in ('04_sentinel1.py','sentinel-1','sentinel1')):
        return 'processing_geospatial_data','Collecting real Sentinel-1 observations...'
    if any(token in lower for token in ('04_sentinel2.py','sentinel-2','sentinel2')):
        return 'processing_geospatial_data','Collecting real Sentinel-2 observations...'
    if '05_soil.py' in lower or 'stage: soil' in lower:
        return 'processing_geospatial_data','Extracting real SoilGrids values for each GP...'
    if '06_lst.py' in lower or 'stage: lst' in lower:
        return 'processing_geospatial_data','Extracting real Landsat surface-temperature values...'
    if '03_gp_landcover.py' in lower or 'landcover' in lower:
        return 'processing_geospatial_data','Extracting real GP land-cover features...'
    if '02_gp_terrain.py' in lower or 'terrain' in lower:
        return 'processing_geospatial_data','Extracting real GP terrain features...'
    if 'stage-2 gp downscaling' in lower or 'stage-2' in lower:
        return 'running_model','Running the existing GP forecast model...'
    if 'current_gp_forecast' in lower or 'generating forecast' in lower:
        return 'generating_forecast','Generating the GP-level forecast...'
    if 'gp spatial data' in lower or 'stage 01' in lower or 'boundary stage' in lower:
        return 'fetching_gps','Fetching official Gram Panchayat boundaries...'
    if any(token in lower for token in ('cleaning','feature dictionary','feature engineering','existing audits')):
        return 'preparing_features','Preparing the existing model features...'
    if 'block data' in lower:
        return 'preparing_weather_data','Preparing block weather data...'
    if 'era5' in lower or 'weather' in lower:
        return 'preparing_weather_data','Preparing weather data...'
    if 'terrain' in lower or 'landcover' in lower or 'sentinel' in lower or 'soil' in lower or 'lst' in lower:
        return 'processing_geospatial_data','Processing real GP geospatial data...'
    return None
def update_processing_job(job_id,**updates):
    with _processing_jobs_lock:
        job=_processing_jobs.get(job_id)
        if job is not None:job.update(updates)
def run_block_processing(job_id,location):
    update_processing_job(job_id,status='resolving_block',message='Resolving the selected official block...')
    auth_error=earth_engine_preflight()
    if auth_error:
        update_processing_job(job_id,status='failed',error=auth_error)
        with _processing_jobs_lock:
            if _processing_jobs_by_location.get(location_key(location['state'],location['district'],location['block']))==job_id:
                _processing_jobs_by_location.pop(location_key(location['state'],location['district'],location['block']),None)
        return
    # A stale forecast must refresh the upstream current-weather inputs as
    # well as Stage 2; otherwise the orchestrator would simply reuse its old
    # completed artifacts.
    command=[sys.executable,'-u',str(ROOT/'run_pipeline.py'),'--state',location['state'],'--district',location['district'],'--block',location['block'],'--to-stage','stage2','--force']
    try:
        configure_cdsapi_credentials()
        process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        failure_hint=None
        active_stage=None
        active_chunk=None
        with process:
            for line in process.stdout:
                encoding=sys.stdout.encoding or 'utf-8'
                safe_line=line.encode(encoding,errors='replace').decode(encoding,errors='replace')
                print(f'[block-processing {job_id}] {safe_line}',end='',flush=True)
                lower=line.casefold()
                if 'sentinel1' in lower or 'sentinel-1' in lower:
                    active_stage='Sentinel-1'
                elif 'sentinel2' in lower or 'sentinel-2' in lower:
                    active_stage='Sentinel-2'
                elif 'soil' in lower:
                    active_stage='SoilGrids'
                elif 'landcover' in lower:
                    active_stage='land cover'
                elif 'terrain' in lower:
                    active_stage='terrain'
                elif 'lst' in lower:
                    active_stage='Landsat surface temperature'
                chunk_match=re.search(r'\bdate chunk:\s*(\d{4}-\d{2}-\d{2})\s*->\s*(\d{4}-\d{2}-\d{2})',line,re.IGNORECASE)
                if chunk_match:
                    active_chunk=f'{chunk_match.group(1)} to {chunk_match.group(2)}'
                progress=processing_status_for_line(line,active_stage,active_chunk)
                if progress:update_processing_job(job_id,status=progress[0],message=progress[1])
                if 'could not initialize earth engine' in lower:
                    failure_hint='Earth Engine authorization is required before this block can be processed. Run `earthengine authenticate` in the backend Python environment, then retry.'
                elif 'no gp polygons returned for block lgd code' in lower:
                    failure_hint='The official Gram Manchitra service returned no GP boundaries for this block LGD code.'
                elif 'cdsapi is not installed' in lower:
                    failure_hint='The ERA5-Land stage requires the project dependency `cdsapi`. Install the project requirements, then retry.'
                elif 'missing/incomplete configuration file' in lower and '.cdsapirc' in lower:
                    failure_hint='The ERA5-Land stage requires Copernicus CDS API credentials. Locally, configure `%USERPROFILE%\\.cdsapirc`; on Render, set the private `CDSAPI_URL` and `CDSAPI_KEY` service variables, then redeploy.'
                elif 'soil qc failed' in lower:
                    failure_hint='SoilGrids QC failed because source values are missing for one or more GP soil properties. No soil values were imputed; check the backend log for the missing-source count.'
                elif 'missing_source_cells' in lower:
                    missing_match=re.search(r'"missing_source_cells"\s*:\s*(\d+)',line,re.IGNORECASE)
                    if missing_match:
                        failure_hint=f'SoilGrids QC failed with {missing_match.group(1)} missing source cells. No soil values were imputed; check the backend log for the affected GP.'
            return_code=process.wait()
        if return_code != 0:
            update_processing_job(job_id,status='failed',message='Block processing failed.',error=failure_hint or f'The existing data pipeline failed (exit code {return_code}). Check backend logs for the failing stage.')
            return
        if not valid_forecast_output(location):
            update_processing_job(job_id,status='failed',error='The pipeline finished without producing a valid forecast for the selected block.')
            return
        update_processing_job(job_id,status='completed',message='GP forecast is ready.')
    except OSError as exc:
        print(f'[block-processing {job_id}] Could not start pipeline: {exc}',flush=True)
        update_processing_job(job_id,status='failed',error='The existing data pipeline could not be started. Check the backend installation.')
    except Exception as exc:
        print(f'[block-processing {job_id}] Pipeline worker failed: {exc}',flush=True)
        update_processing_job(job_id,status='failed',error='Block processing failed unexpectedly. Check backend logs for details.')
    finally:
        with _processing_jobs_lock:
            key=location_key(location['state'],location['district'],location['block'])
            if _processing_jobs_by_location.get(key)==job_id:
                _processing_jobs_by_location.pop(key,None)
def start_block_processing(location):
    key=location_key(location['state'],location['district'],location['block'])
    with _processing_jobs_lock:
        existing_id=_processing_jobs_by_location.get(key)
        if existing_id and _processing_jobs.get(existing_id,{}).get('status') not in ('completed','failed'):
            return existing_id
        job_id=uuid.uuid4().hex
        _processing_jobs[job_id]={'jobId':job_id,'state':location['state'],'district':location['district'],'block':location['block'],'status':'queued','message':'Block processing is queued...','error':None}
        _processing_jobs_by_location[key]=job_id
    thread=threading.Thread(target=run_block_processing,args=(job_id,location),daemon=True)
    thread.start()
    return job_id
def earth_engine_preflight():
    try:
        import ee
    except ImportError:
        return 'The Earth Engine API is missing from the backend Python environment. Install the project requirements and restart the backend.'
    try:
        ee.Initialize()
    except Exception:
        return 'Earth Engine authorization is required before this block can be processed. Run `earthengine authenticate` in the backend Python environment, then retry.'
    return None
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
def _build_block_data(state,district,block):
    folder=block_path(state,district,block)
    storage_district=folder.parent.name
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
    gp_dir=GP_ROOT/state/storage_district/block/'processed'
    landcover_by_id={}
    landcover_file=gp_dir/'gp_landcover_features.csv'
    if landcover_file.is_file():
        fraction_columns=('water_fraction','trees_fraction','grass_fraction','flooded_vegetation_fraction','crops_fraction','shrub_fraction','built_fraction','bare_fraction','snow_ice_fraction')
        try:
            for row in rows(landcover_file):
                fractions={column.removesuffix('_fraction'):value for column in fraction_columns if (value:=number(row.get(column))) is not None}
                if row.get('gp_id') and fractions:
                    landcover_by_id[str(row['gp_id'])]=fractions
        except OSError as exc:
            print(f'[land-cover] Could not read {landcover_file}: {exc}',flush=True)
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
            grouped[key]['landUseFractions']=landcover_by_id.get(key)
    # The pipeline's merged master GeoJSON already carries the spatial summaries.
    # Keep these when available; CSV exports are optional and are not emitted by
    # every pipeline run.
    for key, props in by_id.items():
        item=grouped.get(key)
        if item is None: continue
        item['elevation']=number(props.get('elevation_mean'))
        item['slope']=number(props.get('slope_mean'))
        item['soilClay']=number(props.get('soil_clay_rootzone_0_30cm_pct'))
        item['landCover']=props.get('dominant_landcover') or None
    # Feature extraction is stored as GeoJSON in this repository. Read those
    # published outputs when optional CSV exports are absent.
    merged_path=gp_dir/'gp_merged_features.geojson'
    if merged_path.exists():
        try: features=json.loads(merged_path.read_text(encoding='utf-8-sig')).get('features',[])
        except (OSError,json.JSONDecodeError): features=[]
        for feature in features:
            props=feature.get('properties',{}); item=grouped.get(str(props.get('gp_id','')))
            if item is None: continue
            for field, col in [('elevation','terrain__elevation_mean'),('slope','terrain__slope_mean'),('soilClay','soil__soil_clay_rootzone_0_30cm_pct')]:
                value=number(props.get(col))
                if value is not None: item[field]=value
            item['landCover']=props.get('landcover__dominant_landcover') or item['landCover']
    for filename, field, col in [('gp_terrain_features.csv','elevation','elevation_mean'),('gp_terrain_features.csv','slope','slope_mean'),('gp_soil_features.csv','soilClay','soil_clay_rootzone_0_30cm_pct'),('gp_landcover_features.csv','landCover','dominant_landcover')]:
        p=gp_dir/filename
        if not p.exists(): continue
        for row in rows(p):
            item=grouped.get(str(row.get('gp_id')))
            if item is not None:
                value=number(row.get(col)) if field!='landCover' else (row.get(col) or None)
                if value is not None: item[field]=value
    gp_fc={'type':'FeatureCollection','features':[]}
    for feature in master.get('features',[]):
        if str(feature.get('properties',{}).get('gp_id','')) in grouped:gp_fc['features'].append(feature)
    gp_fc=simplify_geometry(gp_fc)
    block_fc=None
    candidates=list((GP_ROOT/state/storage_district/block/'raw'/'boundaries').glob('gram_manchitra_block*.geojson'))
    if candidates:
        candidate=max(candidates,key=lambda p:p.stat().st_mtime)
        try:block_fc=simplify_geometry(json.loads(candidate.read_text(encoding='utf-8-sig')))
        except (OSError,json.JSONDecodeError):pass
    for item in grouped.values():item['forecast'].sort(key=lambda d:d['day'])
    location=find_location(state,district,block)
    if not location:return None
    meta_path=folder/'stage2_manifest.json'; meta={}
    if meta_path.exists():
        try:meta=json.loads(meta_path.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:pass
    updated=meta.get('forecast_retrieved_at')
    if not updated:
        source=ROOT/'block'/state/storage_district/block/'processed'/'merge'/'block_current_forecast.csv'
        if source.exists():
            try:
                source_rows=rows(source)
                updated=next((r.get('forecast_retrieved_at') for r in source_rows if r.get('forecast_retrieved_at')) ,None)
            except (OSError,KeyError):pass
    return {'location':location,'gps':sorted(grouped.values(),key=lambda x:x['name'].casefold()),'gpGeometry':gp_fc,'blockGeometry':block_fc,'updatedAt':updated}
def block_data(state,district,block):
    """Reuse parsed forecast and geometry payloads across dashboard/API requests."""
    key=location_key(state,district,block)
    now=time.monotonic()
    with _data_cache_lock:
        cached=_block_data_cache.get(key)
        if cached and now < cached[0]:
            return cached[1]
    result=_build_block_data(state,district,block)
    if result is not None:
        with _data_cache_lock:
            _block_data_cache[key]=(time.monotonic()+300,result)
    return result

class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        data=json.dumps(value,separators=(',',':')).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Access-Control-Allow-Origin','*'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/healthz':return self.send_json({'status':'ok'})
        if u.path.startswith('/agri-api/'):
            return self.proxy_advisory('GET',u.path[len('/agri-api'):]+(('?'+u.query) if u.query else ''),None)
        if u.path=='/api/locations':return self.send_json(locations())
        if u.path=='/api/districts':
            state=parse_qs(u.query).get('state',['Maharashtra'])[0]
            return self.send_json(sorted({x['district'] for x in locations() if x['state']==state}))
        if u.path=='/api/blocks':
            q=parse_qs(u.query); state=q.get('state',['Maharashtra'])[0]; district=q.get('district',[''])[0]
            return self.send_json([x for x in locations() if x['state']==state and x['district']==district])
        if u.path=='/api/gp-location':
            gp_id=parse_qs(u.query).get('id',[''])[0]
            if not gp_id:return self.send_json({'error':'id is required'},400)
            for location in discover():
                f=block_path(location['state'],location['district'],location['block'])/'stage2_current_gp_forecast.csv'
                try:
                    if any(str(row.get('gp_id'))==gp_id for row in rows(f)):return self.send_json(location)
                except (OSError,KeyError):continue
            return self.send_json({'error':'No GP forecast found for this id'},404)
        if u.path=='/api/block-processing-status':
            job_id=parse_qs(u.query).get('jobId',[''])[0]
            if not job_id:return self.send_json({'error':'jobId is required'},400)
            with _processing_jobs_lock:job=_processing_jobs.get(job_id)
            if job is None:return self.send_json({'error':'Processing job was not found'},404)
            return self.send_json(job)
        if u.path=='/api/block':
            q=parse_qs(u.query); keys=[q.get(k,[''])[0] for k in ('state','district','block')]
            if not all(keys):return self.send_json({'error':'state, district and block are required'},400)
            location=find_location(*keys)
            if location is None:
                return self.send_json({'error':'The selected state, district and block are not present in the official location registry.'},404)
            try:
                if valid_forecast_output(location):
                    data=block_data(location['state'],location['district'],location['block'])
                    if data is not None:
                        data['forecastStatus']='current' if forecast_is_current(location) else 'latest_available'
                else:
                    data=None
            except (OSError,ValueError,KeyError) as exc:return self.send_json({'error':str(exc)},500)
            if data is not None:return self.send_json(data,200)
            command=f"python run_pipeline.py --state {location['state']} --district {location['district']} --block {location['block']} --to-stage stage2 --force"
            return self.send_json({
                'status':'coming_soon',
                'error':f"Data coming soon for {location['block']} Block, {location['district']}. This block has not been processed yet. Run this command from the project folder: {command}"
            },404)
        # In production, serve the Vite build from this same process.
        rel='index.html' if u.path=='/' or u.path.startswith(('/dashboard','/twin','/comparison','/map','/forecast','/advisory','/insights','/about','/gp/','/block/')) else u.path.lstrip('/')
        path=(DIST/rel).resolve()
        if not str(path).startswith(str(DIST.resolve())) or not path.is_file():return self.send_error(404)
        content=path.read_bytes(); self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream'); self.send_header('Cache-Control','no-cache' if path.suffix=='.html' else 'public, max-age=31536000, immutable'); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
    def do_POST(self):
        u=urlparse(self.path)
        if u.path.startswith('/agri-api/'):
            body=self.rfile.read(int(self.headers.get('Content-Length','0')))
            return self.proxy_advisory('POST',u.path[len('/agri-api'):]+(('?'+u.query) if u.query else ''),body)
        return self.send_json({'error':'Not found'},404)
    def proxy_advisory(self,method,path,body):
        request=Request('http://127.0.0.1:'+os.environ.get('AGRI_ADVISORY_PORT','5000')+path,data=body,method=method,headers={'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=45) as response:
                payload=response.read(); status=response.status
        except Exception as exc:
            payload=json.dumps({'success':False,'error':'Advisory service unavailable','message':str(exc)}).encode(); status=503
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def log_message(self,fmt,*args):print('%s - %s'%(self.address_string(),fmt%args))

class QuietDisconnectHTTPServer(ThreadingHTTPServer):
    def handle_error(self,request,client_address):
        error=sys.exc_info()[1]
        if isinstance(error,(BrokenPipeError,ConnectionAbortedError,ConnectionResetError)):
            return
        super().handle_error(request,client_address)

if __name__=='__main__':
    # This server owns both the API and the production SPA.  Do not accept
    # traffic when the SPA has not been built: otherwise every request to /
    # becomes BaseHTTPRequestHandler's opaque default 404 response.
    index_file=DIST/'index.html'
    if not index_file.is_file():
        raise RuntimeError(
            f'Frontend build is missing: {index_file}. '
            'Run `npm ci --include=dev && npm run build` in frontend before starting the service.'
        )
    host=os.environ.get('HOST','0.0.0.0')
    port=int(os.environ.get('PORT','8001'))
    node=shutil.which('node')
    advisory_dir=ROOT/'agri_advisory'
    advisory_process=None
    if node and (advisory_dir/'node_modules').is_dir():
        advisory_port=int(os.environ.get('AGRI_ADVISORY_PORT','5000'))
        if advisory_is_healthy(advisory_port):
            print(f'Agri advisory API already running on port {advisory_port}; reusing it.')
        else:
            if not port_is_available(advisory_port):
                occupied_port=advisory_port
                advisory_port=unused_local_port()
                print(f'Agri advisory port {occupied_port} is occupied by another service; using available port {advisory_port}.')
            os.environ['AGRI_ADVISORY_PORT']=str(advisory_port)
            env={**os.environ,'PORT':str(advisory_port)}
            advisory_process=subprocess.Popen([node,'server.js'],cwd=advisory_dir,env=env)
            for _ in range(50):
                if advisory_is_healthy(advisory_port):break
                if advisory_process.poll() is not None:break
                time.sleep(0.1)
            if not advisory_is_healthy(advisory_port):
                print(f'Agri advisory API did not become ready on port {advisory_port}.')
            else:
                print(f'Agri advisory API started on port {advisory_port}')
    else:
        print('Agri advisory API is not installed; run npm install in agri_advisory to enable it.')
    print(f'HyperWeather API at http://{host}:{port} (repository: {ROOT})')
    try: QuietDisconnectHTTPServer((host,port),Handler).serve_forever()
    finally:
        if advisory_process: advisory_process.terminate()
