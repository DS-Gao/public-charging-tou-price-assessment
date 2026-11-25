from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'raw_station';OUT.mkdir(exist_ok=True)
B=ROOT/'outputs/56_r422_structure_tables_20261108/R4_22_Structure_Table_Revision'
RAW=ROOT/'31952289/orders_2025-01_public.parquet'
CFG=ROOT/'github_release_20261018_DS_Gao/charging-price-measurement/config/price_construction.json'
FROZEN=B/'analysis_evidence/inputs/module_b_row_level_inputs.csv.gz'
ST='ST_003774'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads(CFG.read_text(encoding='utf-8'))
assert sha(RAW)==config['inputs']['orders_2025-01']['sha256']
d=pd.read_parquet(RAW,filters=[('station_id','==',ST)])
d.insert(0,'station_extract_row',np.arange(len(d)))
assert d.station_id.eq(ST).all()
for price,fee in [('E','total_elec_fee'),('S','total_service_fee'),('T','total_fee')]:d[price]=d[fee]/d.total_elec_kwh
checks={
 'valid_start_station':d.charge_start_time.notna()&d.station_id.notna(),
 'start_in_month':d.charge_start_time.ge('2025-01-01')&d.charge_start_time.lt('2025-02-01'),
 'duration_positive':d.duration_min.gt(0),'energy_positive':d.total_elec_kwh.gt(0),
 'fees_nonnegative':d[['total_elec_fee','total_service_fee','total_fee']].ge(0).all(axis=1),
 'fee_additive':(d.total_fee-d.total_elec_fee-d.total_service_fee).abs().le(.0100001),
 'finite_prices':np.isfinite(d[['E','S','T']]).all(axis=1)}
for k,v in checks.items():d[k]=v
d['price_base']=pd.DataFrame(checks).all(axis=1)
d['time_order_valid']=d.charge_end_time.ge(d.charge_start_time)
d['duration_discrepancy_min']=d.duration_min-(d.charge_end_time-d.charge_start_time).dt.total_seconds()/60
d['segment_id']='';d['segment_start']=pd.NaT;d['segment_end']=pd.NaT
minute=d.charge_start_time.dt.hour*60+d.charge_start_time.dt.minute
for name,start,end,hours in config['native_8_segments']['2025-01']:
 mask=minute.ge(start*60)&minute.lt(end*60) if start<end else minute.ge(start*60)|minute.lt(end*60)
 stamp=d.charge_start_time.dt.normalize()+pd.Timedelta(hours=start)
 if start>end:stamp=stamp-pd.to_timedelta(minute.lt(end*60).astype(int),unit='D')
 d.loc[mask,'segment_id']=name;d.loc[mask,'segment_start']=stamp[mask];d.loc[mask,'segment_end']=stamp[mask]+pd.Timedelta(hours=hours)
d['fully_contained']=d.segment_id.ne('')&d.charge_start_time.ge(d.segment_start)&d.charge_end_time.le(d.segment_end)
d['tariff_date']=(d.charge_start_time-pd.Timedelta(hours=23)).dt.normalize()
d['window']=np.select([d.tariff_date.ge('2025-01-01')&d.tariff_date.lt('2025-01-16'),d.tariff_date.ge('2025-01-16')&d.tariff_date.lt('2025-02-01')],['formation','validation'],default='outside_month_split')
d['eligible']=d.price_base&d.fully_contained&d.window.ne('outside_month_split')
d['calendar_date']=d.charge_start_time.dt.normalize()
d['duplicate_record']=d.duplicated(subset=['station_id','pile_id','charge_start_time','charge_end_time','total_elec_kwh','total_elec_fee','total_service_fee','total_fee'],keep=False)
d.to_csv(OUT/'station_transactions_with_audit_flags.csv.gz',index=False)
g=d[d.eligible].copy()
def summarize(z):
 out={'n_transactions':len(z),'n_tariff_dates':z.tariff_date.nunique(),'n_calendar_dates':z.calendar_date.nunique(),'kwh':z.total_elec_kwh.sum(),'n_piles':z.pile_id.nunique(),'first_start':str(z.charge_start_time.min()),'last_start':str(z.charge_start_time.max()),'min_kwh':z.total_elec_kwh.min(),'median_kwh':z.total_elec_kwh.median(),'max_kwh':z.total_elec_kwh.max()}
 for c in ['E','S','T']:
  for label,fun in [('median','median'),('min','min'),('max','max')]:out[f'{c}_{label}']=getattr(z[c],fun)()
  out[c+'_q10']=z[c].quantile(.1);out[c+'_q90']=z[c].quantile(.9)
 out['tariff_dates']=';'.join(z.tariff_date.dt.strftime('%Y-%m-%d').sort_values().unique())
 out['calendar_dates']=';'.join(z.calendar_date.dt.strftime('%Y-%m-%d').sort_values().unique())
 return out
def grouped(keys,name):
 rows=[]
 for key,z in g.groupby(keys):
  key=key if isinstance(key,tuple) else (key,)
  rows.append(dict(zip(keys,key))|summarize(z))
 result=pd.DataFrame(rows);result.to_csv(OUT/name,index=False);return result
endpoints=grouped(['window','segment_id'],'endpoint_summary.csv')
daily=grouped(['window','segment_id','tariff_date'],'daily_endpoint_summary.csv')
pile=grouped(['window','segment_id','pile_id'],'pile_endpoint_summary.csv')
rates=g.assign(E_rounded=g.E.round(2),S_rounded=g.S.round(2),T_rounded=g['T'].round(2)).groupby(['window','segment_id','E_rounded','S_rounded','T_rounded']).agg(n=('T','size'),kwh=('total_elec_kwh','sum'),first_start=('charge_start_time','min'),last_start=('charge_start_time','max'),piles=('pile_id','nunique')).reset_index()
rates.to_csv(OUT/'rounded_rate_groups.csv',index=False)
ref=pd.read_csv(FROZEN,float_precision='round_trip');ref=ref[ref.station_id.eq(ST)&ref.month.eq('2025-01')]
ref.to_csv(OUT/'frozen_station_january_rows.csv',index=False)
comp=[];bounds=[]
for _,r in ref.iterrows():
 bh=int(r.boundary_hour);q=float(r.expected_direction)
 for window in ['formation','validation']:
  es={}
  for side in ['before','after']:
   seg=r[side+'_segment'];z=g[g.window.eq(window)&g.segment_id.eq(seg)];es[side]=summarize(z)
   fields={}
   if window=='formation':
    fields={f'formation_{c}_{side}':es[side][c+'_median'] for c in ['E','S','T']}
    fields|={f'formation_n_sessions_{side}':len(z),f'formation_n_tariff_days_{side}':z.tariff_date.nunique(),f'kwh_{side}':z.total_elec_kwh.sum()}
   else:fields={f'validation_{side}_elec_price':es[side]['E_median'],f'validation_{side}_total_price_observed':es[side]['T_median']}
   fields[f'{window}_{side}_endpoint_available']=bool(len(z)>=10 and z.tariff_date.nunique()>=3)
   for name,value in fields.items():
    saved=r[name];diff=float(value)-float(saved);comp.append(dict(boundary_hour=bh,window=window,side=side,field=name,raw_recomputed=value,frozen_value=saved,difference=diff,match=abs(diff)<1e-10))
  row=dict(boundary_hour=bh,window=window,expected_direction=q)
  for c in ['E','S','T']:row[c+'_star']=q*(es['after'][c+'_median']-es['before'][c+'_median'])
  row['residual_star']=row['T_star']-row['E_star']-row['S_star'];bounds.append(row)
  name='LOCAL' if window=='formation' else 'Y';value=row['T_star'];diff=value-r[name]
  comp.append(dict(boundary_hour=bh,window=window,side='contrast',field=name,raw_recomputed=value,frozen_value=r[name],difference=diff,match=abs(diff)<1e-10))
pd.DataFrame(comp).to_csv(OUT/'raw_vs_frozen_comparison.csv',index=False)
pd.DataFrame(bounds).to_csv(OUT/'reconstructed_boundary_contrasts.csv',index=False)
assert all(x['match'] for x in comp)
counts=[dict(stage='raw_station',n=len(d))];keep=pd.Series(True,index=d.index)
for k in checks:keep&=d[k];counts.append(dict(stage=k,n=int(keep.sum()),independent_failures=int((~d[k]).sum())))
keep&=d.fully_contained;counts.append(dict(stage='fully_contained',n=int(keep.sum())))
keep&=d.window.ne('outside_month_split');counts.append(dict(stage='tariff_date_window',n=int(keep.sum())))
pd.DataFrame(counts).to_csv(OUT/'eligibility_waterfall.csv',index=False)
# Independent implementation of the aggregation using SQL over this station
# only. Never scans or reprocesses other stations.
sys.path.insert(0,str(OUT.parent/'runtime'))
import duckdb
con=duckdb.connect();con.register('raw',d.iloc[:,:13]);defs=pd.DataFrame(config['native_8_segments']['2025-01'],columns=['segment_id','sh','eh','hours']);con.register('defs',defs)
sql="""WITH p AS(SELECT *,total_elec_fee/total_elec_kwh E,total_service_fee/total_elec_kwh S,total_fee/total_elec_kwh T, hour(charge_start_time)*60+minute(charge_start_time) m FROM raw WHERE station_id IS NOT NULL AND charge_start_time >= '2025-01-01' AND charge_start_time < '2025-02-01' AND duration_min>0 AND total_elec_kwh>0 AND total_elec_fee>=0 AND total_service_fee>=0 AND total_fee>=0 AND abs(total_fee-total_elec_fee-total_service_fee)<=.0100001), j AS(SELECT p.*, defs.*,date_trunc('day',charge_start_time)+sh*INTERVAL '1 hour'-CASE WHEN sh>eh AND m<eh*60 THEN INTERVAL '1 day' ELSE INTERVAL '0 day' END ss,cast(charge_start_time-INTERVAL '23 hour' AS DATE) td FROM p JOIN defs ON ((sh<eh AND m>=sh*60 AND m<eh*60) OR(sh>eh AND (m>=sh*60 OR m<eh*60))) WHERE isfinite(E) AND isfinite(S) AND isfinite(T)), z AS(SELECT *,CASE WHEN td<'2025-01-16' THEN 'formation' ELSE 'validation' END win FROM j WHERE charge_start_time>=ss AND charge_end_time<=ss+hours*INTERVAL '1 hour' AND td>='2025-01-01' AND td<'2025-02-01') SELECT win AS window,segment_id,count(*) n_transactions,count(distinct td) n_tariff_dates,sum(total_elec_kwh) kwh,median(E) E_median,median(S) S_median,median(T) T_median FROM z GROUP BY win,segment_id"""
sql_result=con.execute(sql).df();sql_result.to_csv(OUT/'independent_sql_endpoints.csv',index=False)
joined=endpoints.merge(sql_result,on=['window','segment_id'],suffixes=('_pandas','_sql'))
maxdiff={c:float((joined[c+'_pandas']-joined[c+'_sql']).abs().max()) for c in ['n_transactions','n_tariff_dates','kwh','E_median','S_median','T_median']}
assert max(maxdiff.values())<1e-9
inputs=[RAW,CFG,FROZEN,ROOT/'analysis/price_architecture_v2/02_boundary_signal_core.py',ROOT/'github_release_20261018_DS_Gao/charging-price-measurement/src/prices/reconstruction_engine.py']
record=dict(station=ST,raw_station_rows=len(d),eligible_rows=len(g),time_order_violations=int((d.price_base&~d.time_order_valid).sum()),duplicate_record_rows=int(d.duplicate_record.sum()),duration_max_abs_discrepancy_min=float(d.duration_discrepancy_min.abs().max()),fee_identity_max_abs=float((d.total_fee-d.total_elec_fee-d.total_service_fee).abs().max()),comparison_fields=len(comp),max_frozen_diff=max(abs(x['difference']) for x in comp),independent_SQL_max_differences=maxdiff,inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],python=sys.version,versions=dict(pandas=pd.__version__,numpy=np.__version__,duckdb=duckdb.__version__),raw_read_scope='January parquet filter station_id=ST_003774, all columns; no other stations materialized',source_file_hash_matches_frozen_config=True)
(OUT/'VERIFICATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
print(endpoints[['window','segment_id','n_transactions','n_tariff_dates','kwh','n_piles','E_median','S_median','T_median']].to_string(index=False))
