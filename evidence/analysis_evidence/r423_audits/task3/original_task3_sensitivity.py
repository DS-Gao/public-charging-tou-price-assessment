from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'task3';OUT.mkdir(exist_ok=True)
SRC=ROOT/'analysis/price_architecture_v2/urbanev_external_analysis_20260913'
DATA=ROOT/'outputs/09_price_architecture/urbanev_external_analysis_20260913'
sys.path.insert(0,str(SRC))
from e1_extract import reduce_pairs,RAW
split=json.loads((DATA/'sample_split.json').read_text());ids=split['validation_station_ids']
assert len(ids)==1652
assert all((RAW/f'{s}.csv').is_file() for s in ids)
ep=pd.read_parquet(DATA/'daily_endpoints.parquet')
assert set(ep.station_id.astype(str))<=set(map(str,ids))
panels={};days=[]
for t in [.7,.8,.9]:
 e=ep.copy();e['supported']=e.coverage>=t
 # reduce_pairs originally receives a single station. The full-day flag must
 # use station and day together when batching the saved endpoint table.
 from urbanev_adapter import paired_boundaries
 e['full_day_supported']=e.set_index(['station_id','day']).index.isin(e[e.supported].groupby(['station_id','day']).size().loc[lambda x:x==9].index)
 pairs=paired_boundaries(e,{0:-1,480:1,600:1,720:-1,840:1,1140:-1})
 pairs=pairs[pairs.month.isin(['2022-09','2022-10','2022-11','2022-12','2023-01','2023-02'])]
 keys=['station_id','month','boundary_minute'];agg={'paired_days':('boundary_day','nunique'),'orientation':('orientation','first')}
 for c in ['E','S','T']:
  for side in ['left','right']:agg[f'{c}_{side}']=(f'{c}_{side}','median')
 p=pairs.groupby(keys).agg(**agg).reset_index()
 for c in ['E','S','T']:p[c+'_star']=np.round(p.orientation*(p[c+'_right']-p[c+'_left']),12)
 p['A']=p.paired_days>=10;p['R']=p.A&(p.E_star>.01);p['H']=p.R&(p.T_star>.03)
 p['D30']=np.round(30*(p.T_star-p.E_star),10);p['abs_D30']=p.D30.abs();p['threshold']=t
 panels[t]=p
 days.append(dict(threshold=t,supported_day_segments=int(e.supported.sum()),station_days_any_segment=e[e.supported][['station_id','day']].drop_duplicates().shape[0],station_days_all9=int(e[e.supported].groupby(['station_id','day']).size().eq(9).sum()),paired_boundary_days=len(pairs),supported_stations=p.loc[p.A,'station_id'].nunique()))
 p.to_csv(OUT/f'monthly_boundary_prices_{int(100*t)}.csv.gz',index=False)
 print('threshold',t,'done',flush=True)
base=pd.read_parquet(DATA/'monthly_boundary_panel.parquet');base=base[base.variant=='base']
chk=panels[.8].merge(base,on=keys,suffixes=('_new','_old'))
diff={c:float((chk[c+'_new']-chk[c+'_old']).abs().max()) for c in ['E_star','S_star','T_star','paired_days']}
assert all(x<1e-9 for x in diff.values())
assert int(panels[.8].R.sum())==4921 and int(panels[.8].H.sum())==1377
common=set.intersection(*[set(map(tuple,p.loc[p.A,keys].to_numpy())) for p in panels.values()])
rows=[]
for t,p in panels.items():
 p['common_A']=list(map(lambda x:x in common,map(tuple,p[keys].to_numpy())))
 for scope,sel in [('own_A',p.A),('common_A',p.common_A)]:
  z=p[sel]
  for layer,g in [('pooled',z)]+list(z.groupby(['month','boundary_minute'])):
   r=g[g.R];h=g[g.H]
   rows.append(dict(threshold=t,sample=scope,layer=str(layer),A=len(g),stations=g.station_id.nunique(),R=len(r),H=len(h),H_over_R=len(h)/len(r) if len(r) else np.nan,median_abs_D30_A=g.abs_D30.median(),median_abs_D30_H=h.abs_D30.median(),H_abs_over_090=int((h.abs_D30>.9).sum()),H_abs_over_090_share=(h.abs_D30>.9).mean()))
 # Fixed A membership; qualification remains price-dependent and is reported.
 p.to_csv(OUT/f'monthly_boundary_prices_{int(100*t)}.csv.gz',index=False)
pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
pd.DataFrame(days).to_csv(OUT/'daily_support.csv',index=False)
inputs=[DATA/'daily_endpoints.parquet',DATA/'monthly_boundary_panel.parquet',DATA/'sample_split.json',SRC/'urbanev_adapter.py',SRC/'e1_extract.py']
record=dict(inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],raw_files_present=1652,raw_reprocessing=False,source='Saved pre-threshold daily endpoints contain n/expected_slots and all positive record-count day segments, including unsupported ones. Threshold only changes eligibility, not within-segment medians.',baseline80_max_differences=diff,common_A_units=len(common),analysis_population=59472)
(OUT/'PROVENANCE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("layer=='pooled'").to_string(index=False))
