from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];OUT=P/'raw_station'
d=pd.read_csv(OUT/'station_transactions_with_audit_flags.csv.gz',parse_dates=['charge_start_time','charge_end_time','tariff_date','calendar_date'])
g=d[d.eligible].copy()
old=set(g.loc[g.window.eq('formation'),'pile_id'])
g['pile_cohort']=np.where(g.pile_id.isin(old),'observed_in_formation','first_observed_validation')
records=[]
for keys,z in g.groupby(['window','segment_id','pile_cohort']):
 records.append(dict(zip(['window','segment_id','pile_cohort'],keys))|dict(n=len(z),n_dates=z.tariff_date.nunique(),kwh=z.total_elec_kwh.sum(),E_median=z.E.median(),S_median=z.S.median(),T_median=z['T'].median(),n_piles=z.pile_id.nunique(),first_start=str(z.charge_start_time.min()),last_start=str(z.charge_start_time.max())))
cohort=pd.DataFrame(records);cohort.to_csv(OUT/'cohort_endpoint_prices.csv',index=False)
rawpile=[]
for pile,z in d.groupby('pile_id'):
 rawpile.append(dict(pile_id=pile,cohort='observed_in_formation' if pile in old else 'first_observed_validation',raw_n=len(z),eligible_n=int(z.eligible.sum()),power_kw=';'.join(map(str,sorted(z.rated_gun_power_kw.unique()))),equipment=';'.join(map(str,z.equipment_classification.unique())),first_start=str(z.charge_start_time.min()),last_start=str(z.charge_start_time.max()),calendar_dates=z.calendar_date.nunique(),tariff_dates=z.tariff_date.nunique()))
pd.DataFrame(rawpile).to_csv(OUT/'pile_inventory.csv',index=False)
# Check only the 12 implicated pile identifiers against the raw file's station
# assignment, including rows that may have been assigned to a different station.
RAW=ROOT/'31952289/orders_2025-01_public.parquet'
mapping=pd.read_parquet(RAW,columns=['station_id','pile_id','charge_start_time'],filters=[('pile_id','in',list(d.pile_id.unique()))])
mp=mapping.groupby(['pile_id','station_id']).agg(n=('charge_start_time','size'),first_start=('charge_start_time','min'),last_start=('charge_start_time','max')).reset_index();mp.to_csv(OUT/'raw_pile_station_mapping.csv',index=False)
# Retain old observed piles as a composition diagnostic, not a replacement
# evaluation sample. Pooled medians and supports are recomputed within cohort.
rows=[]
for window in ['formation','validation']:
 for bh,before,after,q in [(7,'23_07','07_10',1),(13,'10_13','13_17',-1)]:
  for cohort_name,selector in [('all',pd.Series(True,index=g.index)),('observed_in_formation',g.pile_id.isin(old)),('first_observed_validation',~g.pile_id.isin(old))]:
   z=g[g.window.eq(window)&selector];a=z[z.segment_id.eq(after)];b=z[z.segment_id.eq(before)]
   r=dict(window=window,boundary_hour=bh,pile_cohort=cohort_name,n_before=len(b),n_after=len(a),days_before=b.tariff_date.nunique(),days_after=a.tariff_date.nunique(),supported=len(b)>=10 and len(a)>=10 and b.tariff_date.nunique()>=3 and a.tariff_date.nunique()>=3)
   for c in ['E','S','T']:r[c+'_star']=q*(a[c].median()-b[c].median()) if len(a) and len(b) else np.nan
   rows.append(r)
pd.DataFrame(rows).to_csv(OUT/'fixed_pile_cohort_contrasts.csv',index=False)
# Direct daily evidence and within-pile persistence; rounding is descriptive
# only and was not used in reconstruction or eligibility.
daily=g.groupby(['tariff_date','segment_id','pile_cohort']).agg(n=('T','size'),n_piles=('pile_id','nunique'),T_median=('T','median'),S_median=('S','median'),E_median=('E','median')).reset_index()
daily.to_csv(OUT/'daily_cohort_prices.csv',index=False)
check=dict(formation_piles=sorted(old),validation_piles=sorted(g.loc[g.window.eq('validation'),'pile_id'].unique()),raw_unique_piles=d.pile_id.nunique(),pile_identifiers_mapping_to_multiple_stations=int(mp.groupby('pile_id').station_id.nunique().gt(1).sum()),raw_target_pile_rows=len(mapping),additional_station_ids=sorted(set(mapping.station_id)-{'ST_003774'}),first_new_pile_transaction=str(d.loc[~d.pile_id.isin(old),'charge_start_time'].min()),caveat='First appearance in the provided January records is not proof of physical installation, operator change, or a verified mapping history.',raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest())
(OUT/'PILE_COMPOSITION_CHECK.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
