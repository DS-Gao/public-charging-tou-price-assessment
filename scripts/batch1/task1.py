from common import *

def task1():
 initialize();c=database()
 a=csv(OLD/'task1/fixed_A_unit_prices.csv')
 fields=[]
 for name in ['orders_2025-01_public.parquet','orders_2025-07_public.parquet','stations_public.parquet']:
  p=source(ROOT/'31952289'/name,'Raw Beijing schema and immutable input')
  for j,f in enumerate(pq.read_schema(p),1):fields.append(dict(dataset='Beijing',file=str(p),position=j,field=f.name,dtype=str(f.type)))
 for name in ['station_information.csv','pile_rated_power.csv']:
  p=source(URAW/name,'Actual UrbanEV attribute schema');g=pd.read_csv(p)
  for j,col in enumerate(g,1):fields.append(dict(dataset='UrbanEV',file=str(p),position=j,field=col,dtype=str(g[col].dtype),records=len(g),missing=int(g[col].isna().sum())))
 source(ROOT/'doi_10_5061_dryad_np5hqc04z__v20260204/README.md','UrbanEV release definitions')
 save(pd.DataFrame(fields),'task1/all_fields.csv')
 attrs=[('Beijing','pile ID','pile_id','present','station+month+pile ID; not gun ID'),('Beijing','rated power','rated_gun_power_kw','present','recorded rated power, not actual power'),('Beijing','AC/DC','equipment_classification','present','released direct type'),('Beijing','station pile counts','piles_num;dc_piles_num;ac_piles_num','present','static date unverified'),('Beijing','location','geocoding','partial','coded field; no longitude/latitude/district columns'),('Beijing','operator/platform','','absent','no group attribution possible'),('Beijing','public/dedicated type','','absent','construction_site is surrounding-POI scene'),('Beijing','fast-charger share','','not_directly_available','DC share reported explicitly; no speed equivalence assumed'),('UrbanEV','position','longitude;latitude;TAZID','present','TAZ not assumed equal to administrative district'),('UrbanEV','pile counts','slow_count;fast_count;charge_count','present','static release'),('UrbanEV','pile properties','pile_id;power;pileType;station_id','present','actual pile_id header differs from README pileNo'),('UrbanEV','operator/platform','','absent','cannot attribute a strategy')]
 save(pd.DataFrame(attrs,columns=['dataset','attribute','fields','status','meaning']),'task1/attribute_availability.csv')
 static=pd.read_parquet(ROOT/'31952289/stations_public.parquet')
 act=c.execute('SELECT month,station_id,count(*) AS raw_transactions,count(*) FILTER(WHERE base_valid) AS base_transactions,count(*) FILTER(WHERE base_valid AND contained) AS contained_transactions FROM tx GROUP BY ALL ORDER BY month,station_id').df()
 counts=[];station=[]
 for m in ['2025-01','2025-07','union']:
  z=a if m=='union' else a.loc[a.month.eq(m)];active=act if m=='union' else act.loc[act.month.eq(m)]
  counts.append(dict(month=m,active_stations=active.station_id.nunique(),A_units=len(z),A_stations=z.station_id.nunique(),R_units=int(z.R_median.sum()),R_stations=z.loc[z.R_median,'station_id'].nunique()))
  if m=='union':continue
  g=static.merge(active.drop(columns='month'),on='station_id',how='left');g['month']=m
  g['active_this_month']=g.raw_transactions.notna()
  g[['raw_transactions','base_transactions','contained_transactions']]=g[['raw_transactions','base_transactions','contained_transactions']].fillna(0)
  g['included_union_A']=g.station_id.isin(a.station_id);g['included_month_A']=g.station_id.isin(z.station_id)
  g['dc_pile_fraction']=g.dc_piles_num/g.piles_num;station.append(g)
 save(pd.DataFrame(counts),'task1/sample_counts.csv');station=pd.concat(station,ignore_index=True)
 save(station,'task1/station_month_membership.csv');rows=[]
 for m,g in station.groupby('month'):
  for definition in ['included_union_A','included_month_A']:
   for active in [True,False]:
    v=g.loc[g.active_this_month] if active else g
    for flag,h in v.groupby(definition):
     rows.append(dict(month=m,membership=definition,active_only=active,included=flag,stations=len(h),active_stations=int(h.active_this_month.sum()),raw_transactions_median=h.raw_transactions.median(),base_transactions_median=h.base_transactions.median(),contained_transactions_median=h.contained_transactions.median(),piles_median=h.piles_num.median(),dc_share_station_mean=h.dc_pile_fraction.mean(),dc_share_station_median=h.dc_pile_fraction.median(),dc_share_pile_weighted=h.dc_piles_num.sum()/h.piles_num.sum(),piles_denominator=h.piles_num.sum()))
 save(pd.DataFrame(rows),'task1/included_vs_other.csv')
 save(c.execute("SELECT month,coalesce(equipment_classification,'MISSING') AS equipment_classification,count(*) AS raw_n,count(*) FILTER(WHERE base_valid AND contained) AS contained_n FROM tx GROUP BY ALL ORDER BY ALL").df(),'task1/acdc_values.csv')
 for n in ['facility_field_dictionary_audit.csv','facility_field_coverage.csv','facility_category_values.csv']:
  shutil.copy2(source(OLD/'task3'/n,'Reuse audited field evidence'),OUT/'task1'/('reused_'+n))
 done(1,dict(A=len(a)==8133,A_stations=a.station_id.nunique()==2368,R=int(a.R_median.sum())==7531,R_stations=a.loc[a.R_median,'station_id'].nunique()==2216,active_union=act.station_id.nunique()==8553,static_matches_active=set(act.station_id)==set(static.station_id)))
 print(pd.DataFrame(counts).to_string(index=False));c.close()

if __name__=='__main__':task1()
