from common import *

def panel_from_endpoints(ep,layer):
 keys=KEY+['restriction','period']
 l=ep.loc[ep.side.eq('L')].drop(columns='side').set_index(keys).add_suffix('_L')
 r=ep.loc[ep.side.eq('R')].drop(columns='side').set_index(keys).add_suffix('_R')
 p=l.join(r,how='outer').reset_index().merge(layer,on=['month','boundary_hour'],validate='many_to_one')
 for side in ['L','R']:
  for col in ['n','days','piles']:p[col+'_'+side]=p[col+'_'+side].fillna(0)
  p['support_'+side]=(p['n_'+side]>=np.where(p.period.eq('fullmonth'),20,10))&(p['days_'+side]>=np.where(p.period.eq('fullmonth'),5,3))
 p['support']=p.support_L&p.support_R
 for x in ['E','S','T']:p[x]=p.q*(p[x+'_R']-p[x+'_L'])
 p['R']=p.support&(p.E>.01);p['H']=p.R&(p['T']>.03);p['D30']=30*(p['T']-p.E)
 p['January_status']=np.where(p.month.eq('2025-01'),'暂定：数据来源机制未决','not_applicable')
 return p

def rates(g,rmask,hmask):
 r=g.loc[rmask];h=g.loc[hmask]
 return dict(supported_units=int(g.support.sum()),R=len(r),retained=int((r['T']>.03).sum()),retention=(r['T']>.03).mean() if len(r) else np.nan,H=len(h),amount_exceed=int((h.D30.abs()>.9).sum()),amount_exceed_share=(h.D30.abs()>.9).mean() if len(h) else np.nan)

def full_metrics(p,a):
 original=a[KEY+['R_median','H_median','E_median','S_median','T_median','D30_median']]
 f=p.loc[p.period.eq('fullmonth')].merge(original,on=KEY,how='inner',validate='many_to_one')
 save(f,'task3/restricted_fullmonth_unit_prices.csv');rows=[]
 for (res,m,b),g in f.groupby(['restriction','month','boundary_hour']):
  g=g.loc[g.support]
  for pop,rm,hm in [('requalified',g.R,g.H),('fixed_original_R_H',g.R_median,g.H_median)]:
   rows.append(dict(restriction=res,month=m,boundary_hour=b,membership=pop,**rates(g,rm,hm)))
  old=g.copy();old['T']=old.T_median;old['E']=old.E_median;old['D30']=old.D30_median
  rows.append(dict(restriction=res,month=m,boundary_hour=b,membership='original_on_restricted_support',**rates(old,old.R_median,old.H_median)))
 metrics=pd.DataFrame(rows);save(metrics,'task3/restriction_metrics_by_layer.csv');summary=[]
 for (res,pop),g in metrics.groupby(['restriction','membership']):
  for month in ['2025-01','2025-07','ALL_10_LAYERS']:
   h=g if month=='ALL_10_LAYERS' else g.loc[g.month.eq(month)]
   summary.append(dict(restriction=res,membership=pop,month=month,layers=len(h),supported_units=h.supported_units.sum(),R=h.R.sum(),H=h.H.sum(),retained=h.retained.sum(),amount_exceed=h.amount_exceed.sum(),retention_pooled=h.retained.sum()/h.R.sum() if h.R.sum() else np.nan,amount_pooled=h.amount_exceed.sum()/h.H.sum() if h.H.sum() else np.nan,retention_layer_equal=h.retention.mean(),amount_layer_equal=h.amount_exceed_share.mean(),nonempty_R_layers=int(h.R.gt(0).sum()),nonempty_H_layers=int(h.H.gt(0).sum())))
 save(pd.DataFrame(summary),'task3/restriction_metrics_summary.csv')
 return f

def crosswindow(p):
 v=csv(OLD/'task1/fixed_V_unit_prices.csv');fullpanel=parquet(OLD/'cache/boundary_prices.parquet')
 f0=fullpanel.loc[fullpanel.period.eq('formation'),KEY+['E_median','T_median']]
 v0=fullpanel.loc[fullpanel.period.eq('validation'),KEY+['T_median']]
 orig=f0.merge(v0,on=KEY,suffixes=('_f','_v'));orig['E_median_f']=orig.pop('E_median')
 vf=v[KEY].assign(original_V=True)
 f=p.loc[p.period.eq('formation')].drop(columns='period');vv=p.loc[p.period.eq('validation')].drop(columns='period')
 z=f.merge(vv,on=KEY+['restriction'],suffixes=('_f','_v'))
 # Frozen V requires support in both halves, not electricity qualification R_f.
 z=z.loc[z.support_f&z.support_v].copy()
 z=z.merge(vf,on=KEY,how='left').merge(orig,on=KEY,how='left');z['original_V']=z.original_V.fillna(False)
 save(z,'task3/same_pile_crosswindow_units.csv');rows=[]
 for res,g in z.groupby('restriction'):
  if res not in ['same_month_pile','formation_locked_pile','unrestricted']:continue
  for pop in ['independent_restricted_V','intersection_original_V']:
   h=g if pop.startswith('independent') else g.loc[g.original_V]
   for version in ['restricted_prices','original_prices_same_members']:
    for (m,b),k in h.groupby(['month','boundary_hour']):
     if version=='restricted_prices':errE=(k.E_f-k.T_v).abs();errL=(k.T_f-k.T_v).abs()
     else:errE=(k.E_median_f-k.T_median_v).abs();errL=(k.T_median_f-k.T_median_v).abs()
     rows.append(dict(restriction=res,membership=pop,version=version,month=m,boundary_hour=b,n=len(k),E_MAE=errE.mean(),LOCAL_MAE=errL.mean(),relative_reduction=1-errL.mean()/errE.mean()))
 metrics=pd.DataFrame(rows);save(metrics,'task3/same_pile_crosswindow_by_layer.csv');summary=[]
 for (res,pop,version),g in metrics.groupby(['restriction','membership','version']):
  for m in ['2025-01','2025-07','ALL_10_LAYERS']:
   h=g if m=='ALL_10_LAYERS' else g.loc[g.month.eq(m)]
   summary.append(dict(restriction=res,membership=pop,version=version,month=m,layers=len(h),n=h.n.sum(),E_MAE=h.E_MAE.mean(),LOCAL_MAE=h.LOCAL_MAE.mean(),relative_reduction=1-h.LOCAL_MAE.mean()/h.E_MAE.mean()))
 save(pd.DataFrame(summary),'task3/same_pile_crosswindow_summary.csv')
 return z

def task3():
 assert (OUT/'validation/task2.json').exists()
 cfg,layer=layers();a=csv(OLD/'task1/fixed_A_unit_prices.csv');c=database();c.register('layers',layer);c.register('Akeys',a[KEY])
 c.execute('''CREATE TEMP TABLE bt AS SELECT t.month,b.boundary_hour,t.station_id,t.pile_id,t.tariff_day,t.half,t.E,t.S,t.T,
 CASE WHEN t.segment_id=b.segment_L THEN 'L' ELSE 'R' END AS side,
 CASE WHEN rated_gun_power_kw IS NULL THEN 'missing' WHEN rated_gun_power_kw<30 THEN 'lt30' WHEN rated_gun_power_kw<120 THEN '30to120' ELSE 'ge120' END AS power_class,
 coalesce(equipment_classification,'missing') AS acdc
 FROM valid t JOIN layers b ON t.month=b.month AND t.segment_id IN(b.segment_L,b.segment_R)''')
 comp=[]
 for field in ['power_class','acdc']:
  counts=c.execute(f'''SELECT t.month,t.boundary_hour,t.station_id,t.side,t.{field} AS category,count(*) AS n FROM bt t JOIN Akeys a USING(month,boundary_hour,station_id) GROUP BY ALL''').df()
  cats=['lt30','30to120','ge120','missing'] if field=='power_class' else ['交流设备','直流设备','missing']
  grid=a[KEY].merge(pd.DataFrame({'category':cats}),how='cross').merge(pd.DataFrame({'side':['L','R']}),how='cross')
  counts=grid.merge(counts,on=KEY+['category','side'],how='left');counts.n=counts.n.fillna(0)
  counts['share']=counts.n/counts.groupby(KEY+['side']).n.transform('sum')
  l=counts.loc[counts.side.eq('L')].drop(columns='side');r=counts.loc[counts.side.eq('R')].drop(columns='side')
  z=l.merge(r,on=KEY+['category'],suffixes=('_L','_R'));z['after_minus_before']=z.share_R-z.share_L;z['classification']=field;comp.append(z)
 comp=pd.concat(comp,ignore_index=True);save(comp,'task3/unit_endpoint_composition.csv');distributions=[]
 onehour={('2025-01',17),('2025-01',22),('2025-01',23),('2025-07',10),('2025-07',22),('2025-07',23)}
 for (m,b,k,cat),g in comp.groupby(['month','boundary_hour','classification','category']):
  distributions.append(dict(month=m,boundary_hour=b,classification=k,category=cat,one_hour_layer=(m,b) in onehour,**dist(g.after_minus_before)))
 d=pd.DataFrame(distributions);save(d,'task3/composition_difference_distribution.csv');save(d.loc[d.one_hour_layer],'task3/six_one_hour_layers.csv')
 c.execute('''CREATE TEMP TABLE common_pile AS SELECT month,boundary_hour,station_id,pile_id FROM bt GROUP BY ALL HAVING count(DISTINCT side)=2''')
 c.execute('''CREATE TEMP TABLE formation_pile AS SELECT month,boundary_hour,station_id,pile_id FROM bt WHERE half='formation' GROUP BY ALL HAVING count(DISTINCT side)=2''')
 restricted=["SELECT *, 'unrestricted' AS restriction FROM bt",
 "SELECT b.*, 'same_month_pile' AS restriction FROM bt b JOIN common_pile USING(month,boundary_hour,station_id,pile_id)",
 "SELECT b.*, 'formation_locked_pile' AS restriction FROM bt b JOIN formation_pile USING(month,boundary_hour,station_id,pile_id)",
 "SELECT *, 'power_'||power_class AS restriction FROM bt WHERE power_class<>'missing'",
 "SELECT *, 'type_'||acdc AS restriction FROM bt WHERE acdc<>'missing'"]
 c.execute('CREATE TEMP VIEW restricted AS '+' UNION ALL '.join(restricted))
 print('Task3 reconstructing restricted endpoints',flush=True)
 ep=c.execute('''WITH expanded AS (SELECT *, 'fullmonth' AS period FROM restricted UNION ALL SELECT *,half AS period FROM restricted WHERE half IN ('formation','validation'))
 SELECT month,boundary_hour,station_id,restriction,period,side,count(*) AS n,count(DISTINCT tariff_day) AS days,count(DISTINCT pile_id) AS piles,median(E) AS E,median(S) AS S,median(T) AS T FROM expanded GROUP BY ALL ORDER BY ALL''').df()
 save(ep,'task3/restricted_endpoints.csv');p=panel_from_endpoints(ep,layer);save(p,'task3/all_restricted_panels.csv')
 f=full_metrics(p,a);z=crosswindow(p)
 baseline=f.loc[f.restriction.eq('unrestricted')]
 delta=max((baseline.E-baseline.E_median).abs().max(),(baseline['T']-baseline.T_median).abs().max())
 original_V=z.loc[z.restriction.eq('unrestricted')]
 done(3,dict(baseline_A=len(baseline)==8133,unrestricted_prices_match=delta<1e-12,baseline_V=len(original_V)==5846,composition_complete=len(comp)==8133*7,restricted_support_subset_A=bool((p.loc[p.period.eq('fullmonth')&p.support&~p.restriction.eq('formation_locked_pile')].merge(a[KEY],on=KEY,how='left',indicator=True)._merge=='both').all())))
 c.close()

if __name__=='__main__':task3()
