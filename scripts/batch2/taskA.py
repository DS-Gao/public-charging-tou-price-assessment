from common import *

CONFIGS=['E','CAL','E_plus_S','LOCAL'];METRICS=['mae','me','p90_abs_error','rmse','root_mean_layer_mse']

def aggregate_layers(d,draws=False):
 keys=['weight','configuration']+(['draw'] if draws else [])
 out=[d]
 for month,g in group_scopes(d):
  if len(g)==0:continue
  rows=g.groupby(keys,sort=False).agg(n=('n','sum'),n_layers=('layer_order','nunique'),**{c:(c,'mean') for c in ['mae','me','p90_abs_error','rmse','mse']}).reset_index()
  rows['root_mean_layer_mse']=np.sqrt(rows.mse);rows['month']=month;rows['layer_order']='equal_layers'
  rows['scope']='summary';out.append(rows)
 return pd.concat(out,ignore_index=True)

def crosswindow():
 f=formation();s=stats()
 orig=csv(PHASE/'references/original_point_estimates.csv');es=csv(PHASE/'task2/primary_metrics.csv')
 orig=orig.loc[orig.scope.eq('layer')&orig.group.eq('all')&orig.configuration.isin(CONFIGS)].copy()
 es=es.loc[es.scope.eq('layer')&es.group.eq('all')].copy()
 es=es.rename(columns={'E_plus_S_mae':'mae','E_plus_S_me':'me','E_plus_S_p90':'p90_abs_error'})
 es['configuration']='E_plus_S'
 keep=['scope','layer_order','month','weight','configuration','n','mae','me','p90_abs_error']
 p=pd.concat([orig[keep],es[keep]],ignore_index=True);p.layer_order=p.layer_order.astype(int)
 rows=[];checks=[]
 for layer,g in f.groupby('layer_order'):
  v=g.loc[g.V]
  for weight in ['ST','TX','KWH']:
   w=s.weights_for_panel(g,v,weight)
   for config in CONFIGS:
    r=p.loc[p.layer_order.eq(layer)&p.weight.eq(weight)&p.configuration.eq(config)].iloc[0].to_dict()
    err=v[config].to_numpy()-v.Y.to_numpy();r.update(mse=float(w@(err**2)),rmse=float(np.sqrt(w@(err**2))))
    r['root_mean_layer_mse']=r['rmse'];rows.append(r)
    checks.append(abs(r['mae']-w@np.abs(err))<1e-12)
 p=aggregate_layers(pd.DataFrame(rows));save(p,'taskA/crosswindow_point_metrics.csv')
 # Original summary draws are reused; only missing quantities use saved multiplicities.
 op=source(PHASE/'references/original_bootstrap_draws.csv.gz')
 parts=[]
 for c in pd.read_csv(op,chunksize=200000,usecols=['scope','layer_order','month','group','weight','configuration','draw','n','mae','me','p90_abs_error']):
  parts.append(c.loc[c.scope.eq('layer')&c.group.eq('all')&c.configuration.isin(['E','CAL','LOCAL'])])
 od=pd.concat(parts,ignore_index=True);od.layer_order=od.layer_order.astype(int)
 mult=csv(PHASE/'task1/formation_2342_multiplicities.csv.gz');assert len(mult)==2000
 phase_draws=csv(PHASE/'task1/formation_2342_draw_metrics.csv.gz')
 batchq=load_module(ROOT/'.submission_github_checkout_20261109/src/prices/fast_bootstrap.py','frozen_fast_bootstrap').batch_quantile
 new=[];escheck=[]
 for layer,g in f.groupby('layer_order'):
  g=g.reset_index(drop=True);v=g.V.to_numpy(bool);m=mult[g.station_id.tolist()].to_numpy(float)
  errs=g.loc[v,CONFIGS].to_numpy()-g.loc[v,'Y'].to_numpy()[:,None]
  for weight in ['ST','TX','KWH']:
   if weight=='ST':w=m[:,v].copy()
   else:
    pref='tx' if weight=='TX' else 'kwh';bm=m*g[pref+'_before'].to_numpy();am=m*g[pref+'_after'].to_numpy()
    w=(.5*bm/bm.sum(axis=1)[:,None]+.5*am/am.sum(axis=1)[:,None])[:,v]
   w=w/w.sum(axis=1)[:,None];mses=w@(errs**2)
   for j,config in enumerate(CONFIGS):
    z=pd.DataFrame(dict(draw=np.arange(2000),layer_order=layer,month=g.month.iloc[0],weight=weight,configuration=config,n=int(v.sum()),mse=mses[:,j],rmse=np.sqrt(mses[:,j]),scope='layer'))
    if config=='E_plus_S':
     z['mae']=w@np.abs(errs[:,j]);z['me']=w@errs[:,j];z['p90_abs_error']=batchq(np.abs(errs[:,j]),w)
     if weight=='ST':
      saved=phase_draws.loc[phase_draws.scope.eq('layer')&phase_draws.layer_order.astype(str).eq(str(layer))].sort_values('draw')
      escheck.append(np.max(np.abs(z.mae.to_numpy()-saved.E_plus_S_mae.to_numpy()))<1e-12)
      z['mae']=saved.E_plus_S_mae.to_numpy()
    else:
     saved=od.loc[od.layer_order.eq(layer)&od.weight.eq(weight)&od.configuration.eq(config)].sort_values('draw')
     assert len(saved)==2000
     for metric in ['mae','me','p90_abs_error']:z[metric]=saved[metric].to_numpy()
    z['root_mean_layer_mse']=z.rmse;new.append(z)
  print('A missing metrics same-draw evaluation layer',layer,flush=True)
 d=aggregate_layers(pd.concat(new,ignore_index=True),True)
 save(d,'taskA/crosswindow_same_draw_metrics.csv.gz')
 keys=['scope','layer_order','month','weight','configuration'];d.layer_order=d.layer_order.astype(str);p.layer_order=p.layer_order.astype(str)
 ci=[]
 for key,g in d.groupby(keys,sort=False):
  r=dict(zip(keys,key));r.update(draws=len(g),seed=20260920,station_frame=2342)
  for metric in METRICS:r[metric+'_low']=g[metric].quantile(.025);r[metric+'_high']=g[metric].quantile(.975)
  ci.append(r)
 ci=pd.DataFrame(ci);full=p.merge(ci,on=keys,validate='one_to_one');save(full,'taskA/crosswindow_metrics_with_intervals.csv')
 gains=[]
 for key,g in p.groupby(['scope','layer_order','month','weight'],sort=False):
  r=dict(zip(['scope','layer_order','month','weight'],key));r['n']=int(g.n.iloc[0])
  gd=d.loc[d.scope.eq(key[0])&d.layer_order.eq(key[1])&d.month.eq(key[2])&d.weight.eq(key[3])]
  e=g.set_index('configuration').loc['E'];l=g.set_index('configuration').loc['LOCAL']
  de=gd.loc[gd.configuration.eq('E')].set_index('draw');dl=gd.loc[gd.configuration.eq('LOCAL')].set_index('draw')
  for metric in ['mae','rmse','root_mean_layer_mse']:
   net=e[metric]-l[metric];ds=de[metric]-dl[metric]
   r[metric+'_gain']=net;r[metric+'_reduction']=net/e[metric]
   for lab,vals in [('gain',ds),('reduction',ds/de[metric])]:
    r[metric+'_'+lab+'_low']=vals.quantile(.025);r[metric+'_'+lab+'_high']=vals.quantile(.975)
  gains.append(r)
 save(pd.DataFrame(gains),'taskA/E_to_LOCAL_gains.csv')
 influence=csv(PHASE/'task2/station_influence_metrics.csv');save(influence.loc[influence.month.eq('2025-07')],'taskA/july_station_influence.csv')
 save(f.loc[f.station_id.eq('ST_003774')&f.month.eq('2025-07'),KEY+['layer_order','V','E','E_plus_S','LOCAL','Y']],'taskA/july_ST_003774_membership.csv')
 cov=csv(PHASE/'task2/support_coverage.csv');cov=cov.loc[cov.layer_order.ne('ALL')].copy();cov.layer_order=cov.layer_order.astype(int)
 mapping=f[['layer_order','month']].drop_duplicates();cov=cov.merge(mapping,on='layer_order',validate='many_to_one')
 covs=[cov]
 for month,g in group_scopes(cov):
  z=g.groupby('group',sort=False)[['entry_n','observable_n','eval_n','missing_target_n']].sum().reset_index()
  z['month']=month;z['layer_order']='equal_layers';z['availability']=z.observable_n/z.entry_n;z['denominator']='summed group formation entry';covs.append(z)
 save(pd.concat(covs,ignore_index=True),'taskA/support_coverage.csv')
 sm=csv(PHASE/'task2/support_groups_metrics.csv');save(sm,'taskA/support_group_metrics.csv')
 return all(checks) and all(escheck),f

def dispersion_decomposition():
 base=ROOT/'.submission_github_checkout_20261109/data/reference/historical'
 d=csv(base/'02_boundary_signal_core/network_signal_dispersion.csv')
 p=csv(base/'component_pairing/a2/a2_point_dispersion.csv');p=p.loc[p.variant.eq('ORIGINAL')]
 z=d[['layer_order','month','boundary_hour','F_IQR_30kwh','F_IQR_30kwh_boot_lower','F_IQR_30kwh_boot_upper','dispersion_class']].merge(p[['layer_order','n_R','F30']],on='layer_order',validate='one_to_one')
 assert np.allclose(z.F30,z.F_IQR_30kwh,atol=1e-9)
 save(z,'taskA/F30_by_layer_reused.csv');rr=[]
 for month,g in group_scopes(z):
  for name in sorted(z.dispersion_class.unique()):rr.append(dict(month=month,layers=len(g),classification=name,n=int(g.dispersion_class.eq(name).sum())))
 save(pd.DataFrame(rr),'taskA/F30_class_counts.csv')
 path=ROOT/'outputs/09_price_architecture/measurement_framework_empirical_extension_20260913/a3'
 pts=parquet(path/'a3_point_estimates.parquet');ci=csv(path/'a3_marginal_intervals.csv');source(path/'a3_manifest.json','Original 2368-station decomposition bootstrap protocol')
 sc=['JANUARY_LAYER_EQUAL','JULY_LAYER_EQUAL','ALL10_LAYER_EQUAL'];pts=pts.loc[pts.scope_type.eq('layer')|pts.scope.isin(sc)]
 ci=ci.loc[ci.scope_type.eq('layer')|ci.scope.isin(sc)]
 save(pts,'taskA/weight_decomposition_points_reused.csv');save(ci,'taskA/weight_decomposition_intervals_reused.csv')
 rows=[]
 for _,r in pts.iterrows():
  z={k:r[k] for k in ['scope_type','scope','layer_order','target']};z.update(ST_median=r.m00,target_median=r.m11,gap=r.observed_gap,composition=r.composition_component,within_group=r.within_component,unit='RMB/30kWh',estimand='A: median abs(B30), common weighted ECDF')
  for name,field in [('ST_median','m00'),('target_median','m11'),('gap','observed_gap'),('composition','composition_component'),('within_group','within_component')]:
   q=ci.loc[ci.scope.eq(r.scope)&ci.target.eq(r.target)&ci.metric.eq(field)]
   assert len(q)==1;z[name+'_low']=q.q025.iloc[0];z[name+'_high']=q.q975.iloc[0]
  rows.append(z)
 result=pd.DataFrame(rows);save(result,'taskA/median_weight_gap_comparison.csv')
 return np.allclose(result.gap,result.composition+result.within_group,atol=1e-10)

def coverage_figure():
 daily=csv(FIRST/'task2/daily_activity.csv');hour=csv(FIRST/'task2/start_hour_histogram.csv');match=csv(FIRST/'task2/january_matched_daily_endpoint_changes.csv');trans=csv(FIRST/'task2/january_key_endpoint_transition_checks.csv')
 endpoints=csv(FIRST/'task2/daily_station_endpoints.csv')
 counts=endpoints.assign(has_base=endpoints.n_base.gt(0),has_contained=endpoints.n_contained.gt(0)).groupby(['month','tariff_day','segment_id']).agg(base_stations=('has_base','sum'),contained_stations=('has_contained','sum')).reset_index()
 for frame,name in [(daily,'daily_eligible_transactions'),(counts,'daily_native_station_counts'),(hour,'half_month_start_hours'),(match,'january_matched_changes'),(trans,'january_key_transitions')]:save(frame,'taskA/'+name+'.csv')
 # One long-format SI table, with distinct panel-specific denominators.
 table=[]
 def add(panel,month,row,metric,value,denom,definition):table.append(dict(panel=panel,month=month,row=row,metric=metric,value=value,denominator=denom,definition=definition))
 for _,r in daily.loc[daily.observable_hours.gt(0)].iterrows():add('A_daily_transactions',r.month,r.day+' / '+r.day_definition,'base_valid_sessions',r.base_transactions,r.observable_hours,'All price-base eligible start sessions; may cross intervals; unobserved padding dates omitted')
 for _,r in counts.iterrows():add('B_station_coverage',r.month,r.tariff_day+' / '+r.segment_id,'base_stations',r.base_stations,np.nan,'23:00 tariff day; at least one base-valid start')
 for _,r in hour.iterrows():add('C_start_hour',r.month,r.day_definition+' / '+r.half+' / '+str(r.start_hour),'base_fraction',r.base_fraction,r.base_n,'Hour share of corresponding half-month base-valid starts')
 wanted=match.loc[match.date_after.isin(['2025-01-14','2025-01-16','2025-01-17','2025-01-27'])]
 for _,r in wanted.iterrows():
  for metric in ['E_median_change','T_median_change','E_within003']:
   add('D_matched_station_prices','2025-01',r.date_after+' / '+r.segment_id+' / '+r.scope+' / n>='+str(r.minimum_each_day),metric,r[metric],r.common_stations,'Same station on date_after and preceding day; chronological price difference')
 save(pd.DataFrame(table),'taskA/SI_coverage_table.csv')
 plt=plotting();fig,axs=plt.subplots(2,3,figsize=(15.5,8.4),layout='constrained')
 colors={'2025-01':'#31688e','2025-07':'#c75b12'}
 for m in ['2025-01','2025-07']:
  z=daily.loc[daily.month.eq(m)&daily.day_definition.eq('calendar')&daily.day.str.startswith(m)].sort_values('day');axs[0,0].plot(pd.to_datetime(z.day).dt.day,z.base_transactions/1000,label=m,color=colors[m])
 axs[0,0].set(title='(a) Eligible sessions by calendar day',xlabel='Day of month',ylabel='Sessions (thousands)');axs[0,0].legend(frameon=False)
 for ax,m,label in [(axs[0,1],'2025-01','(b) January station coverage'),(axs[0,2],'2025-07','(c) July station coverage')]:
  for seg,g in counts.loc[counts.month.eq(m)].groupby('segment_id'):
   t=pd.to_datetime(g.tariff_day);keep=t.dt.strftime('%Y-%m').eq(m);ax.plot(t[keep].dt.day,g.loc[keep,'base_stations'],label=seg.replace('_','-'),lw=1)
  ax.set(title=label,xlabel='Tariff day (starts 23:00)',ylabel='Stations with eligible starts');ax.legend(ncol=2,fontsize=7,frameon=False)
 for m in ['2025-01','2025-07']:
  for half,sty in [('formation','-'),('validation','--')]:
   z=hour.loc[hour.month.eq(m)&hour.half.eq(half)&hour.day_definition.eq('frozen_tariff')]
   axs[1,0].plot(z.start_hour,z.base_fraction,label=m+' '+half,color=colors[m],ls=sty)
 axs[1,0].set(title='(d) Start-hour distribution',xlabel='Start hour',ylabel='Share of eligible half-month starts',xticks=[0,6,12,18,23]);axs[1,0].legend(fontsize=7,frameon=False)
 matched=wanted.loc[wanted.scope.eq('contained')&wanted.minimum_each_day.eq(1)]
 if matched.empty:matched=wanted.loc[wanted.scope.eq('base')&wanted.minimum_each_day.eq(1)]
 for ax,c,title in [(axs[1,1],'E_median_change','(e) Matched electricity-price changes'),(axs[1,2],'T_median_change','(f) Matched total-price changes')]:
  for seg,g in matched.groupby('segment_id'):
   ax.plot(pd.to_datetime(g.date_after).dt.day,g[c],marker='o',ms=3,label=seg.replace('_','-'))
  ax.axhline(0,color='black',lw=.7);ax.set(title=title,xlabel='January date vs preceding day',ylabel='Median same-station change (RMB/kWh)',xticks=[14,16,17,27]);ax.legend(ncol=2,fontsize=7,frameon=False)
 fig.savefig(OUT/'figures/A_coverage_and_matched_prices.png',dpi=190);fig.savefig(OUT/'figures/A_coverage_and_matched_prices.pdf');plt.close(fig)
 (OUT/'taskA/SI_coverage_table_notes.md').write_text('# 补充材料覆盖表说明\n\nSI_coverage_table.csv是一张分四Panel的长表；逐日交易、每日原生时段站数、半月小时分布及四个关键日期同站价差都保留，未用一个总平均代替逐日记录。图A的同站价差优先使用contained、每天至少1笔；表另保留base和更高逐日笔数门槛。日历日和tariff day不混合；月末tariff day在原自然月提取范围内截断。图中曲线展示的是有记录覆盖与同站中位价变化，不识别覆盖变化成因。\n',encoding='utf-8')
 return len(wanted)>0

if __name__=='__main__':
 ok,f=crosswindow();dec=dispersion_decomposition();cv=coverage_figure()
 done('A',dict(fixed_V_5846=f.V.sum()==5846,july_V_4886=f.loc[f.month.eq('2025-07'),'V'].sum()==4886,reused_MAE_and_ES_draws_match=ok,decomposition_closes=dec,coverage_table=cv))
