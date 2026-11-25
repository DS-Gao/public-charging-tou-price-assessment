from common import *

STARTS=[0,480,600,660,720,840,900,1020,1140]
LABELS=['00-08','08-10','10-11','11-12','12-14','14-15','15-17','17-19','19-24']
MONTHS=[str(m) for m in pd.period_range('2022-09','2023-02',freq='M')]

def set_label(g,c,kind):
 v=g[c].min() if kind=='min' else g[c].max()
 return '|'.join(str(int(s)) for s in g.loc[(g[c]-v).abs().le(1e-10),'segment'].sort_values())

def main():
 d=csv(FIRST/'task5/daily_station_classification.csv',dtype={'station_id':str},parse_dates=['day']);d['month']=d.day.dt.strftime('%Y-%m')
 sm=csv(FIRST/'task5/station_month_classification.csv',dtype={'station_id':str})
 p=parquet(UE/'monthly_boundary_panel.parquet');p=p.loc[p.variant.eq('base')].copy();p.station_id=p.station_id.astype(str)
 qualified=set(p.loc[p.paired_days.ge(10)&p.E_star.gt(.01),'station_id']);assert len(qualified)==205
 ep=parquet(UE/'daily_endpoints.parquet');ep.station_id=ep.station_id.astype(str);ep['month']=ep.day.dt.strftime('%Y-%m')
 gooddays=d.loc[d.complete_day,['station_id','day']]
 ec=ep.merge(gooddays,on=['station_id','day'],validate='many_to_one')
 eligible=sm.loc[sm.complete_days.ge(10),['station_id','month','complete_days']]
 ec=ec.merge(eligible,on=['station_id','month'],validate='many_to_one')
 profiles=ec.groupby(['station_id','month','segment']).agg(E=('E','median'),T=('T','median'),days=('day','nunique')).reset_index()
 profiles['interval']=profiles.segment.map(dict(enumerate(LABELS)))
 qprof=profiles.loc[profiles.station_id.isin(qualified)]
 save(qprof,'taskD/qualified205_complete_day_profiles.csv')
 menus=[]
 for (sid,m),g in qprof.groupby(['station_id','month']):
  assert len(g)==9 and g.days.min()>=10 and g.days.nunique()==1
  r=dict(station_id=sid,month=m,complete_days=int(g.days.iloc[0]))
  for c in ['E','T']:
   for kind in ['min','max']:
    r[c+'_'+kind+'_set']=set_label(g,c,kind);r[c+'_'+kind+'_intervals']='|'.join(LABELS[int(i)] for i in r[c+'_'+kind+'_set'].split('|'))
  mn=set(r['T_min_set'].split('|'));mx=set(r['T_max_set'].split('|'));shoulder={'1','4','8'}
  r.update(T_min_excludes_00_08='0' not in mn,T_min_not_unique_00_08=mn!={'0'},T_max_any_shoulder=bool(mx&shoulder),T_max_all_shoulder=bool(mx) and mx<=shoulder,T_min_tie=len(mn)>1,T_max_tie=len(mx)>1)
  menus.append(r)
 menus=pd.DataFrame(menus);save(menus,'taskD/qualified205_monthly_minmax_sets.csv')
 xt=[]
 for month,h in [('ALL',menus)]+list(menus.groupby('month')):
  for kind in ['min','max']:
   z=h.groupby(['E_'+kind+'_intervals','T_'+kind+'_intervals']).agg(station_months=('station_id','size'),stations=('station_id','nunique')).reset_index();z.columns=['E_intervals','T_intervals','station_months','stations'];z['extremum']=kind;z['month']=month;xt.append(z)
 save(pd.concat(xt,ignore_index=True),'taskD/electricity_total_extremum_crosstabs.csv')
 # Midnight uses previous calendar day; both days must independently be complete.
 all_complete_ep=ep.merge(gooddays,on=['station_id','day'],validate='many_to_one')
 opp=[]
 for b,leftseg,rightseg,q in [(0,8,0,-1),(720,3,4,-1)]:
  left=all_complete_ep.loc[all_complete_ep.segment.eq(leftseg)&all_complete_ep.station_id.isin(qualified),['station_id','day','E','T']].copy()
  if b==0:left.day+=pd.Timedelta(days=1)
  right=all_complete_ep.loc[all_complete_ep.segment.eq(rightseg)&all_complete_ep.station_id.isin(qualified),['station_id','day','E','T']]
  paired=left.merge(right,on=['station_id','day'],suffixes=('_before','_after'),validate='one_to_one');paired['month']=paired.day.dt.strftime('%Y-%m')
  for (sid,m),g in paired.groupby(['station_id','month']):
   de=g.E_after.median()-g.E_before.median();dt=g.T_after.median()-g.T_before.median();n=len(g)
   opp.append(dict(station_id=sid,month=m,boundary_minute=b,paired_complete_days=n,delta_E=de,delta_T=dt,E_star=q*de,T_star=q*dt,supported=n>=10,opposite_thresholded=n>=10 and abs(de)>.01 and abs(dt)>.03 and de*dt<0,opposite_numeric=n>=10 and abs(de)>1e-10 and abs(dt)>1e-10 and de*dt<0))
 opp=pd.DataFrame(opp);save(opp,'taskD/qualified205_opposite_boundary_directions.csv')
 counts=[]
 for m,h in [('ALL',menus)]+list(menus.groupby('month')):
  for flag in ['T_min_excludes_00_08','T_min_not_unique_00_08','T_max_any_shoulder','T_max_all_shoulder','T_min_tie','T_max_tie']:
   z=h.loc[h[flag]];counts.append(dict(month=m,indicator=flag,station_months=len(z),stations=z.station_id.nunique(),denominator_station_months=len(h),denominator_stations=h.station_id.nunique()))
 for m,h in [('ALL',opp)]+list(opp.groupby('month')):
  for b,z in h.groupby('boundary_minute'):
   supported=z.loc[z.supported]
   for flag in ['opposite_thresholded','opposite_numeric']:
    k=supported.loc[supported[flag]];counts.append(dict(month=m,indicator=f'{flag}_{b:04d}',station_months=len(k),stations=k.station_id.nunique(),denominator_station_months=len(supported),denominator_stations=supported.station_id.nunique()))
 save(pd.DataFrame(counts),'taskD/qualified205_misalignment_counts.csv')
 coverage=sm.loc[sm.station_id.isin(qualified)].copy();coverage['eligible_for_menu']=coverage.complete_days.ge(10);save(coverage,'taskD/qualified205_complete_day_coverage.csv')
 # Identify genuinely flat-E station-months and the stricter six-month station set.
 non=p.loc[~p.station_id.isin(qualified)].copy();r=[]
 for (sid,m),g in non.groupby(['station_id','month']):
  valid=g.loc[g.paired_days.ge(10)];k=int(valid.E_star.abs().gt(.01).sum())
  r.append(dict(station_id=sid,month=m,supported_boundaries=len(valid),nonflat_boundaries=k,flat_all_six=len(valid)==6 and k==0,reason='flat_all_six' if len(valid)==6 and k==0 else 'observed_nonflat' if k else 'insufficient_boundary_support'))
 flat=pd.DataFrame(r);strict=flat.groupby('station_id').flat_all_six.agg(['sum','count']);ids=set(strict.loc[strict['sum'].eq(6)&strict['count'].eq(6)].index)
 flat['strict_six_month_station']=flat.station_id.isin(ids);save(flat,'taskD/nonqualified_flat_electricity_roster.csv')
 ds=d.merge(flat[['station_id','month','flat_all_six','strict_six_month_station']],on=['station_id','month'],how='inner',validate='many_to_one')
 varied=ds.loc[ds.complete_day].groupby(['station_id','month']).agg(complete_days=('day','size'),total_varies_days=('T_range',lambda s:s.gt(.03+1e-12).sum()),median_daily_T_range=('T_range','median'),max_daily_T_range=('T_range','max')).reset_index()
 flat=flat.merge(varied,on=['station_id','month'],how='left',validate='one_to_one');flat['complete_day_eligible']=flat.complete_days.ge(10)
 save(flat,'taskD/flat_electricity_total_variation_by_station_month.csv')
 flatcounts=[];curve=[]
 for pop,cond in [('strict_six_month_stations',flat.strict_six_month_station),('available_flat_station_months',flat.flat_all_six)]:
  selected=flat.loc[cond&flat.complete_day_eligible]
  for m,g in [('ALL',selected)]+list(selected.groupby('month')):
   vary=g.loc[g.total_varies_days.gt(0)]
   flatcounts.append(dict(population=pop,month=m,stations=g.station_id.nunique(),station_months=len(g),stations_any_total_variation=vary.station_id.nunique(),station_months_any_total_variation=len(vary),complete_days=int(g.complete_days.sum()),total_varies_days=int(g.total_varies_days.sum())))
  pr=profiles.merge(selected[['station_id','month','total_varies_days']],on=['station_id','month'],how='inner',validate='many_to_one')
  for subgroup,h in [('all_flat_E',pr),('has_total_variation',pr.loc[pr.total_varies_days.gt(0)])]:
   for m,g in [('ALL',h)]+list(h.groupby('month')):
    if m=='ALL':g=g.groupby(['station_id','segment']).T.median().rename('T').reset_index()
    for seg,k in g.groupby('segment'):curve.append(dict(population=pop,subgroup=subgroup,month=m,segment=seg,interval=LABELS[int(seg)],stations=k.station_id.nunique(),median_T=k['T'].median(),p25_T=k['T'].quantile(.25),p75_T=k['T'].quantile(.75)))
 save(pd.DataFrame(flatcounts),'taskD/flat_electricity_total_variation_counts.csv');curve=pd.DataFrame(curve);save(curve,'taskD/flat_electricity_total_price_curves.csv')
 plt=plotting();fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
 for m,g in curve.loc[curve.population.eq('strict_six_month_stations')&curve.subgroup.eq('all_flat_E')].groupby('month'):
  ax.step(np.r_[STARTS,1440]/60,np.r_[g.sort_values('segment').median_T,g.sort_values('segment').median_T.iloc[-1]],where='post',label=m,lw=2 if m=='ALL' else 1)
 ax.set(xlabel='Hour',ylabel='Released total price (RMB/kWh)',title='Flat-electricity stations: median total-price profiles',xticks=[0,8,10,12,14,19,24]);ax.legend(fontsize=8,ncol=4,frameon=False)
 fig.savefig(OUT/'figures/D_flat_electricity_total_profiles.png',dpi=180);fig.savefig(OUT/'figures/D_flat_electricity_total_profiles.pdf');plt.close(fig)
 done('D',dict(qualified_roster_205=len(qualified)==205,menus_have9_intervals=qprof.groupby(['station_id','month']).size().eq(9).all(),complete_days_min10=menus.complete_days.ge(10).all(),ties_retained=menus.T_min_set.notna().all(),strict_flat_stations_not_205=not(ids&qualified),crosstabs_close=pd.concat(xt).groupby(['month','extremum']).station_months.sum().loc[('ALL','min')]==len(menus)))

if __name__=='__main__':main()
