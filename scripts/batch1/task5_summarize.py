from common import *
MONTHS=[str(m) for m in pd.period_range('2022-09','2023-02',freq='M')]
STARTS=[0,480,600,660,720,840,900,1020,1140]
CATS=['total_fixed_split','both_change','both_flat','other']

def share_bin(v):
 if not np.isfinite(v):return 'no_complete_days'
 if v==0:return '0'
 if v<=.5:return '(0,50%]'
 if v<.9:return '(50%,90%)'
 if v<1:return '[90%,100%)'
 return '100%'

def summarize():
 d=pd.read_csv(OUT/'task5/daily_station_classification.csv',dtype={'station_id':str},parse_dates=['day']);d['month']=d.day.dt.strftime('%Y-%m')
 p=parquet(UE/'monthly_boundary_panel.parquet');p=p.loc[p.variant.eq('base')].copy();p.station_id=p.station_id.astype(str)
 r=p.loc[p.paired_days.ge(10)&p.E_star.gt(.01)];qualified=set(r.station_id)
 ep=parquet(UE/'daily_endpoints.parquet');ep.station_id=ep.station_id.astype(str);ep['month']=ep.day.dt.strftime('%Y-%m')
 oldcomplete=ep.loc[ep.supported].groupby(['station_id','day']).size().eq(9).rename('old_complete')
 check=d.merge(oldcomplete,on=['station_id','day'],how='left');check['old_complete']=check.old_complete.eq(True)
 coverage_mismatch=int((check.complete_day!=check.old_complete).sum())
 save(check.loc[check.complete_day!=check.old_complete],'validation/urbanev_complete_day_mismatches.csv')
 monthly=[]
 for (sid,m),g in d.groupby(['station_id','month']):
  h=g.loc[g.complete_day];n=len(h);row=dict(station_id=sid,month=m,qualified_205=sid in qualified,complete_days=n,calendar_days=len(g))
  for cat in CATS:row[cat+'_days']=int(h.daily_category.eq(cat).sum());row[cat+'_fraction']=h.daily_category.eq(cat).mean() if n else np.nan
  if n<10:strict=major='insufficient_data'
  else:
   strict=next((cat for cat in CATS if row[cat+'_days']==n),'other')
   major=next((cat for cat in CATS if row[cat+'_days']>n/2),'other')
  row['strict_category']=strict;row['majority_category']=major;row['fixed_split_bin']=share_bin(row['total_fixed_split_fraction']);monthly.append(row)
 monthly=pd.DataFrame(monthly);save(monthly,'task5/station_month_classification.csv');stations=[]
 for sid,g in monthly.groupby('station_id'):
  row=dict(station_id=sid,qualified_205=sid in qualified,complete_days=int(g.complete_days.sum()),months_at_least10=int(g.complete_days.ge(10).sum()))
  for rule in ['strict','majority']:
   values=g[rule+'_category'];row[rule+'_category']=values.iloc[0] if values.nunique()==1 else 'cross_month_change'
   for m in MONTHS:row[rule+'_'+m]=g.loc[g.month.eq(m),rule+'_category'].iloc[0]
  row['fixed_split_days']=int(g.total_fixed_split_days.sum());row['fixed_split_fraction']=row['fixed_split_days']/row['complete_days'] if row['complete_days'] else np.nan
  row['fixed_split_bin']=share_bin(row['fixed_split_fraction']);stations.append(row)
 stations=pd.DataFrame(stations);save(stations,'task5/station_classification.csv');counts=[];bins=[]
 for level,frame in [('station',stations),('station_month',monthly)]:
  for pop,h in [('analysis_1652',frame),('qualified_205',frame.loc[frame.qualified_205])]:
   for rule in ['strict','majority']:
    groups=[('ALL',h)] if level=='station' else [('ALL_MONTHS',h)]+list(h.groupby('month'))
    for month,k in groups:
     for cat,n in k[rule+'_category'].value_counts().items():counts.append(dict(level=level,population=pop,rule=rule,month=month,category=cat,count=int(n),denominator=len(k),fraction=n/len(k)))
   if pop=='qualified_205':
    for cat,n in h.fixed_split_bin.value_counts().items():bins.append(dict(level=level,eligibility='all_complete_days_including_months_below10',bin=cat,count=int(n),denominator=len(h)))
    if level=='station_month':
     h=h.loc[h.complete_days.ge(10)]
     for cat,n in h.fixed_split_bin.value_counts().items():bins.append(dict(level=level,eligibility='at_least10_complete_days',bin=cat,count=int(n),denominator=len(h)))
 save(pd.DataFrame(counts),'task5/classification_counts.csv');save(pd.DataFrame(bins),'task5/qualified205_fixed_split_bins.csv')
 # Profiles: each station is reduced across supported days before cross-station aggregation.
 e=ep.loc[ep.supported&ep.station_id.isin(qualified)].copy()
 sm=e.groupby(['station_id','month','segment']).agg(days=('day','nunique'),E=('E','median'),S=('S','median'),T=('T','median')).reset_index()
 sm['profile_supported']=sm.days.ge(10);save(sm,'task5/qualified205_station_month_profiles.csv');profiles=[]
 for (m,seg),g in sm.loc[sm.profile_supported].groupby(['month','segment']):profiles.append(dict(month=m,segment=seg,start_minute=STARTS[seg],stations=len(g),E=g.E.median(),S=g.S.median(),T=g['T'].median()))
 # All-period curve retains station-equal, not station-day pooled, weights.
 st=e.groupby(['station_id','segment']).agg(days=('day','nunique'),E=('E','median'),S=('S','median'),T=('T','median')).reset_index()
 for seg,g in st.loc[st.days.ge(10)].groupby('segment'):profiles.append(dict(month='ALL',segment=seg,start_minute=STARTS[seg],stations=len(g),E=g.E.median(),S=g.S.median(),T=g['T'].median()))
 profiles=pd.DataFrame(profiles);save(profiles,'task5/qualified205_native_profiles.csv')
 transitions=[];unittrans=[]
 for seg,start in enumerate(STARTS):
  l=e.loc[e.segment.eq((seg-1)%9),['station_id','day','E','S','T']].copy();rr=e.loc[e.segment.eq(seg),['station_id','day','E','S','T']].copy()
  if seg==0:l['day']=l.day+pd.Timedelta(days=1)
  j=l.merge(rr,on=['station_id','day'],suffixes=('_L','_R'));j['month']=j.day.dt.strftime('%Y-%m');j=j.loc[j.month.isin(MONTHS)]
  agg={x+'_'+side:(x+'_'+side,'median') for x in ['E','S','T'] for side in ['L','R']};agg['paired_days']=('day','nunique')
  q=j.groupby(['station_id','month']).agg(**agg).reset_index();q=q.loc[q.paired_days.ge(10)].copy();q['boundary_minute']=start
  for x in ['E','S','T']:q[x+'_after_minus_before']=q[x+'_R']-q[x+'_L']
  unittrans.append(q)
  for m,h in q.groupby('month'):
   transitions.append(dict(month=m,boundary_minute=start,official_boundary=start in [0,480,600,720,840,1140],stations=len(h),paired_median_T_change=h.T_after_minus_before.median(),T_jump_stations=int(h.T_after_minus_before.abs().ge(.03).sum()),T_jump_share=h.T_after_minus_before.abs().ge(.03).mean(),E_median_change=h.E_after_minus_before.median(),S_median_change=h.S_after_minus_before.median()))
 save(pd.concat(unittrans,ignore_index=True),'task5/native_boundary_station_changes.csv');save(pd.DataFrame(transitions),'task5/native_boundary_jump_summary.csv')
 profdiff=[]
 for m,g in profiles.groupby('month'):
  g=g.sort_values('segment').copy();g['T_curve_step']=g['T']-np.roll(g['T'],1);g['step_absolute_ge003']=g.T_curve_step.abs().ge(.03);g['official_boundary']=g.start_minute.isin([0,480,600,720,840,1140]);profdiff.append(g)
 save(pd.concat(profdiff,ignore_index=True),'task5/crossstation_profile_steps.csv')
 # 12:00 original R uses 11-12 before; requested 10-12 is a separate diagnostic.
 twelve=r.loc[r.boundary_minute.eq(720)].copy();save(twelve,'task5/original_12_qualified_units.csv')
 w=pd.read_csv(OUT/'task5/daily_10_12_vs_12_14.csv',dtype={'station_id':str},parse_dates=['day']);w=w.loc[w.supported_both_windows].copy();w['month']=w.day.dt.strftime('%Y-%m')
 agg=w.groupby(['station_id','month','window']).agg(days=('day','nunique'),E=('E','median'),S=('S','median'),T=('T','median')).reset_index()
 l=agg.loc[agg.window.eq('10_12')].drop(columns='window');rr=agg.loc[agg.window.eq('12_14')].drop(columns='window')
 j=twelve.merge(l.merge(rr,on=['station_id','month'],suffixes=('_10_12','_12_14')),on=['station_id','month'],how='left')
 j['requested_window_supported']=j.days_10_12.ge(10)&j.days_12_14.ge(10)
 for x in ['E','S','T']:j[x+'_requested_star']=j[x+'_10_12']-j[x+'_12_14']
 j['requested_residual']=j.T_requested_star-j.E_requested_star-j.S_requested_star
 save(j,'task5/qualified12_requested_window_prices.csv');tw=[]
 for m,g in j.groupby('month'):
  h=g.loc[g.requested_window_supported];row=dict(month=m,original_R=len(g),original_retained=int(g.T_star.gt(.03).sum()),original_neutral=int(g.T_star.between(-.03,.03).sum()),original_reversed=int(g.T_star.lt(-.03).sum()),requested_supported=len(h),requested_reversed=int(h.T_requested_star.lt(-.03).sum()))
  for x in ['E','S','T']:
   for side in ['left','right']:row[x+'_original_'+side+'_crossstation_median']=g[x+'_'+side].median()
   for window in ['10_12','12_14']:row[x+'_'+window+'_crossstation_median']=h[x+'_'+window].median()
   row[x+'_requested_star_median']=h[x+'_requested_star'].median()
  row['requested_residual_abs_max']=h.requested_residual.abs().max();tw.append(row)
 save(pd.DataFrame(tw),'task5/qualified12_monthly_summary.csv')
 a=p.loc[p.paired_days.ge(10)];premise=[]
 for label,mask in [('exact_near_zero_both_1e10',a.E_star.abs().le(1e-10)&a.T_star.abs().le(1e-10)),('both_neutral_003',a.E_star.abs().le(.03)&a.T_star.abs().le(.03)),('weak_E001_neutral_T003',a.E_star.abs().le(.01)&a.T_star.abs().le(.03))]:premise.append(dict(definition=label,count=int(mask.sum()),A_denominator=len(a),fraction=mask.mean()))
 save(pd.DataFrame(premise),'task5/attachment_893_percent_definition_audit.csv')
 plt=plotting();fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
 for ax,m in zip(axes.flat,MONTHS):
  g=profiles.loc[profiles.month.eq(m)].sort_values('segment')
  for x in ['E','S','T']:ax.step(np.r_[g.start_minute/60,24],np.r_[g[x],g[x].iloc[-1]],where='post',label=x)
  ax.set_title(m);ax.set_xticks([0,8,10,12,14,19,24]);ax.grid(alpha=.2);ax.legend()
 fig.supxlabel('Local hour; station-equal median of supported native-period prices');fig.supylabel('Released RMB/kWh');fig.tight_layout();fig.savefig(OUT/'figures/urbanev_205_profiles.png',dpi=180);fig.savefig(OUT/'figures/urbanev_205_profiles.pdf');plt.close(fig)
 done(5,dict(analysis_stations=len(stations)==1652,qualified_stations=len(qualified)==205,station_months=len(monthly)==1652*6,complete_day_matches_frozen=coverage_mismatch==0,original_R=len(r)==4921,original_A=len(a)==49369,strict_and_majority_count_closure=len(stations.loc[stations.qualified_205])==205))
 print(pd.DataFrame(counts).query("level=='station' and population=='qualified_205'").to_string(index=False))

if __name__=='__main__':summarize()
