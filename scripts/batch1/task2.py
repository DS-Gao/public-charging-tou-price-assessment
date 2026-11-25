from common import *

def task2():
 assert (OUT/'validation/task1.json').exists()
 cfg,layer=layers();panel=parquet(OLD/'cache/boundary_prices.parquet')
 cols=['n_L','n_R','days_L','days_R','support_L','support_R','support']
 full=panel.loc[panel.period.eq('fullmonth')].set_index(KEY)
 form=panel.loc[panel.period.eq('formation')].set_index(KEY)
 p=full[cols].join(form[cols],how='outer',lsuffix='_full',rsuffix='_formation').reset_index()
 af=p.loc[p.support_formation.fillna(False)&~p.support_full.fillna(False)]
 ex=[]
 for m,b,n in [('2025-01',7,2),('2025-01',10,1),('2025-07',7,1),('2025-07',10,1)]:ex.append(af.loc[af.month.eq(m)&af.boundary_hour.eq(b)].sort_values('station_id').head(n))
 examples=pd.concat(ex,ignore_index=True);assert len(examples)==5
 examples['reason']='Formation passes 10 transactions/3 tariff dates on both sides; full-month fails 20/5 on at least one side.'
 save(examples,'task2/Af_not_A_five_examples.csv')
 cross=[]
 for (m,b),g in p.groupby(['month','boundary_hour']):
  aa=g.support_full.fillna(False);ff=g.support_formation.fillna(False)
  cross.append(dict(month=m,boundary_hour=b,A=int(aa.sum()),Af=int(ff.sum()),Af_not_A=int((ff&~aa).sum()),A_not_Af=int((aa&~ff).sum()),intersection=int((aa&ff).sum())))
 cross=pd.DataFrame(cross);save(cross,'task2/membership_crosswalk.csv')
 for n in ['jan_formation_validation_availability.csv','jan_qualification_transitions.csv','jan_Rf_unobservable_endpoint_causes.csv','report_daily_activity_by_cohort.csv']:
  shutil.copy2(source(OLD/'task2'/n,'Reuse January trace evidence'),OUT/'task2'/('reused_'+n))
 c=database();c.register('layers',layer)
 print('Aggregating daily endpoints',flush=True)
 c.execute('''CREATE TEMP TABLE daily_ep AS SELECT month,tariff_day,station_id,segment_id,
 count(*) AS n_base,count(*) FILTER(WHERE contained) AS n_contained,
 median(E) AS E_base,median(T) AS T_base,
 median(E) FILTER(WHERE contained) AS E_contained,median(T) FILTER(WHERE contained) AS T_contained
 FROM tx WHERE base_valid GROUP BY ALL''')
 ep=c.execute('SELECT * FROM daily_ep ORDER BY ALL').df();save(ep,'task2/daily_station_endpoints.csv')
 u=c.execute('''WITH l AS (SELECT d.*,b.boundary_hour,b.q,
 d.tariff_day+CASE WHEN b.boundary_hour=23 THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END AS anchor_day
 FROM daily_ep d JOIN layers b ON d.month=b.month AND d.segment_id=b.segment_L),
 r AS (SELECT d.*,b.boundary_hour,b.q,d.tariff_day AS anchor_day
 FROM daily_ep d JOIN layers b ON d.month=b.month AND d.segment_id=b.segment_R)
 SELECT coalesce(l.month,r.month) AS month,coalesce(l.boundary_hour,r.boundary_hour) AS boundary_hour,
 coalesce(l.station_id,r.station_id) AS station_id,coalesce(l.anchor_day,r.anchor_day) AS tariff_day,
 coalesce(l.q,r.q) AS q,coalesce(l.n_base,0) AS n_base_L,coalesce(r.n_base,0) AS n_base_R,
 coalesce(l.n_contained,0) AS n_contained_L,coalesce(r.n_contained,0) AS n_contained_R,
 l.E_base AS E_base_L,r.E_base AS E_base_R,l.T_base AS T_base_L,r.T_base AS T_base_R,
 l.E_contained AS E_contained_L,r.E_contained AS E_contained_R,
 l.T_contained AS T_contained_L,r.T_contained AS T_contained_R
 FROM l FULL OUTER JOIN r ON l.month=r.month AND l.boundary_hour=r.boundary_hour
 AND l.station_id=r.station_id AND l.anchor_day=r.anchor_day''').df()
 u=u.loc[u.tariff_day.dt.strftime('%Y-%m').eq(u.month)].sort_values(['month','boundary_hour','tariff_day','station_id'])
 save(u,'task2/daily_boundary_station_evidence.csv');rows=[]
 lookup={k:g for k,g in u.groupby(['month','boundary_hour','tariff_day'])}
 for b in layer.itertuples():
  native={s[0]:s for s in cfg['native_8_segments'][b.month]};ms=pd.Timestamp(b.month+'-01');me=ms+pd.offsets.MonthBegin(1)
  for day in pd.date_range(ms,me-pd.Timedelta(days=1)):
   g=lookup.get((b.month,b.boundary_hour,day),u.iloc[:0]);exposure={}
   for side in ['L','R']:
    _,start,end,dur=native[getattr(b,'segment_'+side)]
    anchor=day-pd.Timedelta(days=1) if b.boundary_hour==23 and side=='L' else day
    st=anchor+pd.Timedelta(hours=start,days=0 if start==23 else 1);en=st+pd.Timedelta(hours=dur)
    hours=max(0,(min(en,me)-max(st,ms)).total_seconds()/3600)
    exposure['observed_hours_'+side]=hours;exposure['complete_window_'+side]=hours==dur
   for scope in ['base','contained']:
    for minimum in [1,4]:
     h=g.loc[g['n_'+scope+'_L'].ge(minimum)&g['n_'+scope+'_R'].ge(minimum)]
     de=h.q*(h['E_'+scope+'_R']-h['E_'+scope+'_L']);dt=h.q*(h['T_'+scope+'_R']-h['T_'+scope+'_L'])
     row=dict(month=b.month,boundary_hour=b.boundary_hour,tariff_day=day,scope=scope,minimum_each_side=minimum,paired_stations=len(h),E_positive_n=int((de>.03).sum()),T_positive_n=int((dt>.03).sum()),E_positive_share=(de>.03).mean() if len(h) else np.nan,T_positive_share=(dt>.03).mean() if len(h) else np.nan,**exposure)
     for side in ['L','R']:
      for scope2 in ['base','contained']:row['all_'+scope2+'_transactions_'+side]=int(g['n_'+scope2+'_'+side].sum())
      row['E_crossstation_median_'+side]=g['E_'+scope+'_'+side].median()
      row['E_paired_crossstation_median_'+side]=h['E_'+scope+'_'+side].median()
     rows.append(row)
 daily=pd.DataFrame(rows);save(daily,'task2/daily_boundary_summary.csv')
 native=[];e=ep.loc[ep.month.eq('2025-01')]
 for day in pd.date_range('2025-01-01','2025-01-31'):
  for seg,st,en,dur in cfg['native_8_segments']['2025-01']:
   g=e.loc[e.tariff_day.eq(day)&e.segment_id.eq(seg)]
   for scope in ['base','contained']:
    h=g.loc[g['n_'+scope]>0];native.append(dict(tariff_day=day,segment_id=seg,scope=scope,stations=len(h),transactions=h['n_'+scope].sum(),E_crossstation_median=h['E_'+scope].median(),extract_edge_incomplete=day.day==31))
 save(pd.DataFrame(native),'task2/january_native_daily.csv')
 activity=[]
 for mode,expr in [('calendar','CAST(charge_start_time AS DATE)'),('tariff','tariff_day')]:
  z=c.execute(f'''SELECT month,{expr} AS day,count(*) AS raw_transactions,count(*) FILTER(WHERE base_valid) AS base_transactions,count(*) FILTER(WHERE base_valid AND contained) AS contained_transactions FROM tx GROUP BY ALL ORDER BY month,day''').df()
  z['day_definition']=mode;exposure=[]
  for r in z.itertuples():
   ms=pd.Timestamp(r.month+'-01');me=ms+pd.offsets.MonthBegin(1);st=pd.Timestamp(r.day)+pd.Timedelta(hours=23 if mode=='tariff' else 0)
   exposure.append(max(0,(min(st+pd.Timedelta(days=1),me)-max(st,ms)).total_seconds()/3600))
  z['observable_hours']=exposure;activity.append(z)
 activity=pd.concat(activity,ignore_index=True);save(activity,'task2/daily_activity.csv');halves=[]
 for m in ['2025-01','2025-07']:
  for mode,scheme in [('calendar','natural_1_15_vs_16_end'),('tariff','frozen_1_15_vs_16_end'),('tariff','equal_complete_1_15_vs_16_30')]:
   g=activity.loc[activity.month.eq(m)&activity.day_definition.eq(mode)&activity.day.dt.strftime('%Y-%m').eq(m)].copy()
   if scheme.startswith('equal'):g=g.loc[g.day.dt.day.le(30)]
   for name,mask in [('first',g.day.dt.day.le(15)),('second',g.day.dt.day.ge(16))]:
    h=g.loc[mask];halves.append(dict(month=m,scheme=scheme,half=name,observed_days=len(h),observable_hours=h.observable_hours.sum(),base_transactions=h.base_transactions.sum(),contained_transactions=h.contained_transactions.sum(),base_per_exposed_day=h.base_transactions.sum()/(h.observable_hours.sum()/24)))
 halves=pd.DataFrame(halves);save(halves,'task2/half_month_activity.csv')
 changes=halves.loc[halves.half.eq('first')].merge(halves.loc[halves.half.eq('second')],on=['month','scheme'],suffixes=('_first','_second'))
 changes['total_reduction']=1-changes.base_transactions_second/changes.base_transactions_first
 changes['exposure_adjusted_reduction']=1-changes.base_per_exposed_day_second/changes.base_per_exposed_day_first
 save(changes,'task2/second_half_activity_change.csv')
 audit=c.execute('''SELECT month,CAST(charge_start_time AS DATE) AS day,count(*) AS records,
 count(*) FILTER(WHERE charge_start_time IS NULL) AS missing_start,count(*) FILTER(WHERE charge_end_time IS NULL) AS missing_end,
 count(*) FILTER(WHERE charge_end_time<charge_start_time) AS end_before_start,
 count(*) FILTER(WHERE abs(epoch(charge_end_time-charge_start_time)/60-duration_min)>0.017) AS duration_mismatch_over_1second,
 max(abs(epoch(charge_end_time-charge_start_time)/60-duration_min)) AS duration_max_gap_minutes,
 count(*) FILTER(WHERE CAST(charge_start_time AS DATE)<>transaction_date) AS transaction_date_differs_from_start,
 count(*) FILTER(WHERE tariff_day<>CAST(charge_start_time-INTERVAL 23 HOUR AS DATE)) AS tariff_rule_mismatch,
 min(charge_start_time) AS first_start,max(charge_start_time) AS last_start
 FROM tx GROUP BY ALL ORDER BY month,day''').df()
 save(audit,'task2/timestamp_daily_audit.csv')
 offsets=c.execute('''SELECT month,CAST(charge_start_time AS DATE) AS day,date_diff('day',CAST(charge_start_time AS DATE),transaction_date) AS transaction_date_offset_days,count(*) AS records FROM tx GROUP BY ALL ORDER BY ALL''').df()
 save(offsets,'task2/transaction_date_offsets.csv')
 hist=[]
 for mode,expr in [('frozen_tariff','half'),('natural_calendar',"CASE WHEN day(charge_start_time)<=15 THEN 'formation' ELSE 'validation' END")]:
  z=c.execute(f'''SELECT month,{expr} AS half,hour(charge_start_time) AS start_hour,count(*) AS raw_n,count(*) FILTER(WHERE base_valid) AS base_n,count(*) FILTER(WHERE base_valid AND contained) AS contained_n FROM tx GROUP BY ALL ORDER BY ALL''').df();z['day_definition']=mode
  z['base_fraction']=z.base_n/z.groupby(['month','half']).base_n.transform('sum');hist.append(z)
 hist=pd.concat(hist,ignore_index=True);save(hist,'task2/start_hour_histogram.csv')
 switch=[];paired_switch=[]
 for (m,b,scope,n),g in daily.groupby(['month','boundary_hour','scope','minimum_each_side']):
  g=g.sort_values('tariff_day');g=g.loc[g.complete_window_L&g.complete_window_R]
  for i in range(1,len(g)):
   prev=g.iloc[i-1];now=g.iloc[i];d0=prev.tariff_day;d1=now.tariff_day
   h0=lookup.get((m,b,d0),u.iloc[:0]);h1=lookup.get((m,b,d1),u.iloc[:0])
   h0=h0.loc[h0['n_'+scope+'_L'].ge(n)&h0['n_'+scope+'_R'].ge(n)].set_index('station_id')
   h1=h1.loc[h1['n_'+scope+'_L'].ge(n)&h1['n_'+scope+'_R'].ge(n)].set_index('station_id')
   common=h0.index.intersection(h1.index);de0=h0.q*(h0['E_'+scope+'_R']-h0['E_'+scope+'_L']);de1=h1.q*(h1['E_'+scope+'_R']-h1['E_'+scope+'_L'])
   switch.append(dict(month=m,boundary_hour=b,scope=scope,minimum_each_side=n,date_before=d0,date_after=d1,N_before=len(h0),N_after=len(h1),share_before=prev.E_positive_share,share_after=now.E_positive_share,share_change=now.E_positive_share-prev.E_positive_share,common_stations=len(common),common_positive_to_nonpositive=int(((de0.loc[common]>.03)&(de1.loc[common]<=.03)).sum()),common_nonpositive_to_positive=int(((de0.loc[common]<=.03)&(de1.loc[common]>.03)).sum()),common_median_E_change=(de1.loc[common]-de0.loc[common]).median(),at_half_split=d1.day==16))
 switch=pd.DataFrame(switch);save(switch,'task2/all_daily_transition_candidates.csv')
 rank=switch.loc[switch.month.eq('2025-01')&switch.boundary_hour.isin([7,13])].copy();rank['abs_change']=rank.share_change.abs()
 save(rank.sort_values('abs_change',ascending=False).groupby(['boundary_hour','scope','minimum_each_side']).head(5),'task2/january_largest_share_changes.csv')
 c.close()
 finish_task2()

def finish_task2():
 daily=pd.read_csv(OUT/'task2/daily_boundary_summary.csv',parse_dates=['tariff_day'])
 hist=pd.read_csv(OUT/'task2/start_hour_histogram.csv')
 cross=pd.read_csv(OUT/'task2/membership_crosswalk.csv')
 audit=pd.read_csv(OUT/'task2/timestamp_daily_audit.csv')
 changes=pd.read_csv(OUT/'task2/second_half_activity_change.csv')
 plt=plotting()
 for b in [7,10,13]:
  fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
  for scope,style in [('base','--'),('contained','-')]:
   g=daily.loc[daily.month.eq('2025-01')&daily.boundary_hour.eq(b)&daily.scope.eq(scope)&daily.minimum_each_side.eq(1)].copy()
   g.loc[~(g.complete_window_L&g.complete_window_R),['E_positive_share','T_positive_share']]=np.nan
   x=g.tariff_day.dt.day
   ax[0].plot(x,g.paired_stations,style,label=scope+' paired stations')
   ax[1].plot(x,g.E_positive_share,style,label=scope+' oriented E > .03')
   ax[2].plot(x,g.E_crossstation_median_L,style,label=scope+' before')
   ax[2].plot(x,g.E_crossstation_median_R,style,label=scope+' after')
  for a in ax:a.axvline(15.5,color='grey',lw=1);a.axvspan(30.5,31.5,color='grey',alpha=.15);a.legend(fontsize=8);a.grid(alpha=.2)
  ax[0].set_ylabel('Stations');ax[1].set_ylabel('Share');ax[2].set_ylabel('RMB/kWh');ax[2].set_xlabel('January tariff date (starts 23:00)')
  fig.suptitle(f'January {b:02}:00: observed daily evidence; shaded month-end is incomplete')
  fig.tight_layout();fig.savefig(OUT/f'figures/january_{b:02}_daily.png',dpi=180);fig.savefig(OUT/f'figures/january_{b:02}_daily.pdf');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for i,mode in enumerate(['frozen_tariff','natural_calendar']):
  for half,offset in [('formation',-.2),('validation',.2)]:
   g=hist.loc[hist.month.eq('2025-01')&hist.day_definition.eq(mode)&hist.half.eq(half)]
   ax[i].bar(g.start_hour+offset,g.base_fraction,width=.4,label=half)
  ax[i].set_title(mode);ax[i].set_xlabel('Start hour, local clock');ax[i].legend()
 ax[0].set_ylabel('Fraction of eligible transactions');fig.tight_layout();fig.savefig(OUT/'figures/january_hour_histogram.png',dpi=180);plt.close(fig)
 status=dict(identified_code_error=False,identified_timestamp_format_error=False,upstream_timezone_independently_verified=False,edge_truncation_present=True,january_provisional=True,reason='Released natural-month extraction truncates tariff-day edges; sharp observed membership/activity shifts have unresolved source. Conservative provisional label for new January tasks3/4; no frozen-code error identified.')
 (OUT/'task2/january_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 done(2,dict(non_nested_534=int(cross.Af_not_A.sum())==534,Jan205=int(cross.loc[cross.month.eq('2025-01'),'Af_not_A'].sum())==205,July329=int(cross.loc[cross.month.eq('2025-07'),'Af_not_A'].sum())==329,rows_1240=len(daily)==1240,tariff_implementation_matches=int(audit.tariff_rule_mismatch.sum())==0))
 print(changes[['month','scheme','total_reduction','exposure_adjusted_reduction']].to_string(index=False))

if __name__=='__main__':
 if '--finish-only' in sys.argv:finish_task2()
 else:task2()
