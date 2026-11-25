from common import *

def power_class(s):
 return pd.Series(np.select([s.lt(30),s.lt(120),s.ge(120)],['low_lt30','medium_30_lt120','high_ge120'],default='Missing'),index=s.index)

def scene_map(d):
 # One vote per station (not per boundary or transaction).
 x=d[['station_id','construction_site']].drop_duplicates().copy();x.construction_site=x.construction_site.fillna('Missing')
 counts=x.groupby('construction_site').station_id.nunique().rename('stations').reset_index().sort_values(['stations','construction_site'],ascending=[False,True])
 names=counts.construction_site.tolist()
 keep=names if len(names)<=8 else names[:7]
 return {k:k if k in keep else 'Other' for k in names},counts

def qgroups(x,n):
 vals=x.dropna();edges=vals.quantile(np.arange(1,n)/n).to_numpy()
 def apply(v):return pd.Series(np.where(v.notna(),'Q'+(np.searchsorted(edges,v.to_numpy(),side='left')+1).astype(str),'Missing'),index=v.index)
 return apply,edges

def features():
 meta=csv(FIRST/'task1/station_month_membership.csv')
 c=database()
 dyn=c.execute('''SELECT month,station_id,max(rated_gun_power_kw) FILTER(WHERE rated_gun_power_kw>0) AS observed_max_power,
   count(*) FILTER(WHERE half='formation') AS formation_transactions,
   max(rated_gun_power_kw) FILTER(WHERE half='formation' AND rated_gun_power_kw>0) AS formation_max_power,
   count(*) AS full_base_sessions FROM tx WHERE base_valid GROUP BY month,station_id''').df();c.close()
 d=meta.merge(dyn,on=['month','station_id'],how='left',validate='one_to_one')
 # No imputation of an unobserved feature; zero counts are actual no-session station-months.
 d['power_class']=power_class(d.observed_max_power);d['formation_power_class']=power_class(d.formation_max_power)
 d['dc_share']=d.dc_pile_fraction;d['log_piles']=np.log1p(d.piles_num.where(d.piles_num.ge(0)))
 d['log_station_power']=np.log1p(d.station_total_power_kw.where(d.station_total_power_kw.ge(0)))
 d.construction_site=d.construction_site.fillna('Missing')
 save(d,'cache/beijing_station_month_features.csv')
 return d

def lpm_models(d):
 import statsmodels.api as sm
 import patsy
 from scipy.linalg import qr
 rows=[];models=[];dropped=[]
 for scope,g in group_scopes(d):
  for outcome in ['large_D30_H','rho_outside']:
   h=g.loc[g[outcome].notna()].copy()
   terms=['dc_share','log_piles','log_station_power','C(power_class)','C(scene_group)','C(volume_group)']
   h['January']=h.month.eq('2025-01').astype(int)
   formula=outcome+' ~ C(layer_id) + '+' + '.join(terms)
   numeric=['dc_share','log_piles','log_station_power']
   h=h.loc[h[numeric].notna().all(axis=1)&h.power_class.ne('Missing')&h.volume_group.ne('Missing')&h.construction_site.ne('Missing')]
   y,X=patsy.dmatrices(formula,h,return_type='dataframe')
   if scope=='ALL':
    # Explicit treatment-coded feature interactions; month intercept is already in layer FE.
    feature_cols=[c for c in X.columns if c!='Intercept' and not c.startswith('C(layer_id)')]
    for col in feature_cols:X['January:'+col]=X[col]*h.loc[X.index,'January']
    formula+=' + January * (each treatment-coded station-feature column; no redundant month intercept)'
   # Keep all layer FE first. Only truly absent/redundant feature contrasts can be removed.
   keep=[];rank=0
   for i,col in enumerate(X.columns):
    proposed=X.iloc[:,keep+[i]].to_numpy();newrank=np.linalg.matrix_rank(proposed)
    if newrank>rank:keep.append(i);rank=newrank
   excluded=[v for i,v in enumerate(X.columns) if i not in keep]
   for term in excluded:dropped.append(dict(scope=scope,outcome=outcome,term=term,reason='Structural zero or exact linear dependence; coefficient not separately identifiable'))
   X=X.iloc[:,keep];h=h.loc[X.index]
   fit=sm.OLS(y.to_numpy().ravel(),X).fit(cov_type='cluster',cov_kwds={'groups':h.station_id,'use_correction':True,'df_correction':True},use_t=True)
   meta=dict(scope=scope,outcome=outcome,model='LPM',n=len(h),stations=h.station_id.nunique(),r_squared=fit.rsquared,pseudo_r_squared=np.nan,pseudo_r_squared_note='Not applicable to LPM; actual R-squared reported',parameters=X.shape[1],rank=int(np.linalg.matrix_rank(X)),dropped_terms=' | '.join(excluded),formula=formula,events=int(h[outcome].sum()),status='ok')
   models.append(meta);ci=fit.conf_int()
   for term in X.columns:
    rows.append(dict(scope=scope,outcome=outcome,term=term,coefficient=fit.params[term],cluster_se=fit.bse[term],t=fit.tvalues[term],p_value=fit.pvalues[term],ci_low=ci.loc[term,0],ci_high=ci.loc[term,1],n=len(h),stations=h.station_id.nunique(),r_squared=fit.rsquared))
 save(pd.DataFrame(rows),'taskB/lpm_coefficients.csv');save(pd.DataFrame(models),'taskB/lpm_models.csv')
 save(pd.DataFrame(dropped,columns=['scope','outcome','term','reason']),'taskB/nonidentifiable_terms.csv')
 return pd.DataFrame(models)

def main():
 feat=features();a=csv(OLD/'task1/fixed_A_unit_prices.csv');r=a.loc[a.R_median].copy()
 d=r.merge(feat,on=['month','station_id'],how='left',validate='many_to_one')
 assert len(d)==len(r)
 d['rho']=d.T_median/d.E_median;d['abs_rho_minus_one']=abs(d.rho-1)
 d['large_D30_H']=np.where(d.H_median,(d.D30_median.abs()>.9).astype(float),np.nan)
 d['rho_outside']=((d.rho<.5)|(d.rho>1.5)).astype(int)
 d['layer_id']=d.month+'_'+d.boundary_hour.astype(str)
 mapping,freq=scene_map(d);d['scene_group']=d.construction_site.map(mapping)
 save(pd.DataFrame([dict(original=k,merged=v) for k,v in mapping.items()]),'taskB/scene_mapping.csv')
 freqrows=[]
 for pop,frame in [('all_metadata',feat),('R',d)]:
  for scope,g in group_scopes(frame):
   z=g.groupby('construction_site').agg(rows=('station_id','size'),stations=('station_id','nunique')).reset_index();z['population']=pop;z['month']=scope;freqrows.append(z)
 save(pd.concat(freqrows,ignore_index=True),'taskB/all_scene_frequencies.csv')
 edges=[]
 for month,g in d.groupby('month'):
  stat=g.drop_duplicates('station_id')
  for col,new in [('piles_num','pile_group'),('station_total_power_kw','station_power_group'),('base_transactions','volume_group')]:
   fun,cuts=qgroups(stat[col],4);d.loc[g.index,new]=fun(g[col]);edges.append(dict(month=month,feature=col,reference='unique R stations',n_stations=len(stat),q25=cuts[0],q50=cuts[1],q75=cuts[2]))
 d['dc_group']=np.select([d.dc_share.eq(0),d.dc_share.gt(0)&d.dc_share.le(.5),d.dc_share.gt(.5)&d.dc_share.lt(1),d.dc_share.eq(1)],['0','(0,0.5]','(0.5,1)','1'],default='Missing')
 save(pd.DataFrame(edges),'taskB/descriptive_quantile_thresholds.csv')
 save(d,'taskB/R_unit_features_and_outcomes.csv')
 desc=[]
 for scope,g in group_scopes(d):
  for col in ['dc_group','pile_group','station_power_group','power_class','scene_group','volume_group','boundary_hour','month']:
   for val,h in g.groupby(col,dropna=False):
    hh=h.loc[h.H_median]
    desc.append(dict(scope=scope,feature=col,group=val,n_R=len(h),stations_R=h.station_id.nunique(),n_H=len(hh),stations_H=hh.station_id.nunique(),large_D30_events=int(hh.large_D30_H.sum()),large_D30_H_fraction=hh.large_D30_H.mean(),rho_outside_events=int(h.rho_outside.sum()),rho_outside_R_fraction=h.rho_outside.mean(),median_abs_rho_minus_one=h.abs_rho_minus_one.median()))
 save(pd.DataFrame(desc),'taskB/feature_group_descriptives.csv')
 models=lpm_models(d)
 done('B',dict(original_R_preserved=len(d)==len(r),H_outside_is_missing=d.loc[~d.H_median,'large_D30_H'].isna().all(),scene_le8=d.scene_group.nunique()<=8,three_scopes_two_models=len(models)==6,full_rank_models=models['rank'].eq(models.parameters).all(),monthly_counts_match=(feat.full_base_sessions.fillna(0)==feat.base_transactions).all()))

if __name__=='__main__':main()
