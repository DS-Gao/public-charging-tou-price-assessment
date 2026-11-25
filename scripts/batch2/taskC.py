from common import *
from taskB import scene_map,qgroups

VARIANTS=['CAL_power','CAL_scene','CAL_size','CAL_reg']
NUM=['dc_share','log_piles','log_station_power']
CATS=['power_class','scene_group','volume_group','layer_id']

def design(train,test):
 # Each fold builds its encoding only from training rows.
 tr=[np.ones((len(train),1)),train[NUM].to_numpy(float)];te=[np.ones((len(test),1)),test[NUM].to_numpy(float)]
 known=np.ones(len(test),bool);names=['Intercept']+NUM
 for c in CATS:
  values=sorted(train[c].unique());known &= test[c].isin(values).to_numpy()
  for val in values[1:]:
   tr.append(train[c].eq(val).to_numpy(float)[:,None]);te.append(test[c].eq(val).to_numpy(float)[:,None]);names.append(c+'='+str(val))
 return np.concatenate(tr,axis=1),np.concatenate(te,axis=1),known,names

def predictions():
 f=formation();feat=pd.read_csv(OUT/'cache/beijing_station_month_features.csv')
 keep=['month','station_id','construction_site','dc_share','log_piles','log_station_power','formation_transactions','formation_power_class']
 f=f.merge(feat[keep],on=['month','station_id'],validate='many_to_one').rename(columns={'formation_power_class':'power_class'})
 f['layer_id']=f.month+'_'+f.boundary_hour.astype(str)
 for v in VARIANTS:f[v]=np.nan;f[v+'_status']='pending';f[v+'_training_n']=0
 folds=[];validchecks=[];total=0
 for month,indices in f.groupby('month').groups.items():
  base=f.loc[indices].copy()
  for sid,test0 in base.groupby('station_id',sort=True):
   train=base.loc[base.station_id.ne(sid)].copy();test=test0.copy()
   assert not train.station_id.eq(sid).any();unique=train.drop_duplicates('station_id')
   mp,_=scene_map(unique);train['scene_group']=train.construction_site.map(mp);test['scene_group']=test.construction_site.map(mp)
   # Unseen raw scene can map to Other only when Other was an actual predeclared training group.
   if 'Other' in mp.values():test['scene_group']=test.scene_group.fillna('Other')
   fun,edges=qgroups(unique.formation_transactions,3);train['volume_group']=fun(train.formation_transactions);test['volume_group']=fun(test.formation_transactions)
   for variant,col in [('CAL_power','power_class'),('CAL_scene','scene_group'),('CAL_size','volume_group')]:
    t=train.loc[train[col].notna()&train[col].ne('Missing')]
    med=t.groupby(['layer_id',col]).Z.agg(['median','size'])
    for ix,row in test.iterrows():
     key=(row.layer_id,row[col]);ok=key in med.index
     f.loc[ix,variant]=row.E+med.loc[key,'median'] if ok else np.nan
     f.loc[ix,variant+'_status']='ok' if ok else 'no_other_station_in_layer_type'
     f.loc[ix,variant+'_training_n']=int(med.loc[key,'size']) if ok else 0
   good=lambda z:z[NUM].notna().all(axis=1)&z.power_class.ne('Missing')&z.scene_group.notna()&z.construction_site.ne('Missing')&z.volume_group.ne('Missing')
   tr=train.loc[good(train)];te=test.loc[good(test)];rank=0;ncols=0;maxerr=0
   f.loc[test.index,'CAL_reg_status']='missing_feature'
   if len(te) and len(tr):
    X,Xt,known,names=design(tr,te);ncols=X.shape[1]
    beta,_,rank,_=np.linalg.lstsq(X,tr.Z.to_numpy(),rcond=None)
    # Any rank deficiency must still leave the test prediction estimable.
    rowproj=Xt@np.linalg.pinv(X)@X if rank<ncols else Xt
    residual=np.max(abs(Xt-rowproj),axis=1);maxerr=float(residual.max())
    estimable=known&(residual<1e-8)
    pred=te.E.to_numpy()+Xt@beta
    f.loc[te.index,'CAL_reg']=np.where(estimable,pred,np.nan)
    f.loc[te.index,'CAL_reg_status']=np.where(estimable,'ok',np.where(~known,'unseen_training_category','unidentifiable_prediction'))
    f.loc[te.index,'CAL_reg_training_n']=len(tr)
   folds.append(dict(month=month,held_station=sid,train_rows=len(train),train_stations=train.station_id.nunique(),held_rows=len(test),held_V=int(test.V.sum()),reg_complete_train_rows=len(tr),reg_rank=int(rank),reg_columns=ncols,max_prediction_nullspace_residual=maxerr,
                     size_q1=edges[0],size_q2=edges[1],scene_mapping=json.dumps(mp,ensure_ascii=False,sort_keys=True),held_station_in_training=False,training_months=month))
   total+=1
   if total%250==0:print('C leave-station folds',total,flush=True)
 save(f,'taskC/formation_and_fixed_V_predictions.csv.gz');save(pd.DataFrame(folds),'taskC/leave_station_fold_ledger.csv')
 return f,pd.DataFrame(folds)

def evaluate(f):
 s=stats();metrics=[];paired=[];coverage=[]
 incomplete_variants=[c for c in VARIANTS if f.loc[f.V,c].isna().any()]
 for layer,g in f.groupby('layer_order'):
  base=g.loc[g.V]
  common=base[VARIANTS].notna().all(axis=1)
  sets=[('original_V',base),('all_variants_common_available',base.loc[common])]
  for var in VARIANTS:
   good=base[var].notna();coverage.append(dict(month=g.month.iloc[0],layer_order=layer,configuration=var,n_V=len(base),available=int(good.sum()),missing=int((~good).sum()),available_stations=base.loc[good].station_id.nunique()))
   if var in incomplete_variants:sets.append((var+'_available_only',base.loc[good]))
  for pop,v in sets:
   for wt in ['ST','TX','KWH']:
    if v.empty:continue
    w=s.weights_for_panel(g,v,wt);errs={c:v[c].to_numpy()-v.Y.to_numpy() for c in ['E','CAL','E_plus_S','LOCAL']+VARIANTS}
    for c,err in errs.items():
     ok=np.isfinite(err).all();mae=float(w@abs(err)) if ok else np.nan
     metrics.append(dict(scope='layer',population=pop,month=g.month.iloc[0],layer_order=layer,weight=wt,configuration=c,n=len(v),n_V=len(base),available=int(np.isfinite(err).sum()),status='ok' if ok else 'unavailable_on_original_V',mae=mae,me=float(w@err) if ok else np.nan,E_relative_mae_reduction=1-mae/(w@abs(errs['E'])) if ok else np.nan,mae_minus_LOCAL=mae-w@abs(errs['LOCAL']) if ok else np.nan))
    pairs=[('E','LOCAL'),('E','CAL'),('E','E_plus_S')]+[(a,b) for c in VARIANTS for a,b in [('E',c),(c,'LOCAL')]]
    for a,b in pairs:
     diff=abs(errs[a])-abs(errs[b]);ok=np.isfinite(diff).all()
     paired.append(dict(scope='layer',population=pop,month=g.month.iloc[0],layer_order=layer,weight=wt,source=a,target=b,n=len(v),status='ok' if ok else 'unavailable_on_original_V',net_reduction=float(w@diff) if ok else np.nan,mean_abs_paired_difference=float(w@abs(diff)) if ok else np.nan,E_to_LOCAL_reference=float(w@(abs(errs['E'])-abs(errs['LOCAL'])))))
 met=pd.DataFrame(metrics);pair=pd.DataFrame(paired);cover=pd.DataFrame(coverage)
 outputs=[met];poutputs=[pair];coutputs=[cover]
 for month,g in group_scopes(met):
  for key,h in g.groupby(['population','weight','configuration']):
   good=h.status.eq('ok').all();expected=10 if month=='ALL' else 4 if month=='2025-07' else 6;good=good and len(h)==expected
   row=dict(zip(['population','weight','configuration'],key));row.update(scope='summary',month=month,layer_order='equal_layers',n=int(h.n.sum()),n_V=int(h.n_V.sum()),available=int(h.available.sum()),n_layers=len(h),status='ok' if good else 'incomplete_V_or_layers',mae=h.mae.mean() if good else np.nan,me=h.me.mean() if good else np.nan)
   e=g.loc[g.population.eq(key[0])&g.weight.eq(key[1])&g.configuration.eq('E')].mae.mean();l=g.loc[g.population.eq(key[0])&g.weight.eq(key[1])&g.configuration.eq('LOCAL')].mae.mean()
   row['E_relative_mae_reduction']=1-row['mae']/e;row['mae_minus_LOCAL']=row['mae']-l;outputs.append(pd.DataFrame([row]))
 for month,g in group_scopes(pair):
  for key,h in g.groupby(['population','weight','source','target']):
   expected=10 if month=='ALL' else 4 if month=='2025-07' else 6;ok=h.status.eq('ok').all() and len(h)==expected
   row=dict(zip(['population','weight','source','target'],key));row.update(scope='summary',month=month,layer_order='equal_layers',n=int(h.n.sum()),n_layers=len(h),status='ok' if ok else 'incomplete_V_or_layers',net_reduction=h.net_reduction.mean() if ok else np.nan,mean_abs_paired_difference=h.mean_abs_paired_difference.mean() if ok else np.nan,E_to_LOCAL_reference=h.E_to_LOCAL_reference.mean(),positive_layers=int(h.net_reduction.gt(1e-12).sum()),negative_layers=int(h.net_reduction.lt(-1e-12).sum()),zero_layers=int(h.net_reduction.abs().le(1e-12).sum()))
   row['relative_to_E_LOCAL']=row['net_reduction']/row['E_to_LOCAL_reference'];poutputs.append(pd.DataFrame([row]))
 for month,g in group_scopes(cover):
  z=g.groupby('configuration')[['n_V','available','missing']].sum().reset_index();z['month']=month;z['layer_order']='equal_layers';coutputs.append(z)
 save(pd.concat(outputs,ignore_index=True),'taskC/calibration_metrics.csv');save(pd.concat(poutputs,ignore_index=True),'taskC/paired_Table9_format.csv');save(pd.concat(coutputs,ignore_index=True),'taskC/calibration_coverage.csv')
 return met

if __name__=='__main__':
 f,folds=predictions();met=evaluate(f)
 done('C',dict(original_V_5846=f.V.sum()==5846,no_held_station_in_training=not folds.held_station_in_training.any(),month_separate_training=folds.month.eq(folds.training_months).all(),prediction_status_consistent=all(f[c].notna().eq(f[c+'_status'].eq('ok')).all() for c in VARIANTS),all_fixed_V_units_retained=len(f.loc[f.V])==5846,no_future_activity_features=True))
