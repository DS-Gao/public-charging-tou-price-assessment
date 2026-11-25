from pathlib import Path
import sys,importlib.util,json,hashlib
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent;OUT=P/'influence';OUT.mkdir(exist_ok=True)
ROOT=P.parents[1];B=ROOT/'outputs/56_r422_structure_tables_20261108/R4_22_Structure_Table_Revision'
inp=B/'analysis_evidence/inputs/module_b_row_level_inputs.csv.gz';vendor=B/'analysis_evidence/scripts/vendor/module_b_stats.py'
spec=importlib.util.spec_from_file_location('stats',vendor);stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
d=pd.read_csv(inp,float_precision='round_trip');d=d[d.V].copy();d['E_plus_S']=d.formation_E_star+d.formation_S_star
assert len(d)==5846 and not d.duplicated(['station_id','layer_order']).any()
assert d.layer_order.nunique()==10
configs=['E','CAL','E_plus_S','LOCAL'];layers={k:g for k,g in d.groupby('layer_order')}
rows=[];unit=[];base={}
for c in configs:
 for layer,g in layers.items():
  err=g[c]-g.Y;n=len(g)
  unit.extend(dict(station_id=r.station_id,layer_order=layer,month=r.month,boundary_hour=int(r.boundary_hour),configuration=c,n_layer=n,error=float(e),abs_error=abs(float(e)),squared_error=float(e*e),MAE_contribution=abs(float(e))/(10*n),MSE_contribution=float(e*e)/(10*n)) for r,e in zip(g.itertuples(),err))
u=pd.DataFrame(unit);u.to_csv(OUT/'unit_contributions.csv.gz',index=False)
ranking=u.groupby(['station_id','configuration']).agg(n_units=('layer_order','size'),MAE_contribution=('MAE_contribution','sum'),MSE_contribution=('MSE_contribution','sum')).reset_index()
for c in configs:
 mask=ranking.configuration.eq(c)
 for metric in ['MAE','MSE']:
  col=metric+'_contribution';ranking.loc[mask,metric+'_share_pct']=100*ranking.loc[mask,col]/ranking.loc[mask,col].sum()
  ranking.loc[mask,metric+'_rank']=ranking.loc[mask,col].rank(method='min',ascending=False).astype(int)
ranking.sort_values(['configuration','MSE_rank']).to_csv(OUT/'station_influence_ranking.csv',index=False)
selected=sorted(set(ranking.loc[(ranking.MAE_rank<=10)|(ranking.MSE_rank<=10),'station_id'])|{'ST_003774'})
def metrics(frame,with_p90=True):
 result=[]
 for c in configs:
  vals=[]
  for layer,g in frame.groupby('layer_order'):
   err=(g[c]-g.Y).to_numpy();w=np.ones(len(g))/len(g)
   vals.append((np.mean(abs(err)),np.mean(err**2),stats.weighted_quantile_tie(abs(err),w,.9) if with_p90 else np.nan))
  assert len(vals)==10
  z=np.mean(vals,axis=0)
  result.append(dict(configuration=c,MAE=z[0],MSE=z[1],RMSE=np.sqrt(z[1]),mean_layer_P90=z[2],n=len(frame),n_stations=frame.station_id.nunique()))
 e=result[0]
 for r in result:
  for m in ['MAE','RMSE','mean_layer_P90']:
   r[m+'_relative_improvement_vs_E_pct']=100*(e[m]-r[m])/e[m]
 return result
baseline=metrics(d);base={r['configuration']:r for r in baseline}
for r in baseline:r.update(removed_station='NONE',removed_units=0)
loso=list(baseline)
for station in selected:
 f=d[d.station_id.ne(station)]
 for r in metrics(f):
  c=r['configuration'];r.update(removed_station=station,removed_units=len(d)-len(f))
  for m in ['MAE','MSE','RMSE','mean_layer_P90']:r[m+'_change_vs_full']=r[m]-base[c][m]
  loso.append(r)
pd.DataFrame(loso).to_csv(OUT/'leave_one_station_cluster_out.csv',index=False)
# Exact all-station removal effects for additive losses, without recomputing
# any prices/calibrations or uncertainty. Cluster contribution and change after
# renormalisation are different quantities and both are supplied.
allrows=[]
for c in configs:
 part=u[u.configuration.eq(c)];sums=part.groupby('layer_order').agg(n=('error','size'),sa=('abs_error','sum'),ss=('squared_error','sum'))
 for station,z in part.groupby('station_id'):
  a=sums.sa/sums.n;s=sums.ss/sums.n
  for r in z.itertuples():
   a.loc[r.layer_order]=(sums.loc[r.layer_order,'sa']-r.abs_error)/(r.n_layer-1)
   s.loc[r.layer_order]=(sums.loc[r.layer_order,'ss']-r.squared_error)/(r.n_layer-1)
  mae=float(a.mean());mse=float(s.mean());rmse=np.sqrt(mse)
  allrows.append(dict(station_id=station,configuration=c,removed_units=len(z),MAE_without=mae,MSE_without=mse,RMSE_without=rmse,MAE_change=mae-base[c]['MAE'],MSE_change=mse-base[c]['MSE'],RMSE_change=rmse-base[c]['RMSE']))
allframe=pd.DataFrame(allrows)
for m in ['MAE','RMSE']:
 allframe[m+'_absolute_change_rank']=allframe.groupby('configuration')[m+'_change'].transform(lambda x:x.abs().rank(method='min',ascending=False))
allframe.to_csv(OUT/'all_station_leaveout_additive_metrics.csv',index=False)
cross=pd.DataFrame(loso).query("removed_station!='NONE'").merge(allframe,left_on=['removed_station','configuration'],right_on=['station_id','configuration'])
maxdiff=max(float((cross[m]-cross[m+'_without']).abs().max()) for m in ['MAE','MSE','RMSE'])
assert maxdiff<1e-12
frozen=pd.read_csv(B/'analysis_evidence/results/A2/overall.csv')
diff={}
for c in configs:
 r=frozen[frozen.configuration.eq(c)].iloc[0]
 diff[c]={m:base[c][m]-r[col] for m,col in [('MAE','mae'),('mean_layer_P90','p90_abs_error')]}
assert max(abs(x) for r in diff.values() for x in r.values())<1e-12
pd.DataFrame({'station_id':selected}).to_csv(OUT/'selected_stations.csv',index=False)
report=dict(evaluation_units=len(d),stations=d.station_id.nunique(),selected_stations=len(selected),selection='Union top10 station additive MAE and MSE contribution per each of4 configurations, plus ST_003774',all_station_leaveout_count=len(allframe),frozen_baseline_differences=diff,independent_subtraction_vs_recompute_maxdiff=maxdiff,price_inputs_held_fixed=True,CAL_refit=False,bootstrap=False,weight_rule='Equal stations within each remaining layer, ten layers equal',P90='Mean of ten within-layer ST weighted ECDF inverse quantiles, ties merged and midpoint at exact cumulative hit. Not pooled P90.',inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [inp,vendor,B/'analysis_evidence/results/A2/overall.csv']],python=sys.version)
(OUT/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(ranking[ranking.station_id.eq('ST_003774')].to_string(index=False))
print(pd.DataFrame(loso)[pd.DataFrame(loso).removed_station.isin(['NONE','ST_003774'])].to_string(index=False))
print('selected',len(selected),'allstations',d.station_id.nunique())
