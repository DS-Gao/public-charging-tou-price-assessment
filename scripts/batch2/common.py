from pathlib import Path
import sys, os, json, hashlib, shutil, importlib.util, ast
sys.dont_write_bytecode=True
ROOT=Path(os.environ['EV_PROJECT_ROOT']).resolve()
OUT=Path(os.environ.get('EV_OUTPUT_ROOT', str(Path.cwd()/'reproduction'))).resolve()
for name in ['cache','figures','provenance','validation','task1','task2','task3','task5','taskA','taskB','taskC','taskD']:
 (OUT/name).mkdir(parents=True,exist_ok=True)
FIRST=ROOT/'revision_R4.27_diagnostics'
OLD=ROOT/'outputs/R4_26_Targeted_Scientific_Diagnostics_20261116'
PHASE=ROOT/'outputs/R4_27_Phase_A_20261116'
REFINE=ROOT/'outputs/62_r426_contribution_refinement_20261109/R4_26_Contribution_Refinement/analysis_evidence'
UE=ROOT/'outputs/09_price_architecture/urbanev_external_analysis_20260913'
URAW=ROOT/'doi_10_5061_dryad_np5hqc04z__v20260204/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw'
for d in [OLD/'runtime',PHASE/'runtime',FIRST/'runtime',OUT/'runtime']:
 sys.path.insert(0,str(d))
os.environ['MPLCONFIGDIR']=str(OUT/'cache/mpl')
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
KEY=['month','boundary_hour','station_id']
_sources={}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def save(df,p):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.16g')

def source(p,purpose='Reuse frozen evidence'):
 p=Path(p).resolve();key=str(p)
 if key not in _sources:
  _sources[key]=dict(path=key,bytes=p.stat().st_size,sha256=sha(p),purpose=purpose)
  save(pd.DataFrame(_sources.values()),'provenance/consumed_'+Path(sys.argv[0]).stem+'.csv')
 return p

def csv(p,why='Reuse frozen evidence',**kwargs):return pd.read_csv(source(p,why),**kwargs)
def parquet(p,why='Reuse frozen evidence',**kwargs):return pd.read_parquet(source(p,why),**kwargs)

def database():
 p=source(OLD/'cache/audit.duckdb','Beijing verified eligible-session cache; READ_ONLY')
 c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='4GB'")
 c.execute(f"SET temp_directory='{(OUT/'cache').as_posix()}'")
 c.execute(f"ATTACH '{p.as_posix()}' AS frozen (READ_ONLY)")
 c.execute('CREATE TEMP VIEW tx AS SELECT * FROM frozen.tx')
 return c

def layers():
 cfg=json.loads(source(ROOT/'analysis/price_architecture_v2/config_stage2.json').read_text(encoding='utf-8'))
 r=[]
 for order,(m,b,l,rgt,_,__,___,q,selection) in enumerate(cfg['boundary_layers'],1):
  if selection=='PURE_TOU':r.append(dict(month=m,boundary_hour=b,segment_L=l,segment_R=rgt,q=q,layer_order=order))
 return cfg,pd.DataFrame(r)

def load_module(p,name):
 p=source(p,'Original numerical definition');spec=importlib.util.spec_from_file_location(name,p)
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def stats():return load_module(REFINE/'scripts/vendor/module_b_stats.py','frozen_module_b_stats')

def formation():
 f=csv(REFINE/'inputs/module_b_row_level_inputs.csv.gz')
 es=csv(PHASE/'task2/analysis_input_formation.csv.gz')
 f=f.merge(es[KEY+['E_plus_S']],on=KEY,validate='one_to_one')
 assert len(f)==6883 and f.V.sum()==5846 and f.station_id.nunique()==2342
 assert not f.duplicated(KEY).any()
 return f

def group_scopes(d):
 return [(m,d.loc[d.month.eq(m)]) for m in ['2025-07','2025-01']]+[('ALL',d)]

def done(task,checks):
 checks={k:bool(v) for k,v in checks.items()};checks['passed']=all(checks.values())
 (OUT/f'validation/task{task}.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
 assert checks['passed'],checks
 print(f'Task {task}: '+json.dumps(checks),flush=True)

def plotting():
 import pip
 sys.path.insert(0,str(Path(pip.__file__).parent/'_vendor'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 return plt

def fit_cluster_ols(data,formula,outcome,scope,model,task,weight=None):
 import statsmodels.formula.api as smf
 data=data.copy()
 fit=smf.ols(formula,data=data,missing='drop').fit()
 used=data.loc[fit.model.data.row_labels];nc=used.station_id.nunique()
 rank=int(np.linalg.matrix_rank(fit.model.exog));p=fit.model.exog.shape[1]
 meta=dict(task=task,scope=scope,outcome=outcome,model=model,formula=formula,n=len(used),stations=nc,
           r_squared=fit.rsquared,adjusted_r_squared=fit.rsquared_adj,pseudo_r_squared=np.nan,
           pseudo_r_squared_note='Not applicable: ordinary least squares / linear probability model',
           rank=rank,parameters=p,status='ok' if rank==p and nc>=2 else 'rank_deficient' if rank<p else 'too_few_clusters')
 if rank<p or nc<2:return [],meta
 fit=fit.get_robustcov_results(cov_type='cluster',groups=used.station_id,use_correction=True,df_correction=True,use_t=True)
 ci=fit.conf_int();rows=[]
 for i,term in enumerate(fit.model.exog_names):
  rows.append(dict(**{k:meta[k] for k in ['task','scope','outcome','model','n','stations','r_squared']},term=term,
                   coefficient=fit.params[i],cluster_se=fit.bse[i],t=fit.tvalues[i],p_value=fit.pvalues[i],ci_low=ci[i,0],ci_high=ci[i,1]))
 return rows,meta
