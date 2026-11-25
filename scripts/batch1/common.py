from pathlib import Path
import sys, os, json, hashlib, shutil, subprocess, ast
sys.dont_write_bytecode=True
ROOT=Path(os.environ['EV_PROJECT_ROOT']).resolve()
OUT=Path(os.environ.get('EV_OUTPUT_ROOT', str(Path.cwd()/'reproduction'))).resolve()
for name in ['cache','figures','provenance','validation','task1','task2','task3','task5','taskA','taskB','taskC','taskD']:
 (OUT/name).mkdir(parents=True,exist_ok=True)
OLD=ROOT/'outputs/R4_26_Targeted_Scientific_Diagnostics_20261116'
UE=ROOT/'outputs/09_price_architecture/urbanev_external_analysis_20260913'
URAW=ROOT/'doi_10_5061_dryad_np5hqc04z__v20260204/UrbanEVDataset/UrbanEVDataset/20220901-20230228_station-raw'
sys.path.insert(0,str(OLD/'runtime'))
sys.path.insert(0,str(ROOT/'outputs/R4_27_Phase_A_20261116/runtime'))
sys.path.insert(0,str(OUT/'runtime'))
os.environ['MPLCONFIGDIR']=str(OUT/'cache/mpl')
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
KEY=['month','boundary_hour','station_id']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()

def save(df,p):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.16g')

def source(p,purpose):
 p=Path(p).resolve(); ledger=OUT/'provenance/consumed_sources.csv'
 old=pd.read_csv(ledger) if ledger.exists() else pd.DataFrame(columns=['path','bytes','sha256','purpose'])
 if not old.path.eq(str(p)).any():
  save(pd.concat([old,pd.DataFrame([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),purpose=purpose)])],ignore_index=True),'provenance/consumed_sources.csv')
 return p

def csv(p,why='Reused verified evidence'): return pd.read_csv(source(p,why))
def parquet(p,why='Reused verified evidence'): return pd.read_parquet(source(p,why))
def database():
 p=source(OLD/'cache/audit.duckdb','Verified Beijing transaction cache; READ_ONLY')
 c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='4GB'")
 c.execute(f"SET temp_directory='{(OUT/'cache').as_posix()}'")
 c.execute(f"ATTACH '{p.as_posix()}' AS frozen (READ_ONLY)")
 c.execute('CREATE TEMP VIEW tx AS SELECT * FROM frozen.tx')
 c.execute('CREATE TEMP VIEW valid AS SELECT * FROM tx WHERE base_valid AND contained')
 return c

def layers():
 p=source(ROOT/'analysis/price_architecture_v2/config_stage2.json','Frozen definitions')
 cfg=json.loads(p.read_text(encoding='utf-8'));rows=[]
 for order,(m,b,l,r,_,__,___,q,selection) in enumerate(cfg['boundary_layers'],1):
  if selection=='PURE_TOU':rows.append(dict(month=m,boundary_hour=b,segment_L=l,segment_R=r,q=q,layer_order=order))
 return cfg,pd.DataFrame(rows)


def dist(x):
 x=pd.Series(x).dropna()
 return dict(n=len(x),median=x.median(),p10=x.quantile(.1),p25=x.quantile(.25),p75=x.quantile(.75),p90=x.quantile(.9),max=x.max())

def done(task,checks):
 checks={k:bool(v) for k,v in checks.items()}
 checks['passed']=all(bool(v) for v in checks.values())
 (OUT/f'validation/task{task}.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
 assert checks['passed'],checks
 print(f'Task {task} complete: {checks}',flush=True)

def plotting():
 import pip
 sys.path.insert(0,str(Path(pip.__file__).parent/'_vendor'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 return plt
