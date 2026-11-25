from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
from io import BytesIO
MONTHS=[str(m) for m in pd.period_range('2022-09','2023-02',freq='M')]
STARTS=np.array([0,480,600,660,720,840,900,1020,1140])
EXPECTED=np.diff(np.r_[STARTS,1440])//5

def extract_station(sid,qualified):
 path=URAW/'charge_5min'/f'{sid}.csv';data=path.read_bytes()
 x=pd.read_csv(BytesIO(data),usecols=['time','e_price','s_price'])
 nraw=len(x);x['time']=pd.to_datetime(x.time,errors='coerce')
 x['e_price']=pd.to_numeric(x.e_price,errors='coerce');x['s_price']=pd.to_numeric(x.s_price,errors='coerce')
 dup=x.duplicated('time',keep=False)
 valid=x.time.notna()&~dup&np.isfinite(x.e_price)&np.isfinite(x.s_price)&(x.e_price>=0)&(x.s_price>=0)&(x.time.dt.minute%5==0)&(x.time.dt.second==0)
 x=x.loc[valid].copy();x['day']=x.time.dt.normalize();x['minute']=x.time.dt.hour*60+x.time.dt.minute;x['segment']=np.searchsorted(STARTS,x.minute,side='right')-1;x['T']=x.e_price+x.s_price
 x=x.loc[x.day.between('2022-09-01','2023-02-28')]
 grid=x.groupby(['day','segment']).size().rename('n').reset_index();grid['supported']=grid.n>=np.ceil(EXPECTED[grid.segment]*.8)
 complete=grid.loc[grid.supported].groupby('day').size().eq(9)
 d=x.groupby('day').agg(valid_snapshots=('time','size'),Emin=('e_price','min'),Emax=('e_price','max'),Tmin=('T','min'),Tmax=('T','max')).reindex(pd.date_range('2022-09-01','2023-02-28'))
 d.index.name='day';d=d.reset_index();d['station_id']=str(sid);d['complete_day']=d.day.map(complete).fillna(False).astype(bool)
 d['E_range']=d.Emax-d.Emin;d['T_range']=d.Tmax-d.Tmin
 d['daily_category']=np.select([d.T_range.le(.001)&d.E_range.gt(.03),d.T_range.gt(.001)&d.E_range.gt(.03),d.T_range.le(.001)&d.E_range.le(.03)],['total_fixed_split','both_change','both_flat'],default='other')
 d.loc[~d.complete_day,'daily_category']='incomplete_day';d['qualified_205']=str(sid) in qualified
 combined=[]
 if str(sid) in qualified:
  # Exact raw aggregation over 10-12; native 10-11,11-12,12-14 all must be supported.
  supported=grid.loc[grid.supported&grid.segment.isin([2,3,4])].groupby('day').size().eq(3)
  w=x.loc[x.minute.ge(600)&x.minute.lt(840)].copy();w['window']=np.where(w.minute.lt(720),'10_12','12_14')
  combined=w.groupby(['day','window']).agg(n=('time','size'),E=('e_price','median'),S=('s_price','median'),T=('T','median')).reset_index()
  combined['station_id']=str(sid);combined['supported_both_windows']=combined.day.map(supported).fillna(False).astype(bool)
 return d,combined,dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),purpose='Task5 exact daily ranges; raw paired five-minute records'),dict(station_id=str(sid),raw_rows=nraw,valid_rows=len(x),duplicate_timestamp_rows=int(dup.sum()),complete_days=int(d.complete_day.sum()))

def extract_all():
 assert json.loads((OUT/'validation/task4.json').read_text())['passed']
 split=json.loads(source(UE/'sample_split.json','Frozen engineering/analysis separation').read_text())
 ids=[str(s) for s in split['validation_station_ids']];assert len(ids)==1652
 p=parquet(UE/'monthly_boundary_panel.parquet');p=p.loc[p.variant.eq('base')].copy();p.station_id=p.station_id.astype(str)
 qualified=set(p.loc[p.paired_days.ge(10)&p.E_star.gt(.01),'station_id']);assert len(qualified)==205
 source(UE/'daily_endpoints.parquet','Reuse native endpoint grid coverage and price profiles')
 daily=[];windows=[];manifest=[];quality=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(extract_station,s,qualified) for s in ids]
  for i,job in enumerate(as_completed(jobs),1):
   d,w,m,q=job.result();daily.append(d);manifest.append(m);quality.append(q)
   if isinstance(w,pd.DataFrame):windows.append(w)
   if i%200==0:print(f'UrbanEV new raw-range extraction {i}/{len(ids)}',flush=True)
 d=pd.concat(daily,ignore_index=True).sort_values(['station_id','day']);w=pd.concat(windows,ignore_index=True).sort_values(['station_id','day','window'])
 save(d,'task5/daily_station_classification.csv');save(w,'task5/daily_10_12_vs_12_14.csv');save(pd.DataFrame(quality).sort_values('station_id'),'task5/raw_quality.csv')
 save(pd.DataFrame(manifest).sort_values('path'),'provenance/urbanev_raw_sources.csv')
 print('UrbanEV extraction finished',flush=True)

if __name__=='__main__':extract_all()
