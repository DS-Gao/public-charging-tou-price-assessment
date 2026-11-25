"""R4.26 targeted diagnostics. Writes only inside this independent audit directory.
No bootstrap, permutation, regression, raw-data edits or manuscript changes.
Rules transcribed from the hashed existing Stage 1/2 and reconstruction code.
"""
from pathlib import Path
import sys, json, hashlib, platform, time, importlib.util, argparse
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / 'runtime'))
import duckdb
import numpy as np
import pandas as pd

EVID = ROOT/'outputs/62_r426_contribution_refinement_20261109/R4_26_Contribution_Refinement/analysis_evidence'
BASE = ROOT/'outputs/R4_26_AE_Submission_AI_Disclosure_20261111'
CFG = ROOT/'analysis/price_architecture_v2/config_stage2.json'
KEY = ['month','boundary_hour','station_id']
METHODS = ['median','mean','vwap']
checks = []

def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

def save(df, path):
    path = OUT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.16g')

def dump(obj,path):
    (OUT/path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def check(name, actual, expected, tol=0):
    ok = abs(float(actual)-float(expected)) <= tol
    checks.append(dict(check=name,actual=actual,expected=expected,tolerance=tol,passed=bool(ok)))
    save(pd.DataFrame(checks),'validation/checks.csv')
    if not ok: raise AssertionError(f'{name}: {actual} vs {expected} (tol {tol})')

def dist(x):
    a=pd.Series(x).dropna().astype(float)
    if len(a)==0: return {k:np.nan for k in ['n','mean','median','q25','q75','iqr','p90','min','max','abs_max']}
    q=a.quantile([.25,.5,.75,.9])
    return dict(n=len(a),mean=a.mean(),median=q.loc[.5],q25=q.loc[.25],q75=q.loc[.75],iqr=q.loc[.75]-q.loc[.25],p90=q.loc[.9],min=a.min(),max=a.max(),abs_max=a.abs().max())

def label(x,tol=.03): return np.where(x>tol,'positive',np.where(x<-tol,'negative','neutral'))
def amount(x): return np.where(x<-.9,'Attenuation',np.where(x>.9,'Amplification','Within reference range'))

def load_stats():
    p=EVID/'scripts/vendor/module_b_stats.py'
    spec=importlib.util.spec_from_file_location('frozen_stats',p)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def build_db(con,cfg):
    log('Constructing shared raw/price-base/contained tables')
    # No pre-existing research module is executed; all SQL output stays in this DB.
    raw_sql=' UNION ALL '.join(f"SELECT '{m}' AS month, * FROM read_parquet('{(ROOT/cfg['inputs']['orders_'+m]['path']).as_posix()}')" for m in ['2025-01','2025-07'])
    con.execute('CREATE OR REPLACE VIEW raw AS '+raw_sql)
    seg=pd.DataFrame([(m,*x) for m,ss in cfg['native_8_segments'].items() for x in ss],columns=['month','segment_id','start_hour','end_hour','duration_hours'])
    con.register('segment_defs',seg)
    con.execute("""CREATE OR REPLACE TABLE tx AS
    WITH assigned AS (
      SELECT r.*, s.segment_id,s.start_hour,s.end_hour,s.duration_hours,
      total_elec_fee/total_elec_kwh AS E, total_service_fee/total_elec_kwh AS S,total_fee/total_elec_kwh AS T,
      total_fee-total_elec_fee-total_service_fee AS fee_delta,
      CAST(charge_start_time-INTERVAL 23 HOUR AS DATE) AS tariff_day,
      date_trunc('day',charge_start_time)+start_hour*INTERVAL 1 HOUR
        -CASE WHEN start_hour>end_hour AND hour(charge_start_time)<end_hour THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END AS segment_start
      FROM raw r LEFT JOIN segment_defs s ON r.month=s.month AND
      ((start_hour<end_hour AND hour(charge_start_time)>=start_hour AND hour(charge_start_time)<end_hour)
       OR(start_hour>end_hour AND(hour(charge_start_time)>=start_hour OR hour(charge_start_time)<end_hour)))
    ) SELECT *,
      coalesce(station_id IS NOT NULL AND charge_start_time IS NOT NULL
       AND charge_start_time>=CAST(month||'-01' AS TIMESTAMP)
       AND charge_start_time<CAST(month||'-01' AS TIMESTAMP)+INTERVAL 1 MONTH
       AND duration_min>0 AND total_elec_kwh>0 AND total_elec_fee>=0 AND total_service_fee>=0 AND total_fee>=0
       AND abs(fee_delta)<=0.0100001 AND isfinite(E) AND isfinite(S) AND isfinite(T),false) AS base_valid,
      coalesce(charge_start_time>=segment_start AND charge_end_time<=segment_start+duration_hours*INTERVAL 1 HOUR,false) AS contained,
      CASE WHEN tariff_day>=CAST(month||'-01' AS DATE) AND tariff_day<CAST(month||'-01' AS DATE)+INTERVAL 15 DAY THEN 'formation'
           WHEN tariff_day>=CAST(month||'-01' AS DATE)+INTERVAL 15 DAY AND tariff_day<CAST(month||'-01' AS DATE)+INTERVAL 1 MONTH THEN 'validation'
           ELSE 'outside_month_split' END AS half
    FROM assigned""")
    con.execute('CREATE OR REPLACE VIEW valid AS SELECT * FROM tx WHERE base_valid AND contained')
    for name,q,expected in [('raw_transactions','SELECT count(*) FROM tx',8544695),('price_base','SELECT count(*) FROM tx WHERE base_valid',8495859),('contained','SELECT count(*) FROM valid',6262859)]:
        check(name,con.execute(q).fetchone()[0],expected)
    expr=[]
    for x,f in [('E','total_elec_fee'),('S','total_service_fee'),('T','total_fee')]:
        expr += [f'median({x}) AS {x}_median',f'avg({x}) AS {x}_mean',f'sum({f})/sum(total_elec_kwh) AS {x}_vwap']
    con.execute("""CREATE OR REPLACE TABLE endpoints AS
      WITH expanded AS (SELECT *, 'fullmonth' AS period FROM valid UNION ALL SELECT *,half AS period FROM valid WHERE half IN ('formation','validation'))
      SELECT month,station_id,segment_id,period,count(*) AS n,count(DISTINCT tariff_day) AS days,
       sum(total_elec_kwh) AS kwh,count(DISTINCT pile_id) AS devices,
       median(rated_gun_power_kw) AS rated_power_tx_median,
       avg(fee_delta/total_elec_kwh) AS identity_mean,sum(fee_delta)/sum(total_elec_kwh) AS identity_vwap,
       """+', '.join(expr)+" FROM expanded GROUP BY month,station_id,segment_id,period")
    log('Endpoint aggregation complete')

def build_panel(con,cfg):
    ep=con.execute('SELECT * FROM endpoints ORDER BY month,station_id,segment_id,period').df()
    frames=[]
    for order,(m,b,left,right,_,__,scope,q,selection) in enumerate(cfg['boundary_layers'],1):
        if selection!='PURE_TOU': continue
        for period in ['fullmonth','formation','validation']:
            e=ep[ep.month.eq(m)&ep.period.eq(period)]
            l=e[e.segment_id.eq(left)].drop(columns=['month','segment_id','period']).set_index('station_id').add_suffix('_L')
            r=e[e.segment_id.eq(right)].drop(columns=['month','segment_id','period']).set_index('station_id').add_suffix('_R')
            f=l.join(r,how='outer').reset_index()
            f['month']=m;f['boundary_hour']=b;f['layer_order']=order;f['period']=period;f['q']=q
            f['segment_L']=left;f['segment_R']=right
            nmin,dmin=(20,5) if period=='fullmonth' else (10,3)
            for side in ['L','R']:
                for col in ['n','days','kwh','devices']: f[f'{col}_{side}']=f[f'{col}_{side}'].fillna(0)
                f[f'support_{side}']=(f[f'n_{side}']>=nmin)&(f[f'days_{side}']>=dmin)
            f['support']=f.support_L&f.support_R
            for method in METHODS:
                for x in ['E','S','T']: f[f'{x}_{method}']=q*(f[f'{x}_{method}_R']-f[f'{x}_{method}_L'])
                f[f'r_{method}']=f[f'T_{method}']-f[f'E_{method}']-f[f'S_{method}']
                f[f'D30_{method}']=30*(f[f'T_{method}']-f[f'E_{method}'])
                f[f'R_{method}']=f.support&(f[f'E_{method}']>.01)
                f[f'H_{method}']=f[f'R_{method}']&(f[f'T_{method}']>.03)
                f[f'direction_{method}']=label(f[f'T_{method}'])
                f[f'amount_{method}']=amount(f[f'D30_{method}'])
            frames.append(f)
    panel=pd.concat(frames,ignore_index=True).sort_values(KEY+['period'])
    panel.to_parquet(OUT/'cache/boundary_prices.parquet',index=False)
    ep.to_parquet(OUT/'cache/endpoint_prices.parquet',index=False)
    return panel

def baseline(panel):
    full=panel[panel.period.eq('fullmonth')&panel.support].copy()
    check('A',len(full),8133);check('R',full.R_median.sum(),7531);check('H',full.H_median.sum(),7479)
    for g,n in [('Attenuation',1157),('Within reference range',4773),('Amplification',1549)]: check('H_'+g,((full.H_median)&full.amount_median.eq(g)).sum(),n)
    old=pd.read_parquet(ROOT/'outputs/09_price_architecture/02_boundary_signal_core/boundary_signal_panel.parquet')
    dump({'frozen_panel_columns':old.columns.tolist()},'validation/frozen_panel_schema.json')
    # Price rows against historical raw-derived panel (scope selected independently).
    old=old[old.analysis_scope.eq('PURE_TOU')]
    j=full.merge(old,on=KEY,suffixes=('','_frozen'),validate='one_to_one')
    check('frozen_A_key_match',len(j),8133)
    for x in ['E','S','T']:
        c= x+'_star' if x+'_star' in j else {'T':'T_star_observed'}[x]
        check('full_'+x+'_max_abs_vs_frozen',np.max(np.abs(j[x+'_median']-j[c])),0,1e-12)
    frozen=pd.read_csv(EVID/'inputs/module_b_row_level_inputs.csv.gz')
    f=panel[panel.period.eq('formation')&panel.support]
    v=panel[panel.period.eq('validation')&panel.support]
    check('Af',len(f),6883)
    pairs=f.merge(v,on=KEY,suffixes=('_f','_v'),validate='one_to_one')
    check('V_reconstructed',len(pairs),5846)
    a=frozen[frozen.V.astype(bool)]
    j=pairs.merge(a,on=KEY,validate='one_to_one')
    check('frozen_V_key_match',len(j),5846)
    for col,ref in [('E_median_f','E'),('S_median_f','formation_S_star'),('T_median_f','LOCAL'),('T_median_v','Y')]:
        check(col+'_max_abs_vs_frozen',np.max(np.abs(j[col]-j[ref])),0,1e-12)
    pairs=pairs.merge(a[KEY+['CAL','cal_value']],on=KEY,validate='one_to_one')
    save(full,'task1/fixed_A_unit_prices.csv')
    save(pairs,'task1/fixed_V_unit_prices.csv')
    return full,pairs,frozen

def fee_audit(con):
    records=[]
    for scope,where in [('all_raw','true'),('price_base','base_valid'),('contained','base_valid AND contained')]:
        for m in ['2025-01','2025-07','ALL']:
            cond=where+(f" AND month='{m}'" if m!='ALL' else '')
            base=con.execute(f"SELECT count(*) n,count(*) FILTER(WHERE fee_delta=0) exact_zero,count(*) FILTER(WHERE abs(fee_delta)<=1e-12) within_1e12 FROM tx WHERE {cond}").df().iloc[0].to_dict()
            for value,expression,unit in [('fee_delta','fee_delta','RMB'),('absolute_fee_delta','abs(fee_delta)','RMB'),('fee_delta_per_kwh','fee_delta/total_elec_kwh','RMB/kWh'),('absolute_fee_delta_per_kwh','abs(fee_delta/total_elec_kwh)','RMB/kWh')]:
                r=con.execute(f"SELECT count(*) finite_n,avg({expression}) mean,median({expression}) median,quantile_cont({expression},.9) p90,max(abs({expression})) abs_max,min({expression}) min,max({expression}) max FROM tx WHERE {cond} AND total_elec_kwh>0 AND isfinite({expression})" if 'per_kwh' in value else f"SELECT count(*) finite_n,avg({expression}) mean,median({expression}) median,quantile_cont({expression},.9) p90,max(abs({expression})) abs_max,min({expression}) min,max({expression}) max FROM tx WHERE {cond} AND isfinite({expression})").df().iloc[0].to_dict()
                records.append(dict(scope=scope,month=m,variable=value,unit=unit,**base,exact_zero_fraction=base['exact_zero']/base['n'],within_1e12_fraction=base['within_1e12']/base['n'],**r))
    save(pd.DataFrame(records),'task1/fee_identity_audit.csv')
    r=[]
    for m in ['2025-01','2025-07']:
        for field in ['station_id','charge_start_time','charge_end_time','duration_min','total_elec_kwh','total_elec_fee','total_service_fee','total_fee','pile_id','rated_gun_power_kw']:
            f=con.execute(f'SELECT count(*) n,count(*) FILTER(WHERE "{field}" IS NULL) missing FROM tx WHERE month=?',[m]).df().iloc[0].to_dict()
            r.append(dict(month=m,field=field,**f,missing_fraction=f['missing']/f['n']))
    save(pd.DataFrame(r),'task1/raw_field_missingness.csv')
    check('same_sample_component_nonfinite',con.execute('SELECT count(*) FROM valid WHERE NOT(isfinite(E) AND isfinite(S) AND isfinite(T))').fetchone()[0],0)
    ep=con.execute('SELECT * FROM endpoints').df(); res=[]
    for method in ['mean','vwap']:
        err=ep[f'T_{method}']-ep[f'E_{method}']-ep[f'S_{method}']-ep[f'identity_{method}']
        check(method+'_endpoint_identity_max_error',err.abs().max(),0,1e-12)
        for period,g in ep.groupby('period'):
            rr=g[f'T_{method}']-g[f'E_{method}']-g[f'S_{method}']
            res.append(dict(method=method,period=period,quantity='endpoint_residual',**dist(rr)))
    save(pd.DataFrame(res),'task1/additivity_audit.csv')

def task1(full,pairs,stats):
    log('Task 1 summaries and same-V historical comparisons')
    rows=[]; counts=[]; trans=[]
    scopes=[('ALL',full)]+[(f'{m}_{b:02d}',g) for (m,b),g in full.groupby(['month','boundary_hour'])]
    for layer,g in scopes:
        for method in METHODS:
            for population,mask in [('fixed_A',np.ones(len(g),dtype=bool)),('fixed_R_median',g.R_median),('fixed_H_median',g.H_median),('method_R',g[f'R_{method}']),('method_H',g[f'H_{method}'])]:
                sub=g.loc[mask]
                for x in ['E','S','T','r','D30']:
                    rows.append(dict(layer=layer,population=population,method=method,variable=x,unit='RMB/reference30kWh' if x=='D30' else 'RMB/kWh',**dist(sub[f'{x}_{method}'])))
                counts.append(dict(layer=layer,population=population,method=method,n=len(sub),stations=sub.station_id.nunique(),R=int(sub[f'R_{method}'].sum()),H=int(sub[f'H_{method}'].sum()),positive=int((sub[f'T_{method}']>.03).sum()),neutral=int((sub[f'T_{method}'].abs()<=.03).sum()),negative=int((sub[f'T_{method}']<-.03).sum()),attenuation=int((sub[f'D30_{method}']<-.9).sum()),within_reference=int((sub[f'D30_{method}'].abs()<=.9).sum()),amplification=int((sub[f'D30_{method}']>.9).sum())))
        for target in ['mean','vwap']:
            for pop,mask in [('fixed_A',np.ones(len(g),bool)),('fixed_H_median',g.H_median)]:
                sub=g.loc[mask]
                for kind in ['direction','amount','R','H']:
                    t=sub.groupby([f'{kind}_median',f'{kind}_{target}'],dropna=False).size()
                    for (a,b),n in t.items(): trans.append(dict(layer=layer,population=pop,target_method=target,classification=kind,from_median=str(a),to_target=str(b),n=int(n),denominator=len(sub),fraction=n/len(sub) if len(sub) else np.nan))
    d=pd.DataFrame(rows);save(d[d.layer.ne('ALL')],'task1/aggregation_comparison_by_layer.csv');save(d[d.layer.eq('ALL')],'task1/aggregation_comparison_summary.csv')
    save(pd.DataFrame(counts),'task1/aggregation_classification_counts.csv');save(pd.DataFrame(trans),'task1/direction_amount_transition.csv')
    changes=[]
    for layer,g in scopes:
        for source,target,interpretation in [('median','mean','aggregation_function_at_equal_transaction_mass'),('mean','vwap','transaction_to_energy_weight_for_linear_mean')]:
            for x in ['E','S','T','r','D30']:
                z=g[f'{x}_{target}']-g[f'{x}_{source}'];changes.append(dict(layer=layer,source=source,target=target,interpretation=interpretation,variable=x,**dist(z),mean_abs=z.abs().mean(),p90_abs=z.abs().quantile(.9)))
    save(pd.DataFrame(changes),'task1/aggregation_function_weight_changes.csv')
    cw=[];gaps=[];gap_units=[]
    for method in METHODS:
        for (m,b),g in pairs.groupby(['month','boundary_hour']):
            target=g[f'T_{method}_v'].to_numpy(float)
            inputs={'E':g[f'E_{method}_f'].to_numpy(float),'E+S':(g[f'E_{method}_f']+g[f'S_{method}_f']).to_numpy(float),'LOCAL':g[f'T_{method}_f'].to_numpy(float)}
            if method=='median': inputs['CAL']=g.CAL.to_numpy(float)
            errs={}
            for name,pred in inputs.items():
                err=pred-target;errs[name]=err
                cw.append(dict(scope='layer',month=m,boundary_hour=b,method=method,input=name,n=len(g),n_layers=1,mae=np.abs(err).mean(),me=err.mean(),p90_abs_error=stats.weighted_quantile_tie(np.abs(err),np.ones(len(err)),.9),label_disagreement=(label(pred)!=label(target)).mean(),paired_gain=np.nan,mean_abs_paired_loss_difference=np.nan))
            for a,z in [('E','E+S'),('E','LOCAL'),('E+S','LOCAL')]:
                loss=np.abs(errs[a])-np.abs(errs[z])
                cw.append(dict(scope='layer',month=m,boundary_hour=b,method=method,input=a+'→'+z,n=len(g),n_layers=1,mae=np.nan,me=np.nan,p90_abs_error=np.nan,label_disagreement=np.nan,paired_gain=loss.mean(),mean_abs_paired_loss_difference=np.abs(loss).mean()))
            gap=inputs['LOCAL']-inputs['E+S'];loss_gap=errs['LOCAL']-errs['E+S']
            gaps.append(dict(scope='layer',month=m,boundary_hour=b,method=method,n=len(g),**{f'input_gap_{k}':v for k,v in dist(gap).items()},absolute_input_gap_mean=np.abs(gap).mean(),absolute_input_gap_p90=np.quantile(np.abs(gap),.9),error_difference_max=np.abs(loss_gap).max(),absolute_loss_difference_max=np.max(np.abs(np.abs(errs['LOCAL'])-np.abs(errs['E+S']))),tolerance=1e-12,n_within_tolerance=int((np.abs(gap)<=1e-12).sum())))
            u=g[KEY].copy();u['method']=method;u['LOCAL_minus_EplusS']=gap;u['error_difference']=loss_gap;u['absolute_loss_difference']=np.abs(errs['LOCAL'])-np.abs(errs['E+S']);gap_units.append(u)
    d=pd.DataFrame(cw)
    for (method,name),g in d.groupby(['method','input']):
        r=dict(scope='ten_layer_equal',month='ALL',boundary_hour='ALL',method=method,input=name,n=int(g.n.sum()),n_layers=len(g))
        for x in ['mae','me','p90_abs_error','label_disagreement','paired_gain','mean_abs_paired_loss_difference']:r[x]=g[x].mean()
        cw.append(r)
    d=pd.DataFrame(cw);save(d,'task1/cross_window_comparison.csv')
    save(pd.DataFrame(gaps),'task1/EplusS_LOCAL_input_gap.csv');save(pd.concat(gap_units),'task1/EplusS_LOCAL_unit_gap.csv')
    for name,want in [('E',.036470),('E+S',.017447),('LOCAL',.017398)]:
        got=d[(d.scope=='ten_layer_equal')&(d.method=='median')&(d.input==name)].mae.iloc[0];check('median_MAE_'+name,got,want,.0000006)
    # Exact frozen point values, not just rounded manuscript values.
    ref=EVID/'results/A2/a2_overall.csv'
    if ref.exists(): dump({'note':'frozen A2 present','path':str(ref)},'validation/A2_reference.json')

def task2(con,panel):
    log('Task 2 membership, endpoint failures and daily histories')
    frames=[]
    for period,prefix in [('fullmonth','full'),('formation','f'),('validation','v')]:
        g=panel[panel.period.eq(period)].copy().set_index(KEY)
        cols=['support','support_L','support_R','n_L','n_R','days_L','days_R','E_median','S_median','T_median','R_median','H_median','segment_L','segment_R']
        frames.append(g[cols].add_prefix(prefix+'_'))
    allp=pd.concat(frames,axis=1).reset_index()
    for prefix in ['full','f','v']:
        for c in ['support','support_L','support_R','R_median','H_median']:allp[f'{prefix}_{c}']=allp[f'{prefix}_{c}'].eq(True)
        for c in ['n_L','n_R','days_L','days_R']:allp[f'{prefix}_{c}']=allp[f'{prefix}_{c}'].fillna(0)
    allp['A']=allp.full_support;allp['Af']=allp.f_support;allp['R']=allp.full_R_median;allp['Rf']=allp.f_R_median;allp['Ov']=allp.v_support;allp['V']=allp.Af&allp.Ov
    allp['membership']=np.select([allp.A&allp.Af,allp.A&~allp.Af,~allp.A&allp.Af],['A_intersect_Af','A_only','Af_only'],default='neither')
    allp=allp[allp.A|allp.Af].copy()
    check('Af_outside_A_all10',((~allp.A)&allp.Af).sum(),534)
    allrows=[]
    for (m,b,grp),g in allp.groupby(['month','boundary_hour','membership']):
        allrows.append(dict(month=m,boundary_hour=b,membership=grp,n=len(g),stations=g.station_id.nunique(),R=int(g.R.sum()),Rf=int(g.Rf.sum()),Ov=int(g.Ov.sum()),Rf_and_Ov=int((g.Rf&g.Ov).sum()),Af_and_Ov=int((g.Af&g.Ov).sum())))
    save(pd.DataFrame(allrows),'task2/all_layers_membership_crosswalk.csv')
    jan=allp[allp.month.eq('2025-01')&allp.boundary_hour.isin([7,10,13])].copy()
    save(jan,'task2/jan_membership_unit_rows.csv');save(pd.DataFrame(allrows).query("month=='2025-01' and boundary_hour in [7,10,13]"),'task2/jan_membership_crosswalk.csv')
    av=[];fail=[]
    for b,want in [(7,(234,173,26)),(10,(126,48,45)),(13,(137,12,11))]:
        g=jan[jan.boundary_hour.eq(b)]
        for name,n,expected in [('Af',g.Af.sum(),want[0]),('Rf',g.Rf.sum(),want[1]),('Rf_and_Ov',(g.Rf&g.Ov).sum(),want[2])]:check(f'Jan{b}_{name}',n,expected)
        av.append(dict(month='2025-01',boundary_hour=b,A=int(g.A.sum()),R=int(g.R.sum()),Af=int(g.Af.sum()),Rf=int(g.Rf.sum()),Rf_and_Ov=int((g.Rf&g.Ov).sum()),Af_and_Ov=int((g.Af&g.Ov).sum()),Rf_availability=(g.Rf&g.Ov).sum()/g.Rf.sum(),Af_availability=(g.Af&g.Ov).sum()/g.Af.sum(),Af_outside_A=int((g.Af&~g.A).sum())))
    for r in jan.to_dict('records'):
        for window,prefix,nmin,dmin in [('fullmonth','full',20,5),('formation','f',10,3),('validation','v',10,3)]:
            for side in ['L','R']:
                n=r[f'{prefix}_n_{side}'];days=r[f'{prefix}_days_{side}']
                fail.append({**{k:r[k] for k in KEY},'membership':r['membership'],'Rf':r['Rf'],'window':window,'side':side,'n':n,'days':days,'no_eligible_contained_transactions':n==0,'transactions_below_threshold':n<nmin,'dates_below_threshold':days<dmin,'support':n>=nmin and days>=dmin,'price_nonfinite_with_positive_n':n>0 and not np.isfinite(r[f'{prefix}_T_median']) if r[f'{prefix}_n_L']>0 and r[f'{prefix}_n_R']>0 else False})
    fd=pd.DataFrame(fail);save(fd,'task2/jan_support_failure_reasons.csv');save(pd.DataFrame(av),'task2/jan_formation_validation_availability.csv')
    fs=fd.groupby(['boundary_hour','membership','window','side']).agg(n_units=('station_id','size'),n_no_eligible=('no_eligible_contained_transactions','sum'),n_low_transactions=('transactions_below_threshold','sum'),n_low_dates=('dates_below_threshold','sum')).reset_index();save(fs,'task2/jan_support_failure_summary.csv')
    # Daily endpoint cells for union A|Af; include zero-activity dates and raw/eligible stages.
    mapping=[]
    for r in jan.to_dict('records'):
        for side in ['L','R']:
            seg=r[f'full_segment_{side}'] if pd.notna(r[f'full_segment_{side}']) else r[f'f_segment_{side}']
            mapping.append({k:r[k] for k in KEY}|dict(side=side,segment_id=seg,membership=r['membership'],Rf=r['Rf']))
    mp=pd.DataFrame(mapping);con.register('jan_mapping',mp)
    daily=con.execute("""WITH dates AS (SELECT CAST(d AS DATE) tariff_day FROM generate_series(DATE '2024-12-31',DATE '2025-01-31',INTERVAL 1 DAY) t(d)),
    cells AS (SELECT m.*,d.tariff_day FROM jan_mapping m CROSS JOIN dates d),
    agg AS (SELECT month,station_id,segment_id,tariff_day,count(*) raw_start_count,
      count(*) FILTER(WHERE base_valid) base_count,count(*) FILTER(WHERE contained) contained_raw_count,
      count(*) FILTER(WHERE base_valid AND contained) n,
      count(*) FILTER(WHERE contained AND NOT base_valid) invalid_contained_count,
      count(DISTINCT pile_id) FILTER(WHERE base_valid AND contained) devices,
      sum(total_elec_kwh) FILTER(WHERE base_valid AND contained) kwh,
      median(E) FILTER(WHERE base_valid AND contained) E_median,
      median(S) FILTER(WHERE base_valid AND contained) S_median,
      median(T) FILTER(WHERE base_valid AND contained) T_median
      FROM tx WHERE month='2025-01' GROUP BY month,station_id,segment_id,tariff_day)
    SELECT c.*,coalesce(raw_start_count,0) raw_start_count,coalesce(base_count,0) base_count,coalesce(contained_raw_count,0) contained_raw_count,
      coalesce(n,0) n,CAST(coalesce(n,0)>0 AS INTEGER) valid_date_indicator,coalesce(invalid_contained_count,0) invalid_contained_count,
      coalesce(devices,0) devices,coalesce(kwh,0) kwh,E_median,S_median,T_median,
      CASE WHEN c.tariff_day<DATE '2025-01-01' THEN 'outside_month_split' WHEN c.tariff_day<DATE '2025-01-16' THEN 'formation' ELSE 'validation' END half
    FROM cells c LEFT JOIN agg a USING(month,station_id,segment_id,tariff_day) ORDER BY boundary_hour,station_id,side,tariff_day""").df()
    save(daily,'task2/jan_daily_endpoint_activity.csv')
    # Summaries explain zero eligible cells without inventing missing-price causes.
    stages=daily.groupby(['boundary_hour','station_id','side','half']).agg(raw_start_count=('raw_start_count','sum'),base_count=('base_count','sum'),contained_raw_count=('contained_raw_count','sum'),eligible_contained_count=('n','sum'),invalid_contained_count=('invalid_contained_count','sum'),valid_dates=('valid_date_indicator','sum')).reset_index()
    save(stages,'task2/jan_endpoint_screening_stages.csv')
    # Deterministic trace examples: one per membership and one Rf-unobservable per boundary.
    selected=[]
    for b,g in jan.groupby('boundary_hour'):
        for cat,z in g.groupby('membership'):
            z=z.assign(score=(z.full_E_median-z.f_E_median).abs().fillna(-1)).sort_values(['score','station_id'],ascending=[False,True])
            rr=z.iloc[0].to_dict();rr['selection_reason']=cat+'_max_abs_full_vs_formation_E';selected.append(rr)
        z=g[g.Rf&~g.Ov].sort_values('station_id')
        if len(z):rr=z.iloc[0].to_dict();rr['selection_reason']='first_Rf_without_target';selected.append(rr)
    traces=pd.DataFrame(selected);save(traces,'task2/jan_unit_trace_examples.csv')
    trace_keys=traces[KEY].drop_duplicates();save(daily.merge(trace_keys,on=KEY),'task2/jan_trace_daily_records.csv')
    # Daily device composition on selected units; no disclosure of users (none provided).
    trmap=mp.merge(trace_keys,on=KEY);con.register('trace_mapping',trmap)
    td=con.execute("""SELECT m.month,m.boundary_hour,m.station_id,m.side,v.tariff_day,v.pile_id,count(*) n,sum(total_elec_kwh) kwh,median(rated_gun_power_kw) rated_power,median(E) E,median(S) S,median(T) T
    FROM trace_mapping m JOIN valid v USING(month,station_id,segment_id) GROUP BY m.month,m.boundary_hour,m.station_id,m.side,v.tariff_day,v.pile_id ORDER BY boundary_hour,station_id,side,tariff_day,pile_id""").df()
    save(td,'task2/jan_trace_device_daily.csv')
    return jan,allp

def task3(con,full,cfg):
    log('Task 3 field availability, joins, overlap and conditional H profiles')
    station_path=ROOT/cfg['inputs']['stations']['path']
    stations=con.execute('SELECT * FROM read_parquet(?)',[str(station_path)]).df()
    dump({str(station_path):[dict(name=c,dtype=str(stations[c].dtype)) for c in stations], 'orders':con.execute('DESCRIBE raw').df().to_dict('records')},'task3/raw_schema.json')
    save(stations[stations.duplicated('station_id',keep=False)],'task3/duplicate_station_rows.csv')
    check('duplicate_station_keys',stations.station_id.duplicated().sum(),0)
    static=['station_total_power_kw','piles_num','dc_piles_num','charging_gun_num','dc_charging_gun_num','construction_site']
    coverage=[];categories=[]
    for col in static:
        s=stations[col]
        r=dict(scope='station_file',month='static_unknown_date',field=col,n=len(s),missing=int(s.isna().sum()),missing_fraction=s.isna().mean(),unique_values=s.nunique(dropna=True),nonpositive=np.nan,noninteger=np.nan)
        if col!='construction_site':
            a=pd.to_numeric(s,errors='coerce');r.update(dist(a));r['n']=len(s);r['nonpositive']=int((a<=0).sum());r['noninteger']=int((a.notna()&(a!=np.floor(a))).sum())
        else:
            for value,n in s.fillna('<MISSING>').value_counts(dropna=False).items():categories.append(dict(field=col,value=str(value),n=int(n),fraction=n/len(s)))
        coverage.append(r)
    save(pd.DataFrame(categories),'task3/facility_category_values.csv')
    for m in ['2025-01','2025-07']:
        for col in ['rated_gun_power_kw','pile_id']:
            query=f"SELECT count(*) n,count(*) FILTER(WHERE {col} IS NULL) missing,count(DISTINCT {col}) unique_values FROM tx WHERE month='{m}'"
            r=con.execute(query).df().iloc[0].to_dict();r.update(scope='raw_transactions',month=m,field=col,missing_fraction=r['missing']/r['n'])
            if col=='rated_gun_power_kw':
                vals=con.execute(f"SELECT median({col}) median,quantile_cont({col},.25) q25,quantile_cont({col},.75) q75,min({col}) min,max({col}) max,count(*) FILTER(WHERE {col}<=0 OR NOT isfinite({col})) nonpositive_or_nonfinite FROM tx WHERE month='{m}'").df().iloc[0].to_dict();r.update(vals)
                cats=con.execute(f"SELECT {col} AS field_value,count(*) AS n FROM tx WHERE month='{m}' GROUP BY {col}").df()
                cats['month']=m;cats['field']=col;save(cats,f'task3/rated_power_values_{m}.csv')
            coverage.append(r)
    # Composite (month,station,pile) makes cross-station labels unambiguous; no claim of physical continuity.
    pid=con.execute("SELECT month,pile_id,count(DISTINCT station_id) stations,count(DISTINCT rated_gun_power_kw) rated_power_values FROM tx WHERE pile_id IS NOT NULL GROUP BY month,pile_id").df()
    save(pid[pid.stations.gt(1)],'task3/pile_labels_multiple_stations.csv')
    conflicts=con.execute("SELECT month,station_id,pile_id,count(DISTINCT rated_gun_power_kw) power_values FROM tx WHERE pile_id IS NOT NULL GROUP BY month,station_id,pile_id HAVING count(DISTINCT rated_gun_power_kw)>1").df();save(conflicts,'task3/device_power_record_conflicts.csv')
    joins=[]
    for m in ['2025-01','2025-07']:
        for pop,ids in [('raw_active',con.execute('SELECT DISTINCT station_id FROM tx WHERE month=?',[m]).df().station_id),('A',full[full.month.eq(m)].station_id),('H',full[full.month.eq(m)&full.H_median].station_id)]:
            z=pd.DataFrame({'station_id':ids.drop_duplicates()}).merge(stations,on='station_id',how='left',indicator=True,validate='one_to_one')
            joins.append(dict(month=m,population=pop,n_unique_stations=len(z),matched=int(z._merge.eq('both').sum()),coverage=z._merge.eq('both').mean()))
    save(pd.DataFrame(joins),'task3/station_attribute_join_audit.csv')
    g=full.merge(stations[['station_id']+static],on='station_id',how='left',validate='many_to_one')
    # Persist endpoint device sets only internally; output overlap counts and Jaccard.
    devices=con.execute('SELECT month,station_id,segment_id,list(DISTINCT pile_id) FILTER(WHERE pile_id IS NOT NULL) ids,count(*) FILTER(WHERE pile_id IS NULL) missing_pile FROM valid GROUP BY month,station_id,segment_id').df()
    lookup={(r.month,r.station_id,r.segment_id):(set(r.ids) if r.ids is not None else set(),r.missing_pile) for r in devices.itertuples()}
    overlap=[]
    for r in g.itertuples():
        a,ma=lookup[(r.month,r.station_id,r.segment_L)];b,mb=lookup[(r.month,r.station_id,r.segment_R)]
        overlap.append(dict(month=r.month,boundary_hour=r.boundary_hour,station_id=r.station_id,left_unique=len(a-b),right_unique=len(b-a),shared=len(a&b),union=len(a|b),jaccard=len(a&b)/len(a|b) if len(a|b) and ma+mb==0 else np.nan,missing_pile_records=ma+mb,identifier_scope='station_month_released_identifier',H=bool(r.H_median),amount_group=r.amount_median))
    ov=pd.DataFrame(overlap);save(ov,'task3/charger_overlap_profile.csv');g=g.merge(ov[KEY+['left_unique','right_unique','shared','jaccard']],on=KEY,validate='one_to_one')
    g['transaction_count_ratio_R_L']=g.n_R/g.n_L;g['kwh_ratio_R_L']=g.kwh_R/g.kwh_L
    g['transaction_count_difference_R_L']=g.n_R-g.n_L;g['kwh_difference_R_L']=g.kwh_R-g.kwh_L
    g['min_support_days']=g[['days_L','days_R']].min(axis=1)
    for c in static:
        coverage.append(dict(scope='fixed_H_units',month='ALL',field=c,n=int(g.H_median.sum()),missing=int(g.loc[g.H_median,c].isna().sum()),missing_fraction=g.loc[g.H_median,c].isna().mean(),unique_values=g.loc[g.H_median,c].nunique()))
    save(pd.DataFrame(coverage),'task3/facility_field_coverage.csv')
    h=g[g.H_median];profile=[]
    features=static[:-1]+['rated_power_tx_median_L','rated_power_tx_median_R','n_L','n_R','kwh_L','kwh_R','days_L','days_R','devices_L','devices_R','jaccard','left_unique','right_unique','shared','transaction_count_ratio_R_L','kwh_ratio_R_L','transaction_count_difference_R_L','kwh_difference_R_L','min_support_days']
    scopes=[('ALL',h)]+[(f'{m}_{b:02d}',z) for (m,b),z in h.groupby(['month','boundary_hour'])]
    for layer,z in scopes:
        for group,part in z.groupby('amount_median'):
            for col in features:profile.append(dict(layer=layer,amount_group=group,n_units=len(part),n_stations=part.station_id.nunique(),variable=col,variable_class='facility_snapshot' if col in static else ('observed_transaction_power' if col.startswith('rated_') else 'observed_activity'),missing=int(part[col].isna().sum()),**dist(part[col])))
            for val,n in part.construction_site.fillna('<MISSING>').value_counts().items():profile.append(dict(layer=layer,amount_group=group,n_units=len(part),n_stations=part.station_id.nunique(),variable='construction_site_raw_code',variable_class='released_poi_site_category',category=str(val),count=int(n),fraction=n/len(part)))
    pr=pd.DataFrame(profile);save(pr,'task3/H_amount_group_profile.csv')
    save(pr[pr.variable_class.eq('observed_activity')],'task3/activity_support_profile.csv')
    rr=[]
    r=g[g.R_median]
    for layer,z in [('ALL',r)]+[(f'{m}_{b:02d}',z) for (m,b),z in r.groupby(['month','boundary_hour'])]:
        for direction,part in z.groupby('direction_median'):rr.append(dict(layer=layer,direction=direction,n_units=len(part),n_stations=part.station_id.nunique(),denominator=len(z),fraction=len(part)/len(z)))
    save(pd.DataFrame(rr),'task3/R_direction_profile.csv');save(g,'task3/analysis_unit_attributes.csv')
    check('R_neutral',len(r[r.direction_median.eq('neutral')]),27);check('R_reversed',len(r[r.direction_median.eq('negative')]),25)
    return g

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-db',action='store_true');args=parser.parse_args()
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    files=[CFG,ROOT/'analysis/price_architecture_v2/01_menu_feasibility.py',ROOT/'analysis/price_architecture_v2/02_boundary_signal_core.py',EVID/'provenance/public_code/reconstruction_engine.py',EVID/'scripts/vendor/module_b_stats.py',EVID/'inputs/module_b_row_level_inputs.csv.gz',ROOT/'outputs/09_price_architecture/02_boundary_signal_core/boundary_signal_panel.parquet']
    files += [ROOT/cfg['inputs'][x]['path'] for x in ['orders_2025-01','orders_2025-07','stations']]
    files += [p for p in BASE.rglob('*') if p.is_file()]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(before,'validation/protected_input_hashes_before.json')
    dump({'python':sys.version,'duckdb':duckdb.__version__,'pandas':pd.__version__,'numpy':np.__version__,'platform':platform.platform(),'audit_script_sha256':sha(Path(__file__)),'date':'2026-11-16','bootstrap_runs':0,'regression_runs':0},'validation/run_environment.json')
    for key in ['orders_2025-01','orders_2025-07','stations']:
        check('hash_'+key,before[cfg['inputs'][key]['path']]==cfg['inputs'][key]['sha256'],True)
    con=duckdb.connect(str(OUT/'cache/audit.duckdb'));con.execute("SET threads=1");con.execute("SET memory_limit='3GB'");con.execute(f"SET temp_directory='{(OUT/'cache/duckdb_temp').as_posix()}'")
    if not args.reuse_db:build_db(con,cfg)
    else:
        for name,query,expected in [('raw_transactions','SELECT count(*) FROM tx',8544695),('price_base','SELECT count(*) FROM tx WHERE base_valid',8495859),('contained','SELECT count(*) FROM valid',6262859)]:check(name,con.execute(query).fetchone()[0],expected)
    panel=build_panel(con,cfg);full,pairs,frozen=baseline(panel)
    log('Baseline reconstructed successfully')
    fee_audit(con);task1(full,pairs,load_stats());jan,allp=task2(con,panel);task3(con,full,cfg)
    con.close()
    after={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(after,'validation/protected_input_hashes_after.json')
    check('all_protected_files_unchanged',before==after,True)
    save(pd.DataFrame(checks),'validation/checks.csv')
    log(f'Completed {len(checks)} checks. All requested numeric outputs written.')

if __name__=='__main__': main()
