"""Export three SI1 tables from the frozen output-57/58 audit summaries.

No statistical analysis or raw-data reconstruction is run. This reads relative
paths in the source or Supplementary Data package. Full-precision display CSVs
and TeX fragments are written; insertion into the flat master is a separate step.
"""
from pathlib import Path
import csv,json,hashlib
BASE=Path(__file__).resolve().parents[1]
A=BASE/'r423_audits'
T=BASE/'tables';T.mkdir(exist_ok=True)
O=BASE/'results/r423_integrated';O.mkdir(parents=True,exist_ok=True)
used=[]
def read(rel):
    p=A/rel;used.append({'path':str(p.relative_to(BASE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(name,rows):
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def spec(widths):
    n=len(widths);return '@{}'+''.join('>{\\'+('raggedright' if i==0 else 'raggedleft')+r'\arraybackslash}p{\dimexpr\textwidth/100*'+str(w)+f'-{2*(n-1)}\\tabcolsep/{n}'+r'\relax}' for i,w in enumerate(widths))+'@{}'
def begin(caption,label):
    return r'\begin{table}[!htbp]\centering\begin{minipage}{\textwidth}\raggedright'+'\n'+r'\begingroup\linespread{1}\fontsize{9}{11}\selectfont\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.14}'+'\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n'
def tab(widths,head,rows):return '\\begin{tabular}{'+spec(widths)+'}\\toprule\n'+head+r'\\\midrule'+'\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'
def end(note):return '\\SharedNote{'+note+'}\n'+r'\endgroup\end{minipage}\end{table}'+'\n'

data=read('task2/results/summary_metrics.csv')
keys=['E','CAL','E_plus_S','LOCAL'];schemes=['layer_equal','month_equal','pooled_unit']
actual={r['scheme'] for r in data};print('Aggregation schemes',actual)
if 'pooled_unit_equal' in actual:schemes[2]='pooled_unit_equal'
if 'pooled_equal' in actual:schemes[2]='pooled_equal'
if 'pooled_equalunit' in actual:schemes[2]='pooled_equalunit'
assert set(schemes)==actual
lookup={(r['scheme'],r['configuration']):r for r in data}
assert all(int(r['n'])==5846 for r in data)
f=begin('Sensitivity of cross-window assessment to aggregation weights and error metrics.','tab:r423-aggregation')
header=r'Aggregation scheme & E & CAL & E+S & LOCAL & \shortstack[r]{E+S vs E\\reduction} & \shortstack[r]{LOCAL vs E\\reduction}'
rows_out=[]
for panel,metric in [('A','mae'),('B','rmse')]:
    f+=f'\\PanelTitle{{\\textbf{{Panel {panel}.}} {metric.upper()} (RMB/kWh)}}\n'
    rows=[]
    for scheme,name in zip(schemes,['Ten-layer equal','Month equal','Pooled-unit equal']):
        vals=[lookup[scheme,k] for k in keys];redkey=f'relative_E_{metric.upper()}_improvement_pct'
        rows.append([name]+[f'{float(v[metric]):.9f}' for v in vals]+[f'{float(vals[i][redkey]):.4f}\\%' for i in [2,3]])
        rows_out.append({'metric':metric,'scheme':scheme,**{k:vals[i][metric] for i,k in enumerate(keys)},'E_plus_S_reduction_pct':vals[2][redkey],'LOCAL_reduction_pct':vals[3][redkey],'n':5846})
    f+=tab([18,14,14,14,14,13,13],header,rows)
f+=end(r'All rows use the same 5,846 units, fixed inputs and subsequent observed total-price target. Weights are defined in Section~\ref{sec:r423-aggregation}. Reductions are relative to E within each row; RMSE is the square root of the weighted MSE. No new intervals are estimated.')
(T/'r423_aggregation.tex').write_text(f,encoding='utf8');csvout('aggregation_display.csv',rows_out)

summ=read('task3/summary.csv');daily=read('task3/daily_support.csv')
f=begin('Sensitivity to daily native-interval grid completeness requirements.','tab:r423-grid')
rows=[];out=[]
for thresh in [.7,.8,.9]:
    r=next(x for x in summ if float(x['threshold'])==thresh and x['sample']=='own_A' and x['layer']=='pooled')
    d=next(x for x in daily if float(x['threshold'])==thresh)
    out.append({**d,**r})
    rows.append([f'{thresh:.0%}'.replace('%',r'\%')]+[f'{int(d[k]):,}' for k in ['supported_day_segments','station_days_any_segment','station_days_all9']]+[f'{int(r[k]):,}' for k in ['A','stations','R','H']]+[f'{float(r["H_over_R"])*100:.4f}\\%'])
header=r'Grid rule & \shortstack[r]{Valid daily\\intervals} & \shortstack[r]{Site-days:\\any interval} & \shortstack[r]{Site-days:\\all nine} & $A$ & Stations & $R$ & $H$ & $H/R$'
f+=tab([8,14,14,14,11,10,8,8,13],header,rows)
f+=end(r'The grid rule applies to valid/expected five-minute observations within each native daily interval, not the full 288-point day. $A$ counts locally supported station--month--boundary units; $R$ is electricity-qualified and $H$ direction-retained. Stations are unique members of $A$; $H/R$ is pooled across units, not layer-equal.')
(T/'r423_grid.tex').write_text(f,encoding='utf8');csvout('urbanev_grid_display.csv',out)

cohort=read('station/raw_station/cohort_endpoint_prices.csv')
endpoint=read('station/raw_station/endpoint_summary.csv')
contrast=read('station/raw_station/fixed_pile_cohort_contrasts.csv')
rank=read('station/influence/station_influence_ranking.csv')
loo=read('station/influence/leave_one_station_cluster_out.csv')
f=begin('Station influence and observed pile composition: ST\_003774.','tab:r423-station-influence')
f+=r'\PanelTitle{\textbf{Panel A.} January validation contrasts and transaction counts by observed pile group}'+'\n'
rows=[];out=[]
for hour,before,after in [(7,'23_07','07_10'),(13,'10_13','13_17')]:
    vals=[]
    for seg in [before,after]:
        counts=[]
        for group in ['observed_in_formation','first_observed_validation']:
            r=next((x for x in cohort if x['window']=='validation' and x['segment_id']==seg and x['pile_cohort']==group),None)
            counts.append(int(r['n']) if r else 0)
        vals+=counts
    med=[float(next(x for x in endpoint if x['window']=='validation' and x['segment_id']==seg)['T_median']) for seg in [before,after]]
    cs=[float(next(x for x in contrast if x['window']=='validation' and int(x['boundary_hour'])==hour and x['pile_cohort']==group)['T_star']) for group in ['all','observed_in_formation']]
    rows.append([f'{hour:02d}:00',f'{vals[0]} / {vals[1]}',f'{vals[2]} / {vals[3]}',f'{med[0]:.6f} $\\to$ {med[1]:.6f}',f'{cs[0]:.6f}',f'{cs[1]:.6f}'])
    out.append(dict(boundary_hour=hour,before_original_tx=vals[0],before_first_observed_tx=vals[1],after_original_tx=vals[2],after_first_observed_tx=vals[3],before_T=med[0],after_T=med[1],T_star=cs[0],fixed_original_T_star=cs[1]))
f+=tab([10,14,14,28,17,17],r'Boundary & \shortstack[r]{Before tx:\\original / first} & \shortstack[r]{After tx:\\original / first} & \shortstack[r]{Station total-price medians\\before $\to$ after} & \shortstack[r]{Oriented\\contrast} & \shortstack[r]{Fixed original-\\pile contrast}',rows)
csvout('station_composition_display.csv',out)
f+=r'\PanelTitle{\textbf{Panel B.} Contribution shares and whole-station removal}'+'\n'
rows=[];out=[]
for k in keys:
    r=next(x for x in rank if x['station_id']=='ST_003774' and x['configuration']==k)
    full=next(x for x in loo if x['removed_station']=='NONE' and x['configuration']==k)
    minus=next(x for x in loo if x['removed_station']=='ST_003774' and x['configuration']==k)
    rows.append([k.replace('E_plus_S','E+S'),f'{float(r["MAE_share_pct"]):.4f}\\%',f'{float(r["MSE_share_pct"]):.4f}\\%',f'{float(full["MAE"]):.9f} $\\to$ {float(minus["MAE"]):.9f}',f'{float(full["RMSE"]):.9f} $\\to$ {float(minus["RMSE"]):.9f}'])
    out.append({'input':k,'mae_share_pct':r['MAE_share_pct'],'mse_share_pct':r['MSE_share_pct'],'original_MAE':full['MAE'],'leaveout_MAE':minus['MAE'],'original_RMSE':full['RMSE'],'leaveout_RMSE':minus['RMSE'],'removed_units':minus['removed_units']})
f+=tab([8,14,14,32,32],r'Input & \shortstack[r]{MAE share} & \shortstack[r]{MSE share} & \shortstack[r]{MAE\\original $\to$ leave-one-out} & \shortstack[r]{RMSE\\original $\to$ leave-one-out}',rows)
f+=end(r'Prices and errors are RMB/kWh. Panel A counts transactions, not piles: original = the two identifiers observed in formation; first = the ten first observed in validation. Contrasts use the boundary orientation. Panel B removes all seven evaluation units of this station and renormalizes remaining within-layer ST weights, retaining ten equal layer weights and all fixed inputs, including CAL. This is an influence diagnostic, not a refit or a replacement sample.')
(T/'r423_station_influence.tex').write_text(f,encoding='utf8');csvout('station_influence_display.csv',out)
(O/'PROVENANCE.json').write_text(json.dumps({'sources':used,'operation':'format frozen audited summaries; no raw reconstruction or statistical reestimation','generator':'analysis_evidence/scripts/build_r423_tables.py'},indent=2),encoding='utf8')
print('Three tables and four display CSVs exported.')
