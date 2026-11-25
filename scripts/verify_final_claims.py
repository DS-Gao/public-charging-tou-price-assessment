"""Check manuscript claims from bundled derived/reference data, never raw acquisition."""
from pathlib import Path
import csv,json,re,statistics
R=Path(__file__).resolve().parents[1]
def rows(p):return list(csv.DictReader((R/p).open(encoding='utf-8-sig')))
checks=[]
def check(name,a,b,tol=0):
 ok=abs(a-b)<=tol if isinstance(a,(int,float)) else a==b
 checks.append(dict(check=name,actual=a,expected=b,passed=ok));assert ok,(name,a,b)
u=rows('data/derived/common/unit_level_signed.csv')
for city,expected in [('Beijing',[8133,7531,7479,3916]),('UrbanEV',[49369,4921,1377,48180])]:
 a=[x for x in u if x['system']==city and x['A']=='True'];r=[x for x in a if x['R']=='True'];h=[x for x in a if x['H']=='True']
 check(city+' support',[len(a),len(r),len(h),sum(x['C']=='True' for x in a)],expected)
 if city=='Beijing':
  check('direction',round(100*len(h)/len(r),2),99.31)
  check('amount',round(100*sum(abs(float(x['D30']))>.9 for x in h)/len(h),2),36.18)
  check('complete-menu loss',round(100*sum(x['C']!='True' for x in a)/len(a),1),51.9)
 else:
  check('overall loss',round(100*sum(x['C']!='True' for x in a)/len(a),2),2.41)
  check('H loss',round(100*sum(x['C']!='True' for x in h)/len(h),2),49.24)
for x in rows('analysis/phase_a/task5/Table9_panelA.csv'):
 check(x['operator']+' members',int(x['n']),5846)
 check(x['operator']+' positive paired point',float(x['E_to_E_plus_S'])>0,True)
 if x['operator']=='median':
  check('E MAE',round(float(x['E_mae']),5),.03647);check('E+S MAE',round(float(x['E_plus_S_mae']),5),.01745)
 elif x['operator']=='mean':check('mean paired reduction',round(float(x['E_to_E_plus_S']),6),.019527)
 else:check('VWAP paired reduction',round(float(x['E_to_E_plus_S']),6),.019325)
m=rows('analysis/phase_a/task2/primary_metrics.csv')
check('positive layer points',sum(float(x['E_to_E_plus_S'])>0 for x in m if x['scope']=='layer' and x['weight']=='ST' and x['group']=='all'),10)
n=rows('data/additions/batch1/task5/qualified12_monthly_summary.csv')
check('noon reversal/qualified',[sum(int(x[k]) for x in n) for k in ['original_reversed','original_R']],[676,685])
j=next(x for x in rows('data/additions/batch1/task2/second_half_activity_change.csv') if x['month']=='2025-01' and x['scheme']=='natural_1_15_vs_16_end')
check('January daily increase',round(-100*float(j['exposure_adjusted_reduction']),2),121.09)
for x in rows('data/additions/batch1/task2/reused_jan_formation_validation_availability.csv'):
 if int(x['boundary_hour'])==13:check('January13 membership',[int(x[k]) for k in ['A','R','Af','Rf','Af_and_Ov']],[776,634,137,12,136])
c=rows('data/additions/batch2/taskC/calibration_metrics.csv')
for config,expected in [('E_plus_S',63.88),('CAL_scene',.106),('CAL_size',.008),('CAL_power',.011)]:
 population='CAL_power_available_only' if config=='CAL_power' else 'original_V'
 x=next(x for x in c if x['scope']=='summary' and x['population']==population and x['month']=='2025-07' and x['weight']=='ST' and x['configuration']==config)
 check('July '+config,round(float(x['E_relative_mae_reduction'])*100,2 if config=='E_plus_S' else 3),expected)
screen=next(x for x in rows('data/reference/application/screening_pooled.csv') if x['system']=='Beijing' and x['population']=='A' and float(x['nominal_fraction'])==0.2)
check('monitor shortlist counts',[int(screen[k]) for k in ['E_selected_n','T_selected_n','intersection_n','symmetric_difference_n']],[1634,1631,861,1543])
check('monitor shortlist',round((int(screen['E_selected_n'])-int(screen['intersection_n']))/int(screen['E_selected_n'])*100,1),47.3)
bj=[x for x in u if x['system']=='Beijing' and x['A']=='True']
check('residual above .03',sum(abs(float(x['residual_star_reconstructed']))>.03 for x in bj),45)
cls=lambda t: 1 if t>.03 else -1 if t<-.03 else 0
check('total class changes on R',sum(cls(float(x['T_star'])) != cls(float(x['T_star'])-float(x['residual_star_reconstructed'])) for x in bj if x['R']=='True'),9)
claims=rows('CLAIMS_TO_FILES.csv');missing=[]
for x in claims:
 for name in x['file'].split(';'):
  if not (R/name).is_file():missing.append([x['claim_id'],name])
check('mapped files present',missing,[])
print(json.dumps({'passed':True,'numerical_checks':checks,'mapped_claims':len(claims)},indent=2))
