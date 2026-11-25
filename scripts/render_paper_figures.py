"""Display-only edits to saved quantities. No new estimation or selection."""
from pathlib import Path
import hashlib,json,re,sys
R=Path(__file__).resolve().parents[1]
W=R
def read(p):return p.read_text(encoding='utf-8-sig')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
P=R
import argparse
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
SRC=args.output.resolve();SRC.mkdir(parents=True,exist_ok=True);Q=SRC
if SRC==R or R in SRC.parents:raise SystemExit('Output must be outside the package')
blue='#255A78';gold='#C68A2F';teal='#008775';gray='#93999E';red='#AD4B43'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.3,'ytick.labelsize':8.4,'legend.fontsize':8.4,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.axisbelow':True,'savefig.facecolor':'white'})
def save(fig,num):
    fig.savefig(SRC/f'Figure_{num:02}.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(Q/f'Figure_{num:02}.png',dpi=180)
    plt.close(fig)
u=P/'data/derived/common/unit_level_signed.csv';d=pd.read_csv(u,low_memory=False);d=d[d.A];b=d[d.system=='Beijing'];s=d[d.system=='UrbanEV']
# Reuse original figure 3/4 display code verbatim apart from requested text.
import ast
old=ast.literal_eval(read(R/'scripts/display_components_frozen.py').split('DISPLAY_SOURCE = ',1)[1])+'# F5 use'
block=old[old.index('# F3:'):old.index('# F5 use')]
block=block.replace("['Smaller','Within band','Larger']","['Smaller (< −0.90)','Within ±0.90','Larger (> 0.90)']")
block=block.replace("ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.52,-.48),frameon=False,handlelength=1.2)","ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.52,-.48),frameon=False,handlelength=1.2,fontsize=8,title='RMB per 30 kWh' if k else None,title_fontsize=8)")
block=block.replace("'Dashed reference: RMB 0.90'","'RMB 0.90'")
block=block.replace("figsize=(6.25,4.25)","figsize=(6.25,4.55)").replace("bottom=.19,top=.91","bottom=.24,top=.91")
block=re.sub(r';rec\(3,.*?\)\n','\n',block)
block=re.sub(r';refcode=.*?\n','\n',block)
block=block.replace("' (RMB per reference 30 kWh)'", "' (RMB per reference 30 kWh)\\n(symmetric log scale)'")
exec(block,globals())
# Original two heatmaps, keeping all cells and hatching.
heat=P/'data/reference/historical/urbanev/e1_layers.csv';d=pd.read_csv(heat)
months=sorted(d.month.unique());hours=sorted(d.boundary_minute.unique())
fig,axes=plt.subplots(2,1,figsize=(6.25,5.9));small=[]
for ax,col,denom,title,cmap in zip(axes,['H_over_R','H_absD30_over_090'],['R','H'],['(a) Direction-retaining share among electricity-qualified comparisons','(b) Amount discrepancy among direction-retaining comparisons'],['Blues','Oranges']):
    vals=d.pivot(index='month',columns='boundary_minute',values=col).reindex(index=months,columns=hours).to_numpy()*100
    n=d.pivot(index='month',columns='boundary_minute',values=denom).reindex(index=months,columns=hours).to_numpy()
    im=ax.imshow(vals,vmin=0,vmax=100,cmap=cmap,aspect='auto')
    for i in range(len(months)):
        for j in range(len(hours)):
            tiny=0<n[i,j]<5
            if tiny:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=(1,1,1,.45),edgecolor='#7a858d',hatch='///',linewidth=.5))
                small.append({'month':months[i],'boundary_minute':int(hours[j]),'denominator':denom,'n':int(n[i,j])})
            label=f'{vals[i,j]:.0f}%\n{denom}={n[i,j]:.0f}'+('*' if tiny else '') if np.isfinite(vals[i,j]) else f'NA\n{denom}=0'
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if not tiny and np.isfinite(vals[i,j]) and vals[i,j]>65 else '#17232b')
    ax.set_yticks(range(len(months)),months);ax.set_xticks(range(len(hours)),[f'{h//60:02}:{h%60:02}' for h in hours])
    ax.set_title(title,loc='left',fontsize=8.7,fontweight='bold',pad=8)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.025).set_label('Percent')
axes[1].set_xlabel('Reference clock boundary')
fig.subplots_adjust(left=.13,right=.91,bottom=.13,top=.94,hspace=.34)
fig.text(.13,.033,'NA: empty denominator     * Hatched: n < 5',fontsize=7.6)
save(fig,5)
# Same 156 observed grid records, verified in the preserved precheck.
trace=P/'data/derived/external/station1006_display.csv'
x=pd.read_csv(trace,parse_dates=['time']);assert len(x)==156
start=pd.Timestamp('2022-09-01 19:00');mid=pd.Timestamp('2022-09-02 00:00');end=pd.Timestamp('2022-09-02 08:00')
assert x.time.min()==start and x.time.max()==end-pd.Timedelta(minutes=5)
at=lambda t: x.loc[x.time==pd.Timestamp(t),['e_price','s_price','T']].iloc[0]
assert np.allclose(at('2022-09-01 22:55'),[.7402,.2598,1])
assert np.allclose(at('2022-09-01 23:00'),[.2962,.7038,1])
assert np.allclose(at('2022-09-02 00:00'),[.2962,1.1438,1.44])
fig,ax=plt.subplots(figsize=(6.25,6.25/2.6))
fig.subplots_adjust(left=.105,right=.955,bottom=.27,top=.97)
ax.axvspan(start,mid,color='#eef3f6',zorder=0);ax.axvspan(mid,end,color='#f9f4e9',zorder=0)
for col,label,color in [('e_price','Electricity',blue),('s_price','Service',gold),('T','Total',teal)]:
    ax.step(list(x.time)+[end],list(x[col])+[x[col].iloc[-1]],where='post',label=label,color=color,lw=1.45,zorder=3)
ax.axvline(mid,color='#626b70',ls='--',lw=.85,zorder=2)
ax.set(xlim=(start,end),ylim=(0,2.55),ylabel='Price (RMB/kWh)',yticks=[0,.5,1,1.5,2])
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.03),ncol=3,frameon=False,fontsize=8.2)
ax.text(pd.Timestamp('2022-09-01 21:30'),2.05,'Before boundary (19:00–24:00)',ha='center',fontsize=7.7)
ax.text(pd.Timestamp('2022-09-02 04:00'),2.05,'After boundary (00:00–08:00)',ha='center',fontsize=7.7)
ax.text(mid+pd.Timedelta(minutes=10),1.82,'Midnight boundary',ha='left',va='center',fontsize=7.5,color='#50585b')
ticks=pd.to_datetime(['2022-09-01 19:00','2022-09-01 22:00','2022-09-02 00:00','2022-09-02 03:00','2022-09-02 06:00','2022-09-02 08:00'])
ax.set_xticks(ticks);ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.text(2.5/13,-.25,'1 Sep',transform=ax.transAxes,ha='center',fontsize=8)
ax.text(9/13,-.25,'2 Sep',transform=ax.transAxes,ha='center',fontsize=8)
save(fig,6)
protected=True  # This renderer never writes Figure 1.
js(Q/'figure_changes.json',{'Figure1_unchanged':protected,'colors_from_original_Figure2_code':{'E':blue,'S':gold,'T':teal},'frozen_inputs':[{'file':str(p),'sha256':sha(p)} for p in [u,heat,trace]],'figures_changed':[3,4,5,6],'Figure3_scope':'pooled Beijing and Shenzhen','trace_rows':156,'service_rises_at_2300':True,'service_rises_at_midnight':True,'midnight_electricity_unchanged':True,'heatmap_cells':72,'small_denominator_cells':small})
print('Four figures rendered from frozen values. Figure 1 protected.')

# Export all finalized vector assets. The independently redrawn panels above
# remain alongside these assets for comparison with their saved data.
import shutil
for asset in sorted((R/'figures').rglob('*.pdf')):
    target=SRC/'final_vectors'/asset.relative_to(R/'figures')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(asset,target)
print('Exported 20 final figure assets, including author Figure 1 without alteration.')
