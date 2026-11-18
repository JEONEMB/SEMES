"""Scientific figures matching the reference PPT's analysis and graph families."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image

BG='#f7f7f9';COLORS=['#0070e8','#d84a00','#0f766e','#7c3aed','#0891b2','#be185d']


def main(out,previous,root):
    target=out/'figures';target.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/NotoSansKR-VF.ttf')
    if font.exists():
        # FreeType/Matplotlib may use the variable font's thin default axis.
        # Pin the actual outlines to regular weight for readable scientific axes.
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        regular=Path(tempfile.gettempdir())/'offaxis-NotoSansKR-Regular.ttf'
        if not regular.exists():
            instantiateVariableFont(TTFont(font),{'wght':400},inplace=True).save(regular)
        font_manager.fontManager.addfont(regular)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(regular)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'figure.facecolor':BG,'axes.facecolor':BG,
        'axes.edgecolor':'#dce1e8','axes.labelcolor':'#475569','xtick.color':'#475569','ytick.color':'#475569',
        'grid.color':'#dce1e8','grid.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':BG,'legend.frameon':False,'legend.fontsize':7.5})
    inv=pd.read_csv(out/'frames.csv');m=pd.read_csv(out/'mean_displacements.csv');f=pd.read_csv(out/'linear_fits.csv')
    r=pd.read_csv(out/'frame_linear_residuals.csv');n=pd.read_csv(out/'point_noise.csv');st=pd.read_csv(out/'steps.csv')
    pts=pd.read_csv(out/'displacements.csv.gz');ec=pd.read_csv(out/'ecc.csv')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    groups=json.loads((out/'groups.json').read_text(encoding='utf-8'))
    exported={}
    def save(fig,name):
        fig.tight_layout(pad=.7);fig.savefig(target/(name+'.png'),dpi=190);plt.close(fig);exported[name]=name+'.png'
    def style(ax,x='',y=''):
        ax.set(xlabel=x,ylabel=y);ax.grid(True,alpha=.8);ax.set_axisbelow(True)
    def paths(method,scope='new'):
        fig,axs=plt.subplots(2,1,figsize=(7.1,5.2))
        for ax,axis,fixed in zip(axs,['y_mm','theta_deg'],['theta_deg','y_mm']):
            gg=[g for g in groups if g['scope']==scope and g['varying']==axis]
            for i,g in enumerate(gg):
                data=m[(m.method==method)&m.path.isin(g['paths'])]
                ax.scatter(data.dx,data.dy,s=5,color=COLORS[i],label=(f"Θ={g['fixed_value']:g}°" if axis=='y_mm' else f"Y={g['fixed_value']:g}mm") if scope=='new' else g['group'][4:])
                pose=data.groupby(axis)[['dx','dy']].mean().sort_index()
                ax.plot(pose.dx,pose.dy,lw=.7,color=COLORS[i])
            ax.scatter([0],[0],marker='*',s=45,color='#111827',label='center1' if scope=='new' else 'center')
            selected_paths=[p for gd in gg for p in gd['paths']]
            view=m[m.path.isin(selected_paths)&m.method.isin(['no_diagonal630','ECC'])]
            for dim,setter in [('dx',ax.set_xlim),('dy',ax.set_ylim)]:
                lo,hi=view[dim].min(),view[dim].max();pad=max((hi-lo)*.07,.001);setter(lo-pad,hi+pad)
            # Numeric labels stay in raw pixel units; one vertical pixel of motion
            # occupies 100 times the screen distance of one horizontal pixel.
            ax.set_aspect(100,adjustable='datalim')
            ax.set_title(('Y 분석 그룹' if axis=='y_mm' else 'Θ 분석 그룹')+'  가로 ×1 / 세로 ×100',loc='left',fontsize=9)
            style(ax,'가로 이동 (px)','세로 이동 (px)');ax.legend(loc='upper left',ncol=3,fontsize=6.5)
        save(fig,'path_'+scope+'_'+method)
    paths('no_diagonal630');paths('ECC');paths('no_diagonal630','old');paths('ECC','old')
    # Source slide 6: remove slope, retain the extrapolated intercept.
    fig,axs=plt.subplots(1,2,figsize=(8.2,4.5))
    for ax,axis in zip(axs,['y_mm','theta_deg']):
        for i,(group,g) in enumerate(r[(r.scope=='new')&(r.method=='no_diagonal630')&(r.varying==axis)].groupby('group')):
            ax.scatter(g.command,g.slope_removed_dx,s=6,color=COLORS[i],label=group.replace('new_',''))
            model=f[(f.group==group)&(f.method=='no_diagonal630')].iloc[0]
            ax.axhline(model.intercept_dx,color=COLORS[i],lw=.8,ls='--')
        ax.scatter([0],[0],marker='*',c='#111827',s=50,label='center1')
        style(ax,'명목 Y (mm)' if axis=='y_mm' else '명목 Θ (°)','직선 기울기 제거 후 가로 이동 (px)');ax.legend(fontsize=6)
    save(fig,'zero_intercepts')
    # Source slide 9: constant-increment pools only, first center step excluded.
    fig,axs=plt.subplots(2,1,figsize=(7.1,5.0))
    for ax,axis in zip(axs,['y_mm','theta_deg']):
        ss=st[(st.scope=='new')&(st.varying==axis)&(st.method=='no_diagonal630')&~st.first_from_center]
        for i,(step,g) in enumerate(ss.groupby('command_step')):
            ax.scatter(g.to_command,g.step_dx,s=10,color=COLORS[i%6],label=f'Δ={step:g}'+('mm' if axis=='y_mm' else '°'))
            ax.axhline(g.mean_dx.iloc[0],color=COLORS[i%6],lw=.6)
            for sign in [-1,1]:ax.axhline(g.mean_dx.iloc[0]+sign*3*g.sigma_dx.iloc[0],color=COLORS[i%6],lw=.5,ls='--',alpha=.5)
        style(ax,'명령값 (도착 위치)','한 칸 가로 이동 (px)');ax.legend(ncol=3,fontsize=6)
    save(fig,'steps')
    # Chronology proxy and brightness, analogous to source slide 11.
    common=r[(r.scope=='new')&(r.varying=='theta_deg')&(r.method=='no_diagonal630')].merge(inv[['path','time_proxy_minutes']],on='path')
    b=pd.read_csv(previous/'brightness.csv').merge(inv[['path','time_proxy_minutes']],on='path')
    fig,axs=plt.subplots(2,2,figsize=(12.8,6.4))
    for i,(y,g) in enumerate(inv[inv.category=='mixed'].groupby('y_mm')):
        axs[0,0].scatter(g.time_proxy_minutes,g.theta_deg/2,s=7,color=COLORS[i],label=f'Y={y:g}mm')
    refs=inv[inv.category=='new-reference'];axs[0,0].scatter(refs.time_proxy_minutes,[0]*len(refs),marker='*',color='#111827',s=45,label='기준점')
    axs[0,0].legend(ncol=3,fontsize=7);axs[0,0].set_title('촬영 순서 대리 지표 (파일 수정 시각)',loc='left',fontsize=11)
    style(axs[0,0],'center1 이후 파일 시각 (분)','Θ / 2°')
    fi=f[(f.scope=='new')&(f.varying=='theta_deg')&(f.method=='no_diagonal630')]
    axs[0,1].barh(fi.group.str.replace('new_',''),fi.intercept_dx,color=COLORS[:len(fi)])
    style(axs[0,1],'0점 외삽 절편 (px)','');axs[0,1].set_title('0점 어긋남 (직선 절편)',loc='left',fontsize=11)
    ss=st[(st.scope=='new')&(st.varying=='theta_deg')&(st.method=='no_diagonal630')&st.command_step.eq(1)]
    for i,(g,gg) in enumerate(ss.groupby('group')):axs[1,0].scatter(gg.repeat,gg.step_dx,s=15,color=COLORS[i],label=g.replace('new_',''))
    style(axs[1,0],'반복 파일 번호','Θ 1° 한 칸 가로 이동 (px)');axs[1,0].set_xticks([1,2,3]);axs[1,0].legend(fontsize=7)
    axs[1,0].set_title('같은 1° 구간의 반복 차이',loc='left',fontsize=11)
    bb=b[b.category=='mixed'].groupby('path',as_index=False).agg(time=('time_proxy_minutes','first'),brightness=('mean_gray','mean'),y=('y_nominal_mm','first'))
    for i,(y,g) in enumerate(bb.groupby('y')):axs[1,1].scatter(g.time,g.brightness,s=7,color=COLORS[i],label=f'Y={y:g}')
    style(axs[1,1],'center1 이후 파일 시각 (분)','QR 패치 평균 밝기');axs[1,1].set_title('밝기 변화',loc='left',fontsize=11)
    save(fig,'timing_brightness')
    # Source slide 12: vertical relation, common residual, point histogram, diagonal effect.
    fig,axs=plt.subplots(2,2,figsize=(12.8,6.4))
    mm=m[(m.category=='mixed')&(m.method=='no_diagonal630')]
    for i,(y,g) in enumerate(mm.groupby('y_mm')):axs[0,0].scatter(g.dx,g.dy,s=6,color=COLORS[i],label=f'Y={y:g}')
    c=np.polyfit(mm.dx,mm.dy,1);xx=np.linspace(mm.dx.min(),mm.dx.max(),100);axs[0,0].plot(xx,np.polyval(c,xx),ls='--',lw=1,color='#111827')
    style(axs[0,0],'가로 이동 (px)','세로 이동 (px)');axs[0,0].set_title('세로 이동',loc='left',fontsize=11)
    for i,(name,g) in enumerate(common.groupby('group')):axs[0,1].scatter(g.time_proxy_minutes,g.residual_dx,s=7,color=COLORS[i],label=name.replace('new_',''))
    style(axs[0,1],'center1 이후 파일 시각 (분)','사진 공통 가로 residual (px)');axs[0,1].legend(fontsize=7,ncol=2);axs[0,1].set_title('사진 공통 흔들림 (직선 잔차)',loc='left',fontsize=11)
    nn=n[(n.scope=='new')&(n.varying=='theta_deg')&~n.diagonal]
    for dim,color in [('dx','#0070e8'),('dy','#d84a00')]:
        values=nn['sigma_'+dim];axs[1,0].hist(values,bins=35,alpha=.5,color=color,label=('가로' if dim=='dx' else '세로')+f' 중앙값 {values.median():.3f}px')
        axs[1,0].axvline(values.median(),color=color,lw=.8)
    style(axs[1,0],'점 하나의 잔차 표준편차 (px)','점 × 분석 그룹 수');axs[1,0].legend(fontsize=7);axs[1,0].set_title('점 하나 흔들림 (공통 성분 제거)',loc='left',fontsize=11)
    nd=n[(n.scope=='new')&(n.varying=='theta_deg')]
    values=[nd[~nd.diagonal].sigma_dx.median(),nd[nd.diagonal].sigma_dx.median()]
    axs[1,1].barh(['대각선 아닌 점','대각선 125점'],values,color=['#9ca3af','#ea580c'])
    for i,v in enumerate(values):axs[1,1].text(v,i,f' {v:.3f}px',va='center',fontsize=9)
    style(axs[1,1],'점 고유 가로 σ의 중앙값 (px)','');axs[1,1].set_title('대각선 모서리',loc='left',fontsize=11)
    save(fig,'noise_decomposition')
    # Source slide 19: feature-position noise maps, separate fixed-axis group families.
    fig,axs=plt.subplots(1,2,figsize=(12.8,3.9))
    vmax=n[(n.scope=='new')&~n.diagonal].sigma_dx.quantile(.98)
    for ax,axis in zip(axs,['y_mm','theta_deg']):
        g=n[(n.scope=='new')&(n.varying==axis)&~n.diagonal].groupby('key',as_index=False).agg(x=('x','first'),y=('y','first'),noise=('sigma_dx','mean'))
        sc=ax.scatter(g.x,g.y,c=g.noise,cmap='Blues',vmin=0,vmax=vmax,s=7)
        ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(('Y 변화' if axis=='y_mm' else 'Θ 변화')+' 분석 그룹의 점 고유 흔들림',fontsize=10)
        style(ax,'원본 x (px)','원본 y (px)');fig.colorbar(sc,ax=ax,orientation='horizontal',label='점 고유 가로 σ (px)',fraction=.08,pad=.19)
    save(fig,'point_noise_map')
    # Same QR source photos, measured data annotations only.
    image=np.array(Image.open(root/'기준 위치/center1.bmp'))
    base=pts[pts.path=='기준 위치/center1.bmp']
    for kind in ['reference','features','ecc']:
        fig,ax=plt.subplots(figsize=(7.1,3.4));ax.imshow(image);ax.set_xlim(1500,2300);ax.set_ylim(1210,930);ax.axis('off')
        if kind=='features':
            for diagonal,color in [(False,'#16a34a'),(True,'#f97316')]:
                g=base[base.diagonal==diagonal];ax.scatter(g.x,g.y,s=3 if not diagonal else 5,c=color,linewidths=0)
        for slot,g in base.groupby('qr'):
            if kind=='ecc':
                ref=ec[(ec.path=='기준 위치/center1.bmp')&(ec.qr==slot)].iloc[0]
                x={'L':1644,'C':1902,'R':2160}[slot]+ref.old_dx;y={'L':1068,'C':1069,'R':1070}[slot]+ref.old_dy;w,h=204,216
                ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,ec='#008cff',lw=1))
            else:
                x,y=(g.x.min()+g.x.max())/2,(g.y.min()+g.y.max())/2;w,h=g.x.max()-g.x.min(),g.y.max()-g.y.min()
                ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,ec='#008cff',lw=.7))
            if kind!='features':ax.scatter([x],[y],c='#008cff',s=12);ax.text(x,y-h/2-10,slot,color='#008cff',ha='center',fontsize=8)
        save(fig,'photo_'+kind)
    # Source slide 18: always-pass / occasionally-pass feature overlays per trial.
    fig,axs=plt.subplots(2,2,figsize=(12.8,6.4))
    for ax,gdef in zip(axs.ravel(),[g for g in groups if g['scope']=='new' and g['varying']=='theta_deg']):
        g=pts[pts.path.isin(gdef['paths'])].groupby('key').accepted.agg(['min','max'])
        bp=base.join(g,on='key',rsuffix='_all');ax.imshow(image)
        for condition,color in [(bp['min'].eq(True),'#16a34a'),(bp['min'].eq(False)&~bp.diagonal,'#f97316'),(bp.diagonal,'#d84a00')]:
            q=bp[condition];ax.scatter(q.x,q.y,c=color,s=2.5,linewidths=0)
        ax.set_xlim(1500,2300);ax.set_ylim(1200,940);ax.axis('off')
        ax.set_title(gdef['group'].replace('new_','')+f"  항상 통과 {int(g['min'].sum())} / 755점",loc='left',fontsize=10)
    save(fig,'fit_group_maps')
    # Original slide 24's method-difference plot (corner 630 vs QR ECC).
    mm=m.pivot(index='path',columns='method',values=['dx','dy'])
    selected=inv[inv.category.isin(['new-reference','mixed'])].sort_values(['category','y_mm','theta_deg','repeat']).path
    comp=pd.DataFrame({'path':selected})
    for dim in ['dx','dy']:comp[dim+'_630_minus_ecc']=mm[dim].loc[selected,'no_diagonal630'].to_numpy()-mm[dim].loc[selected,'ECC'].to_numpy()
    comp.to_csv(out/'method_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.1,3.8));ax.scatter(np.arange(len(comp)),comp.dx_630_minus_ecc,s=7,c='#0070e8',label='630점 − ECC');ax.axhline(0,c='#111827',lw=.6)
    style(ax,'사진 (기준점 / Y / Θ / 반복 번호 순)','가로 이동량 차이 (px)');ax.legend();save(fig,'method_difference')
    patch=pd.read_csv(out/'patch_comparison.csv')
    pp=patch.pivot(index=['path','qr'],columns='patch',values=['relative_dx','relative_dy'])
    delta=pd.DataFrame({'path':[i[0] for i in pp.index], 'qr':[i[1] for i in pp.index],
        'dx_label_minus_module':pp.relative_dx.label_204x216.to_numpy()-pp.relative_dx.module_extent_plus8.to_numpy(),
        'dy_label_minus_module':pp.relative_dy.label_204x216.to_numpy()-pp.relative_dy.module_extent_plus8.to_numpy()})
    delta.to_csv(out/'patch_differences.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.1,3.8))
    for i,(slot,g) in enumerate(delta.groupby('qr')):ax.scatter(np.arange(len(g)),g.dx_label_minus_module,s=6,c=COLORS[i],label=slot)
    ax.axhline(0,c='#111827',lw=.6);style(ax,'사진 (기존 center / 명목 좌표 순)','가로 이동량 차이 (px)');ax.legend();save(fig,'patch_sensitivity')
    # Reference-point perturbation: exact evaluation of each affine transform.
    perturb=[]
    for row in ec[ec.category=='mixed'].itertuples():
        ref=ec[(ec.path=='기준 위치/center1.bmp')&(ec.qr==row.qr)].iloc[0]
        A=np.array([[row.a00,row.a01],[row.a10,row.a11]])
        B=np.array([[ref.a00,ref.a01],[ref.a10,ref.a11]])
        change=(A@np.linalg.inv(B)-np.eye(2))@np.array([1.,.75])
        perturb.append(dict(path=row.path,qr=row.qr,dx_difference=change[0],dy_difference=change[1]))
    pd.DataFrame(perturb).to_csv(out/'reference_point_perturbation.csv',index=False)
    (out/'figures.json').write_text(json.dumps(exported,indent=2),encoding='utf-8')
    print('figures',len(exported))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.output,a.previous,a.root)
