"""Reconstruct PPT slide 6: subtract a common axis slope, retain zero intercepts."""
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


def main(previous,out):
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    for image in (previous/'figures').glob('*.png'):
        if not image.name.startswith('label_residual_'):shutil.copy2(image,out/'figures'/image.name)
    font=Path('C:/Windows/Fonts/NotoSansKR-VF.ttf');regular=Path(tempfile.gettempdir())/'offaxis-NotoSansKR-Regular.ttf'
    if font.exists():
        if not regular.exists():instantiateVariableFont(TTFont(font),{'wght':400},inplace=True).save(regular)
        font_manager.fontManager.addfont(regular);plt.rcParams['font.family']=font_manager.FontProperties(fname=str(regular)).get_name()
    plt.rcParams.update({'font.size':9,'axes.unicode_minus':False,'figure.facecolor':'#f7f7f9','axes.facecolor':'#f7f7f9','savefig.facecolor':'#f7f7f9','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    f=pd.read_csv(previous/'linear_fits.csv');f=f[f.method.eq('no_diagonal630')].copy()
    r=pd.read_csv(previous/'frame_linear_residuals.csv');r=r[r.method.eq('no_diagonal630')].copy()
    # A single slope per axis and session preserves group-to-group slope differences.
    # Averaging the original six trial slopes recovers the reference graph's scale.
    common=f.groupby(['scope','varying']).slope_dx.mean().rename('common_slope_dx').reset_index()
    f=f.merge(common,on=['scope','varying'],validate='many_to_one')
    r=r.merge(f[['group','intercept_dx','slope_dx','common_slope_dx']],on='group',validate='many_to_one')
    r['slope_removed_dx_common']=r.dx-r.common_slope_dx*r.command
    r['display_line_dx']=r.intercept_dx+(r.slope_dx-r.common_slope_dx)*r.command
    # This ordinate retains the fitted intercept; subtracting it gives the OLS residual.
    assert np.allclose(r.slope_removed_dx_common-r.display_line_dx,r.residual_dx,atol=1e-10)
    sy=common[(common.scope=='new')&(common.varying=='y_mm')].common_slope_dx.iloc[0]
    st=common[(common.scope=='new')&(common.varying=='theta_deg')].common_slope_dx.iloc[0]
    # Mixed-axis extension: remove both commanded motion components, never the intercept.
    r['other_axis_component']=np.where(r.varying.eq('y_mm'),st*r.fixed_value,sy*r.fixed_value)
    r['mixed_axis_slope_removed_dx']=r.slope_removed_dx_common-r.other_axis_component
    r['mixed_axis_display_line_dx']=r.display_line_dx-r.other_axis_component
    r.to_csv(out/'zero_position_points.csv',index=False);f.to_csv(out/'zero_position_fits.csv',index=False)
    common.to_csv(out/'common_axis_slopes.csv',index=False)
    colors=['#0f766e','#a85b00','#0070e8','#e64b00']
    def label(g,axis,scope):
        if scope=='new':return (f'Θ={g.fixed_value.iloc[0]:g}° 고정' if axis=='y_mm' else f'Y={g.fixed_value.iloc[0]:g}mm 고정')
        return g.group.iloc[0].replace('old_','')
    for scope in ['old','new']:
        fig,axs=plt.subplots(1,2,figsize=(8.2,4.5))
        for ax,axis in zip(axs,['y_mm','theta_deg']):
            for i,(_,g) in enumerate(r[r.scope.eq(scope)&r.varying.eq(axis)].groupby('group')):
                ax.scatter(g.command,g.slope_removed_dx_common,s=9,c=colors[i],label=label(g,axis,scope))
                xx=np.array([min(0,g.command.min()),max(0,g.command.max())]);row=g.iloc[0]
                ax.plot(xx,row.intercept_dx+(row.slope_dx-row.common_slope_dx)*xx,color=colors[i],lw=.9)
                ax.scatter([0],[row.intercept_dx],marker='D',s=30,c=colors[i],edgecolors='white',linewidths=.5,zorder=4)
            ax.scatter([0],[0],marker='*',s=90,c='#111827',label='center' if scope=='old' else 'center1',zorder=5)
            ax.axvline(0,color='#9ca3af',lw=.6,ls=':');ax.axhline(0,color='#9ca3af',lw=.6)
            ax.set(xlabel='명령한 Y (mm)' if axis=='y_mm' else '명령한 Θ (°)',ylabel='공통 기울기를 뺀 가로 이동 (px)')
            ax.set_title('Y: 직선에서 남은 가로 이동' if axis=='y_mm' else 'Θ: 직선에서 남은 가로 이동',loc='left',fontsize=10)
            ax.grid(True,color='#dce1e8',lw=.6);ax.set_axisbelow(True);ax.legend(fontsize=6.5,loc='best')
        fig.tight_layout(pad=.8);fig.savefig(out/'figures'/f'zero_position_{scope}.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.2,4.5))
    adjusted=[]
    for ax,axis in zip(axs,['y_mm','theta_deg']):
        for i,(name,g) in enumerate(r[r.scope.eq('new')&r.varying.eq(axis)].groupby('group')):
            p=g.iloc[0];a=p.intercept_dx-p.other_axis_component
            ax.scatter(g.command,g.mixed_axis_slope_removed_dx,s=10,c=colors[i],label=label(g,axis,'new'))
            xx=np.array([0,g.command.max()]);ax.plot(xx,a+(p.slope_dx-p.common_slope_dx)*xx,color=colors[i],lw=.9)
            ax.scatter([0],[a],marker='D',s=35,c=colors[i],edgecolors='white',lw=.5,zorder=4)
            adjusted.append({'group':name,'varying':axis,'fixed_value':p.fixed_value,'intercept_after_other_axis_removal_px':a})
        ax.scatter([0],[0],marker='*',s=90,c='#111827',label='center1',zorder=5)
        ax.axvline(0,color='#9ca3af',lw=.6,ls=':');ax.axhline(0,color='#9ca3af',lw=.6)
        ax.set(xlabel='명령한 Y (mm)' if axis=='y_mm' else '명령한 Θ (°)',ylabel='두 축 기울기를 뺀 가로 이동 (px)')
        ax.set_title('Y: 명령 성분 제거 후 가로 이동' if axis=='y_mm' else 'Θ: 명령 성분 제거 후 가로 이동',loc='left',fontsize=10)
        ax.grid(True,color='#dce1e8',lw=.6);ax.set_axisbelow(True);ax.legend(fontsize=6.5,loc='best')
    fig.tight_layout(pad=.8);fig.savefig(out/'figures/zero_position_new_additive.png',dpi=190);plt.close(fig)
    pd.DataFrame(adjusted).to_csv(out/'mixed_zero_intercepts.csv',index=False)
    # Small multiples expose per-image fluctuations despite different mixed-axis offsets.
    fig,axs=plt.subplots(2,4,figsize=(12,6))
    for row,axis in enumerate(['y_mm','theta_deg']):
        for col,(_,g) in enumerate(r[r.scope.eq('new')&r.varying.eq(axis)].groupby('group')):
            ax=axs[row,col];p=g.iloc[0];ax.scatter(g.command,g.slope_removed_dx_common,s=13,c=colors[col])
            xx=np.array([0,g.command.max()]);ax.plot(xx,p.intercept_dx+(p.slope_dx-p.common_slope_dx)*xx,c=colors[col],lw=.8)
            ax.scatter([0],[p.intercept_dx],marker='D',s=35,c=colors[col]);ax.set_title(label(g,axis,'new')+f'\n절편 {p.intercept_dx:.3f}px',fontsize=10)
            ax.set(xlabel='명령 Y (mm)' if axis=='y_mm' else '명령 Θ (°)',ylabel='기울기 제거 후 (px)');ax.grid(True,alpha=.3);ax.ticklabel_format(useOffset=False,axis='y')
    fig.tight_layout();fig.savefig(out/'figures/zero_position_new_detail.png',dpi=190);plt.close(fig)
    pages=json.loads((previous/'deck_content.json').read_text(encoding='utf-8'))
    new=f[f.scope.eq('new')]
    pages[4].update(title='이동 관계 – 0점 위치',figure='zero_position_new_additive',rows=[['세로축','dx − sY·Y − sΘ·Θ (px)'],['제거한 기울기',f'Y {sy:.6f}px/mm / Θ {st:.6f}px/°'],['혼합축 처리','고정된 다른 축의 명령 이동 성분도 제거'],['다이아몬드','보정 후 그룹 직선의 명령 0 절편'],['별','center1 기준 (0, 0)'],['남긴 성분','절편·비선형성·coupling·관측 흔들림'],['해석','0점 절편은 양의 좌표에서 외삽한 값']],caption='단독축 원본을 혼합축에 확장했다. 두 축 공통 기울기만 제거하고 절편은 남겼다. 다른 축 미제거 그래프와 표시 수치를 별도 제공한다. 순수 기구 영점이나 센서 잡음만을 뜻하지 않는다.')
    yzero=[q for q in adjusted if q['varying']=='y_mm'];tzero=[q for q in adjusted if q['varying']=='theta_deg']
    pages[4]['rows']=[['0점이란','기울기 제거 후 그룹 직선의 명령 0 절편'],['Y 그룹 0점','\n'.join(f"Θ={q['fixed_value']:g}° 고정: {q['intercept_after_other_axis_removal_px']:+.3f}px" for q in yzero)],['Θ 그룹 0점','\n'.join(f"Y={q['fixed_value']:g}mm 고정: {q['intercept_after_other_axis_removal_px']:+.3f}px" for q in tzero)],['그룹 간 차이',f"Y 최대 {np.ptp([q['intercept_after_other_axis_removal_px'] for q in yzero]):.3f}px / Θ 최대 {np.ptp([q['intercept_after_other_axis_removal_px'] for q in tzero]):.3f}px"],['혼합축 처리','dx − sY·Y − sΘ·Θ, 절편 유지'],['기울기',f'Y {sy:.6f}px/mm\nΘ {st:.6f}px/°'],['해석','관측 흔들림·비선형·coupling도 남음\n양의 좌표에서의 절편 외삽']]
    def compact(values,symbol):
        entries=[f"{symbol}{q['fixed_value']:g}: {q['intercept_after_other_axis_removal_px']:+.3f}" for q in values]
        return ' / '.join(entries[:2])+'px\n'+' / '.join(entries[2:])+'px'
    pages[4]['rows'][1]=['Y 그룹 0점',compact(yzero,'Θ')]
    pages[4]['rows'][2]=['Θ 그룹 0점',compact(tzero,'Y')]
    pages[4]['rows'][6]=['해석','절편은 외삽값이며 비선형·coupling·흔들림 포함']
    old=f[f.scope.eq('old')]
    pages[13].update(title='기존 데이터의 0점 위치 재현',figure='zero_position_old',full=False,header=False,rows=[['Y 그룹 절편',' / '.join(f'{x:+.3f}' for x in old[old.varying.eq('y_mm')].intercept_dx)+' px'],['Θ 그룹 절편',' / '.join(f'{x:+.3f}' for x in old[old.varying.eq('theta_deg')].intercept_dx)+' px'],['Y 공통 기울기',f'{common.query("scope == \'old\' and varying == \'y_mm\'").common_slope_dx.iloc[0]:.6f} px/mm'],['Θ 공통 기울기',f'{common.query("scope == \'old\' and varying == \'theta_deg\'").common_slope_dx.iloc[0]:.6f} px/°'],['原 PPT 대조','Y −0.54 / +0.11 / −0.45 / −0.43px\nΘ −0.84 / −0.19px']],caption='기존 281장으로 첨부한 0점 위치 페이지를 재구성했다. 원본의 공통 기울기 수치는 명시되지 않아 trial 기울기의 평균을 사용했다. 절편은 원본 반올림 값과 부합한다.')
    (out/'deck_content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False),encoding='utf-8')
    note='# 0점 위치 그래프 수정 (v3)\n\n사용자가 지정한 원본 PPT 6쪽을 기준으로 했다. 가로축은 명령 Y(mm) 또는 θ(°), 세로축은 측정 가로변위 dx에서 공통 축 기울기 s×명령값을 뺀 값(px)이다. 그룹 절편은 남긴다. 다이아몬드는 명령 0에서의 그룹 직선 절편, 별은 기준 영상이다. 이는 dx–dy 이동 경로도, 절편까지 제거한 OLS 잔차도 아니다.\n\n원본의 공통 제거 기울기 수치는 미기재다. 각 세션·축의 그룹별 직선 기울기를 산술 평균하여 사용했다. 기존 자료에서 Y=14.375964px/mm, θ=80.539419px/°이며 원본의 그래프 추세를 복원한다. 앞선 v2 0점 그래프는 그룹마다 그 그룹의 기울기를 제거하여 추세 차이를 평평하게 만든 한계가 있었다. 이번에는 하나의 공통 기울기를 제거한다. 그룹 절편 자체는 바뀌지 않는다.\n\n신규 혼합 자료는 다른 축을 고정한 그룹이다. 절편에는 고정된 축의 실제 이동량이 포함되므로 순수한 기구 영점 오차로 읽을 수 없다. 원본 단독축 trial의 절편과 의미가 동일하지 않다. 고정축 기여를 임의로 빼지 않고 원시 수준을 보존했다. 그룹별 확대 그림도 제공한다.\n\nzero_position_points.csv: 측정값 dx, 공통 기울기 common_slope_dx, 표시값 slope_removed_dx_common, 표시 직선 display_line_dx, 별도 순수 잔차 residual_dx. zero_position_fits.csv: 그룹별 기울기·절편. common_axis_slopes.csv: 제거한 공통 기울기.\n\n수치 검증: 표시값 − 표시 직선 = 원래 OLS 잔차, 오차 허용 1e-10px. 원본 fitting이나 변위는 다시 변경하지 않았다.\n\n재현: python tool/offaxis/zero_position_revision.py --previous docs/offaxis-eda/ppt-method-v2 --output docs/offaxis-eda/zero-position-v3\n'
    note+='\n## 혼합축 주 그래프\n\n주 그래프 zero_position_new_additive.png는 `dx−sY·Y−sTheta·theta`이다. 원본은 한 축만 움직였지만 신규 자료는 두 축이 동시에 움직여서 다른 고정축의 큰 이동량까지 제거해야 0점 차이를 볼 수 있다. sY·sTheta는 위에서 정의한 신규 그룹 기울기의 평균이다. 절편은 제거하지 않는다. 본 값에는 0점 offset, coupling과 비선형 모델 오차, 관측 흔들림이 남는다. 실제 원점을 두 축에서 새로 촬영한 것과 구분한다. fixed-axis 그룹의 보정 절편은 mixed_zero_intercepts.csv에 제공한다. 다른 축의 기여를 남긴 기본 그래프 zero_position_new.png와 원시 수준 CSV도 함께 보존한다.\n'
    (out/'README.md').write_text(note,encoding='utf-8');print('zero-position figures and identities verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.previous,a.output)
