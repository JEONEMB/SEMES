"""Render all group results and input-point examples, without changing measurements."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def dump(name,data): (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f(v,d=6): return '—' if v is None else f'{v:.{d}f}'
TARGETS={'upper_structure':'상단','lower_structure':'하단'}

def main():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    metrics=read('correlations.json');summary=read('summary.json');fits=read('fit_summary.json')
    images=read('manifest.json');measures=read('measurements.json')
    image_map={r['id']:r for r in images};measure_map={(r['id'],r['target']):r for r in measures}
    primary=[r for r in metrics if r['role']=='primary' and r['status']=='descriptive_fit']
    valid=[r for r in metrics if r['status']=='descriptive_fit']
    grouped={}
    for r in valid: grouped.setdefault(r['id'],{})[r['target']]=r
    figs=HERE/'figures';figs.mkdir(exist_ok=True)

    # Independent recomputation of each group's OLS and Pearson coefficients.
    max_coef_diff=0.;max_r_diff=0.
    for r in valid:
        x=np.array([m['x'] for m in r['members']],dtype=float);z=np.array([m['image_angle_deg'] for m in r['members']],dtype=float)
        coeff=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),z,rcond=None)[0]
        a=r['angle_regression']
        max_coef_diff=max(max_coef_diff,abs(coeff[0]-a['intercept']),abs(coeff[1]-a['slope']))
        max_r_diff=max(max_r_diff,abs(float(np.corrcoef(x,z)[0,1])-a['pearson_r']))
        assert np.allclose(coeff,[a['intercept'],a['slope']],rtol=1e-10,atol=1e-12)
        assert abs(float(np.corrcoef(x,z)[0,1])-a['pearson_r'])<1e-12
        assert len(x)==r['n'] and len(set(x))==r['distinct_levels']

    charts={}
    for gid,pair in grouped.items():
        fig,axes=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
        first=next(iter(pair.values()))
        for col,target in enumerate(TARGETS):
            r=pair[target];a=r['angle_regression'];members=r['members']
            x=np.array([m['x'] for m in members]);z=np.array([m['image_angle_deg'] for m in members])
            residual=np.array(a['residuals'])
            repeats=[m['repeat'] or 0 for m in members]
            for rep in sorted(set(repeats)):
                idx=np.array([v==rep for v in repeats])
                label=f'사진 번호 {rep}' if rep else '개별 이미지'
                axes[0,col].scatter(x[idx],z[idx],s=18,alpha=.8,label=label)
                axes[1,col].scatter(x[idx],residual[idx],s=18,alpha=.8)
            xx=np.array([x.min(),x.max()])
            axes[0,col].plot(xx,a['intercept']+a['slope']*xx,color='#222',lw=1.3,label='구간별 OLS')
            axes[0,col].set_title(f'{TARGETS[target]}: r={a["pearson_r"]:+.6f}, 변화율={a["slope"]:+.6f}\n잔차 RMS={a["rmse"]:.6f}° / 사진 {r["n"]}장')
            axes[0,col].set_ylabel('영상 직선 각도 φ (°)')
            axes[0,col].legend(fontsize=8,loc='best')
            axes[1,col].axhline(0,color='#333',lw=.8)
            axes[1,col].set_ylabel('회귀 잔차 (영상 °)')
            for row in range(2):
                axes[row,col].set_xlabel('Y 라벨 (mm)' if r['axis']=='Y' else 'θ 라벨 (°)')
                axes[row,col].grid(alpha=.2)
                axes[row,col].ticklabel_format(axis='y',style='plain',useOffset=False)
        role=' / 같은 θ 목록 비교' if first['role']!='primary' else ''
        fig.suptitle(first['label']+role,fontsize=15)
        rel=f'figures/{gid}.png';fig.savefig(HERE/rel,dpi=135);plt.close(fig)
        charts[gid]={'label':first['label'],'role':first['role'],'axis':first['axis'],'path':rel}

    # Sensitivity comparisons retain every segment as its own value.
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for row,axis in enumerate(['theta','Y']):
        for col,target in enumerate(TARGETS):
            rr=[r for r in primary if r['axis']==axis and r['target']==target]
            vals=[r['predicted_angle_change_per_step_deg'] for r in rr]
            ax=axes[row,col];ax.axvline(0,color='#555',lw=.8)
            ax.scatter(vals,range(len(rr)),c=['#1565c0' if r['dataset']==1 else '#b45615' if r['dataset']==2 else '#2e7d32' for r in rr],s=40)
            ax.set_yticks(range(len(rr)),[r['label'] for r in rr],fontsize=9);ax.invert_yaxis()
            ax.set_title(f'{TARGETS[target]} / '+('θ 0.01° 이동' if axis=='theta' else 'Y 0.05mm 이동'))
            ax.set_xlabel('회귀식이 예측한 영상 각도 변화량 (°)')
            ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useOffset=False)
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('조건별 민감도 — 모든 촬영 구간을 각각 표시',fontsize=16)
    fig.savefig(HERE/'sensitivity_overview.png',dpi=145);plt.close(fig)

    # Physical-boundary visual checks: endpoint conditions, not statistical exclusions.
    selections=[]
    first=[r for r in images if r['dataset']==1 and not r['is_reference']]
    for axis in ['Y','theta']:
        rr=[r for r in first if r['axis']==axis];field='y_mm' if axis=='Y' else 'theta_deg'
        selections.extend([min(rr,key=lambda r:r[field]),max(rr,key=lambda r:r[field])])
    for y in [.05,1.]:
        for theta in [.01,2.]:
            selections.append(next(r for r in images if r['dataset']==2 and r['y_mm']==y and r['theta_deg']==theta and r['repeat']==1))
    third=[r for r in images if r['dataset']==3 and r['repeat']==1]
    selections.extend([min(third,key=lambda r:r['theta_deg']),max(third,key=lambda r:r['theta_deg'])])
    worst=max(measures,key=lambda r:r['band_endpoint_points']);selections.append(image_map[worst['id']])
    unique={r['id']:r for r in selections};points=np.load(HERE/'points.npz',allow_pickle=False)
    overlays=[]
    for iid,im in unique.items():
        with Image.open(im['path']) as src: gray=np.asarray(src.convert('L'))
        fig,axes=plt.subplots(2,1,figsize=(15,5),layout='constrained')
        for ax,target in zip(axes,TARGETS):
            p=points[iid+'_'+target];r=measure_map[iid,target]
            x,y=p[:,0].astype(float),p[:,1].astype(float)
            y0=max(0,int(y.min())-30);y1=min(gray.shape[0],int(y.max())+31)
            ax.imshow(gray[y0:y1],cmap='gray',vmin=0,vmax=255,aspect='auto',
                      extent=(-.5,gray.shape[1]-.5,y1-.5,y0-.5))
            ax.scatter(x,y,s=2,alpha=.6,color='#ffa43c')
            xx=np.array([x.min(),x.max()]);ax.plot(xx,r['intercept_px']+r['slope']*xx,color='#00ddff',lw=1.7)
            ax.set_ylim(y1-.5,y0-.5);ax.set_ylabel('영상 y (px)')
            ax.set_title(f'{TARGETS[target]} φ={r["image_angle_deg"]:.6f}° / 입력 {len(x)}점 / 탐색 띠 끝점 {r["band_endpoint_points"]}개')
        axes[-1].set_xlabel('영상 x (px)')
        fig.suptitle(f'{iid}: {Path(im["file"]).name} / 세로 확대 표시',fontsize=14)
        rel=f'figures/overlay_{iid}.png';fig.savefig(HERE/rel,dpi=135);plt.close(fig)
        overlays.append({'id':iid,'label':f'{im["dataset"]}차 {Path(im["file"]).name}','path':rel})

    def table_rows(rr):
        return ''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [escape(r['label']),TARGETS[r['target']],r['n'],
            r['distinct_levels'],f'{min(r["levels"]):g} ~ {max(r["levels"]):g}',
            f(r['angle_regression']['pearson_r']),f(r['angle_regression']['slope'],8),
            f(r['predicted_angle_change_per_step_deg'],8),f(r['angle_regression']['rmse']),
            f(r['angle_regression']['max_abs_residual'])])+'</tr>' for r in rr)
    def table(rr,axis):
        unit='영상 °/장비 °' if axis=='theta' else '영상 °/mm'
        step='θ 0.01°' if axis=='theta' else 'Y 0.05mm'
        return f'<div class="scroll"><table><thead><tr><th>구간</th><th>구조물</th><th>사진 수</th><th>이동값 수</th><th>라벨 범위</th><th>Pearson r</th><th>변화율 ({unit})</th><th>{step}당 영상 변화 (°)</th><th>RMS (영상 °)</th><th>최대 절대 잔차 (영상 °)</th></tr></thead><tbody>'+table_rows(rr)+'</tbody></table></div>'
    theta=[r for r in primary if r['axis']=='theta'];ys=[r for r in primary if r['axis']=='Y']
    common=[r for r in valid if r['role']!='primary']
    options=''.join(f'<option value="{escape(v["path"])}">{escape(v["label"])} / {"Y 변화" if v["axis"]=="Y" else "θ 변화"}{" / 공통 θ 조건" if v["role"]!="primary" else ""}</option>' for v in charts.values())
    firstchart=next(iter(charts.values()))['path']
    overlay_html=''.join(f'<details><summary>{escape(v["label"])} ({v["id"]})</summary><a href="{v["path"]}"><img loading="lazy" src="{v["path"]}" alt="경계점과 직선"></a></details>' for v in overlays)
    missing=''.join(f'<tr><td>{escape(r["label"])}</td><td>{r["axis"]}</td><td>{r["n"]}</td><td>{escape(str(r["levels"]))}</td></tr>'
                    for r in summary['not_computable'] if r['target']=='upper_structure')
    quick={}
    for target in TARGETS:
        rr=[r for r in theta if r['target']==target];yr=[r for r in ys if r['target']==target]
        quick[target]={'theta_pearson_range':[min(r['angle_regression']['pearson_r'] for r in rr),max(r['angle_regression']['pearson_r'] for r in rr)],
                       'theta_sensitivity_range':[min(r['angle_regression']['slope'] for r in rr),max(r['angle_regression']['slope'] for r in rr)],
                       'theta_step_range':[min(r['predicted_angle_change_per_step_deg'] for r in rr),max(r['predicted_angle_change_per_step_deg'] for r in rr)],
                       'theta_residual_rms_range':[min(r['angle_regression']['rmse'] for r in rr),max(r['angle_regression']['rmse'] for r in rr)],
                       'max_abs_y_step_response':max(abs(r['predicted_angle_change_per_step_deg']) for r in yr),
                       'y_sensitivity_range':[min(r['angle_regression']['slope'] for r in yr),max(r['angle_regression']['slope'] for r in yr)]}
    dump('quick_results.json',quick)
    document=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>외부 구조물 기울기와 Y·θ 상관관계·민감도</title><style>
body{{margin:0;background:#f3f5f9;color:#253248;font-family:"Malgun Gothic",sans-serif;line-height:1.75}}
main{{max-width:1300px;background:white;margin:auto;padding:30px 24px 70px}}h1{{font-size:28px}}h2{{margin-top:38px}}
.lead{{background:#eef5ff;border-left:5px solid #1966aa;padding:18px 23px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{border:1px solid #d6dfeb;padding:9px;text-align:right;white-space:nowrap}}th{{background:#edf2f8}}th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;height:auto}}details{{padding:10px;border:1px solid #d6dfeb;margin:8px 0}}summary{{cursor:pointer}}a{{color:#005aa5}}select{{padding:9px;max-width:100%;font-size:15px}}code{{overflow-wrap:anywhere}}.note{{color:#58647a}}.formula{{padding:12px;background:#f4f6fa}}</style></head>
<body><main><h1>외부 상·하단 구조물과 Y·θ의 상관관계·민감도</h1>
<div class="lead"><b>θ 변화에 대한 선형 반응이 1·2·3차 모두에서 확인됐습니다.</b><br>
θ 증가 시 <b>상단 영상 각도는 감소</b>, <b>하단 영상 각도는 증가</b>합니다.<br>
Y 변화의 반응은 선택한 이동 단위 기준으로 더 작지만, 특히 2차 일부 조건에서 방향성 있는 변화가 관측됐습니다.</div>
<p>575장(1차 281장·2차 60장·3차 234장), 상·하단 1,150개 직선.
모든 직선이 최대 50회·종료값 10⁻⁸ 조건을 충족했고 비유한 결과나 예외는 없었습니다.
원본 해시를 모두 확인했으며 기준 이미지 4장의 8개 경계점 집합과 직선 결과를 이전 결과와 대조했습니다.</p>
<h2>1. 이번에 수행한 계산</h2>
<p>기존 탐색 영역·중앙 가림 마스크 → 가로 1px마다 부호 있는 세로 Sobel 최댓값 → 일반 Huber 세로 잔차 피팅.
Gaussian, 추가 점 제거, 최소 점 개수 제한, 이미지 정합·회전 보정·탐색 영역 추적을 추가하지 않았습니다.</p>
<p>Huber: t=1.345, MAD(center=0) 척도 갱신, OLS 시작값, maxiter=50, tol=10⁻⁸, conv=dev.
현재 OpenCV {read('provenance.json')['opencv_version']}, statsmodels {fits['statsmodels_version']}.
상단·하단 각도는 φ=atan(m)×180/π, 영상 좌표는 +y가 아래입니다. 영상 각도 φ와 장비 θ는 서로 다른 양입니다.</p>
<div class="formula">이미지별 직선: y<sub>px</sub> = m·x<sub>px</sub> + b<br>
Y 조건별 관계: φ = a<sub>Y</sub>·Y + c<br>θ 조건별 관계: φ = a<sub>θ</sub>·θ + c<br>
단위 이동 반응: a<sub>Y</sub>×0.05mm 또는 a<sub>θ</sub>×0.01°</div>
<p>Huber는 영상의 직선 추정에 사용했습니다. 그 결과와 이동 라벨의 관계에는 앞서 합의한 Pearson r·절편 포함 단순선형회귀(OLS)·잔차 RMS를 계산했습니다.
각 사진이 관측값 한 개이며, 수천 개 경계점을 독립 사진처럼 세지 않았습니다. 반복 사진은 평균내지 않고 모두 표시했습니다.
같은 조건의 반복 사진을 독립 이동 실험으로 가정하는 p값·신뢰구간·동등성 판정은 산출하지 않았습니다.</p>
<h2>2. 데이터 구분</h2><ul>
<li>1차: 기존 파일 수정시각 기반 2·3·6·7번 Y 구간과 4·5번 θ 구간을 각각 분석했습니다. 4번도 포함했습니다.
단일 축 촬영에서 다른 축이 구간 내 고정됐다는 전제를 사용했으며, 확인되지 않은 다른 축 절대좌표를 0으로 채우지 않았습니다.</li>
<li>2차: θ가 같은 사진으로 Y 관계를, Y가 같은 사진으로 θ 관계를 구했습니다. 같은 라벨 사진 3장 모두 사용했습니다.</li>
<li>3차: Y=0.05mm 고정, θ=0.04~0.82° 중 제공된 78조건×3장. 0.10°가 빠진 목록은 그대로 사용했습니다.
확인된 0.32° 파일명 및 0.57° TXT(T_Actual=-1.100) 정정을 반영했으며 원본 파일은 수정하지 않았습니다.</li>
<li>차수·촬영 구간을 합친 하나의 회귀식은 만들지 않았습니다. 구간별 절편도 각각 추정했습니다.</li></ul>
<h2>3. θ 변화와 영상 각도의 관계</h2>{table(theta,'theta')}
<p>상단은 모든 구간에서 음의 변화율, 하단은 모든 구간에서 양의 변화율입니다.
전 범위 상관계수가 높다는 사실만으로 한 틱 판별 또는 ±0.005° 보정 정확도가 검증되는 것은 아닙니다. 회귀 잔차도 함께 표시했습니다.</p>
<h2>4. Y 변화와 영상 각도의 관계</h2>{table(ys,'Y')}
<p>1차 네 Y 구간의 상관계수는 상단 약 −0.223~+0.020, 하단 약 −0.135~+0.202입니다.
2차 상단에서는 θ=0.01·0.1·1° 조건에 음의 상관이 관측됩니다. 변화량 자체는 표의 a<sub>Y</sub>와 0.05mm당 반응을 함께 읽어야 합니다.
이 결과만으로 Y 영향이 정확히 0이라고 하거나 특정 기계적 원인을 지목하지 않았습니다.</p>
<h2>5. Y 고정값에 따른 θ 민감도 비교</h2>
<p>Y=0.05·0.1·0.5·1mm 모두에 있는 공통 θ=0.01·0.1·1·2°에서만 비교했습니다(각 Y당 12장).
촬영된 θ 범위가 서로 달라 생기는 비교 차이를 줄이기 위한 조건 일치이며, 나머지 사진도 위의 전체 그룹 분석에 포함됐습니다.</p>
{table(common,'theta')}
<p>공통 조건에서 변화율의 최댓값−최솟값은 상단 <b>{summary['interactions']['upper_structure']['a_theta_max_minus_min']:.9f}</b>,
하단 <b>{summary['interactions']['lower_structure']['a_theta_max_minus_min']:.9f}</b> 영상 °/장비 °입니다.
이는 관측된 차이이며 Y와 θ의 상호작용 유무에 대한 유의성 검정 결과는 아닙니다.</p>
<a href="sensitivity_overview.png"><img src="sensitivity_overview.png" alt="모든 구간별 민감도"></a>
<h2>6. 구간별 산점도와 잔차</h2>
<p>위쪽: 개별 사진의 영상 각도와 해당 그룹의 회귀선. 아래쪽: 각 사진이 회귀선에서 벗어난 정도.
사진 번호는 파일의 (1)·(2)·(3) 표기이며 서로 독립적인 촬영 세션을 뜻한다고 가정하지 않았습니다.</p>
<select id="chartSelect" aria-label="분석 구간">{options}</select><br>
<a id="chartLink" href="{firstchart}"><img id="chart" src="{firstchart}" alt="선택한 구간의 산점도와 잔차"></a>
<h2>7. 원본 위 경계점·직선 확인</h2>
<p>각 차수의 이동 범위 끝 조건과 탐색 띠 끝점이 가장 많은 이미지를 표시했습니다. 통계에는 전체 사진이 사용됐습니다.
주황은 입력점 전체, 하늘색은 구조물당 단일 직선입니다. 중앙 가림 구간을 잇는 선은 추정 직선의 연장입니다.
보기 쉽게 세로를 확대했으므로 화면 기울기를 실제 각도로 읽으면 안 됩니다.</p>
<p>총 {sum(r['n'] for r in measures):,}개 점 중 기존 탐색 띠의 끝 행이 선택된 점은 {summary['extraction_endpoint_counts']['total']}개,
해당 직선은 {summary['extraction_endpoint_counts']['lines_with_endpoint_points']}개입니다. 이 점들도 제거하지 않았습니다.</p>
<p>원본 확인에서 D2_058(Y=1mm·θ=2°) 등 일부 끝 조건은 중앙 장착부가 기존 가림 마스크 밖으로 이동해
그 주변의 점도 입력에 포함된 것을 확인했습니다. 영상 가장자리에서도 주 경계에서 벗어난 점이 보입니다.
설정한 마스크나 점 선택을 바꾸지 않고 Huber 가중만 적용한 결과이며, 이 점들의 영향만 따로 분리하는 실험은 하지 않았습니다.</p>{overlay_html}
<h2>8. 이동값이 하나뿐인 조건</h2>
<p>아래 조건은 사진을 측정했지만 해당 축 값이 변하지 않아 상관계수·변화율을 계산할 수 없습니다.
2차 기준 3장은 Y=0·θ=0뿐이므로 Y=0.05 고정 그룹 등에 임의로 넣지 않았습니다.
1차 center는 각 이동 구간에 중복 삽입하지 않고 기준 재현 확인에 사용했습니다.</p>
<div class="scroll"><table><thead><tr><th>조건</th><th>확인하려던 축</th><th>사진 수</th><th>존재하는 이동값</th></tr></thead><tbody>{missing}</tbody></table></div>
<h2>9. 근거·결과 파일</h2><ul>
<li><a href="https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/correlat.pdf">NIST Pearson 상관계수</a> ·
<a href="https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm">NIST 선형 최소제곱 회귀</a> ·
<a href="https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.fit.html">statsmodels RLM.fit</a></li>
<li><a href="settings.json">전체 설정과 고정 탐색 영역</a> · <a href="manifest.json">575장 경로·라벨·정정 내역</a> · <a href="provenance.json">원본 해시·재현 근거</a></li>
<li><a href="measurements.json">1,150개 직선 측정값</a> · <a href="correlations.json">그룹별 모든 관측값·기울기 m 및 각도 φ 회귀·잔차</a> · <a href="summary.json">요약</a></li>
<li><a href="fit_summary.json">Huber 종료 상태·기준값 재현</a> · <a href="verification.json">수치 교차 확인</a></li>
<li>재현 순서: <a href="extract.py">extract.py</a> → <a href="fit.py">fit.py</a> → <a href="analyze.py">analyze.py</a> → <a href="make_report.py">make_report.py</a>.
fit.py는 기존 huber_convergence_20261101/.venv의 Python, 나머지는 기존 기본 Python을 사용합니다.</li></ul>
<p class="note">새 임의 합격선, 데이터 증강, Gaussian, 이상점 제거, 기계적 오차 추정, θ 보정 알고리즘 변경은 수행하지 않았습니다.</p>
</main><script>document.getElementById('chartSelect').addEventListener('change',function(){{document.getElementById('chart').src=this.value;document.getElementById('chartLink').href=this.value;}});</script></body></html>'''
    (HERE/'report.html').write_text(document,encoding='utf-8')
    verification={'source_image_hashes_all_match':True,'reference_point_sets_match':8,
                  'reference_coefficients_max_abs_difference':max(abs(c['slope_delta']) for c in fits['reference_checks']),
                  'regression_groups_crosschecked':len(valid),
                  'max_independent_ols_coefficient_difference':float(max_coef_diff),
                  'max_independent_pearson_difference':float(max_r_diff),
                  'all_rows_retained':True,'all_huber_tolerance_met':fits['tolerance_met']==1150,
                  'figures':charts,'overlays':overlays}
    dump('verification.json',verification)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'] and not v.startswith('http'))
    parser=Links();parser.feed(document)
    missing_links=[p for p in parser.links if not (HERE/p).is_file()]
    assert not missing_links,missing_links
    verification.update(local_links_checked=len(parser.links),missing_links=missing_links)
    dump('verification.json',verification)
    dump('output_hashes.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    print(json.dumps({'quick':quick,'verification':{k:v for k,v in verification.items() if k not in ['figures','overlays']}},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
