"""Render measured C++ evaluation outputs as Markdown tables and PNG figures.
No training or image inference is performed by this reporting script.
"""
from pathlib import Path
import csv, math, statistics, json, hashlib
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'results'
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

def read(name):
    with (RESULT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter='\t'))
def num(x):
    try:return float(x)
    except:return math.nan
def finite(x):return math.isfinite(num(x))
def pct(x):return f'{100*x:.1f}%'
def quantile(v,p):
    v=sorted(v)
    return v[max(0,math.ceil(len(v)*p)-1)] if v else math.nan
def fmt(x):return f'{x:.5f}' if math.isfinite(x) else '측정 불가'
def metrics(rows,axis,limit):
    errors=[abs(num(r['pred_'+axis])-num(r[axis])) for r in rows if finite(r.get('pred_'+axis))]
    passed=sum(e<=limit+1e-12 for e in errors)
    return [len(rows),len(errors),f'{passed}/{len(rows)} ({pct(passed/len(rows))})' if rows else '자료 없음',fmt(statistics.mean(errors)) if errors else '—',fmt(quantile(errors,.95)),fmt(max(errors)) if errors else '—']
def table(headers,rows):
    cell=lambda x:str(x).replace('|',r'\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(cell,r))+' |' for r in rows])
HEAD=['조건','전체','유효','통과 수/전체','MAE','p95','최대오차']

pred=read('predictions.tsv')
if (RESULT/'frozen_predictions.tsv').exists():pred+=read('frozen_predictions.tsv')
measure={r['index']:r for r in read('measurements.tsv')}
by=defaultdict(list)
for r in pred:by[r['model']].append(r)
base={r['index']:r for r in pred if r['model']=='joint_v2_retrained'}
# Frozen measurements that failed remain in the denominator.
for model in ['joint_v2_frozen_development','control_v1_frozen_development','control_v2_frozen_development']:
    have={r['index'] for r in by[model]}
    for i,r in base.items():
        if i not in have:by[model].append(dict(r,model=model,pred_Y='nan',pred_T='nan',error='measurement unavailable'))

def timing(rows,kind):
    vals=[];decs=[]
    for r in rows:
        m=measure[r['index']];v=num(m[kind]);p=num(r.get('predict_ms'))
        if math.isfinite(v):vals.append(v+(p if math.isfinite(p) else 0))
        if finite(m['decode_ms']):decs.append(num(m['decode_ms']))
    return [fmt(statistics.median(vals)) if vals else '—',fmt(quantile(vals,.95)),fmt(statistics.median(decs)) if decs else '—',len(vals)]
def repeat_table(rows,axis,limit):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(r)
    out=[]
    for g,rs in groups.items():
        if len(rs)<2:continue
        vs=[num(r['pred_'+axis]) for r in rs if finite(r['pred_'+axis])]
        dev=max(abs(x-statistics.mean(vs)) for x in vs) if vs else math.nan
        width=max(vs)-min(vs) if vs else math.nan
        out.append([g,len(rs),len(vs),fmt(dev),fmt(width),'통과' if len(vs)==len(rs) and dev<=limit else '실패/측정불가'])
    return table(['그룹','사진 수','유효','평균 중심 최대편차','전폭','반복 판정'],out) if out else '동일 자세 반복 사진이 없어 이 조건의 반복정밀도는 측정할 수 없다.'

common='''작성일: 2026-11-04. 실행 환경: Windows x64, Intel Core Ultra 9 185H, MSVC v141(19.16), OpenCV 3.4.15, Release, OpenCV 1스레드. 원본 BMP를 다시 읽어 영상 특징을 측정했다. v1/v2 구조 재학습에는 동일 seed20261104 그룹 분할을 사용해 외부 test에 해당하지 않는 train만으로 fold별 모델을 새로 학습했다. LSJ와 배포 모델은 별도로 원래 계수를 고정해 실행했다. center 4장은 이번 회귀 학습·평가에서 제외했다.

정답은 파일명 명령이다. 실제 자세를 독립 계측한 정답과 구분한다. Y 허용범위는 ±0.045, T는 ±0.01(T 1틱)이다. 통과율은 실패를 포함한 전체 배정 사진을 분모로 한다. MAE/p95/최대오차는 유효 예측 기준이며 유효 수를 함께 쓴다. p95는 오름차순 ceil(0.95N)번째다. v1/v2 재학습 표는 모든 사진이 한 번씩 test에 등장하는 OOF이며, LSJ와 원본 배포 모델은 고정 모델 개발 재생으로 별도 표시한다. 새 날짜·신규 독립 데이터 검증은 아니다.

시간은 이 PC에서 한 번 순회하며 측정한 개발 실행값이다. 카메라 촬영·장비 이동·정지 대기·네트워크·운영 GUI는 포함하지 않는다. 이미지 읽기+그레이스케일 디코딩은 별도다. 측정 도중 빌드 작업이 일부 겹쳤으므로 단독 실행 최적 성능이나 하드 실시간 상한으로 해석하지 않는다. joint v1/v2는 같은 영상 측정값을 공유하고 회귀 계산만 다르다. control T=0 변형은 공통 tracker+P12+mean 측정 파이프라인 시간이며 원본 공개 API 전체 시간과 동일하다고 주장하지 않는다.

1029 파일명 T=0.32 2장/T=0.33 4장 예외를 그대로 유지했다. 같은 파일명이 두 개지만 영상 해시가 다른 BMP를 서로 다른 경로로 보존했고 파일명 정답을 임의 수정하지 않았다. 이 사례는 촬영자 확인 전 라벨 정확성에 제한이 있다.
'''
return_section='''## 평가 2. 실제 두 단계 센터 복귀

| 항목 | 이번 결과 | 이유 |
|---|---|---|
| 실제 센터 복귀 성공률 | 미측정 | 로봇 이동·카메라 재촬영·독립 위치 계측 세션 없음 |
| 도착까지 step 수 | 미측정 | 실제 이동 이력 없음 |
| 도착까지 총 시간 | 미측정 | 이동·정지·촬영 시간이 없음 |
| 최종 Y/T 실제 잔차 | 미측정 | 이동 후 실제 사진/독립 계측 없음 |

계산상 두 번의 명령이 필요하다고 실제 2-step 성공이라고 쓰지 않는다. 현재 정지 사진 571장은 각 사진에서 추정이 얼마나 맞는지는 보여주지만 그 명령으로 움직인 뒤 어떤 영상이 찍힐지는 제공하지 않는다. 오차를 수식으로 빼서 만든 잔차도 물리 복귀 실측이 아니다.

실기 시험은 `촬영 → joint T 추정 → T 이동 → 정지/재촬영 → T 조건 확인 → control Y 추정 → Y 이동 → 정지/재촬영 → Y/T 동시 검증` 순서다. 보정 이동 1회를 1 step으로 세며, 재시도까지 최대 20 step을 사용한다. 시작 촬영부터 최종 판정까지의 벽시계 시간을 총 시간으로 기록한다. Y ±0.045와 T ±0.01을 동시에 만족해야 성공이다. 실패·보류·20회 초과도 전체 시행 분모에 포함한다. 같은 시작 자세에서 반복 복귀해 평균 편향, 평균 중심 최대편차, 전폭을 따로 보고한다.

실제 위치 판정에는 가능한 한 모델과 독립된 기준기를 사용한다. 자기 추정값만 0이면 `영상 자기 점검`이며 실제 위치 정확도 확정과 구분한다. 장비 명령의 부호·단위를 확인하고 촬영 대기시간·속도·시작 자세·반복 횟수를 실행 전에 고정한다. [실기 기록 양식](return_trials_template.csv)에 실제 실행값을 채운 후 이 표를 갱신한다.
'''

summary=[]
for version in [1,2]:
    jname=f'joint_v{version}_retrained'+('_GLS' if version==1 else '')
    cname='control_v1_T0_four_Y_branches' if version==1 else 'control_v2_T0_no_T_correction'
    jr=by[jname];cr=by[cname]
    text=f'# final v{version} 평가 보고서\n\n'+common
    if version==1:
        text+='''\n## 평가한 구현과 원본 차이\n\njoint는 v1의 102관측+상호작용 204차원, degree 1 전방 GLS와 상관행렬 축소 0.01을 C++ 평가기에서 재적합했다. 원본 측정기는 v2에서도 동일 구조이므로 공통 Observer로 측정했다. 원본 배포 모델의 재학습 없는 출력도 별도 표에 싣는다. 재구현 학습기와 원본 학습기의 수치 동등성 전체 검증을 완료한 것은 아니므로 `v1 구조 재학습`으로 표시한다.\n\ncontrol 원본은 Y/T 여섯 가지 모델이며 T 자료가 있어야 T 가지도 학습할 수 있다. 사용자 요청 T=0-only를 지키기 위해 Y 4개 원점 가지와 공유 2차 반응만 학습하고 영상 잔차로 가지를 선택했다. 원래 Y1/Y2 촬영 구분과 부호 제약은 유지했다. 잔차 공분산 축소는 기존 0.05이며 이번 Y-only 조건의 자유도는 학습 장수-6이다. 원본 6가지 배포 모델과 같은 알고리즘이라고 부르지 않는다.\n'''
    else:
        text+='''\n## 평가한 구현과 원본 차이\n\njoint는 원본 v2의 QR 평균 가로 이동 s와 상·하단 바 기울기 차 d로 `[1,s,d]→(Y,T)` 최소제곱을 fold별 재학습했다. 기존 도구의 |T|≤1 필터는 사용하지 않고 전체 범위를 포함했다.\n\ncontrol v2 원본은 T≠0 자료로 잔여 T 회귀와 Y 보정계수를 학습한다. 사용자 요청 T=0-only에서는 그 계수를 식별할 수 없다. 따라서 T=0 train의 s→Y 직선만 재학습하고 잔여 T 보정은 끈 변형을 평가했다. 이것은 원본 v2의 보정 효과를 평가한 점수가 아니다. 원본 배포 v2의 보정 포함 출력은 아래 개발 재생 표에 별도로 싣는다.\n'''
    text+='\n## 평가 1A. joint 전체 test의 T 정확도\n\n'
    text+=table(HEAD,[[d]+metrics([r for r in jr if d=='전체' or r['date']==d],'T',.01) for d in ['20261014','20261020','20261029','전체']])
    text+='\n\n### fold별 joint T 정확도\n\n'+table(HEAD,[[f'fold {f:02}']+metrics([r for r in jr if int(r['fold'])==f],'T',.01) for f in range(1,6)])
    text+='\n\n## 평가 1B. control T=0 기본 Y 정확도\n\n'
    text+=table(HEAD,[['T=0 / 1014 Y 80장']+metrics([r for r in cr if num(r['T'])==0],'Y',.045)])
    text+='\n\n각 fold T=0 train 64장만 사용했다. test는 16장씩 총 80장이다. 1020/1029에는 center 제외 T=0 자료가 없어 기본 성적을 계산할 수 없다. 각 자세 한 장이므로 T=0 반복정밀도는 추가 반복 촬영이 필요하다.\n'
    bins=[('T=0',lambda t:t==0),('0<|T|≤0.01',lambda t:0<abs(t)<=.01+1e-10),('0.01<|T|≤0.03',lambda t:.01+1e-10<abs(t)<=.03+1e-10),('0.03<|T|≤0.10',lambda t:.03+1e-10<abs(t)<=.1+1e-10),('0.10<|T|≤0.30',lambda t:.1+1e-10<abs(t)<=.3+1e-10),('0.30<|T|≤1',lambda t:.3+1e-10<abs(t)<=1+1e-10),('|T|>1',lambda t:abs(t)>1+1e-10)]
    text+='\n## 평가 1C. control 잔여 T 범위별 Y 정확도\n\nT=0으로 학습한 모델을 고정하고 나머지 T≠0 사진에 적용했다. 정답 T로 나눈 사후 분석이며 추론에 T 정답을 알려주지 않는다.\n\n'
    text+=table(HEAD,[[label]+metrics([r for r in cr if fn(num(r['T']))],'Y',.045) for label,fn in bins])
    text+='\n\n### 날짜별 T≠0 성적\n\n'+table(HEAD,[[d]+metrics([r for r in cr if r['date']==d and num(r['T'])!=0],'Y',.045) for d in ['20261014','20261020','20261029']])
    near=[r for r in cr if 0<abs(num(r['T']))<=.01+1e-10]
    nearpass=sum(finite(r['pred_Y']) and abs(num(r['pred_Y'])-num(r['Y']))<=.045+1e-12 for r in near)
    text+=f'\n\n**팀원용 해석:** T=0에서 잘 맞는다는 사실만으로 두 단계 복귀가 보장되지는 않는다. T가 1틱 이내로 조금 남은 {len(near)}장에서는 Y 통과가 {nearpass}장이다. 이 구간의 표본은 적고, 더 큰 잔여 T 구간에서는 Y 오차가 커진다. 실제 제어에서는 T 단계의 잔차와 control 적용 조건을 함께 확인해야 한다. 원본 v2의 잔여 T 보정 기능을 끈 시험이라는 점도 함께 읽어야 한다.\n'
    text+=f'\n\n![잔여 T와 control Y 오차](control_range_v{version}.png)\n'
    fig,ax=plt.subplots(figsize=(10,5))
    for d,col in [('20261014','#496de3'),('20261020','#efaa45'),('20261029','#26a494')]:
        sub=[r for r in cr if r['date']==d and finite(r['pred_Y'])]
        ax.scatter([num(r['T']) for r in sub],[num(r['pred_Y'])-num(r['Y']) for r in sub],label=d,s=13,alpha=.7,color=col)
    ax.axhspan(-.045,.045,color='#26a494',alpha=.15,label='Y 허용범위 ±0.045')
    ax.set_xlabel('정답 T 명령');ax.set_ylabel('Y 추정 - Y 정답');ax.set_title(f'control v{version} T=0-only 변형: 잔여 T에 따른 Y 오차');ax.legend();fig.tight_layout();fig.savefig(ROOT/f'control_range_v{version}.png',dpi=170);plt.close(fig)
    text+='\n\n### 원본 배포 모델 개발 재생 (OOF 재학습과 구분)\n\n학습 때 이미 사용한 사진이 포함되므로 아래 수치를 독립 검증이나 같은 5-fold OOF로 부르지 않는다. 원본 control은 T=0-only 학습 조건과 다르다.\n\n'
    frozenj=by[f'joint_v{version}_frozen_development'];frozenc=by[f'control_v{version}_frozen_development']
    text+=table(HEAD,[[f'원본 joint v{version} / 전체 T']+metrics(frozenj,'T',.01),[f'원본 control v{version} / T=0 Y']+metrics([r for r in frozenc if num(r['T'])==0],'Y',.045),[f'원본 control v{version} / T≠0 Y']+metrics([r for r in frozenc if num(r['T'])!=0],'Y',.045)])
    text+='\n\n원본 control의 잔여 T 구간별 개발 재생:\n\n'+table(HEAD,[[label]+metrics([r for r in frozenc if fn(num(r['T']))],'Y',.045) for label,fn in bins])
    text+='\n\nT≠0 전체에는 Y=0인 1014/T 200장도 포함되므로 혼합 자세만의 성능과 같지 않다. T=2의 12장은 외삽·정합 위험이 있는 큰 자세로 분리해 읽는다. 원본 v2의 보정 효과는 이 개발 재생에서 확인하되 T=0-only 재학습과 섞어 한 점수로 보고하지 않는다.\n'
    text+='\n\n### 처리시간 (ms)\n\n'+table(['측정 파이프라인','측정+예측 중앙값','측정+예측 p95','읽기/디코딩 중앙값(별도)','측정 유효 수'],[['joint v'+str(version)]+timing(jr,'joint_observe_ms'),['control T=0 변형 (전수 적용)']+timing(cr,'control_track_ms')])
    text+='\n\nOpenCV 스레드 1개, 동일 실행의 공통 측정 캐시를 사용했다. 추론 전 모델/기준 영상 초기화와 fold 학습 시간은 `results/environment.txt`, `results/fold_*.yml`에서 확인한다. 원본 control v2는 추가 바 측정이 있으므로 위 T=0 변형 시간을 원본 v2 시간으로 인용하면 안 된다.\n'
    if (RESULT/f'benchmark_v{version}.tsv').exists():
        bench=read(f'benchmark_v{version}.tsv')
        br=[]
        for stage in ['joint','control']:
            vs=[num(r['infer_ms']) for r in bench if r['stage']==stage and finite(r['infer_ms'])]
            ds=[num(r['decode_ms']) for r in bench if r['stage']==stage and finite(r['decode_ms'])]
            br.append([f'원본 v{version} {stage}',len(vs),fmt(statistics.median(vs)),fmt(quantile(vs,.95)),fmt(statistics.median(ds))])
        text+='\n### 원본 공개 API 별도 시간 측정\n\n각 날짜 4장씩 사전에 정한 동일 12장을 사용했다. 초기화 후 첫 영상으로 joint/control을 한 번씩 워밍업하고, 원본 `joint()`·`control()` 공개 함수를 호출했다. 이미지 읽기·디코딩은 별도이며 학습 시간과 장비 이동은 포함하지 않는다. 이는 12장 표본 실행이므로 571장 전수 시간 분포와 구분한다.\n\n'+table(['공개 함수','표본 수','추론 중앙값 ms','추론 p95 ms','읽기/디코딩 중앙값 ms'],br)+'\n\n[원본 API 시간·출력](results/benchmark_v'+str(version)+'.tsv), [원본 API 측정 코드](src/bench.cpp)를 참조한다. 12장 표본에서 공개 함수와 기존 모델 재생의 joint Y/T 및 control Y 출력 차이는 모두 0이었다([대응 검사](results/public_api_parity.json)). 이는 원본 재생 경로 검사이며 새 학습기 전체의 동등성 증명과 구분한다.\n'
    text+='\n### 성능 시각화\n\n![모델별 평가 비교](comparison.png)\n'
    text+='\n### 반복정밀도 상세\n\nY는 반복 평균 중심 최대편차 0.045, T는 0.01을 보고 기준으로 사용하며 전폭도 병기한다. 정확도 통과와 별개다.\n\n'
    text+='joint T:\n\n'+repeat_table(jr,'T',.01)+'\n\ncontrol Y (T≠0 참고 조건):\n\n'+repeat_table(cr,'Y',.045)+'\n\n'
    text+=return_section
    text+='\n## 산출물과 재현\n\n[사진별 예측](results/predictions.tsv), [원본 배포 모델 예측](results/frozen_predictions.tsv), [측정·시간·실패](results/measurements.tsv), [학습/평가 소스](src/eval.cpp), [원본 모델 재생 소스](src/frozen.cpp), [입력 라벨·분할](input.tsv), [실행 안내](README.md)를 함께 확인한다. 기존 assets와 활성 모듈 선택은 변경하지 않았다.\n'
    (ROOT/f'보고서_final_v{version}.md').write_text(text,encoding='utf-8')
    summary.append([f'v{version} joint / 전체 T']+metrics(jr,'T',.01))
    summary.append([f'v{version} control T0변형 / T=0 Y']+metrics([r for r in cr if num(r['T'])==0],'Y',.045))

# Figures compare strict protocols, never frozen training reuse.
fig,axs=plt.subplots(1,2,figsize=(12,5));fig.suptitle('동일 5-fold 재학습 평가 · 정지 사진',fontsize=18,fontweight='bold')
for ax,models,axis,limit,title in [(axs[0],['joint_v1_retrained_GLS','joint_v2_retrained'],'T',.01,'joint 전체 T ±0.01'),(axs[1],['control_v1_T0_four_Y_branches','control_v2_T0_no_T_correction'],'Y',.045,'control T=0 Y ±0.045')]:
    vals=[]
    for model in models:
        rs=[r for r in by[model] if axis=='T' or num(r['T'])==0]
        vals.append(100*sum(finite(r['pred_'+axis]) and abs(num(r['pred_'+axis])-num(r[axis]))<=limit+1e-12 for r in rs)/len(rs))
    ax.bar(['v1 구조','v2 구조'],vals,color=['#496de3','#26a494']);ax.set_ylim(0,110);ax.set_ylabel('통과율 (%)');ax.set_title(title)
    for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}%',ha='center')
fig.text(.5,.025,'control은 T=0-only 변형 · 원본 v2의 잔여 T 보정은 포함하지 않음',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)

if (RESULT/'lsj_new.tsv').exists():
    rs=read('lsj_new.tsv')
    for r in rs:
        r['pred_Y']=r['pred_Y_nuisance']
        r['group']=base[r['index']]['group']
    text='# LSJ 정확도·실행 상태 보고서\n\n'+common
    text+='''\n## 어떤 코드를 평가했나?\n\n`S15P31A507-LSJ` 커밋 8738cac의 두 구현을 구분했다.\n\n1. 최신 `experiments/LSJ/algorithm/theta_huber_20261101`: 상·하단 Huber 직선 각도 두 개로 Y/T를 동시에 푼다. Y는 `nuisance_y_mm` 내부 보조값이며 공개 API의 Y 보정 명령은 아니다. T 추정과 반대 부호 T 보정 명령을 구분한다.\n2. 루트 `src/`: 이전 θ→Y 순차 명령 구현이다. 한 장에서 다음 한 축 명령과 명령 가능 여부를 계산했다. 실제로 그 명령을 수행하거나 같은 정지 사진을 움직인 다음 프레임처럼 재사용하지 않았다.\n\nLSJ 계수는 새로 재학습하지 않고 해당 커밋의 고정 설정을 사용했다. 학습에 사용된 사진이 들어 있으므로 전체 571장 성적을 독립 OOF라고 부르지 않는다. v1/v2 재학습 성적과 평가 노출 조건이 다르다. LSJ가 사용하는 명목 mm/도 라벨과 파일명 명령을 비교했으며 임의 물리 단위 환산은 하지 않았다.\n'''
    text+='\n## 최신 동시 해석의 정확도\n\n'+table(HEAD,[[d+' / T']+metrics([r for r in rs if d=='전체' or r['date']==d],'T',.01) for d in ['20261014','20261020','20261029','전체']]+[[d+' / 내부 Y']+metrics([r for r in rs if d=='전체' or r['date']==d],'Y',.045) for d in ['20261014','20261020','20261029','전체']])
    with (ROOT/'input.tsv').open(encoding='utf-8-sig',newline='') as f: inputs=list(csv.DictReader(f,delimiter='\t'))
    held=[r for r in rs if r['date']=='20261029' or (r['date']=='20261020' and '(3).bmp' in inputs[int(r['index'])]['path'])]
    text+='\n\n### 계수 적합 사진과 구분한 참고 부분집합\n\n기존 LSJ 계수는 1020 반복 (1)·(2)로 적합했다. 아래는 1020 반복 (3)과 1029 전체이며 center를 제외해 253장이다. 같은 자세의 다른 반복이 학습에 있어 그룹 독립 OOF와는 다르다.\n\n'+table(HEAD,[['기존 보류 부분집합 / T']+metrics(held,'T',.01),['기존 보류 부분집합 / 내부 Y']+metrics(held,'Y',.045)])
    both=sum(finite(r['pred_Y']) and finite(r['pred_T']) and abs(num(r['pred_Y'])-num(r['Y']))<=.045+1e-12 and abs(num(r['pred_T'])-num(r['T']))<=.01+1e-12 for r in rs)
    times=[num(r['ms']) for r in rs if finite(r['ms'])]
    text+=f'\n\nY/T 동시 통과: **{both}/{len(rs)} ({pct(both/len(rs))})**. 이는 정지 사진의 내부 자세 추정 결과이며 Y/T 동시 이동 명령 성공률이 아니다.\n\n'
    text+=table(['시간 구간','중앙값 ms','p95 ms','유효 수'],[['최신 Huber 영상 측정+동시 해석 (읽기/디코딩 제외)',fmt(statistics.median(times)) if times else '—',fmt(quantile(times,.95)),len(times)]])
    if (RESULT/'lsj_root.tsv').exists():
        roots=read('lsj_root.tsv')
        for r in roots:r['pred_Y']='nan'
        text+='\n\n## 루트 순차 명령의 정지 사진 진단\n\n'+table(HEAD,[[d+' / T']+metrics([r for r in roots if d=='전체' or r['date']==d],'T',.01) for d in ['20261014','20261020','20261029','전체']])
        statuses=defaultdict(int)
        for r in roots:statuses[r['status'].split(';seq:')[-1]]+=1
        text+='\n\n'+table(['한 장에서 계산한 다음 명령','장수'],[['명령 가능',sum(r['sequence_available']=='1' for r in roots)],['명령 불가/측정 실패',sum(r['sequence_available']!='1' for r in roots)]])
        text+='\n\n'+table(['상태','장수'],sorted(statuses.items(),key=lambda x:-x[1]))
        text+='\n\n루트 Y는 자세 추정값 대신 목표 범위로 가기 위한 보정 명령과 허용 구간을 낸다. 이 명령을 현재 Y 추정값이라고 임의 치환해 정확도를 만들지 않았다. Y 단계 진입과 실제 최종 도착은 새 영상이 있어야 검증할 수 있다.\n'
    text+='\n## 실제 여러 step 복귀\n\n최신 LSJ 모듈은 Y를 내부적으로 계산해도 Y 보정 명령을 제공하지 않는다. 따라서 이 모듈 단독의 Y/T 동시 이동·다중 step 센터 복귀 성공은 현재 코드로 측정하지 않았다. 아래 θ→Y 절차는 Y 명령이 있는 LSJ 루트 순차 구현의 향후 실기 시험에 해당한다.\n\n'+return_section.split('## 평가 2. 실제 두 단계 센터 복귀\n',1)[1].replace('joint T 추정','LSJ 루트 θ 추정').replace('control Y 추정','LSJ 루트 Y 명령 계산')
    text+='\n## 반복정밀도와 근거\n\nT 반복:\n\n'+repeat_table(rs,'T',.01)+'\n\n내부 보조 Y 반복 (실제 Y 명령의 반복 복귀 정밀도와 구분):\n\n'+repeat_table(rs,'Y',.045)+'\n\n[최신 사진별 Y/T 출력](results/lsj_new.tsv), [루트 사진별 명령 진단](results/lsj_root.tsv), [최신 모듈 호출 코드](src/lsj_new.cpp), [루트 호출 코드](src/lsj.cpp). LSJ 소스·계수는 변경하거나 커밋하지 않았다.\n'
    (ROOT/'보고서_LSJ.md').write_text(text,encoding='utf-8')
    summary.extend([['LSJ 고정 / 전체 T']+metrics(rs,'T',.01),['LSJ 고정 / 내부 Y']+metrics(rs,'Y',.045)])

overview='# 성능 비교 요약\n\n'+table(HEAD,summary)+'\n\ncontrol은 T=0-only 재학습 변형이고 LSJ는 고정 계수 개발 재생이다. 동일 알고리즘·동일 학습 노출 조건으로 오해하지 않는다. 실제 복귀 step·총 시간·최종 실측 정확도는 세 보고서 모두 미측정이다.\n\n![재학습 비교](comparison.png)\n\n- [final v1 보고서](보고서_final_v1.md)\n- [final v2 보고서](보고서_final_v2.md)\n- [LSJ 보고서](보고서_LSJ.md)\n'
bt=[]
for version in [1,2]:
    if (RESULT/f'benchmark_v{version}.tsv').exists():
        b=read(f'benchmark_v{version}.tsv')
        for stage in ['joint','control']:
            vs=[num(r['infer_ms']) for r in b if r['stage']==stage and finite(r['infer_ms'])]
            bt.append([f'원본 v{version} {stage}',len(vs),fmt(statistics.median(vs)),fmt(quantile(vs,.95))])
if (RESULT/'lsj_new.tsv').exists():
    vs=[num(r['ms']) for r in read('lsj_new.tsv') if finite(r['ms'])]
    bt.append(['LSJ 최신 내부 Y/T 해석',len(vs),fmt(statistics.median(vs)),fmt(quantile(vs,.95))])
overview+='\n## 처리시간 (영상 읽기/디코딩·장비 이동 제외)\n\n'+table(['측정 대상','사진 수','중앙값 ms','p95 ms'],bt)+'\n\nv1/v2 원본 공개 API는 동일 12장 표본, LSJ는 571장 순회다. 시간 표의 원본 control과 위 정확도 표의 T=0 전용 변형을 구분한다. 원본 모델의 정확도는 각 보고서의 개발 재생 표에 따로 실었다.\n'
(ROOT/'성능비교_요약.md').write_text(overview,encoding='utf-8')
print(overview.split('![')[0])
