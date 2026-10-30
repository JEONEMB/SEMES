import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.workbook.properties import CalcProperties
H = Path(__file__).parent; B = H.parent
J = pd.read_pickle(H/'judge.pkl'); res, Ys, Ts = J['res'], J['Ys'], J['Ts']
est = pd.read_csv(H/'estimate_result.csv')
short = {n: n.split(' ')[0] for n in res}; BEST = 'v2-a2 (판 전체 원근 변환 14개)'
# ---------- 그림: |Y| 구간별 구분 불가 비율 ----------
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
bins=[0,0.25,0.5,0.75,1,1.5,2.01]; labs=['0~0.25','0.25~0.5','0.5~0.75','0.75~1','1~1.5','1.5~2']
M=[]
for n,r in res.items():
    o=r['out']['Y'].assign(v=Ys.value.abs()); o=o[o.규칙2.isin(['가능','불가'])]
    M.append(o.groupby(pd.cut(o.v,bins),observed=True).규칙2.apply(lambda s:(s=='불가').mean()).values)
M=np.array(M)
fig,a=plt.subplots(figsize=(11,4.6))
im=a.imshow(M,cmap='Blues',vmin=0,vmax=1,aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a.text(j,i,f'{M[i,j]*100:.0f}%',ha='center',va='center',fontsize=11,color='white' if M[i,j]>0.55 else '#1f1f1e')
a.set_xticks(range(len(labs))); a.set_xticklabels([f'|Y| {l}mm' for l in labs],fontsize=10)
a.set_yticks(range(len(res))); a.set_yticklabels(list(res),fontsize=10)
a.set_title('Y 사진 중 "QR 위치가 같은 Θ 사진과 구분 불가"인 비율 (규칙2: 통계 거리)',loc='left',fontsize=13,pad=10)
for s in a.spines.values(): s.set_visible(False)
a.set_xticks(np.arange(-.5,len(labs)),minor=True); a.set_yticks(np.arange(-.5,len(res)),minor=True)
a.grid(which='minor',color='white',lw=2); a.tick_params(which='minor',length=0)
fig.text(0.01,0.01,'구간마다 Y 사진 10~20장. 짝 = QR 3개 가로 평균 px가 같은 Θ 지점(보간). 판정 기준 99.73% (3σ 상당).',fontsize=9,color='#5f5e58')
plt.tight_layout(rect=(0,0.04,1,1)); plt.savefig(B/'24_v2_구분불가비율_히트맵.png',dpi=140,facecolor='white')
# ---------- 엑셀 ----------
F='Arial'; base=Font(name=F,size=10); bold=Font(name=F,size=10,bold=True)
hdr=Font(name=F,size=10,bold=True,color='FFFFFF'); hfill=PatternFill('solid',fgColor='1F4E78')
bad=PatternFill('solid',fgColor='FCE4D6'); thin=Side(style='thin',color='D9D9D9')
def H1(ws,r,heads,widths):
    for j,h in enumerate(heads,1):
        c=ws.cell(r,j,h); c.font=hdr; c.fill=hfill; c.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
    for j,w in enumerate(widths,1): ws.column_dimensions[ws.cell(r,j).column_letter].width=w
wb=Workbook(); S=wb.active; S.title='요약'
S['A1']='Y, Θ 구분표 v2: 특징점 조합과 판정 알고리즘 비교'; S['A1'].font=Font(name=F,size=14,bold=True)
S['A2']='Y 사진 80장, Θ 사진 71장(QR 이동량이 Y 범위 안인 것)에 대해, QR 위치가 같은 상대 축 지점과 구분되는지 판정. 숫자가 작을수록 좋음.'; S['A2'].font=base
heads=['버전','사용한 값','값 개수','규칙1 Y 불가','규칙1 Θ 불가','규칙1 같은 축 오판율','규칙2 Y 불가','규칙2 Θ 불가','규칙2 같은 축 오판율','추정 Y 오차 RMS (µm)','추정 Y 최대 (µm)','Y 오차 90µm 초과 (280장 중)']
H1(S,4,heads,[34,46,8,10,10,12,10,10,12,13,12,14])
desc={'v1':'QR 네모 중심 가로 위치 (좌, 중, 우)','v2-b':'v1 + QR 네모 세로 위치, 폭, 높이','v2-a1':'QR마다 내부 픽셀 전체로 맞춘 이동, 기울기, 크기 (ECC affine)','v2-a2':'QR 3개 판 전체를 원근 변환으로 맞춘 귀퉁이 4개와 QR 중심 3개의 이동','v2-a3':'v2-b + v2-a1 + v2-a2 전부','v2-d':'v1 + 뒤쪽 볼트 가로 위치 (QR 밖, 참고용)'}
for i,(n,r) in enumerate(res.items()):
    R=5+i; o=r['out']; e=est[est.버전==n].iloc[0]
    cnt=lambda sh,k:int((o[sh][k]=='불가').sum())
    vals=[n,desc[short[n]],len(r['F']),cnt('Y','규칙1'),cnt('Θ','규칙1'),r['fp1'],cnt('Y','규칙2'),cnt('Θ','규칙2'),r['fp2'],e.Y_RMS_um,e.Y_최대_um,int(e.Y_90um초과)]
    for j,v in enumerate(vals,1):
        c=S.cell(R,j,v); c.font=bold if n==BEST else base; c.border=Border(bottom=thin)
        if j in (6,9): c.number_format='0.0%'
        if j in (10,11): c.number_format='0'
r0=5+len(res)+1
notes=['규칙1: 값마다 "같은 축 사진끼리의 잡음 x 3" 안이면 구분 불가 (v1 방식).',
 '규칙2 (v2-c): 값들의 잡음 크기와 서로 얽힌 정도를 반영한 통계 거리(마할라노비스) 하나로 판정. 기준 99.73% (3σ 상당).',
 '같은 축 오판율: 같은 축 사진끼리 비교했는데 "구분 가능"으로 나온 비율. 낮을수록 판정을 믿을 수 있음. 규칙1은 값이 많아질수록 우연히 하나가 넘을 확률이 커져서 높아짐.',
 '추정 Y 오차: 같은 값들로 Θ와 Y를 직접 계산(선형 최소제곱, 짝/홀 2겹 교차검증). 같은 촬영 구간 사진이 학습과 채점에 섞여 있어 실제보다 좋게 나올 수 있음. 정답값 자체에 약 40µm 오차가 있음.',
 '굵은 행 = QR만 쓰는 버전 중 가장 균형 좋은 v2-a2 (구분 불가가 가장 적은 축에 들고, 오판율이 낮음). Y, Θ 시트의 굵은 행 = v2-a2로도 여전히 구분 불가인 사진.',
 '특징 추출 코드: analysis_v2/extract_homography.py, 판정: analysis_v2/judge.py, 추정: analysis_v2/estimate.py']
for k,t in enumerate(notes): S.cell(r0+k,1,t).font=base
for sh,s in [('Y',Ys),('Θ',Ts)]:
    ws=wb.create_sheet(sh); unit='mm' if sh=='Y' else 'Θ'
    heads=['인덱스','파일명',f'값 ({unit})','좌 QR px','중 QR px','우 QR px','좌중 평균','좌우 평균','중우 평균','좌중우 평균','v1 규칙1 (v1 엑셀과 같음)']+[f'{short[n]} 규칙2' for n in res]+[f'{short[BEST]} 통계 거리']
    H1(ws,1,heads,[7,13,9,10,10,10,10,10,10,11,13]+[10]*len(res)+[11]); ws.row_dimensions[1].height=40; ws.freeze_panes='C2'
    for i,row in s.iterrows():
        R=i+2; still=res[BEST]['out'][sh].loc[i,'규칙2']=='불가'
        vals=[i+1,row.file,row.value,row.dLx,row.dCx,row.dRx,f'=AVERAGE(D{R}:E{R})',f'=AVERAGE(D{R},F{R})',f'=AVERAGE(E{R}:F{R})',f'=AVERAGE(D{R}:F{R})',
              res[list(res)[0]]['out'][sh].loc[i,'규칙1']]+[r['out'][sh].loc[i,'규칙2'] for r in res.values()]
        d=res[BEST]['out'][sh].loc[i,'거리']; vals.append(None if pd.isna(d) else float(d))
        for j,v in enumerate(vals,1):
            c=ws.cell(R,j,v); c.font=bold if still else base; c.border=Border(bottom=thin)
            if v=='불가': c.fill=bad
            if 4<=j<=10: c.number_format='0.000'
            if j==len(vals): c.number_format='0.0'
    ws.cell(len(s)+3,1,f'통계 거리 기준값 ({short[BEST]}): {res[BEST]["dthr"]:.1f} 이하면 구분 불가').font=base
wb.calculation=CalcProperties(fullCalcOnLoad=True)
wb.save(B/'23_YΘ구분표_v2.xlsx'); print('ok')
