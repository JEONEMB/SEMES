"""Build a graph-by-graph PPT protocol report and presentation content data."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def fmt(x,n=3):return f'{x:.{n}f}'
def span(x,n=3):return f'{fmt(min(x),n)} ~ {fmt(max(x),n)}'


def main(out,previous):
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'));m=pd.read_csv(out/'mean_displacements.csv');f=pd.read_csv(out/'linear_fits.csv')
    n=pd.read_csv(out/'point_noise.csv');steps=pd.read_csv(out/'step_statistics.csv');ec=pd.read_csv(out/'ecc.csv');comp=pd.read_csv(out/'method_comparison.csv')
    inv=pd.read_csv(out/'frames.csv');fits=f[(f.scope=='new')&(f.method=='no_diagonal630')]
    fn=fits[fits.varying=='theta_deg'];nn=n[(n.scope=='new')&(n.varying=='theta_deg')]
    counts=pd.read_csv(out/'fitting_summary.csv');raw=pd.read_csv(out/'points.csv.gz')
    qr=raw[raw.path=='기준 위치/center1.bmp'].groupby('qr').agg(xmin=('x','min'),xmax=('x','max'),ymin=('y','min'),ymax=('y','max'))
    def rows_ref():return [[slot,f'({(r.xmin+r.xmax)/2:.2f}, {(r.ymin+r.ymax)/2:.2f})',f'{r.xmax-r.xmin:.2f} × {r.ymax-r.ymin:.2f}'] for slot,r in qr.iterrows()]
    newcorr=ec[ec.category.isin(['mixed','new-reference'])].correlation
    pages=[]
    def add(title,fig=None,rows=None,caption='',section='② 분석에 필요한 정보',full=False,header=False):
        pages.append(dict(title=title,figure=fig,rows=rows,caption=caption,section=section,full=full,header=header))
    add('기초 정보','photo_reference',[
        ['새 데이터','혼합 57장 + 기준 3장 = 60장'],['전체 자료','기존 281장 포함 341장'],['좌표','19곳 × 반복 3장'],['QR','FP11L / FP11C / FP11R'],['영상','3840×2160 RGB BMP'],['범위','Y +0.05~+1.00mm / Θ +0.01~+2.00°'],['기준','새 center1 한 장으로 고정'],['단위','Y: mm, Θ: degree(°) 확정']],
        '원본 PPT처럼 한 장을 기준으로 변위를 구한다. center2/3은 원점 재현 관측으로 남기며 평균 기준으로 섞지 않는다.',section='① 기초 정보')
    add('기준 위치 – 전체 QR 기준','photo_reference',[
        ['QR','중심 좌표 (px)','폭 × 높이 (px)'],*rows_ref(),['기준','새 center1','고정'],['center2/3','별도 원점 관측','기준 평균에 포함하지 않음']],
        '사진과 좌표는 새 center1의 원본 픽셀 계측이다. 거리·초점거리의 절대값을 가정해 표에 넣지 않았다.',header=True)
    add('이동 관계 – 특징점 분석','photo_features',[
        ['전체 모서리','755점 (L258 / C247 / R250)'],['대각선 제외','630점 (L212 / C210 / R208)'],['제외 모서리','대각선 125점'],['위치 fitting','원본 2×2 모듈 경계 밝기 모델'],['창 처리','9×9 / 11×11 각각 위치 적합 후 창 재중심화'],['주 집계','진단과 무관한 고정 630점 전체 평균'],['원본 대조','기준 영상 통과 600/755\nPPT 597/755와 차이를 공개']],
        '초록: 사용 630점 / 주황: 대각선 125점. 원본 PPT의 계측 규칙을 복원했으며 비공개 최적화 세부 설정까지 같다고 주장하지 않는다.')
    add('이동 관계 – ECC 분석','photo_ecc',[
        ['패치','L/C/R 각각 204×216px'],['측정량','패치 affine 변환의 중심점 이동'],['시작 위치','영상 template matching'],['신규 QR 성공','180 / 180'],['신규 correlation','최저 '+fmt(newcorr.min(),6)+'\n중앙값 '+fmt(newcorr.median(),6)],['전처리','입력 smoothing 없음\nECC Gaussian kernel=1'],['비교 제한','PPT의 Gaussian 세부 설정은 미기재']],
        '원본 PPT와 같은 template matching + affine ECC 계열이다. 입력 무필터 조건을 지켜 correlation의 수치 일치만을 목표로 설정을 바꾸지 않았다.')
    add('이동 관계 – 0점 위치','zero_intercepts',[
        ['0점의 정의','직선 기울기를 제거한 trial 절편\n실제 0점 측정과 구분'],['Y 변화 절편',span(fits[fits.varying=='y_mm'].intercept_dx)+'px'],['Θ 변화 절편',span(fn.intercept_dx)+'px'],['해석','fixed-axis 분석 그룹별 차이'],['신규의 한계','양의 좌표만 있어 0점은 외삽'],['보존','raw 변위·직선 계수·잔차 별도 CSV']],
        '원본 PPT처럼 기울기만 제거한 점과 절편선을 함께 표시한다. 절편을 빼서 그래프를 0에 맞추지 않는다.')
    for method,name in [('no_diagonal630','특징점'),('ECC','ECC')]:
        ff=f[(f.scope=='new')&(f.method==method)]
        add('이동 관계 – 이동 경로 ('+name+')','path_new_'+method,[
            ['가로/세로 표시','가로 ×1 / 세로 ×100'],['Y 기울기',span(ff[ff.varying=='y_mm'].slope_dx,2)+'px/mm'],['Θ 기울기',span(ff[ff.varying=='theta_deg'].slope_dx,2)+'px/°'],['Y 직선 잔차 σ',span(ff[ff.varying=='y_mm'].residual_sigma_dx)+'px'],['Θ 직선 잔차 σ',span(ff[ff.varying=='theta_deg'].residual_sigma_dx)+'px'],['선의 의미','명목값 순서의 좌표 평균 연결\n촬영 시간 순서가 아님']],
            '원본 PPT와 같은 가로 변위–세로 변위 XY 경로다. 숫자 축은 원래 px 단위를 유지하고 화면의 세로 배율만 100배로 했다.')
    add('이동 관계 – 한 칸 이동량','steps',[
        ['계산','같은 분석 그룹의 이웃 명목값 차이'],['신규 반복','같은 반복 파일 번호끼리 비교'],['평균 / ±3σ','실제 명령 간격이 같은 칸만 합침'],['다른 간격','0.01°, 0.09°, 1° 등을 섞지 않음'],['기준 첫 칸','혼합 좌표의 시작점은 Y도 0이 아님\n단독 축 첫 칸처럼 합치지 않음'],['자료','모든 pair와 outlier flag를 CSV로 저장']],
        '점은 각 이미지 쌍의 고정 특징점 평균 차이. 가로축은 한 칸이 끝나는 명목값이며, 실선은 같은 간격 평균·점선은 ±3σ다.')
    add('노이즈 – 분석 그룹별 노이즈',rows=[
        ['항목','수치 / 기준','해석'],['0점 어긋남','Θ 분석 그룹 절편 '+span(fn.intercept_dx)+'px','0점에서의 직선 외삽값'],['사진 공통 흔들림','Θ 그룹 σ '+span(fn.residual_sigma_dx)+'px','직선 모델 잔차의 표본 σ'],['점 하나 흔들림','비대각선 가로 중앙값 '+fmt(s['new_point_sigma_median'])+'px','점별 직선 잔차에서 공통 성분 제거'],['대각선 모서리','가로 중앙값 '+fmt(s['new_diagonal_sigma_median'])+'px','같은 잔차 분해로 직접 재계산'],['밝기 변화','원본 QR 패치 평균 밝기','PPT의 판 전체 ROI는 미공개'],['센서 잡음','원본 PPT의 독립 추정식 미기재','새 센서 잡음 수치를 임의로 만들지 않음']],
        caption='대각선은 fitting 진단에서 모두 탈락했다. 작은 잔차 σ를 높은 정확도로 해석할 수 없다. 직선 잔차에는 비선형 모델 오차도 포함된다.',section='③ 노이즈',full=True,header=True)
    add('촬영 시각·0점·반복·밝기','timing_brightness',caption='파일 수정 시각은 촬영 시각의 대리 지표다. 실제 촬영 로그로 확정된 시간 축은 아니다. 같은 1° 구간의 반복 차이와 밝기를 별도로 표시한다.',section='③ 노이즈',full=True)
    add('세로 이동·공통 흔들림·점 고유 흔들림','noise_decomposition',caption='원본 PPT와 같은 4패널 구성. 점 하나의 σ는 점별 직선 잔차에서 630점 공통 성분을 뺀 뒤 구한 표본 표준편차다.',section='③ 노이즈',full=True)
    add('이번 데이터에서 알게 된 것, 모르는 것',rows=[
        ['알게 된 것','계산 결과','확인하지 못한 것'],['혼합 운동','19좌표, 반복 3장, 고정 center1','음의 혼합 영역 일반화'],['기준점 차이','center1/2/3을 독립 관측으로 유지','기구 영점·센서 잡음의 독립 분리'],['계측 비교','755 / 630 / 진단 통과 / ECC 모두 계산','서로 일치해도 절대 정확도 인증은 아님'],['노이즈 분해','직선 잔차 σ와 공통 제거 후 점별 σ','비선형 잔차와 랜덤 잡음의 완전한 분리'],['θ 단위','degree(°), 사용자 확인','없음'],['재현 한계','PPT에 적힌 기준과 수식을 복원','비공개 optimizer / P12 / split의 완전 재현']],
        caption='수치를 맞추려고 원본을 smoothing하거나 점을 사후 제거하지 않았다. 공개되지 않은 구현 설정은 재현 한계로 따로 적는다.',section='④ 정리',full=True,header=True)
    add('기존 281장으로 확인한 수치 대조',rows=[
        ['항목','원본 PPT','이번 방법 복원'],['모서리 수','755 / 630 / 대각선 125','755 / 630 / 대각선 125'],['기준 진단 통과','597 / 755',str(s['center_pass'])+' / 755'],['Y 기울기','14.36 ~ 14.39px/mm',span(s['old_y_slopes'],4)+'px/mm'],['Θ 기울기','80.30 / 80.78px/°',' / '.join(fmt(x,4) for x in s['old_theta_slopes'])+'px/°'],['직선 잔차 σ','Y 0.015~0.034 / Θ 0.029~0.039px',span(s['old_common_sigma_range'],4)+'px (6그룹)'],['원본 코드','첨부되지 않음','동일 문서화 규칙의 복원 구현']],
        caption='새 데이터 결과를 보기 전에 기존 데이터의 기준 항목을 다시 계산했다. 일치하지 않는 세부 수치를 숨기거나 원본 수치에 맞춰 튜닝하지 않았다.',section='④ 정리',full=True,header=True)
    add('그래프별 분석 정의',rows=[
        ['원본 PPT 장','이번 계산','주요 구분'],['6 0점','y−slope·command와 직선 절편','실측 원점과 외삽 절편 구분'],['7–8 경로','dx–dy XY / 화면 세로 100배','Θ–dx 그래프로 대체하지 않음'],['9 한 칸','같은 point ID의 이웃 영상 차이','명령 간격별 평균·σ'],['11 시각/밝기','파일 시각 + 원본 패치 밝기','촬영 로그·판 ROI의 미공개 명시'],['12 공통/개별','점별 OLS 후 630점 공통 제거','3회 반복 RMS로 대체하지 않음'],['17–20 품질/비교','755점 2창 재중심화 및 4방법','진단 탈락도 fixed-point 집계 유지']],
        caption='수치 정의와 실제 촬영 구조를 함께 기록했다. 반복 3장을 서로 독립적인 촬영 trial이라고 임의로 단정하지 않는다.',section='④ 정리',full=True,header=True)
    add('부록',rows=[['측정 방법','품질 지도 / 점별 σ / 4방법 비교 / 기준점·패치 민감도']],section='부록',full=True)
    failures=pd.read_csv(out/'diagnostic_failures.csv')
    add('부록 1. QR 모서리 fitting 진단 기준',rows=[
        ['항목','PPT에 적힌 기준으로 복원'],['모델','원본 2×2 검정/흰색 모듈 경계의 광학 blur 밝기 모델'],['위치 창','9×9 / 11×11 각각 fitting 후 창을 이동해 재적합'],['통과 기준','수렴, RMSE≤15, 대비≥45, 창 차이≤0.65px, 시작점 거리≤3.6px, σ≤2.6'],['σ 하한','PPT에 없는 추가 탈락 기준을 넣지 않음'],['대각선','125점도 동일 방식으로 측정하되 주 평균 630점에서 제외'],['raw 보존','통과/탈락 좌표·모든 진단 이유를 함께 기록'],['전체 진단 실패',' / '.join(f'{r.criterion}: {r.failed:,}' for r in failures.itertuples())]],
        caption='동일한 수식 계열과 명시 기준을 복원한 구현이다. 원본 optimizer의 초기화·반복 수·허용 오차는 공개되지 않아 완전 동일 구현으로 부르지 않는다.',section='부록',full=True,header=True)
    add('부록 2. 모서리 맞춤 전수 결과','fit_group_maps',caption='초록: 해당 Y 고정 그룹의 모든 사진에서 진단 통과 / 주황: 한 번이라도 불확실 / 짙은 주황: 대각선 모서리. 위치는 center1에 표시했다.',section='부록',full=True)
    add('부록 3. 점 고유 흔들림 지도','point_noise_map',caption='진할수록 공통 성분을 뺀 점 고유 가로 σ가 크다. 대각선 125점은 지도에서 제외한다. Y 변화와 Θ 변화 그룹을 나누어 집계했다.',section='부록',full=True)
    comparison=[]
    for method,label in [('all755','모서리 755'),('no_diagonal630','대각선 제외 630'),('accepted_dynamic','진단 통과점'),('ECC','QR 패치 ECC')]:
        ff=f[(f.scope=='new')&(f.varying=='theta_deg')&(f.method==method)]
        a=steps[(steps.scope=='new')&(steps.method==method)&steps.varying.eq('theta_deg')&steps.command_step.eq(1)]
        yy=steps[(steps.scope=='new')&(steps.method==method)&steps.varying.eq('y_mm')&steps.command_step.eq(.5)]
        comparison.append([label,fmt(a.mean_dx.iloc[0],4)+' (σ '+fmt(a.sigma_dx.iloc[0],4)+')',fmt(yy.mean_dx.iloc[0],4)+' (σ '+fmt(yy.sigma_dx.iloc[0],4)+')',span(ff.residual_sigma_dx,4)])
    add('부록 4. 측정 방법 4가지 비교',rows=[['방법','Θ 1° 한 칸 (px)','Y 0.5mm 한 칸 (px)','Θ 그룹 공통 σ (px)'],*comparison],
        caption='원본 PPT와 같이 실제 촬영 간격의 한 칸 이동량과 σ를 비교한다. 신규 자료에 공통으로 존재하는 1°·0.5mm 구간을 골랐으며 전체 간격별 값은 CSV에 있다.',section='부록',full=True,header=True)
    add('부록 5. 계측 방법별 차이','method_difference',[
        ['비교','고정 630점 평균 − 3QR ECC 평균'],['새 60장 가로 RMS',fmt(np.sqrt(np.mean(comp.dx_630_minus_ecc**2)),6)+'px'],['새 60장 가로 최대',fmt(abs(comp.dx_630_minus_ecc).max(),6)+'px'],['해석','같은 reference와 raw 이동량으로 비교'],['주의','서로 다른 계측이 공유하는 편향은\n이 차이만으로 확인할 수 없음']],
        '원본 PPT처럼 사진 번호 축에 방법 간 차이를 표시한다. raw residual을 사후 필터링하지 않았다.',section='부록')
    add('부록 6. 판정 정확도 재현 범위',rows=[
        ['원본 PPT 항목','현재 자료에서 가능한 범위'],['원본 21–22쪽','P12 요약 계수 / 학습 잔차 공분산 / 단독축 비교'],['누락된 자료','원본 P12 정의·코드·저장 분할·반복 seed 미제공'],['새 데이터 구조','두 축이 동시에 움직이는 혼합 자료'],['제시하지 않는 값','다른 모델의 정확도를 원본 방식의 정확도라고 바꿔 쓰지 않음'],['이번 산출','원본 계측·그래프·잔차 σ·한 칸·4방법 비교는 모두 재계산'],['정확한 후속 재현','원본 분석 코드/CSV가 확보되면 동일 분할과 모델로 추가 대조 가능']],
        caption='기존의 단독축 판정 문제와 새로운 혼합 운동 판정 문제는 같은 시험이 아니다. 원본 구현이 없는 정확도 수치를 임의로 생성하지 않았다.',section='부록',full=True,header=True)
    perturb=pd.read_csv(out/'reference_point_perturbation.csv')
    selected=perturb.groupby('path',as_index=False)[['dx_difference','dy_difference']].mean().iloc[[0,5,12,24,36,50]]
    add('부록 7. 기준점이 1px 달라질 때의 영향',rows=[['사진','가로 이동량 차이 (px)','세로 이동량 차이 (px)'],*[[r.path.split('/')[-1],fmt(r.dx_difference,6),fmt(r.dy_difference,6)] for r in selected.itertuples()]],
        caption='같은 affine 변환에서 기준 평가점을 가로 1px·세로 0.75px 이동시켜 계산했다. 변환을 다시 fitting한 실험과 구분한다.',section='부록',full=True,header=True)
    patch=pd.read_csv(out/'patch_comparison.csv');patch=patch[patch.path!='center.bmp']
    pdiff=pd.read_csv(out/'patch_differences.csv')
    pb=json.loads((out/'patch_bounds.json').read_text())
    add('부록 8. ECC 패치 범위 민감도','patch_sensitivity',[
        ['방법','라벨 204×216 / 모듈 경계+8px'],['모듈 범위 창',', '.join(q['slot']+' '+str(q['bounds'][2])+'×'+str(q['bounds'][3]) for q in pb)],['correlation 최저',' / '.join(fmt(g.correlation.min(),6) for _,g in patch.groupby('patch'))],['correlation 중앙값',' / '.join(fmt(g.correlation.median(),6) for _,g in patch.groupby('patch'))],['가로 차이 최대',fmt(abs(pdiff.dx_label_minus_module).max(),6)+'px (QR별)'],['정확한 재현 한계','원본 PPT의 고정창 좌표는 미기재\n이번 두 창의 좌표는 공개']],
        '원본 PPT의 패치 민감도 분석 형태를 따른다. 원본에 없는 고정창 좌표를 같다고 추정하지 않고, 비교 창을 명시적으로 정의했다.',section='부록')
    assert len(pages)==23,len(pages)
    (out/'deck_content.json').write_text(json.dumps(pages,indent=2,ensure_ascii=False),encoding='utf-8')
    maprows=[('6','절편','dx−slope·command','zero_intercepts.png'),('7–8','이동 경로','dx–dy, 화면 세로 100배','path_new_*.png'),('9','한 칸','동일 point ID의 이웃 이미지 차이, 간격별 μ±3σ','steps.png'),('11','시각/밝기','파일 수정 시각 대리축 / 원본 QR 평균','timing_brightness.png'),('12','노이즈','점별 OLS residual → 630점 공통 → 개별 σ','noise_decomposition.png'),('17–18','진단','755점 2창 재중심화, 공개된 통과 기준','fit_group_maps.png'),('19','점 지도','공통 제거 후 점별 σ 지도','point_noise_map.png'),('20','방법 비교','755 / 630 / 통과점 / ECC','step_statistics.csv'),('21–22','판정 정확도','원본 코드·P12·분할 미제공으로 재현 불가','제한 명시'),('23','기준점','affine 평가점을 (1,.75)px 변경','reference_point_perturbation.csv'),('24','패치','명시한 두 ROI의 affine ECC 비교','patch_sensitivity.png')]
    text='# 원본 PPT 분석 방법 기준 재계산 (v2)\n\nθ 단위는 사용자 확인에 따라 **degree(°)**로 확정했다.\n\n'
    text+='앞선 v1은 배치 위주로 맞추고 일부 통계량도 달랐다. 이번에는 755점 전체 재계측, 2창 재중심화, 고정 630점 전체 평균, 한 장 center 기준, OLS 절편·기울기·표본 σ, 간격별 한 칸 이동, 공통/개별 residual 분해로 다시 계산했다.\n\n'
    text+='| 원본 장 | 항목 | 이번 계산 | 산출물 |\n|---|---|---|---|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in maprows)+'\n\n'
    text+='## 핵심 기준\n\n- 신규 기준은 `기준 위치/center1.bmp` 한 장이다. center2/3 평균으로 기준을 바꾸지 않는다.\n- 고정 630점 평균은 QR별 균등 평균이 아닌 **630점 전체를 동일 가중**한다. ECC는 3QR 패치 중심 이동의 평균이다.\n- σ는 ddof=1 표본 표준편차이다. 기존 v1의 반복 RMS와 바꾸어 읽지 않는다.\n- 점별 `d_ij = a_j+b_j*q_i+e_ij`를 적합한 후 `c_i = mean_630(e_ij)`, `u_ij=e_ij-c_i`로 분해한다. 공통 흔들림은 std(c), 점 하나 흔들림은 std(u)이다. 직선 잔차에는 비선형도 섞인다.\n- 새 fixed-axis 그룹은 분석 목적의 그룹이다. 실제 연속 촬영 trial이라고 단정하지 않는다.\n- 새 간격은 불규칙하다. 동일한 명령 간격끼리만 한 칸 μ·σ·3σ 초과 여부를 계산한다.\n- 원본 이미지는 smoothing/denoising하지 않았다. raw residual과 진단 탈락점도 보존한다.\n\n'
    text+='## 원본 수치 대조와 재현 한계\n\n기준 영상 통과는 이번 '+str(s['center_pass'])+'/755, 원본 PPT는 597/755이다. 같은 문서화 기준을 복원했으나 원본의 미공개 optimizer 설정까지 동일하다고 주장하지 않는다.\n\n'
    text+='원본 PPT는 fitting 세부 초기화·반복 수·허용 오차, ECC Gaussian 설정, 판 전체 밝기 ROI, 고정창 좌표, P12 정의와 저장 분할을 제공하지 않는다. 해당 항목은 차이를 명시하며 원본 수치를 맞추기 위해 tuning하지 않았다. 센서 잡음이나 판정 정확도를 임의로 재현했다고 표시하지 않는다.\n\n'
    text+='## 재현\n\n```powershell\npython tool/offaxis/ppt_extract.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2\npython tool/offaxis/ppt_patch_compare.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2\npython tool/offaxis/ppt_analysis.py --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2\npython tool/offaxis/ppt_figures.py --root "data/SSAFY Test Image Set" --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2\npython tool/offaxis/ppt_report.py --previous docs/offaxis-eda --output docs/offaxis-eda/ppt-method-v2\n```\n\n'
    text+='`points.csv.gz`는 재계측 raw fitting, `displacements.csv.gz`는 두 세션의 기준점·변위, `point_linear_residuals.csv.gz`는 공통/개별 분해 전후의 residual이다. 그래프별 집계는 일반 CSV로 제공한다. 파일 시각은 촬영 로그 대신 사용한 대리 지표이다.\n'
    text+='\n## 참고 브랜치 확인\n\n인증 후 `origin/chore/hw-merge` (ef01a82d98d16a077c5ad67443ae4bd2192aa9f5)를 확인했다. `src/alignment.cpp`는 y와 theta를 0으로 반환하는 placeholder이며 원본 PPT 분석 코드나 CSV는 없다. `chore/hye/hw-merge` 이름의 원격 브랜치는 확인되지 않았다.\n\n대각선 125점은 fitting 진단에서 모두 탈락했다. 이 점들의 작은 직선 잔차 σ는 부정확한 위치 모델이나 초기점 고정에 의해서도 생기므로 높은 계측 정확도의 근거로 해석하지 않는다.\n'
    (out/'README.md').write_text(text,encoding='utf-8')
    print('deck pages',len(pages)+1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();main(a.output,a.previous)
