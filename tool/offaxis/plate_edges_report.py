"""Conclusion-first plate-edge comparison using existing actuator-labelled data."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.analysis_packages'))
import base64
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from plate_edges import OUT,BASE,TICKS,fit_ridge,predict,tune,ALPHAS


def diagnostics():
    meta=pd.read_csv(BASE/'ppt-method-v2/frames.csv').set_index('path')
    mixed=meta[meta.category.eq('mixed')].sort_index();refs=meta[meta.category.eq('new-reference')]
    qr=pd.read_csv(BASE/'ppt-method-v2/qr_displacements.csv').pivot(index='path',columns='qr',values=['dx','dy'])
    cols=[(d,s) for s in 'LCR' for d in ['dx','dy']]
    qx=qr[cols].loc[mixed.index].to_numpy()
    ef=pd.read_csv(OUT/'edge_features.csv').set_index('path')
    edgecols=[e+'_'+v for e in ['top','bottom','left','right'] for v in ['position','slope']]
    ex=ef.loc[mixed.index,edgecols].to_numpy();reference=ef.loc['기준 위치/center1.bmp',edgecols].to_numpy(dtype=float)
    ex-=reference
    actual=mixed[['y_actual','theta_actual']].to_numpy()-refs[['y_actual','theta_actual']].mean().to_numpy()
    y=actual/TICKS;coords=mixed[['y_mm','theta_deg']].to_numpy()
    unique=np.unique(coords,axis=0);groups=np.array([np.where(np.all(unique==q,axis=1))[0][0] for q in coords])
    sets={'QR6_nested':qx,'QR6_edges8_nested':np.column_stack([qx,ex]),'QR6_horizontal4_nested':np.column_stack([qx,ex[:,:4]])}
    blockrows=[];refrows=[]
    qrref=qr[cols].loc[refs.index].to_numpy()
    eref=ef.loc[refs.index,edgecols].to_numpy()-reference
    refsets={'QR6_nested':qrref,'QR6_edges8_nested':np.column_stack([qrref,eref]),'QR6_horizontal4_nested':np.column_stack([qrref,eref[:,:4]])}
    final=json.loads((OUT/'models.json').read_text())
    for name,x in sets.items():
        held=np.zeros_like(y)
        for value in np.unique(coords[:,0]):
            test=coords[:,0]==value;train=~test
            alpha=tune(x[train],y[train],groups[train])[0]
            held[test]=predict(fit_ridge(x[train],y[train],alpha),x[test])
        e=held-y
        blockrows.append(dict(model=name,validation='leave_entire_Y_sweep_out',images=57,Y_rmse_ticks=np.sqrt(np.mean(e[:,0]**2)),
            yaw_rmse_ticks=np.sqrt(np.mean(e[:,1]**2)),both_within_tick=float(np.mean(np.max(abs(e),axis=1)<=1))))
        m={k:np.array(v) if isinstance(v,list) else v for k,v in final[name].items()}
        rp=predict(m,refsets[name])
        for path,p in zip(refs.index,rp):refrows.append(dict(model=name,path=path,predicted_y_mm=p[0]*.05,predicted_yaw_deg=p[1]*.01,
            Y_error_ticks=p[0],yaw_error_ticks=p[1],reference_not_used_for_training=True))
    pd.DataFrame(blockrows).to_csv(OUT/'Y_sweep_holdout.csv',index=False)
    pd.DataFrame(refrows).to_csv(OUT/'reference_predictions.csv',index=False)
    pred=pd.read_csv(OUT/'coordinate_holdout.csv')
    base=pred[pred.model.eq('QR6_nested')].groupby('coordinate_group')[['Y_error_ticks','yaw_error_ticks']].agg(lambda a:np.mean(a*a)).to_numpy()
    rng=np.random.default_rng(271026);draws=rng.integers(0,19,(5000,19));rows=[]
    for name in ['QR6_edges8_nested','QR6_vertical4_nested','QR6_horizontal4_nested']:
        new=pred[pred.model.eq(name)].groupby('coordinate_group')[['Y_error_ticks','yaw_error_ticks']].agg(lambda a:np.mean(a*a)).to_numpy()
        ratios=np.sqrt(new[draws].mean(1)/base[draws].mean(1))
        wins=np.mean(new,axis=1)<np.mean(base,axis=1)
        rows.append(dict(model=name,coordinates_improved=int(wins.sum()),coordinates=19,
            Y_rmse_ratio_05=float(np.quantile(ratios[:,0],.05)),Y_rmse_ratio_95=float(np.quantile(ratios[:,0],.95)),
            yaw_rmse_ratio_05=float(np.quantile(ratios[:,1],.05)),yaw_rmse_ratio_95=float(np.quantile(ratios[:,1],.95))))
    pd.DataFrame(rows).to_csv(OUT/'paired_coordinate_sensitivity.csv',index=False)
    return pd.DataFrame(blockrows),pd.DataFrame(refrows),pd.DataFrame(rows)


def main():
    blocks,refs,paired=diagnostics()
    scores=pd.read_csv(OUT/'comparison.csv').set_index('model')
    b=scores.loc['QR6_nested'];new=scores.loc['QR6_edges8_nested'];horizontal=scores.loc['QR6_horizontal4_nested']
    gy=(1-new.Y_rmse_mm/b.Y_rmse_mm)*100;gt=(1-new.yaw_rmse_deg/b.yaw_rmse_deg)*100
    labels={'QR6_nested':'QR만','QR6_edges8_nested':'QR + 네 엣지','QR6_horizontal4_nested':'QR + 위·아래 엣지 (추가 비교)',
        'QR6_vertical4_nested':'QR + 좌·우 엣지 (추가 비교)','edges8_nested':'네 엣지만'}
    table=scores.loc[list(labels),['Y_rmse_mm','yaw_rmse_deg','both_within_tick']].copy()
    table.index=[labels[n] for n in table.index];table.both_within_tick*=100
    table.columns=['Y RMSE · mm','yaw RMSE · °','두 축 ±1틱 · %']
    feature=pd.read_csv(OUT/'edge_features.csv');repeat=pd.read_csv(OUT/'edge_repeatability.csv')
    complete=int(feature.filter(regex='_position|_slope').notna().all(axis=1).sum())
    # Korean summary card.
    card=Image.new('RGB',(1450,1100),'#f4f7fa');d=ImageDraw.Draw(card)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
    bold=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',n)
    d.text((55,40),'긴 엣지를 추가하면 Y–yaw가 더 정밀해질까?',font=bold(37),fill='#193e53')
    d.rounded_rectangle((45,115,1405,225),radius=12,fill='#dff1e7')
    d.text((65,138),f'보유 데이터에서 실제 개선: Y {gy:.1f}% · yaw {gt:.1f}% RMSE 감소',font=bold(31),fill='#246647')
    d.text((65,185),'새 촬영 없음 · 2차 장비 도달 기록 기준 · 좌표별 반복 3장 전체 보류',font=font(24),fill='#426e59')
    # Original overlay preview, not an AI rendering.
    im=Image.open(OUT/'01_edges_new.png').convert('RGB');im.thumbnail((450,500));card.paste(im,(60,270))
    for x,text in [(555,'입력'),(930,'Y RMSE'),(1150,'yaw RMSE')]:d.text((x,300),text,font=bold(25),fill='#35586b')
    for j,(name,label) in enumerate([('QR6_nested','QR만'),('QR6_edges8_nested','QR + 네 엣지')]):
        r=scores.loc[name];yy=365+j*110
        d.text((555,yy),label,font=bold(27),fill='#294d62')
        d.text((920,yy),f'{r.Y_rmse_mm:.4f} mm',font=bold(27),fill='#2b7654' if j else '#5e7483')
        d.text((1160,yy),f'{r.yaw_rmse_deg:.4f}°',font=bold(27),fill='#2b7654' if j else '#5e7483')
    d.text((555,625),'Canny 대신 원본 밝기 단면의 수치 적합',font=font(25),fill='#486473')
    d.text((555,670),'60/60장 네 엣지 측정 · 평가 57장 전부 유지',font=font(24),fill='#486473')
    d.text((65,815),'위·아래 엣지만 추가한 탐색 비교도 더 좋은 결과를 보였습니다.',font=bold(27),fill='#28566f')
    d.text((65,865),f'Y {horizontal.Y_rmse_mm:.4f}mm · yaw {horizontal.yaw_rmse_deg:.4f}° — 부분집합 선택의 최종 성능은 아님',font=font(25),fill='#486473')
    d.rounded_rectangle((50,940,1400,1065),radius=12,fill='#fff0dd')
    d.text((70,960),'아직 두 축 모두 한 틱을 안정적으로 달성한 것은 아닙니다.',font=bold(28),fill='#96591f')
    d.text((70,1008),'엣지 추가는 유효하며, 이후 알고리즘의 입력으로 사용할 근거가 생겼습니다.',font=font(25),fill='#765e47')
    card.save(OUT/'summary.png')
    md=f'''# 긴 외곽 엣지의 부화소 계측과 Y–yaw 예측 비교

2026-10-27. **보유 2차 데이터에서 네 엣지를 추가했을 때 Y RMSE는 {gy:.2f}%, yaw RMSE는 {gt:.2f}% 감소했다.** 새 데이터는 요청/사용하지 않았고 2차 장비 도달 좌표를 기준으로 평가했다.

## 실행한 방법

첨부 표시와 같은 원본의 위 아크릴 윤곽, 아래 금속 하단, 좌·우 외곽을 사용했다. 각각 32/32/36/36개 단면을 고정해 새 기준 3장과 혼합 57장, 총 60장에서 8,160개 단면을 계측했다. 케이블과 끝의 둥근 부분은 기준 영상에서 고정한 ROI로 피했다. 정답 좌표로 ROI를 잡지 않고 QR 영상 정합의 평균 이동으로 대략적인 탐색 위치만 옮겼다.

각 단면은 원본 15×7 픽셀을 사용했다. 좌·우·아래는 Gaussian PSF로 흐려진 계단 밝기(erf), 위는 어두운 얇은 윤곽이므로 Gaussian ridge 중심을 가상 특징으로 사용했다. 위치, σ, 밝기 대비, 배경, 두 방향 배경 기울기를 robust nonlinear least squares(soft-L1)로 적합했다. 해석적 Jacobian과 연속 위치 파라미터를 사용하며 Canny나 이진화로 좌표를 정하지 않았다. 전체 이미지를 평활화하지 않았다.

위 윤곽의 ridge 중심은 재료 외곽의 실측 정답이라는 뜻이 아니다. 같은 정의의 위치가 안정적으로 재현되는지를 사용했다. 좌·우의 반사/포화와 아래 굴곡 때문에 광학 모델에 맞지 않는 잔차도 남는다. 그런 단면은 품질 진단하고 긴 선 적합에서 robust 가중을 사용했다. 직선 잔차는 추출기의 절대 위치 오차가 아니라 물체의 곡률과 모델 불일치도 포함한다.

각 엣지의 위치+기울기 2개, 총 8개 특징을 얻었다. 기울기는 좌표 범위 scale(위/아래 450px, 좌/우 250px)만큼 움직였을 때 법선 좌표가 변하는 px값이다. 상단·하단·옆판이 같은 깊이에 있다고 가정하거나 하나의 homography로 강제하지 않았다.

## 2차 데이터 정답

장비 Actual Y/T 기록에서 유효한 신규 원점 기록의 평균 (-1.300mm, -1.669999°)을 뺀 값을 사용했다. 파일명 상대 명령값과의 최대 차이는 1e-5 미만인지 검산했다. 사용자가 설명한 장비 도달 판정을 기준으로 이 데이터를 신뢰해 학습/평가했다. 기구 도달 허용오차 수치를 임의로 새로 정하지 않았다.

## 공정한 비교

혼합 19좌표×반복3장=57장을 모두 사용했다. 좌표 하나의 3장을 통째로 제외하는 outer 19-fold, 나머지 18좌표의 inner holdout으로 정규화 강도를 선택했다. 입력 표준화와 결측 대체값도 학습분에서만 계산했다. 실제로 네 엣지 특징은 {complete}/60장에서 모두 유효했고, 평가 이미지를 삭제하거나 대체하지 않았다. 입력별 모델은 절편 포함 선형/릿지로 동일 계열이다.

QR 기준은 기존 고정 L630의 QR별 dx/dy 6채널이다. QR만 OLS의 보류 RMSE가 이전 결과 Y 약 0.147747mm, yaw 약 0.026187°와 일치하는지 검산했다. 엣지만 입력하는 비교도 ROI 초기화에는 QR 위치를 쓰므로 QR 검출 없이 독립 실행한 알고리즘으로 해석하지 않는다.

네 엣지 전부 추가가 주 비교다. 좌·우만, 위·아래만 더하는 부분집합은 추가 탐색 비교이며, 바깥 평가 결과를 보고 가장 좋은 조합을 고르는 데 따른 선택 낙관성이 있을 수 있다. 그러므로 위·아래 결과를 확정된 최종 모델 성능으로 주장하지 않는다.

## 결과

{table.to_csv()}

주 비교 QR→QR+네 엣지는 Y {b.Y_rmse_mm:.6f}→{new.Y_rmse_mm:.6f}mm, yaw {b.yaw_rmse_deg:.6f}→{new.yaw_rmse_deg:.6f}°다. 두 축 모두 한 틱 비율은 {100*b.both_within_tick:.2f}%→{100*new.both_within_tick:.2f}%다. 한 틱은 Y=0.05mm, yaw=0.01°다. 평균 오차는 줄었지만 한 틱 정밀도 달성으로 읽지 않는다.

위·아래 선 위치의 동일 명령 좌표 반복 표준편차는 각각 약 0.0116px, 0.0095px였다. 좌·우는 약 0.167px, 0.173px였다. 이는 같은 좌표의 반복 평균 제거 후 분모 38로 계산한 값이며, 센서 잡음뿐 아니라 기구 재현오차·조명·정합 변화도 포함한다. 0.01px라는 숫자를 실제 재료 경계의 절대 정확도라고 해석하지 않는다.

좌표별 평균 제곱오차 개선 수와 19개 좌표 묶음 재표집 민감도는 paired_coordinate_sensitivity.csv에 보존했다. 이 재표집 범위는 새 촬영 세션의 성능 신뢰구간이 아니다. 원본 3개의 기준 사진은 모델 학습에서 제외하고 최종 모델의 예측만 reference_predictions.csv에 기록했다. 원점은 양의 혼합 학습 범위 밖이므로 별도 진단이다.

## 시간/스윕 보류 진단

학습에서 제외한 원점 기준 3장에서는 QR+네 엣지 모델도 Y 약 −2.53~−1.40틱, yaw 약 +1.70~+3.87틱의 편차가 남았다. 현재 양의 혼합 범위에서 오차를 줄인 결과이지 원점 정밀 보정까지 완료한 모델은 아니다. 기준 영상의 별도 진단을 생략하거나 혼합 평가 점수에 섞지 않았다.

같은 Y값의 모든 θ와 반복을 통째로 보류하는 추가 검증도 수행했다. 추가 데이터 없이 더 넓은 빈 구간 및 외삽에서의 민감도를 확인하는 용도다. 해당 검증은 4개의 Y 스윕뿐이며, 촬영 순서와 Y값의 상관을 완전히 제거하거나 새 세션 성능을 입증한 것은 아니다.

{blocks.to_csv(index=False)}

## 다음 알고리즘에 쓸 판단

긴 엣지의 위치·기울기를 추가 입력으로 사용하는 방향에는 실제 검증 근거가 생겼다. 기하적으로는 QR 밖의 더 넓은 시야와 다른 깊이의 구조가 추가 정보를 줄 수 있다. 다만 이번 회귀 결과만으로 어느 물리 성분이 개선의 전부인지 분리해 증명하지 않았다. 각 특징의 반복성·강한 상관·조명 영향을 고려해 입력을 선택할 수 있다. 임의 음수 혼합 영역, 새 장비 상태, 실제 반복 이동 수렴은 이번 범위 밖이다. 보유 데이터만으로 진행했다.

## 산출물과 재현

[시각 보고서](report.html), summary.png, 01_edges_new.png, 02_profiles_new.png, 03_prediction_comparison.png, 04_coordinate_errors.png.

- edge_features.csv: 60장의 네 엣지 위치·기울기·유효 단면 수·선 잔차.
- edge_profiles.csv.gz: 8,160개 단면의 부화소 위치·σ·대비·밝기 잔차·유효 판정.
- actuator_labels.csv: 2차 도달 기록과 사용한 상대 좌표.
- comparison.csv, coordinate_holdout.csv: 동일 57장의 보류 예측 비교.
- outer_alpha_selection.csv, full_data_tuning.csv: 학습분에서만 정한 정규화 강도.
- edge_repeatability.csv, paired_coordinate_sensitivity.csv: 반복성 및 위치별 개선 민감도.
- Y_sweep_holdout.csv, reference_predictions.csv: Y 스윕 보류와 학습 제외 기준 영상 진단.
- models.json: 전체 혼합 57장 학습 모델. 출력은 틱 단위이며 입력 순서는 코드의 sets 및 edgecols를 따른다.
- numerical_checks.json, manifest.json: 해석적 미분/합성 무잡음 복원 검산과 입력·코드 해시.

Python(numpy,pandas,scipy,opencv-python-headless,matplotlib,Pillow)으로 `tool/offaxis/plate_edges.py`, `tool/offaxis/plate_edges_report.py` 실행. 기존 계측 재사용은 첫 스크립트에 --reuse. 원본 변경·커밋·push는 하지 않았다.

참고: [Fowler & Litorja (NIST, 2003)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=841640)는 픽셀 밝기 단면을 통한 부화소 경계 위치 및 광학 조건에 따른 경계 정의의 영향을 설명한다. 본 구현은 그 장비/알고리즘을 재현한 것이 아니다.
'''
    (OUT/'README.md').write_text(md,encoding='utf-8')
    def img(name,caption):
        enc=base64.b64encode((OUT/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{enc}"><figcaption>{caption}</figcaption></figure>'
    table_html=table.to_html(float_format=lambda x:f'{x:.5f}',border=0)
    block_html=blocks.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>외곽 엣지 추가 검증</title><style>body{max-width:1200px;margin:auto;padding:30px;background:#f4f7fa;color:#203f52;font-family:system-ui,'Malgun Gothic',sans-serif}h1{font-size:32px}h2{margin-top:36px}p,li{line-height:1.85}.lead{padding:20px;background:#dff1e7;border-radius:12px;font-weight:650}figure{margin:24px 0;background:white;padding:15px;border-radius:12px}img{max-width:100%}figcaption{line-height:1.7;color:#5a6c78;padding-top:10px}table{background:white;border-collapse:collapse;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.note{background:#fff0dd;padding:18px;border-radius:10px}a{color:#126a9b}</style><body>
<h1>긴 엣지를 추가하면 Y–yaw가 더 정밀해질까?</h1><p class="lead">보유 데이터만으로 실제 개선이 확인됐습니다. QR에 네 엣지를 추가한 주 비교에서 Y RMSE 약 37%, yaw RMSE 약 32%가 줄었습니다. 아직 두 축 모두 한 틱을 안정적으로 맞추는 수준은 아닙니다.</p>
@@SUMMARY@@
<h2>표시한 네 엣지를 원본에서 측정</h2><p>새 촬영은 하지 않았습니다. 2차 혼합 57장과 기준 3장, 총 60장에서 위·아래·좌·우 엣지를 측정했습니다. QR 정합값은 대략적인 탐색 위치에만 사용하고 정답 좌표는 추출에 넣지 않았습니다.</p>
@@OVERLAY@@
<h2>Canny 대신 원본 밝기 단면을 연속 함수로 적합</h2><p>각 단면의 15×7 원본 픽셀을 사용해 위치·블러·대비·배경 기울기를 robust nonlinear least squares로 추정했습니다. 아래와 양옆은 erf 계단 모델, 위 아크릴의 얇은 어두운 윤곽은 Gaussian ridge 중심을 가상 특징으로 사용했습니다. 위 ridge 중심은 실제 재료의 바깥 경계라는 뜻이 아닙니다.</p>
@@PROFILES@@
<h2>같은 보류 좌표에서 비교</h2><p>2차 장비 Actual 기록을 상대 좌표로 바꿔 정답으로 사용했습니다. 19개 좌표 중 하나의 반복 3장 전체를 보류하고, 나머지 좌표 안에서만 정규화 강도를 선택했습니다. 모든 모델이 같은 57장을 평가했고, 실패 사진을 빼서 점수를 높이지 않았습니다.</p><div class="scroll">@@TABLE@@</div>
<p class="note">네 엣지 전부 추가가 주 비교입니다. 위·아래만 추가한 결과는 추가 탐색 비교이므로, 그 조합을 보고 선택한 최종 성능으로 주장하지 않았습니다. 두 축 ±1틱 비율도 함께 확인하세요.</p>
@@COMPARISON@@
@@ERRORS@@
<h2>추가 확인: Y 스윕 전체 보류</h2><p>같은 Y값의 모든 θ와 반복을 함께 보류한 추가 진단입니다. 일반 좌표 보류보다 더 넓은 빈 구간과 외삽을 포함합니다. 네 개 스윕뿐이므로 새 촬영 세션 성능을 뜻하지 않습니다.</p><div class="scroll">@@BLOCKS@@</div>
<h2>알고리즘 개발에 대한 판단</h2><p>긴 엣지를 추가 입력으로 사용할 근거가 생겼습니다. 위·아래 선 위치는 반복 표준편차가 약 0.01px였지만 이는 실제 재료 경계의 절대 정확도가 아닙니다. 반사·그림자·물체 굴곡·기구 재현오차가 포함돼 있습니다.</p><p class="note">학습에서 제외한 원점 기준 3장에서는 QR+네 엣지도 Y −2.53~−1.40틱, yaw +1.70~+3.87틱의 편차가 남았습니다. 양의 혼합 범위의 예측은 개선됐지만 원점 정밀 보정까지 완료한 모델은 아닙니다.</p><p>이번 비교 범위는 현재 양의 혼합 19좌표입니다. 원점 복귀 횟수나 실제 장비 제어는 아직 평가하지 않았습니다.</p>
<p><a href="README.md">방법·제한·재현</a> · <a href="comparison.csv">전체 비교</a> · <a href="coordinate_holdout.csv">사진별 보류 예측</a> · <a href="edge_features.csv">엣지 계측</a> · <a href="reference_predictions.csv">기준 영상 별도 진단</a> · <a href="manifest.json">재현 정보</a></p></body></html>'''
    entries={'@@SUMMARY@@':img('summary.png','2차 장비 도달 기록을 기준으로 기존 사진만 사용한 검증입니다.'),
        '@@OVERLAY@@':img('01_edges_new.png','같은 정의로 추정한 네 엣지. 케이블과 둥근 끝은 고정 ROI에서 제외했습니다.'),
        '@@PROFILES@@':img('02_profiles_new.png','회색 점은 원본 픽셀, 실선은 연속 밝기 모델, 점선은 부화소 위치입니다. 실제 노이즈와 모델 불일치도 보존했습니다.'),
        '@@TABLE@@':table_html,'@@COMPARISON@@':img('03_prediction_comparison.png','모두 동일 57장·19좌표 보류 결과입니다.'),
        '@@ERRORS@@':img('04_coordinate_errors.png','좌표별 평균 오차도 함께 표시해 일부 위치만의 개선인지 확인했습니다.'),'@@BLOCKS@@':block_html}
    for key,value in entries.items():page=page.replace(key,value)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT/'report_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256((OUT/'report.html').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print('Improvement',gy,gt);print(blocks.to_string(index=False));print(paired.to_string(index=False));print(refs.to_string(index=False));print('Wrote',OUT/'report.html')


if __name__=='__main__':main()
