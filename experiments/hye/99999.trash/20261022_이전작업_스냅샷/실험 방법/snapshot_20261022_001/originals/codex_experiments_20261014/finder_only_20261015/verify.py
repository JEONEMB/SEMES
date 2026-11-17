from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(B/'deps'),str(B)]
import numpy as np,pandas as pd
import zero_experiment as Z
z=np.load(P/'tracks.npz');pts=z['points'];x=(pts[:,0]-1906)/486;y=(pts[:,1]-1070)/122
A=np.array([x*0+1,x,y,x*x,x*y,y*y]).T
X=np.array([np.linalg.lstsq(A[v],d[v],rcond=None)[0].reshape(12) for d,v in zip(z['D'],z['valid'])])
saved=np.load(P/'finder_quadratic_model.npz')
branches={a:(a,saved[a+'_base'],saved[a+'_slope'],saved[a+'_quadratic']) for a in ['Y','T']}
names=z['names'].astype(str);ids=np.flatnonzero(names!='center.bmp')
# Deliberately meaningless metadata: inference may only depend on X and frozen model.
m=pd.DataFrame(dict(file=['hidden']*len(names),axis=['?']*len(names),value=np.full(len(names),999.),session=['unknown']*len(names)))
r=Z.predict(m,X,ids,(branches,saved['W']),'center')
p=pd.read_csv(P/'predictions.csv');full=p[(p.method=='finder_quadratic')&(p.protocol=='full280_resubstitution')]
assert np.array_equal(r.pred_axis.to_numpy(),full.pred_axis.to_numpy())
assert np.allclose(r.pred,full.pred,atol=1e-12,rtol=0)
assert len(full)==280 and len(np.unique(names))==281
assert np.all(X[names=='center.bmp']==0)
summary=[]
for directory,method in [(B/'center_fixed_20261015','center_quadratic'),(P,'finder_quadratic')]:
    pp=pd.read_csv(directory/'predictions.csv');rep=pp[(pp.method==method)&pp.protocol.str.startswith('repeated5_seed')]
    assert len(rep)==5600
    summary.append(dict(method=method,protocol='repeated5_20_pooled',**Z.score(rep)))
df=pd.DataFrame(summary);df.to_csv(P/'repeated_cv_summary.csv',index=False)
(P/'verification.json').write_text(json.dumps(dict(frozen_predictions_reproduced=True,metadata_independence=True,all_280_retained=True,center_zero=True,repeated_cv_predictions_per_method=5600),indent=2))
print(df[['method','Y_axis_sign_accuracy','T_axis_sign_accuracy','strict_pass09','vector_Y_max','vector_T_max']].to_string(index=False))
html=(P/'report.html').read_text(encoding='utf-8')
html+='<h2>반복 5분할 20회 통합</h2><p>각 방법 280장 × 20회 = 5600개 시험 예측이며, 독립 사진 5600장이 아님.</p>'+df.to_html(index=False)
html+='<h2>판정</h2><p>알고리즘 적용은 가능하나 현 기준의 정확도 목표 미달. 사각형 전용 2차 모델 재대입 262/280 통과, Y 축·부호 65/80, T 축·부호 197/200. 기존 전체 내부점은 274/280. 사각형 제한 시 현재 자료에서 성능 저하. 새 랜덤 QR 일반화는 미검증. 고정 모델 예측 재현, 시험 메타데이터 독립성, 전체 280장 포함과 center 영점 확인.</p>'
(P/'report.html').write_text(html,encoding='utf-8')
