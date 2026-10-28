"""Axis 2:1 split with most mixed poses in training; separate session intercepts."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

TICKS=np.array([.05,.01])


def fit(x,y,session,alpha):
    centers={}
    xx=x.copy(); yy=y.copy()
    for s in np.unique(session):
        mask=session==s
        xc,yc=x[mask].mean(0),y[mask].mean(0)
        centers[str(s)]={'x':xc.tolist(),'y':yc.tolist()}
        xx[mask]-=xc; yy[mask]-=yc
    scale=xx.std(0); scale=np.where(scale>1e-12,scale,1.)
    a=xx/scale
    u,v,vt=np.linalg.svd(a,full_matrices=False)
    w=v/(v*v+alpha) if alpha else np.divide(1.,v,out=np.zeros_like(v),where=v>v.max()*1e-12)
    coef=(vt.T*w)@(u.T@(yy/TICKS))
    return dict(centers=centers,scale=scale.tolist(),coefficients=coef.tolist(),alpha=alpha)


def predict(m,x,session):
    out=np.zeros((len(x),2))
    for s in np.unique(session):
        mask=session==s;c=m['centers'][str(s)]
        out[mask]=((x[mask]-c['x'])/m['scale'])@np.array(m['coefficients'])*TICKS+c['y']
    return out


def metrics(name,g):
    e=g[['error_y_mm','error_theta_deg']].to_numpy()
    rms=np.sqrt(np.mean(e*e,0))
    return dict(subset=name,images=len(g),y_rmse_mm=float(rms[0]),theta_rmse_deg=float(rms[1]),
                y_rmse_ticks=float(rms[0]/.05),theta_rmse_ticks=float(rms[1]/.01),
                both_within_one_tick=float(np.mean(np.all(abs(e)<=TICKS,axis=1))))


def main(source,output):
    output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(source/'frames.csv').set_index('path')
    qr=pd.read_csv(source/'qr_displacements.csv')
    cols=[(d,s) for s in ['L','C','R'] for d in ['dx','dy']]
    channels=[f'{s}_{d}' for d,s in cols]
    wide=qr.pivot(index='path',columns='qr',values=['dx','dy'])[cols]
    df=meta.join(wide.set_axis(channels,axis=1))
    df=df[df.category.isin(['Y-only','theta-only','mixed'])].copy()
    df['session']=np.where(df.category.eq('mixed'),'new','old')
    df['split']='train';df['inner_fold']=-1
    # Each signed axis is split independently, preserving positive/negative coverage.
    for cat,axis in [('Y-only','y_mm'),('theta-only','theta_deg')]:
        for sign in [-1,1]:
            g=df[df.category.eq(cat)&(np.sign(df[axis])==sign)].copy()
            paths=g.loc[g[axis].abs().sort_values().index].index
            for i,path in enumerate(paths):
                if i%3==2:df.loc[path,'split']='test_axis'
            remaining=[p for p in paths if df.loc[p,'split']=='train']
            for i,path in enumerate(remaining):df.loc[path,'inner_fold']=i%3
    held_coords=[(.1,.1),(.5,1.)]
    for y,t in held_coords:
        df.loc[df.category.eq('mixed')&df.y_mm.eq(y)&df.theta_deg.eq(t),'split']='test_mixed'
    assert (df.split=='test_mixed').sum()==6
    assert ((df.category=='mixed')&(df.split=='train')).sum()==51
    train=df.split.eq('train').to_numpy()
    x=df[channels].to_numpy(); y=df[['y_mm','theta_deg']].to_numpy();s=df.session.to_numpy()
    tuning=[]
    for alpha in [0.,1e-8,1e-6,1e-4,.01,1.,100.]:
        errors=[]
        for fold in range(3):
            val=train&(df.inner_fold.to_numpy()==fold)
            learn=train&~val
            model=fit(x[learn],y[learn],s[learn],alpha)
            errors.extend(((predict(model,x[val],s[val])-y[val])/TICKS).tolist())
        tuning.append(dict(alpha=alpha,inner_axis_mse_ticks=float(np.mean(np.square(errors)))))
    best=min(tuning,key=lambda z:z['inner_axis_mse_ticks'])
    model=fit(x[train],y[train],s[train],best['alpha'])
    pred=predict(model,x,s)
    df[['predicted_y_mm','predicted_theta_deg']]=pred
    df[['error_y_mm','error_theta_deg']]=pred-y
    assert np.isfinite(pred).all()
    assert not set(df[train].index)&set(df[~train].index)
    scores=[metrics(name,g) for name,g in df.groupby('split')]
    scores += [metrics(cat,df[(~train)&df.category.eq(cat)]) for cat in ['Y-only','theta-only']]
    df.to_csv(output/'predictions_and_split.csv')
    pd.DataFrame(scores).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(tuning).to_csv(output/'training_tuning.csv',index=False)
    bundle=dict(model=model,input_channels=channels,tick_sizes=TICKS.tolist(),held_mixed_coordinates=held_coords,
                source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ['frames.csv','qr_displacements.csv']},
                session_required=True,summary=scores)
    (output/'model.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    note=f'''# 축 2:1 분할 + 1사분면 대부분 학습

사용자 지정 방식으로 재학습했다. 축 데이터는 각 축·부호별 절댓값 오름차순에서
연속 세 점 중 앞 두 점 학습, 세 번째 점 평가로 고정했다. 동일 좌표 반복 3장이 아닌 서로 다른 축 좌표 3점이다.
1사분면은 두 좌표 (0.10mm, 0.10°), (0.50mm, 1.00°)의 반복 6장만 평가에 남기고 51장은 학습한다.
좌표 두 곳은 평가값을 보기 전에 지정했으며 다른 좌표로 교체하지 않았다.

전체 학습 {int(train.sum())}장, 평가 {int((~train).sum())}장.
기준 영상 4장은 학습/평가에 넣지 않는다. 원본 데이터와 기존 결과는 변경하지 않았다.

## 모델과 기준

6채널 선형/릿지 회귀. 축 데이터는 기존 center, 혼합 데이터는 center1에 대한 계측이므로
촬영 세션별 가상 입력·출력 중심(학습 평균)과 절편을 사용한다. 6채널 계수는 공유한다.
임의로 두 촬영 세션의 영점이 같다고 강제하지 않는다. 추론 시 old/new 세션 식별이 필요하다.
실제 기구 원점 추정 모델이 아닌 명령 좌표 보정 모델이다.
정규화 강도는 학습 축 점만 3분할해 선택하며, 이때 혼합 학습 51장은 항상 학습에 유지한다.
외부 평가 축 점과 혼합 두 좌표는 튜닝·중심 계산·정규화에 사용하지 않았다.
선택 alpha={best['alpha']:g}. 계수는 SVD로 계산한다.

## 평가 해석

summary.csv의 train은 학습 오차, test_axis는 축 평가, test_mixed는 1사분면 두 좌표 평가다.
각 축 평가에는 실제 0인 반대쪽 축의 추정 오차도 포함한다. Y-only/theta-only도 별도로 기록했다.
이전 결과와 학습·평가 구성이 달라 직접적인 개선 배수 비교는 하지 않는다.
1사분면 평가는 두 좌표뿐이므로 전체 혼합 공간의 정확도로 일반화할 수 없다.
실제 좌표 정답은 독립 실측이 아닌 명령값이다. 가설 검증은 하지 않았다.

## 파일

- predictions_and_split.csv: 모든 학습·평가 이미지의 고정 분할, 예측, 오차
- summary.csv: 축/혼합 평가 및 학습 성능
- training_tuning.csv: 학습 자료 내부 alpha 선택
- model.json: 입력 순서, 세션별 가상 중심, 계수, 출처 SHA256

재현: `python tool/offaxis/refit_axis_split.py --source docs/offaxis-eda/ppt-method-v2 --output docs/offaxis-eda/refit-axis-split`
커밋과 브랜치 생성은 수행하지 않았다.
'''
    (output/'README.md').write_text(note,encoding='utf-8')
    print(pd.DataFrame(scores).to_string(index=False));print('alpha',best['alpha'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
