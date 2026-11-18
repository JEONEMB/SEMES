"""Validate the compiled C ABI and visualize held-coordinate image predictions."""
import argparse
import base64
import ctypes as ct
import json
import hashlib
import os
import sys
import time
from pathlib import Path


class Pose(ct.Structure):_fields_=[('y_mm',ct.c_double),('yaw_deg',ct.c_double)]
class Result(ct.Structure):_fields_=[('features',ct.c_double*10),('pose',Pose),('qr_points',ct.c_int),('upper_valid',ct.c_int),('lower_valid',ct.c_int),('upper_line_rms',ct.c_double),('lower_line_rms',ct.c_double)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis-root',type=Path,required=True);p.add_argument('--reuse-images',action='store_true');a=p.parse_args()
    module=Path(__file__).resolve().parents[1];root=a.analysis_root.resolve();sys.path.insert(0,str(root/'.analysis_packages'))
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    build=module/'build/Release';out=module/'results';out.mkdir(exist_ok=True)
    handles=[]
    if os.name=='nt':handles.append(os.add_dll_directory(str(build)))
    core=ct.CDLL(str(build/'qr_common_core.dll'));lib=ct.CDLL(str(build/'qr_common_image.dll'))
    core.qc_predict_features.argtypes=[ct.POINTER(ct.c_double),ct.c_int,ct.POINTER(Pose)]
    lib.qc_predict_image.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.POINTER(Result)]
    golden=pd.read_csv(module/'tests/golden_features.csv');cols=list(golden.columns[2:12]);rows=[]
    max_frozen=np.zeros(2)
    for r in golden.itertuples(index=False):
        x=(ct.c_double*10)(*[getattr(r,c) for c in cols]);pose=Pose()
        assert core.qc_predict_features(x,r.fold,ct.byref(pose))==0
        max_frozen=np.maximum(max_frozen,abs(np.array([pose.y_mm,pose.yaw_deg])-[r.python_Y_mm,r.python_yaw_deg]))
        rows.append(dict(path=r.path,fold=r.fold,Y_mm=r.Y_mm,yaw_deg=r.yaw_deg,C_Y_mm=pose.y_mm,C_yaw_deg=pose.yaw_deg))
    pd.DataFrame(rows).to_csv(out/'frozen_feature_predictions.csv',index=False)
    assert max_frozen.max()<1e-9,(max_frozen.tolist(),'C regression differs from frozen Python predictions')
    pose=Pose();assert core.qc_predict_features(None,-1,ct.byref(pose))==1
    invalid=(ct.c_double*10)(*([float('nan')]*10));assert core.qc_predict_features(invalid,-1,ct.byref(pose))==5
    assert core.qc_predict_features((ct.c_double*10)(*([0]*10)),19,ct.byref(pose))==1
    bad=Result();assert lib.qc_predict_image(None,None,-1,ct.byref(bad))==1
    assert lib.qc_predict_image(b'nonexistent-test-input.bmp',str(module/'calibration').encode(),-1,ct.byref(bad))==2
    metadata=pd.read_csv(root/'docs/offaxis-eda/ppt-method-v2/frames.csv').set_index('path')
    hashes={}
    for path in golden.path:
        digest=hashlib.sha256((root/'data/SSAFY Test Image Set'/path).read_bytes()).hexdigest()
        assert digest==metadata.loc[path,'sha256'],('Image changed',path)
        hashes[path]=digest
    (out/'source_image_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    if not a.reuse_images:
        started=time.time();images=[]
        for i,r in enumerate(golden.itertuples(index=False)):
            result=Result();status=lib.qc_predict_image(str(root/'data/SSAFY Test Image Set'/r.path).encode(),str(module/'calibration').encode(),r.fold,ct.byref(result))
            entry=dict(path=r.path,fold=r.fold,Y_mm=r.Y_mm,yaw_deg=r.yaw_deg,status=status,
                C_Y_mm=result.pose.y_mm,C_yaw_deg=result.pose.yaw_deg,python_Y_mm=r.python_Y_mm,python_yaw_deg=r.python_yaw_deg,
                upper_valid=result.upper_valid,lower_valid=result.lower_valid,qr_points=result.qr_points,
                upper_line_rms=result.upper_line_rms,lower_line_rms=result.lower_line_rms)
            entry.update({c:result.features[j] for j,c in enumerate(cols)})
            images.append(entry)
            if (i+1)%5==0:print(f'Raw image C validation {i+1}/57, {time.time()-started:.1f}s',flush=True)
        pd.DataFrame(images).to_csv(out/'image_predictions.csv',index=False)
    data=pd.read_csv(out/'image_predictions.csv');assert len(data)==57 and data.status.eq(0).all(),data[['path','status']].to_string()
    assert data.qr_points.eq(630).all()
    def metrics(g):
        e=g[['C_Y_mm','C_yaw_deg']].to_numpy()-g[['Y_mm','yaw_deg']].to_numpy()
        return dict(Y_rmse_mm=float(np.sqrt(np.mean(e[:,0]**2))),yaw_rmse_deg=float(np.sqrt(np.mean(e[:,1]**2))),both_within_tick=float(np.mean((abs(e[:,0])<=.05)&(abs(e[:,1])<=.01))))
    image_metrics=metrics(data);frozen_metrics=metrics(pd.DataFrame(rows))
    differences=data[['C_Y_mm','C_yaw_deg']].to_numpy()-data[['python_Y_mm','python_yaw_deg']].to_numpy()
    feature_difference=data[cols].to_numpy()-golden[cols].to_numpy()
    summary=dict(images=57,coordinate_folds=19,C_core_language='C17',image_adapter='C++17 OpenCV 3.4.15; decoding and coarse affine only',
        frozen_feature_C_metrics=frozen_metrics,raw_image_C_metrics=image_metrics,
        frozen_feature_max_C_Python_delta=max_frozen.tolist(),raw_image_C_Python_rmse_delta=np.sqrt(np.mean(differences**2,axis=0)).tolist(),
        raw_image_C_Python_max_abs_delta=np.max(abs(differences),axis=0).tolist(),
        feature_max_abs_delta=dict(zip(cols,np.max(abs(feature_difference),axis=0).tolist())),
        Canny_used=False,original_images_modified=False,invalid_inputs_rejected=True,
        caution='Held coordinate predictions use fold-specific models. Deployment API uses all57 fit. No closed-loop actuator validation.')
    (out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # One point for the input image's true label, one for the C prediction; join pairs.
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    for row in data.itertuples():
        axes[0].plot([row.Y_mm,row.C_Y_mm],[row.yaw_deg,row.C_yaw_deg],c='#687b8c',alpha=.55,lw=.9,zorder=1)
    axes[0].scatter(data.Y_mm,data.yaw_deg,marker='o',s=48,facecolors='none',edgecolors='#1377b3',linewidths=1.5,label='Input image label (Actual)',zorder=3)
    axes[0].scatter(data.C_Y_mm,data.C_yaw_deg,marker='x',s=28,c='#de702d',linewidths=1.3,label='C image prediction (held coordinate)',zorder=4)
    axes[0].set(xlabel='Y (mm)',ylabel='yaw / theta (deg)',title='Y–THETA plane: true label → C prediction')
    axes[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False);axes[0].grid(alpha=.2)
    ey=(data.C_Y_mm-data.Y_mm)/.05;et=(data.C_yaw_deg-data.yaw_deg)/.01
    axes[1].add_patch(plt.Rectangle((-1,-1),2,2,color='#28a275',alpha=.15,label='Both axes within one tick'))
    axes[1].scatter(ey,et,c=data.fold,cmap='tab20',s=28);axes[1].axhline(0,color='#bbb',lw=1);axes[1].axvline(0,color='#bbb',lw=1)
    axes[1].set(xlabel='Y error (ticks; 1 tick = 0.05 mm)',ylabel='yaw error (ticks; 1 tick = 0.01°)',title='Residuals centered on each true label');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle(f'57 raw images · 19 held-coordinate folds · Y RMSE {image_metrics["Y_rmse_mm"]:.5f} mm · yaw RMSE {image_metrics["yaw_rmse_deg"]:.5f}° · both within tick {100*image_metrics["both_within_tick"]:.1f}%',fontsize=12)
    fig.savefig(out/'y_theta_pairs.png',dpi=180);fig.savefig(out/'y_theta_pairs.svg');plt.close(fig)
    svg=out/'y_theta_pairs.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    picture=base64.b64encode((out/'y_theta_pairs.png').read_bytes()).decode()
    table=data[['path','fold','Y_mm','yaw_deg','C_Y_mm','C_yaw_deg','upper_valid','lower_valid']].to_html(index=False,float_format=lambda v:f'{v:.6f}')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C 구현 · Y–THETA 좌표 대응</title>
    <style>body{{font:16px/1.7 system-ui;color:#263b48;margin:auto;padding:28px;max-width:1400px}}img{{width:100%}}table{{font-size:12px;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd}}pre{{background:#f1f4f7;padding:16px;overflow:auto}}.scroll{{overflow:auto}}</style>
    <h1>QR + 공통 경계 4개 특징의 C 구현</h1>
    <p><strong>파란 원은 입력 이미지의 정답 좌표, 주황 ×는 C 실행 파일이 그 이미지에서 추출·예측한 좌표다.</strong> 선이 두 점을 연결한다. 같은 좌표의 반복 3장을 함께 보류한 19개 평가 모델을 사용했다. 학습 이미지 자체를 예측한 성능과 구분한다.</p>
    <img src="data:image/png;base64,{picture}" alt="Y theta plane paired labels and C predictions">
    <p>C 원본 이미지 경로: Y RMSE <strong>{image_metrics['Y_rmse_mm']:.6f} mm</strong>, yaw RMSE <strong>{image_metrics['yaw_rmse_deg']:.6f}°</strong>, 동시 ±1틱 <strong>{100*image_metrics['both_within_tick']:.1f}%</strong>.</p>
    <p>저장된 동일 10개 특징을 C 회귀에 넣으면 기존 Python 0.075169mm / 0.014019° / 52.6%를 재현한다. 아래 검증 기록은 이 회귀 일치 검사와, 원본 BMP부터 다시 계산한 C 경로를 구분한다. 수치 최적화기·OpenCV 버전 차이로 이미지 경로는 완전한 비트 일치를 목표로 하지 않는다.</p>
    <p>수치 계산은 C17이다. OpenCV가 제공하는 이미지 디코딩·템플릿 정합·초기 affine ECC만 C++17 어댑터를 사용한다. 실행 시 Python이나 이미지별 정답·특징점 캐시를 읽지 않는다. 캘리브레이션에는 기준 이미지의 QR 패턴/점·템플릿과 원점 경계, 학습 계수가 들어 있다. 경계는 원본 밝기 함수의 수치 적합이며 Canny를 사용하지 않는다.</p>
    <p>원점까지의 반복 제어와 장비 간 일반화는 검증하지 않았다. 평가 fold는 정답 좌표를 알고 있는 평가기만 선택한다. 새 이미지의 일반 추론은 fold=-1의 전체 학습 모델을 사용한다.</p>
    <p><a href="image_predictions.csv">이미지별 C 예측</a> · <a href="frozen_feature_predictions.csv">동일 특징 C 회귀 예측</a> · <a href="y_theta_pairs.svg">벡터 그래프</a> · <a href="validation.json">검증 JSON</a></p>
    <h2>검증 기록</h2><pre>{json.dumps(summary,ensure_ascii=False,indent=2)}</pre><h2>57장별 결과</h2><div class="scroll">{table}</div></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
