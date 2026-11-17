"""Frozen improved joint models on the two user-selected archives.
Reuses byte-verified original boundary measurements; does not refit any model.
"""
from pathlib import Path
import sys,json,hashlib,html
import numpy as np,pandas as pd,joblib
H=Path(__file__).resolve().parent;B=H.parent;ROOT=B.parent
MODEL_DIR=B/'joint_yt_improved_20261022'
sys.path.insert(0,str(MODEL_DIR));sys.path.insert(0,str(ROOT/'motion_web_20261022'))
from engine import extract
from fullfield import field_features
from batch_jobs import truth_label

CACHE=ROOT/'motion_web_20261022/runs/72a03bae6d1c47858525d6ae278d8cef'

def summarize(g):
    good=g[g.status=='ok'];passed=good[(good.eY<=.09+1e-12)&(good.eT<=.09+1e-12)]
    return dict(total=len(g),inference_success=len(good),inference_failed=len(g)-len(good),both_within09=len(passed),pass_fraction_all=len(passed)/len(g),Y_MAE=float(good.eY.mean()) if len(good) else None,Y_max_successful=float(good.eY.max()) if len(good) else None,T_MAE=float(good.eT.mean()) if len(good) else None,T_max_successful=float(good.eT.max()) if len(good) else None)

def main():
    job=json.loads((CACHE/'job.json').read_text(encoding='utf-8'));by_index={r['index']:r for r in job['rows']}
    source_audit=json.loads((H/'source_audit.json').read_text(encoding='utf-8'))
    manifest=pd.read_csv(H/'source_manifest.csv').set_index('index')
    assert source_audit['all_zip_members_sha256_match_extracted'] and len(manifest)==341
    assert source_audit['dataset_counts']['20261020']['training_overlap']==0
    assert hashlib.sha256((CACHE/'job.json').read_bytes()).hexdigest()==source_audit['source_job_sha256']
    assert len(job['sources'])==341
    bundles={k:joblib.load(MODEL_DIR/fn) for k,fn in [('improved_nonlinear','fixedfeature_model.joblib'),('coordinate_linear','coordinate_model.joblib')]}
    model_hashes={fn:hashlib.sha256((MODEL_DIR/fn).read_bytes()).hexdigest() for fn in ['fixedfeature_model.joblib','coordinate_model.joblib']}
    records=[];measurements=[]
    for index,source in enumerate(job['sources']):
        archive='20261020' if 'zip_20261020.zip/' in source['relative'] else '20261014'
        name=source['name'];label=truth_label(name);assert label is not None,name
        verified=manifest.loc[index]
        assert verified['name']==name and str(verified['dataset'])==archive
        overlap='training_image' if verified['original_training_overlap'] else ('reference_center' if verified['reference_center_overlap'] else 'new_archive')
        prior=by_index[index];path=CACHE/f'result_{index}.json';raw=None
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==verified['cache_sha256']
            raw=json.loads(path.read_text(encoding='utf-8'));trace=raw['trace'];assert trace['measurement']=='blurred_grid_boundary'
            points=np.array(trace['points']);D=np.array(trace['displacements']);q=np.array(trace['quality'])[:,:3];ids=np.array(trace['ids'])
            assert D.shape==(630,2) and np.isfinite(D).all()
            measurements.append(dict(index=index,name=name,archive=archive,valid=raw['valid'],total=raw['total'],cache=str(path),measurement_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for model_name,bundle in bundles.items():
            base=dict(index=index,archive=archive,file=name,scope=label['scope'],true_Y=label['Y'],true_T=label['T'],model=model_name,training_overlap=overlap,reference='original_fixed_center',evaluation_basis='filename_commands')
            if raw is None:
                records.append(dict(**base,status='measurement_failed',error=prior.get('error','Missing boundary measurement'),pred_Y=np.nan,pred_T=np.nan,correction_Y=np.nan,correction_T=np.nan,eY=np.nan,eT=np.nan,both_within09=False,valid=np.nan));continue
            assert np.array_equal(ids,bundle['ids']) and np.allclose(points,bundle['points'])
            c=bundle['config']
            feature=extract(points,D,q,c['measurement'],c['feature']) if 'measurement' in c else field_features(points,D,q,c['feature'])
            y=bundle['model'].predict(feature[None])[0]
            if trace['identical_reference']:y=np.zeros(2)
            e=abs(y-[label['Y'],label['T']])
            records.append(dict(**base,status='ok',error='',pred_Y=float(y[0]),pred_T=float(y[1]),correction_Y=float(-y[0]),correction_T=float(-y[1]),ideal_residual_Y=float(label['Y']-y[0]),ideal_residual_T=float(label['T']-y[1]),eY=float(e[0]),eT=float(e[1]),both_within09=bool(np.all(e<=.09+1e-12)),valid=raw['valid']))
        if index%60==0:print('evaluated',index+1,341,flush=True)
    p=pd.DataFrame(records);p.to_csv(H/'predictions.csv',index=False,encoding='utf-8-sig')
    mainp=p[p.model=='improved_nonlinear'];mainp.to_csv(H/'robot_corrections.csv',index=False,encoding='utf-8-sig')
    p[(p.status!='ok')|(~p.both_within09)].to_csv(H/'failures.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for (archive,model),g in p.groupby(['archive','model']):
        summaries.append(dict(archive=archive,model=model,subset='all',**summarize(g)))
        for scope,gg in g.groupby('scope'):summaries.append(dict(archive=archive,model=model,subset=scope,**summarize(gg)))
    summary=pd.DataFrame(summaries);summary.to_csv(H/'summary.csv',index=False,encoding='utf-8-sig')
    condition=p.copy();condition['condition']=condition.true_Y.astype(str)+','+condition.true_T.astype(str)
    conditions=pd.DataFrame([dict(archive=a,model=m,truth_Y=float(g.true_Y.iloc[0]),truth_T=float(g.true_T.iloc[0]),**summarize(g)) for (a,m,c),g in condition.groupby(['archive','model','condition'])]);conditions.to_csv(H/'conditions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(measurements).to_csv(H/'measurement_cache.csv',index=False,encoding='utf-8-sig')
    # Frozen model application is resubstitution for the original training images.
    oof=pd.read_csv(MODEL_DIR/'fixedfeature_oof.csv');oof['eY']=abs(oof.pred_Y-oof.true_Y);oof['eT']=abs(oof.pred_T-oof.true_T)
    oof.to_csv(H/'original280_oof_reference.csv',index=False,encoding='utf-8-sig')
    checks=dict(model_hashes=model_hashes,model_config={k:b['config'] for k,b in bundles.items()},cached_measurements=len(measurements),strict_measurement_failures=341-len(measurements),input_images=341,fit_called=False,reference_override='only image pixels identical to original center yield exact(0,0)',source_cache=str(CACHE),limits=['20261014 contains calibration images; frozen-model inference is resubstitution','20261020 is not used to refit these frozen models','T+2 commands exceed original T training range[-1,1] and12 images fail original+-130px NCC search','Errors use filename commands; correction outputs are mathematical negatives, not executed robot commands','Successful-case maximum errors exclude12 unmeasurable images; missing estimates have no finite numerical bound'])
    checks['source_audit_sha256']=hashlib.sha256((H/'source_audit.json').read_bytes()).hexdigest()
    for fn,digest in model_hashes.items():assert hashlib.sha256((MODEL_DIR/fn).read_bytes()).hexdigest()==digest
    assert len(p)==682 and len(mainp)==341
    (H/'audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    write_report(p,summary,conditions,oof)
    print(summary.to_string(index=False),flush=True)

def write_report(p,summary,conditions,oof):
    cols=['archive','model','total','inference_success','inference_failed','Y_MAE','Y_max_successful','T_MAE','T_max_successful','both_within09']
    labels={'archive':'자료','model':'모델','total':'전체','inference_success':'추정 성공','inference_failed':'추정 실패','Y_MAE':'Y 평균오차','Y_max_successful':'Y 최대오차(성공분)','T_MAE':'T 평균오차','T_max_successful':'T 최대오차(성공분)','both_within09':'두 축≤0.09 통과'}
    models={'improved_nonlinear':'개선 비선형(주모델)','coordinate_linear':'좌표 선형(참고)'}
    view=summary[summary.subset=='all'][cols].copy();view['order']=view.model.map({'improved_nonlinear':0,'coordinate_linear':1});view=view.sort_values(['order','archive']).drop(columns='order');view.model=view.model.map(models);view=view.rename(columns=labels)
    per=p[p.model=='improved_nonlinear'].copy();per['자료']=per.archive;per['구분']=per.scope.map({'single':'단일축','mixed':'혼합','center':'center'});per['통과']=per.both_within09.map({True:'통과',False:'실패'})
    show=per[['자료','file','구분','true_Y','true_T','pred_Y','pred_T','correction_Y','correction_T','eY','eT','status','통과']].rename(columns={'file':'파일','true_Y':'명령 Y','true_T':'명령 T','pred_Y':'추정 Y','pred_T':'추정 T','correction_Y':'반대 보정 Y','correction_T':'반대 보정 T','eY':'Y 오차','eT':'T 오차','status':'처리'})
    center_table=show[(per.archive=='20261020')&(per.scope=='center')].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>두 데이터 묶음 Y·T 추정</title><style>body{font:16px system-ui,"Malgun Gothic";max-width:1450px;margin:36px auto;padding:0 24px;line-height:1.7;color:#183247}h1,h2{color:#145577}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border:1px solid #d7e0e8;text-align:right}td:nth-child(2){text-align:left}th{background:#edf4f8;position:sticky;top:0}.scroll{overflow:auto;max-height:700px}.note{padding:18px;background:#fff2d8;border-left:5px solid #c89439}a{color:#14719b}button,input,select{font:inherit;padding:6px;margin:8px}code{background:#eff4f8}</style>
    <h1>20261014 · 20261020 사진 → Y·T 추정과 반대 보정값</h1>
    <p><a href="../../ALGORITHM_RULES.md">알고리즘 규칙</a> · 저장된 개선 비선형 모델을 그대로 적용. 학습·보정계수 갱신 없음.</p>
    <div class="note"><b>모든 사진이 두 축0.09 이내인 것은 아니다.</b> 평균오차가0.09보다 작아도 개별 사진의 최대오차가 넘을 수 있다. 아래 통과 수는 Y·T 절대오차가 모두0.09 이하인 사진 수다. 분류 정확도가 아니다.</div>
    <h2>결과</h2>'''+view.to_html(index=False,float_format=lambda v:f'{v:.6f}')+'''
    <p><b>20261014:281장</b>은 기존 학습280장+고정center1장이다. 저장된 최종 모델로 다시 추정한 재대입 성적이다. 이 자료에서 각 사진을 학습에서 제외했던 OOF는251/280통과, Y최대0.3100·T최대0.0685였다. 재대입 성적을 새 사진 성능으로 해석하지 않는다.</p>
    <p><b>20261020:60장</b>은 실제 Y+T혼합57장+center3장이다. 이번 고정 모델에는 학습시키지 않았다. 원래±130px 초기 탐색에서 T+2.0사진12장은 QR위치를 찾지 못했다. 실패12장도 전체60장 분모에 포함했다. 수치 오차의 평균·최대는 추정된48장 기준이며, 실패 사진은 오차가0이라는 뜻이 아니다. T+2.0은 원래 학습범위±1을 벗어난다.</p>
    <p>입력은 기존 고정center와 현재 사진이다. 새 자료의center사진을 새 기준으로 바꾸지 않았다. 오차 정답은 파일명 명령값이다. 반대 보정값은(-추정Y,-추정T)이며 실제 로봇에 실행하지 않았다. '이상적 잔여 오차' 열은 명령이 정확히 적용된다는 가정의 정답−추정값으로, 보정 후 실측 자세가 아니다.</p>
    <h2>로봇 보정 관점에서의 결과</h2>
    <p><b>새 60장 중 두 축 허용오차 동시 통과는 14장(23.3%)이다.</b> 34장은 추정값이 나왔지만 허용오차를 넘었고, 12장은 측정에 실패했다. 혼합 자세만 보면 57장 중 14장 통과, center 3장은 모두 실패다.</p>
    <p><code>Y+1.00 T+1.00 (1).bmp</code>의 명령값은 (1.00, 1.00)이지만 추정값은 (0.00000, 0.976875)다. 이 결과로 반대 보정 (0.00000, −0.976875)을 적용한다고 가정하면 Y가 1.00만큼 남는다. 현재 모델로 센터 복귀가 된다고 볼 수 없다.</p>
    <p>새 center 사진 3장도 Y를 약 0.095~0.111로 추정한다. 기준 영상과 새 촬영 사이의 차이를 모델이 이동량으로 해석하는 문제가 관찰된다. 이 자료만으로 실제 기계 원점 변화와 영상·모델 편향을 분리할 수는 없다.</p>'''+center_table+'''
    <p>평가 방법: 원본 ZIP과 추출 사진 341장의 바이트 일치를 검증한 뒤, 동일 사진에 대한 기존 고정 630점 경계 측정 329건을 재사용해 두 저장 모델의 추론을 실행했다. 실패 12건도 그대로 보존했다. 20261020 사진·정답으로 모델을 다시 학습하거나 원점을 바꾸지 않았다.</p>
    <h2>주모델 사진별 결과</h2><p>예: 추정(Y,T)를 보고 반대 보정값을 낸다. 표시된 정답은 추론 뒤 채점에만 사용했다.</p>
    <label>자료<select id="dataset"><option value="">전체</option><option>20261014</option><option>20261020</option></select></label><label>파일<input id="filter" placeholder="파일명 검색"></label><label><input id="failed" type="checkbox">실패만 보기</label><div class="scroll" id="rows">'''+show.to_html(index=False,float_format=lambda v:f'{v:.5f}',na_rep='추정 실패')+'''
    </div><h2>저장 파일</h2><p><a href="robot_corrections.csv">주모델 사진별 추정·반대 보정 CSV</a> · <a href="predictions.csv">두 모델 전체 결과</a> · <a href="failures.csv">실패 목록</a> · <a href="conditions.csv">명령 조건별 결과</a> · <a href="source_audit.json">원본·학습중복 검증</a> · <a href="audit.json">모델 보존·측정 출처</a> · <a href="original280_oof_reference.csv">기존280장 OOF참고</a></p>
    <script>const d=document.querySelector('#dataset'),f=document.querySelector('#filter'),e=document.querySelector('#failed');function update(){for(const r of document.querySelectorAll('#rows tbody tr')){const t=[...r.cells].map(c=>c.textContent.trim());r.hidden=(d.value&&t[0]!==d.value)||(f.value&&!t[1].toLowerCase().includes(f.value.toLowerCase()))||(e.checked&&t[t.length-1]!=='실패')}}for(const x of[d,f,e])x.addEventListener('input',update);</script></html>'''
    (H/'report.html').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
