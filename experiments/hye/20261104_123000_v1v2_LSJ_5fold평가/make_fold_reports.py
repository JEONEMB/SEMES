"""Expand existing measurements into model/fold/date/condition tables."""
import runpy
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'results'
d=runpy.run_path(str(ROOT/'make_reports.py'))
by,metrics,table,timing,num,read=(d[k] for k in ['by','metrics','table','timing','num','read'])
HEAD=['fold/조건','전체','유효','통과 수/전체','MAE','p95','최대오차']
DETAIL=ROOT/'fold별_상세';DETAIL.mkdir(exist_ok=True)
conditions=[('T=0',lambda t:t==0),('T≠0 전체',lambda t:t!=0),('0<abs(T)≤0.01',lambda t:0<abs(t)<=.01+1e-10),('0.01<abs(T)≤0.03',lambda t:.01+1e-10<abs(t)<=.03+1e-10),('0.03<abs(T)≤0.10',lambda t:.03+1e-10<abs(t)<=.1+1e-10),('0.10<abs(T)≤0.30',lambda t:.1+1e-10<abs(t)<=.3+1e-10),('0.30<abs(T)≤1',lambda t:.3+1e-10<abs(t)<=1+1e-10),('abs(T)>1',lambda t:abs(t)>1+1e-10)]
master=['# 모델별·fold별 전체 결과','',
 '실측한 기존 결과를 모두 펼친 표다. 재학습·추론을 새로 수행한 것은 아니다. T 허용범위 ±0.01, Y ±0.045. 실패도 전체 분모에 포함하고 MAE/p95/최대오차는 유효 출력 기준이다. 처리시간은 ms다.',
 '', 'control은 T=0 전용 재학습 변형이다. 원본 배포 모델과 LSJ 고정 모델의 fold별 표는 같은 사진 묶음별 집계이며 해당 모델의 OOF 재학습이라고 부르지 않는다. 실제 복귀 step·시간·정확도는 장비 세션 부재로 미측정이다.','']
allrows=[]
def block(title,rs,axis,limit,predicate=lambda r:True):
    rows=[]
    for f in [1,2,3,4,5,None]:
        sel=[r for r in rs if (f is None or int(r['fold'])==f) and predicate(r)]
        label=f'fold {f:02}' if f else '전체'
        m=metrics(sel,axis,limit);rows.append([label]+m)
        allrows.append([title,label,axis]+m)
    return ['## '+title,'',table(HEAD,rows),'']

for v in [1,2]:
    jn=f'joint_v{v}_retrained'+('_GLS' if v==1 else '')
    cn='control_v1_T0_four_Y_branches' if v==1 else 'control_v2_T0_no_T_correction'
    jr,cr=by[jn],by[cn]
    master+=block(f'v{v} joint 재학습 · 전체 T',jr,'T',.01)
    master+=block(f'v{v} control T=0 전용 변형 · T=0 Y',cr,'Y',.045,lambda r:num(r['T'])==0)
    master+=block(f'v{v} control T=0 전용 변형 · T≠0 Y',cr,'Y',.045,lambda r:num(r['T'])!=0)
    master+=['### fold별 처리시간','',table(['fold','joint 중앙값','joint p95','control 중앙값','control p95','읽기/디코딩 중앙값(별도)'],[[f'fold {f:02}']+timing([r for r in jr if int(r['fold'])==f],'joint_observe_ms')[:2]+timing([r for r in cr if int(r['fold'])==f],'control_track_ms')[:2]+[timing([r for r in jr if int(r['fold'])==f],'joint_observe_ms')[2]] for f in range(1,6)]),'']
    for f in range(1,6):
        j=[r for r in jr if int(r['fold'])==f];c=[r for r in cr if int(r['fold'])==f]
        s=[f'# final v{v} · fold {f:02} 상세 결과','',
           f'학습 모델: [fold_{f}.yml](../results/fold_{f}.yml). 이 파일에 해당 fold에서 새로 적합한 v1/v2 계수와 학습 수량이 있다. control 학습은 T=0 64장이다. 다른 fold 모델로 이 test를 예측하지 않았다.','',
           '## joint T 정확도 · 날짜별','',table(HEAD,[[date]+metrics([r for r in j if date=='전체' or r['date']==date],'T',.01) for date in ['20261014','20261020','20261029','전체']]),'',
           '## control Y 정확도 · T 조건별','',table(HEAD,[[label]+metrics([r for r in c if fn(num(r['T']))],'Y',.045) for label,fn in conditions]),'',
           '## control Y 정확도 · 날짜×T 조건','',table(HEAD,[[date+' / '+label]+metrics([r for r in c if r['date']==date and fn(num(r['T']))],'Y',.045) for date in ['20261014','20261020','20261029'] for label,fn in conditions[:2]]),'',
           '## 영상 처리시간 (ms)','',table(['대상','중앙값','p95','읽기/디코딩 중앙값(별도)','유효 측정 수'],[['joint']+timing(j,'joint_observe_ms'),['control T=0 변형 (test 전체 적용)']+timing(c,'control_track_ms')]),'',
           '원본 공개 API 12장 별도 시간과 위 공통 영상 측정 파이프라인 시간을 구분한다. 초기화·학습·장비 이동은 위 추론 시간에서 제외했다.','',
           '## 통과하지 못한 사진','']
        failures=[]
        for stage,rs,axis,limit in [('joint',j,'T',.01),('control T=0', [r for r in c if num(r['T'])==0],'Y',.045)]:
            for r in rs:
                p=num(r['pred_'+axis]);err=abs(p-num(r[axis]));
                if not d['math'].isfinite(p) or err>limit+1e-12:
                    failures.append([stage,r['date'],r['group'],r['index'],r[axis],r['pred_'+axis],d['fmt'](err),r.get('error','')])
        s+=[table(['단계','날짜','그룹','사진 index','정답','예측','절대오차','실패 사유'],failures) if failures else 'joint 및 control T=0 기본 평가의 허용오차 초과/측정 실패 없음.','',
            'index는 [input.tsv](../input.tsv)의 0부터 시작하는 데이터 행 번호다. 원본 BMP 경로·파일명은 해당 행에서 확인한다. control T≠0 전체 사진별 출력은 [predictions.tsv](../results/predictions.tsv)에 있으며 날짜·자세별 오차를 누락 없이 보존했다.','',
            '## 실제 센터 복귀','', 'step·총 시간·최종 실제 정확도: 미측정. 정지 사진 평가로 실제 이동 후 결과를 대신하지 않는다.','',
            '[전체 모델·fold 비교로 돌아가기](../모델별_fold별_전체결과.md)']
        name=f'final_v{v}_fold_{f:02}.md';(DETAIL/name).write_text('\n'.join(s),encoding='utf-8')
    master+=['### 개별 fold의 날짜·조건·실패 사진 상세','']+[f'- [v{v} fold {f:02}](fold별_상세/final_v{v}_fold_{f:02}.md)' for f in range(1,6)]+['']
    p=ROOT/f'보고서_final_v{v}.md';s=p.read_text(encoding='utf-8');p.write_text(s+'\n\n## 모델별·fold별 전체 상세\n\n[전체 모델·fold 표](모델별_fold별_전체결과.md)\n\n'+'\n'.join(f'- [fold {f:02}: 날짜별·T 조건별·실패 사진·시간](fold별_상세/final_v{v}_fold_{f:02}.md)' for f in range(1,6))+'\n',encoding='utf-8')

master+=['# 원본 배포 모델: fold 사진 묶음별 개발 재생','', '아래는 기존 단일 배포 모델을 고정한 출력이다. fold마다 다시 학습한 점수가 아니며 재학습 OOF 표와 섞지 않는다.','']
for v in [1,2]:
    master+=block(f'원본 v{v} joint 고정 모델 · T',by[f'joint_v{v}_frozen_development'],'T',.01)
    for label,fn in conditions[:2]:
        master+=block(f'원본 v{v} control 고정 모델 · {label} Y',by[f'control_v{v}_frozen_development'],'Y',.045,lambda r,fn=fn:fn(num(r['T'])))

rs=read('lsj_new.tsv')
for r in rs:r['pred_Y']=r['pred_Y_nuisance'];r['group']=d['base'][r['index']]['group']
master+=['# LSJ 고정 모델: fold 사진 묶음별 결과','', '동일 고정 계수로 측정했다. 내부 Y는 보조 추정이며 Y 이동 명령이 아니다.','']
master+=block('LSJ 최신 고정 · T',rs,'T',.01)+block('LSJ 최신 고정 · 내부 Y',rs,'Y',.045)
for f in range(1,6):
    sub=[r for r in rs if int(r['fold'])==f]
    rows=[[date+' / '+axis]+metrics([r for r in sub if date=='전체' or r['date']==date],axis,lim) for axis,lim in [('T',.01),('Y',.045)] for date in ['20261014','20261020','20261029','전체']]
    times=[num(r['ms']) for r in sub if d['finite'](r['ms'])]
    s=[f'# LSJ 고정 모델 · fold {f:02} 사진 묶음','', '외부 fold 재학습이 아닌 기존 모델의 해당 사진 묶음 개발 재생이다. 내부 Y는 제어용 Y 명령과 구분한다.','',table(HEAD,rows),'',
       table(['추론 유효 수','중앙값 ms','p95 ms'],[[len(times),d['fmt'](d['statistics'].median(times)),d['fmt'](d['quantile'](times,.95))]]),'',
       '실제 여러 step 도착·총 시간·최종 실제 위치 정확도는 미측정이다.','', '[전체 결과](../모델별_fold별_전체결과.md)']
    (DETAIL/f'LSJ_fold_{f:02}.md').write_text('\n'.join(s),encoding='utf-8')
master+=['### LSJ 개별 fold 상세','']+[f'- [LSJ fold {f:02} 날짜별 Y/T·시간](fold별_상세/LSJ_fold_{f:02}.md)' for f in range(1,6)]+['']
rootrs=read('lsj_root.tsv')
master+=['## LSJ 루트 순차 코드: 다음 명령 가능 여부','',table(['fold','전체 사진','명령 가능','명령 불가/측정 실패'],[[f'fold {f:02}',len(sub:= [r for r in rootrs if int(r['fold'])==f]),sum(r['sequence_available']=='1' for r in sub),sum(r['sequence_available']!='1' for r in sub)] for f in range(1,6)]),'','명령 가능은 실제 센터 복귀 성공이 아니다. 이동을 실행하지 않았다.']
(ROOT/'모델별_fold별_전체결과.md').write_text('\n'.join(master),encoding='utf-8')
with (RESULT/'model_fold_metrics.tsv').open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['model_condition','fold','axis','total','valid','pass','MAE','p95','max']);w.writerows(allrows)
p=ROOT/'보고서_LSJ.md';p.write_text(p.read_text(encoding='utf-8')+'\n\n## fold 사진 묶음별 상세\n\n[전체 모델·fold 결과](모델별_fold별_전체결과.md)\n\n'+'\n'.join(f'- [LSJ fold {f:02}](fold별_상세/LSJ_fold_{f:02}.md)' for f in range(1,6))+'\n',encoding='utf-8')
print('\n'.join(master[:80]))
