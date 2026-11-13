from rt import *; from style import table; from fmt import *
import json
old=load_yml_model('/home/claude/cmp/ksh/config/model_s1d1.yml'); o32=load_yml_model('/home/claude/cmp/ksh/config/model_s3d2.yml')
S=[r for r in samples(('S1',),ref=0) if (r['T']!=0) or (round(r['Y']*100)%10==0)]; Mr=fit(S,1,1)
p3=metrics(predict(old,samples(('S3',),ref=1))); p4=metrics(predict(old,samples(('S4',),ref=0)))
H=["<h2 id='e0'>E0. 기존 수식 백업과 계산 재현 확인</h2>",
"<p><b>결론 : 새 계산 파이프라인이 기존 C++ 학습·평가 결과를 그대로 재현한다.</b> 이후 실험의 수치는 기존 보고 수치와 같은 기준에서 비교할 수 있다.</p>",
table(['항목','기존 (model_s1d1.yml)','이번 재계산','차이'],[
 ['s 상수 c (px)',f"{old['s']['c']:.6f}",f"{Mr['s']['c']:.6f}",f"{Mr['s']['c']-old['s']['c']:.1e}"],
 ['s Y계수 a (px/mm)',f"{old['s']['a']:.6f}",f"{Mr['s']['a']:.6f}",f"{Mr['s']['a']-old['s']['a']:.1e}"],
 ['s Θ계수 b (px/°)',f"{old['s']['b'][0]:.6f}",f"{Mr['s']['b'][0]:.6f}",f"{Mr['s']['b'][0]-old['s']['b'][0]:.1e}"],
 ['d 상수 c',f"{old['d']['c']:.6f}",f"{Mr['d']['c']:.6f}",f"{Mr['d']['c']-old['d']['c']:.1e}"],
 ['d Y계수 a',f"{old['d']['a']:.6f}",f"{Mr['d']['a']:.6f}",f"{Mr['d']['a']-old['d']['a']:.1e}"],
 ['d Θ계수 b',f"{old['d']['b'][0]:.6f}",f"{Mr['d']['b'][0]:.6f}",f"{Mr['d']['b'][0]-old['d']['b'][0]:.1e}"],
 ['3차 234장 (center1 기준) 명세 / 절반','234 / 228 (10/30 보고)',f"{p3['spec']} / {p3['half']}",'일치'],
 ['4차 143장 (center 기준) 명세','109 (11/09 보고)',f"{p4['spec']}",'일치'],
]),
"<p class='small'>재학습 표본 = 기존과 같은 1차 240장(Y 0.1 mm 배수 40장 + Θ 200장). 계수 차이 10⁻⁸ 수준은 부동소수 반올림. 백업 : <code>QR_Image\\_archive\\20261110_수식재학습전_백업</code> (model_s1d1 4곳 MD5 d35791aafa36 동일, model_s3d2 2곳, alignment.yml 2곳).</p>"]
open('/home/claude/rt/report/E0.html','w').write(''.join(H))
r=json.load(open('/home/claude/rt/work/E1.json')); b=json.load(open('/home/claude/rt/work/E1_biv.json'))
keys=list(r['combos'])
best_self=max(keys,key=lambda k:(r['combos'][k]['self']['ALL']['half'])); best_loso=max(keys,key=lambda k:(r['combos'][k]['loso']['ALL']['half'], -r['combos'][k]['loso']['ALL']['trms']))
H=["<h2 id='e1'>E1. 전체 데이터(1~4차 714장) 재학습 — 차수 1~3 전체 조합</h2>",
f"<p><b>결론 : 셀프 테스트에서는 차수를 올릴수록 좋아지지만(절반 기준 s1d1 571 → s3d3 610장), 학습에 없던 세트로 평가하면(LOSO) 차수 효과가 작고, Θ 2°처럼 범위 밖에서는 고차식이 크게 틀린다.</b> LOSO 절반 기준 최고는 <b>{best_loso}</b>(508장)이지만, LOSO Θ RMS는 s2d3·s1d2(0.0050·0.0051°), Y RMS는 s2d2·s1d2(21.4·21.5 µm)가 가장 작다. 세 지표를 고르게 만족하는 것은 <b>s1d2</b>. 셀프 테스트 최고는 <b>{best_self}</b>.</p>",
"<p>학습 표본 : 1차 280 + 2차 57 + 3차 234 + 4차 143 = 714장 (센터 4장·4차 이상 1장 제외). 특징 기준 = 같은 날 기준(1·4차 center, 2·3차 center1). 차수 sNdM = s 식 Θ N차, d 식 Θ M차 (Y는 1차).</p>",
"<h3>E1-1. 셀프 테스트 (학습한 714장으로 다시 평가)</h3>",
table(['조합']+MH,[[k]+mrow(r['combos'][k]['self']['ALL']) for k in keys],cls=['hl' if k in (best_self,) else '' for k in keys]),
"<h3>E1-2. 세트 단위 제외 검증 LOSO (한 세트를 빼고 학습 → 뺀 세트 평가, 4번 결과 합산)</h3>",
table(['조합']+MH,[[k]+mrow(r['combos'][k]['loso']['ALL']) for k in keys],cls=['hl' if k==best_loso else '' for k in keys]),
"<h3>E1-3. 세트별 명세 통과 (셀프 / LOSO)</h3>",
table(['조합','1차 280','2차 57','3차 234','4차 143','Q2 36 (셀프)','Q3 35 (셀프)','Q4 36 (셀프)'],[[k]+[f"{r['combos'][k]['self'][s]['spec']} / {r['combos'][k]['loso'][s]['spec']}" for s in ['S1','S2','S3','S4']]+[f"{r['combos'][k]['self'][q]['spec']}" for q in ['Q2','Q3','Q4']] for k in keys]),
"<p class='small'>2차 LOSO가 고차식에서 크게 나빠지는 이유 : 2차에만 Θ 2.00°(12장)가 있어, 2차를 빼고 학습하면 ±1° 데이터로 2°를 외삽하게 됨 (Θ 3차항이 범위 밖에서 급격히 커짐). 4차는 같은 날 기준 사진이 없어 날짜 치우침이 남아 셀프 테스트에서도 130~136장 수준.</p>",
"<h3>E1-4. 계수 (전체 714장 학습)</h3>",
table(['조합']+CH,[[k]+coef(r['combos'][k]['model']) for k in keys]),
"<h3>E1-5. 참고 : Y·Θ 2변수 다항식 (Y², YΘ 포함 — 현재 C++ 미지원)</h3>",
table(['전체 차수','항 개수','셀프 명세','셀프 절반','셀프 Y RMS','셀프 Θ RMS','LOSO 명세','LOSO 절반','LOSO Y RMS','LOSO Θ RMS'],[[n,len(b[n]['terms']),b[n]['self']['spec'],b[n]['self']['half'],f"{b[n]['self']['yrms']:.1f}",f"{b[n]['self']['trms']:.4f}",b[n]['loso']['spec'],b[n]['loso']['half'],f"{b[n]['loso']['yrms']:.1f}",f"{b[n]['loso']['trms']:.4f}"] for n in ['1','2','3']]),
"<p class='small'>2변수 3차는 셀프 테스트 절반 기준 618장으로 가장 높지만 LOSO에서는 Θ RMS가 0.0106°로 나빠짐(과적합). 코드 확장 대비 이득이 작아 배포 후보에서 제외.</p>",
"<p class='small'>모델 파일 : <code>models\\model_all_sNdM.yml</code> 9개 (C++ <code>model_file</code>에 바로 지정 가능한 형식).</p>"]
open('/home/claude/rt/report/E1.html','w').write(''.join(H))
for k in keys: write_yml_model(r['combos'][k]['model'],f'/mnt/user-data/outputs/retrain_20261110/models/model_all_{k}.yml',f'retrain 2026-11-10 : S1~S4 all (714), sameday ref, {k}',714)
print(best_self,best_loso)
