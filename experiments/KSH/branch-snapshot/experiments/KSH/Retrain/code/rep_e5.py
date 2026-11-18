from style import table
import json
R=json.load(open('/home/claude/rt/work/E5.json'))
H=["<h2 id='e5'>E5. 반복 학습 횟수 (1·3·10회 → 문헌 근거 횟수)</h2>",
"<p><b>결론 : ① 지금 학습은 최소제곱 해를 한 번에 구하는 방식이라 같은 데이터로 1·3·10번 학습하면 계수가 완전히 같다(차이 0). ② 표본을 다시 뽑아 반복(부트스트랩 배깅)하면 반복 횟수가 늘수록 계수의 흔들림이 1/√횟수로 줄어 200회에서 1회의 약 1/15이 되지만, 평균 계수는 전체 데이터로 한 번 맞춘 값으로 수렴할 뿐 정확도는 좋아지지 않는다. ③ 반복이 실제로 쓸모 있는 곳은 '정확도 추정'의 재현성이다 — 10겹 교차검증을 10번 반복(문헌 권장)하면 1번일 때보다 추정 편차가 1/3로 줄어든다.</b></p>",
"<h3>E5-1. 반복 횟수의 근거 문헌 (웹 검색)</h3>",
table(['문헌','내용','이번 실험에 반영'],[
 ["<a href='https://cameron.econ.ucdavis.edu/mmabook/transparencies/ct11_bootstrap.pdf'>Efron & Tibshirani (1993), <i>An Introduction to the Bootstrap</i>, p.52</a> (Cameron & Trivedi 강의 자료 재인용)","표준오차 추정에는 부트스트랩 B = 50이면 대개 충분, B = 200이면 거의 항상 충분. 신뢰구간·검정은 꼬리를 쓰므로 더 많이(예 999회)","부트스트랩 반복 50 · 200회 추가"],
 ["<a href='https://researchcommons.waikato.ac.nz/handle/10289/1451'>Bouckaert & Frank (2004), PAKDD</a>","재현성(같은 실험을 다시 해도 같은 결론) 기준으로 <b>10겹 교차검증 10회 반복(10×10)</b> + 보정 t-검정 권장","10겹×10 교차검증"],
 ["<a href='https://ijcai.org/proceedings/1995/147'>Kohavi (1995), IJCAI</a>","모델 선택에는 <b>10겹 층화 교차검증</b>이 가장 좋음 (더 많은 겹이 가능해도)","층화 10겹 사용"],
 ["<a href='https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf'>Dietterich (1998), Neural Computation</a>","알고리즘을 10번 돌릴 수 있으면 <b>5×2 교차검증</b>(2겹 5회) 검정 권장","5×2겹 비교"],
 ["<a href='https://icml.cc/Conferences/2004/proceedings/papers/61.pdf'>Bouckaert (2004), ICML</a>","교차검증 반복 횟수를 10 → 100으로 늘릴수록 재현성이 일관되게 높아짐","반복 횟수별 추정 편차 측정"],
]),
"<p class='small'>재설정한 횟수 : 계수 안정성 → 부트스트랩 200회 (Efron & Tibshirani), 정확도 추정 → 층화 10겹 × 10회 (Bouckaert & Frank, Kohavi). 사용자 지정 1·3·10회와 나란히 비교.</p>"]
for tag in ['s1d1','s3d2']:
    r=R[tag]; nb=len(r['full'])
    labels=['s c','s a','s b₁','s b₂','s b₃','d c','d a','d b₁','d b₂'] 
    os_,od=int(tag[1]),int(tag[3]); names=['s c','s a']+[f's b{k}' for k in range(1,os_+1)]+['d c','d a']+[f'd b{k}' for k in range(1,od+1)]
    H.append(f"<h3>E5-2 ({tag}). 같은 데이터로 반복 학습 — 결정론 확인</h3>")
    H.append(table(['반복','계수 최대 차이 (1회 대비)'],[['1회','0 (기준)'],['3회',f"{r['det']['max_abs_diff_3']:.1e}"],['10회',f"{r['det']['max_abs_diff_10']:.1e}"]]))
    H.append(f"<h3>E5-3 ({tag}). 부트스트랩 반복(배깅) — 시드 30개로 반복해 본 계수 표준편차와 정확도</h3>")
    rows=[]
    for B in ['1','3','10','50','200']:
        b=r['boot'][B]; rows.append([f"{B}회"+(' (Efron & Tibshirani)' if B=='200' else '')]+[f"{b['sd'][i]:.4f}" for i in range(nb)]+[f"{b['spec_mean']:.1f} ± {b['spec_sd']:.2f}",f"{b['half_mean']:.1f} ± {b['half_sd']:.2f}"])
    rows.append(['전체 데이터 1회 최소제곱 (참고)']+[f"{v:+.4f}" for v in r['full']]+['-','-'])
    H.append(table(['반복 횟수']+[n+' 표준편차' for n in names]+['셀프 명세 (714장)','셀프 절반'],rows,cls=['','','','','hl','']))
    H.append(f"<p class='small'>200회 배깅 계수와 전체 1회 최소제곱 계수의 최대 차이 : {max(r['boot']['200']['dev_from_full']):.4f} — 표본을 다시 뽑아 평균해도 결국 같은 답으로 수렴.</p>")
    H.append(f"<h3>E5-4 ({tag}). 반복 층화 교차검증 — 정확도 추정값과 그 재현성 (시드 30개)</h3>")
    rows=[[k,v['fits'],f"{100*v['spec_mean']:.2f}% ± {100*v['spec_sd']:.2f}",f"{100*v['half_mean']:.2f}% ± {100*v['half_sd']:.2f}",f"{v['yrms_mean']:.2f} ± {v['yrms_sd']:.3f}",f"{v['trms_mean']:.5f} ± {v['trms_sd']:.6f}"] for k,v in r['cv'].items()]
    H.append(table(['방식','학습 횟수','명세 통과율 (평균 ± 시드간 편차)','절반 기준 통과율','Y RMS (µm)','Θ RMS (°)'],rows,cls=['','','hl','']))
H.append("<p class='small'>교차검증은 사진 단위로 무작위 분할하므로 같은 자세 3장이 학습·시험에 나뉠 수 있어 세트 단위 LOSO(E1)보다 낙관적이다. 추정값 자체보다 '반복 횟수에 따른 편차 감소'를 보는 용도.</p>")
open('/home/claude/rt/report/E5.html','w').write(''.join(H))
