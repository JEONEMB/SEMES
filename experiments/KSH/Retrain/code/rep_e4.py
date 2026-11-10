from style import table; from fmt import *
import json
J=json.load(open('/home/claude/rt/work/E4.json')); R=J['res']; D=json.load(open('/home/claude/rt/work/E4b.json'))
VN={'A':'A. 전체 그대로 (Q1 편향)','B':'B. 사분면 균일 — 가중 최소제곱','C':'C. 사분면 균일 — Q1 무작위 축소 (200회 평균)'}
G=['ALL','Q1','Q2','Q3','Q4','AX']
def hrow(v,h): return [VN[v]]+[f"{h[g]['spec']:.1f}/{h[g]['n']:.0f} ({100*h[g]['spec']/h[g]['n']:.1f}%)" for g in G]+[f"{h['ALL']['half']:.1f}",f"{h['ALL']['yrms']:.1f}",f"{h['ALL']['trms']:.4f}"]
def srow(v,s): return [VN[v]]+[f"{s[g]['spec']:.1f}" if isinstance(s[g]['spec'],float) and v=='C' else f"{s[g]['spec']}" for g in ['ALL','Q1','Q2','Q3','Q4','AX','S1','S2','S3','S4']]+[f"{s['ALL']['half']:.1f}" if v=='C' else f"{s['ALL']['half']}"]
H=["<h2 id='e4'>E4. 사분면 균일 학습 vs 전체(1사분면 편향) 학습</h2>",
f"<p>학습 가능 714장의 구성 : Q1 {J['cnt']['Q1']} · Q2 {J['cnt']['Q2']} · Q3 {J['cnt']['Q3']} · Q4 {J['cnt']['Q4']} · 축(1차 단독) {J['cnt']['AX']}. Q2~Q4는 모두 4차 세트에서만 나온다.</p>",
"<p><b>결론 : ① 2~4사분면 데이터가 아예 없는 것(예전 상황)과 조금이라도 있는 것의 차이는 크다 — Q2~Q4 107장 명세 통과 83% → 95%. ② 하지만 이미 사분면마다 35장 이상 있는 지금은, Q1 편향을 균일하게 맞춰도 차이가 작다 — 홀드아웃 명세 통과 1차식 96.9% → 96.7%, s3d2 97.8% → 98.6%. ③ 균일 학습은 Q2·Q3을 조금 올리고 Q1·2차 세트를 조금 내리는 맞교환이다.</b></p>",
"<h3>E4-1. 반복 층화 홀드아웃 (그룹마다 30%를 시험용으로 떼고 나머지로 학습, 200회 평균) — 명세 통과</h3>"]
for tag in ['s1d1','s3d2']:
    H.append(f"<p><b>{tag}</b></p>"+table(['학습 방식','전체','Q1','Q2','Q3','Q4','축','절반 기준 전체','Y RMS','Θ RMS'],[hrow(v,R[tag]['holdout'][v]) for v in ['A','B','C']]))
H.append("<h3>E4-2. 셀프 테스트 (714장 전체 평가) — 명세 통과 장수</h3>")
for tag in ['s1d1','s3d2']:
    H.append(f"<p><b>{tag}</b></p>"+table(['학습 방식','전체 714','Q1 327','Q2 36','Q3 35','Q4 36','축 280','1차 280','2차 57','3차 234','4차 143','절반 기준'],[srow(v,R[tag]['self_'+v]) for v in ['A','B','C']]))
H.append("<h3>E4-3. 극단 비교 : 2~4사분면 데이터 없이 학습한 경우 (Q1 + 축만으로 학습 → Q2~Q4 107장 평가)</h3>")
rows=[]
for tag in ['s1d1','s3d2']:
    d=D[tag]['D']; h=R[tag]['holdout']['A']
    rows.append([tag+' · Q2~Q4 없이 학습',*(f"{d[g]['spec']}/{d[g]['n']} ({100*d[g]['spec']/d[g]['n']:.0f}%)" for g in ['ALL','Q2','Q3','Q4']),f"{d['ALL']['yrms']:.1f}",f"{d['ALL']['trms']:.4f}"])
    q=['Q2','Q3','Q4']; sp=sum(h[g]['spec'] for g in q); n=sum(h[g]['n'] for g in q)
    rows.append([tag+' · 전체 학습 (홀드아웃, 참고)',f"{sp:.1f}/{n:.0f} ({100*sp/n:.0f}%)",*(f"{h[g]['spec']:.1f}/{h[g]['n']:.0f} ({100*h[g]['spec']/h[g]['n']:.0f}%)" for g in q),'-','-'])
H.append(table(['경우','Q2~Q4 전체','Q2','Q3','Q4','Y RMS','Θ RMS'],rows))
H.append("<p class='small'>주의 : Q2~Q4는 모두 4차(같은 날 기준 사진 없음)에서만 나와, '사분면 효과'와 '4차 날짜 치우침'이 섞여 있다. Q2~Q4 데이터를 넣었을 때 좋아지는 몫의 상당 부분은 상수항이 4차 치우침을 일부 흡수한 효과일 수 있다. 같은 날 센터와 함께 2~4사분면을 다시 촬영하면 분리해서 확인할 수 있다.</p>")
c=R['s1d1']['coef']
H.append("<h3>E4-4. 계수 비교 (s1d1)</h3>"+table(['학습 방식','s c','s a','s b₁','d c','d a','d b₁'],[[VN[k] if k in VN else k]+[f"{m['s']['c']:+.4f}",f"{m['s']['a']:.4f}",f"{m['s']['b'][0]:.4f}",f"{m['d']['c']:+.4f}",f"{m['d']['a']:.4f}",f"{m['d']['b'][0]:+.4f}"] for k,m in [('A',c['A']),('B',c['B']),('C',c['C_mean'])]]+[['C의 200회 표준편차']+[f"{c['C_sd']['s']['c']:.4f}",f"{c['C_sd']['s']['a']:.4f}",f"{c['C_sd']['s']['b'][0]:.4f}",f"{c['C_sd']['d']['c']:.4f}",f"{c['C_sd']['d']['a']:.4f}",f"{c['C_sd']['d']['b'][0]:.4f}"]]))
H.append("<p class='small'>B(가중)의 Q 가중치 = 가장 적은 사분면 장수 ÷ 그 사분면 장수 (Q1 = 35/327 ≈ 0.107), 축 데이터는 가중 1로 유지. C는 Q1 327장 중 35장을 무작위로 뽑아 사분면마다 35장으로 맞춤.</p>")
open('/home/claude/rt/report/E4.html','w').write(''.join(H))
