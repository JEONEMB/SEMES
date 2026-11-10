from rt import *; from style import table; from fmt import *
import json, copy
r=json.load(open('/home/claude/rt/work/E1.json'))
old=load_yml_model('/home/claude/cmp/ksh/config/model_s1d1.yml'); o32=load_yml_model('/home/claude/cmp/ksh/config/model_s3d2.yml')
ALL=samples()
M={'기존 s1d1 (1차 240장)':old,'기존 s3d2 (1차 240장)':o32,'새 s1d1 (전체 714장)':r['combos']['s1d1']['model'],'새 s3d2 (전체 714장)':r['combos']['s3d2']['model'],'새 s1d2 (전체 714장)':r['combos']['s1d2']['model']}
res={}
for k,m in M.items():
    P=predict(m,copy.deepcopy(ALL)); res[k]={g:metrics([x for x in P if (x['set']==g or x['q']==g or g=='ALL')]) for g in ['ALL','S1','S2','S3','S4','Q1','Q2','Q3','Q4','AX']}
    # Θ ≤ 1° 만 (2차 2° 제외)
    res[k]['IN']=metrics([x for x in P if abs(x['T'])<=1.0])
json.dump(res,open('/home/claude/rt/work/E2.json','w'),default=float,indent=1)
for k in M: print(k,' '.join('%s %d/%d'%(g,res[k][g]['spec'],res[k][g]['n']) for g in ['ALL','S1','S2','S3','S4','IN']), 'half',res[k]['ALL']['half'])
n,o=r['combos']['s1d1']['model'],old
def dd(a,b): return f"{a-b:+.4f} ({100*(a-b)/abs(b):+.2f}%)"
rows=[['s 상수 c (px)',f"{o['s']['c']:+.4f}",f"{n['s']['c']:+.4f}",dd(n['s']['c'],o['s']['c']),'기준 사진에서 읽히는 영점'],
['s Y계수 a (px/mm)',f"{o['s']['a']:.4f}",f"{n['s']['a']:.4f}",dd(n['s']['a'],o['s']['a']),'Y 1 mm당 QR 이동'],
['s Θ계수 b (px/°)',f"{o['s']['b'][0]:.4f}",f"{n['s']['b'][0]:.4f}",dd(n['s']['b'][0],o['s']['b'][0]),'Θ 1°당 QR 이동'],
['d 상수 c (px)',f"{o['d']['c']:+.4f}",f"{n['d']['c']:+.4f}",dd(n['d']['c'],o['d']['c']),''],
['d Y계수 a (px/mm)',f"{o['d']['a']:.4f}",f"{n['d']['a']:.4f}",dd(n['d']['a'],o['d']['a']),'시차 감도 (Y 분리의 핵심)'],
['d Θ계수 b (px/°)',f"{o['d']['b'][0]:+.4f}",f"{n['d']['b'][0]:+.4f}",dd(n['d']['b'][0],o['d']['b'][0]),''],
['s 잔차 σ (px)',f"{o['s']['sigma']:.4f}",f"{n['s']['sigma']:.4f}",dd(n['s']['sigma'],o['s']['sigma']),'학습 표본 다양해져 증가'],
['d 잔차 σ (px)',f"{o['d']['sigma']:.4f}",f"{n['d']['sigma']:.4f}",dd(n['d']['sigma'],o['d']['sigma']),'']]
# 같은 (s,d)를 두 식으로 역산했을 때의 차이 (대표 자세)
eff=[]
for Y,T in [(0,0),(0.5,0),(0,0.5),(1,1),(-1,1),(1.6,-0.96)]:
    s=ev(o['s'],Y,T)[0]; d=ev(o['d'],Y,T)[0]; y2,t2=invert(n,s,d); eff.append([f"Y {Y:+.2f}, Θ {T:+.2f}",f"{(y2-Y)*1000:+.1f}",f"{t2-T:+.4f}"])
keys=list(M)
H=["<h2 id='e2'>E2. 기존 수식 대비 전체 데이터 학습 수식의 차이</h2>",
f"<p><b>결론 : 기울기 계수(a, b)는 0.1~3.4%만 바뀌었고 상수항은 s −0.070 px · d −0.028 px(상대로 12~25%) 바뀌었다. 기존 수식이 약했던 2차·4차에서 명세 통과가 늘었다.</b> 전체 714장 명세 통과 : 기존 s1d1 {res[keys[0]]['ALL']['spec']}장 → 새 s1d1 {res[keys[2]]['ALL']['spec']}장 → 새 s3d2 {res[keys[3]]['ALL']['spec']}장. 단, 새 수식의 수치는 학습에 쓴 데이터로 평가한 것이라 개선 폭이 과대평가될 수 있음 (객관 비교는 E1 LOSO 참고).</p>",
"<h3>E2-1. 계수 비교 (1차식 s1d1)</h3>",table(['계수','기존 (1차 240장)','새 (전체 714장)','차이','의미'],rows),
"<h3>E2-2. 같은 측정값을 두 수식으로 읽으면 결과가 얼마나 달라지나</h3><p class='small'>기존 수식이 정확하다고 가정한 (s, d)를 새 수식으로 역산했을 때의 차이 = 수식 교체가 출력에 주는 영향</p>",
table(['자세','Y 차이 (µm)','Θ 차이 (°)'],eff),
"<h3>E2-3. 세트별 명세 통과 (모두 같은 날 기준 특징, 4차는 center 기준)</h3>",
table(['수식','전체 714','1차 280','2차 57','3차 234','4차 143','|Θ|≤1° 702','절반 기준 전체'],[[k]+[pc(res[k][g]['spec'],res[k][g]['n']) for g in ['ALL','S1','S2','S3','S4','IN']]+[pc(res[k]['ALL']['half'],res[k]['ALL']['n'])] for k in keys],cls=['','','hl','',''],left=1),
"<h3>E2-4. 전체 714장 정확도 상세</h3>",table(['수식']+MH,[[k]+mrow(res[k]['ALL']) for k in keys]),
"<p class='small'>2차 Θ 2.00° 12장은 기존 두 수식 모두 0/12 통과 (Θ RMS : s1d1 0.0197°, s3d2 0.0288° — 학습 범위 ±1° 밖 외삽). 새 수식은 2°까지 학습에 포함해 2차 통과가 늘었음. 절반 기준(±45 µm·±0.005°)으로는 새 s1d1(571)이 기존 s1d1(578)보다 오히려 적음 — 1차 단독 축 데이터에 맞춰져 있던 계수가 4차의 날짜 치우침 쪽으로 끌려간 영향.</p>"]
open('/home/claude/rt/report/E2.html','w').write(''.join(H))
