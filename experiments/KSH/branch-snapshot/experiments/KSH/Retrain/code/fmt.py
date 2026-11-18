def pc(a,n): return f"{a}/{n} ({100*a/n:.1f}%)" if n else '-'
def mrow(m): return [m['n'], pc(m['spec'],m['n']), pc(m['goal'],m['n']), pc(m['half'],m['n']), f"{m['ym']:+.1f}", f"{m['yrms']:.1f}", f"{m['ymax']:.1f}", f"{m['tm']:+.4f}", f"{m['trms']:.4f}", f"{m['tmax']:.4f}"]
MH=['장수','명세 ±90µm·±0.01°','목표 ±45µm·±0.01°','절반 ±45µm·±0.005°','Y 평균(µm)','Y RMS','Y 최대','Θ 평균(°)','Θ RMS','Θ 최대']
def coef(M): 
    s,d=M['s'],M['d']
    return [f"{s['c']:+.4f}",f"{s['a']:.4f}",', '.join(f'{b:+.4f}' for b in s['b']),f"{s['sigma']:.3f}",f"{d['c']:+.4f}",f"{d['a']:.4f}",', '.join(f'{b:+.4f}' for b in d['b']),f"{d['sigma']:.4f}"]
CH=['s 상수 c','s Y계수 a (px/mm)','s Θ계수 b₁, b₂, b₃','s 잔차 σ','d 상수 c','d Y계수 a','d Θ계수 b₁, b₂, b₃','d 잔차 σ']
