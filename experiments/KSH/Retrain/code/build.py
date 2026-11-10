import os,glob
from style import page
order=['E0','E1','E2','E3','E4','E5','E6','SUM']
names={'E0':'E0 백업·재현','E1':'E1 전체 재학습(1~3차)','E2':'E2 기존 수식 대비','E3':'E3 기준 센터','E4':'E4 사분면 균일','E5':'E5 반복 학습','E6':'E6 기존 결과 대조','SUM':'요약'}
frags={k:open(f'/home/claude/rt/report/{k}.html').read() for k in order if os.path.exists(f'/home/claude/rt/report/{k}.html')}
done=[k for k in order if k in frags]
toc="<div class='card toc'><b>목차</b> · "+' · '.join(f"<a href='#{k.lower()}'>{names[k]}</a>" for k in done if k!='SUM')+f"<br><span class='small'>진행 상황 : {', '.join(names[k] for k in done)} 완료 · 요구사항 정의서 : <code>00_요구사항정의서.html</code></span></div>"
body=(frags.get('SUM','')+toc+''.join(frags[k] for k in order if k in frags and k!='SUM'))
open('/mnt/user-data/outputs/retrain_20261110/1차_결과보고서.html','w',encoding='utf-8').write(page('수식 재학습 1차 결과 보고서',body,'작성 2026-11-10 · 데이터 1~4차 719장 · 실험이 끝날 때마다 갱신'))
print('built',done)
