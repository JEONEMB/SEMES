from pathlib import Path
import json,zipfile
P=Path(__file__).resolve().parent
entries=json.loads((P/'sequences.json').read_text(encoding='utf-8'))
assert len(entries)==6
html='<meta charset="utf-8"><title>새 QR 모서리 이동 6구간</title><style>body{font-family:Malgun Gothic;background:#17191f;color:white;margin:24px}img{max-width:100%}a{color:#8bfa83}</style><h1>새 QR 모서리 755점 · 6구간 이동</h1><p>각 사진은 center에서 직접 추적. 선은 명령값 크기순 측정 연결이며 시간순 로봇 궤적이 아님. 위 실제변위 ×1/×1, 아래 확대변위 ×1/×100. 추적 기준 실패 0건은 축·이동량 정확도 검증이 아님. center 불확실 147점도 포함.</p><p>Y 0.05 계열은 1.95까지이며 2.00은 별도 끝점 점선. T는 ±1.00까지.</p><p><a href="tracking_failures.csv">실패 목록</a> · <a href="checks.json">추적 설정</a> · <a href="sequences.json">사진 목록</a> · <a href="../../ALGORITHM_RULES.md">공통 규칙</a></p>'
with zipfile.ZipFile(P/'발표용_이동6구간.zip','w',zipfile.ZIP_DEFLATED) as z:
    for e in entries:
        fn=e['slug']+'_발표용.png';assert (P/'slide_panels'/fn).exists()
        z.write(P/'slide_panels'/fn,fn)
        html+=f'<h2>{e["slug"]}</h2><a href="slide_panels/{fn}">발표용 PNG</a> · <a href="slide_panels/{e["slug"]}_representative36_dy100.png">대표36점 확대 그림 (center 품질순·QR별4분면 각3점·점간격20px)</a><img src="slide_panels/{fn}">'
(P/'report.html').write_text(html,encoding='utf-8')
print('Verified and packaged 6 slides',[(e['slug'],len(e['files'])) for e in entries])
