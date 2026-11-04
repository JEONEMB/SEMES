from pathlib import Path
import json,zipfile
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;SRC=P/'slide_panels_bold';OUT=P/'발표그림_라벨정리_20261015';OUT.mkdir(exist_ok=True)
entries=json.loads((P/'sequences.json').read_text(encoding='utf-8'))
font=lambda s:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',s)
rows=[]
for no,e in enumerate(entries,1):
    direction='음의방향' if e['sign']<0 else '양의방향'
    family='0.05시작계열' if e['axis']=='Y' and abs(e['start'])==.05 else ('0.10시작계열' if e['axis']=='Y' else '0.01간격')
    title=f'{no:02d}  |  {e["axis"]} {direction}  |  {e["start"]:+.2f} → {e["end"]:+.2f}'
    filename=f'2026-10-15_{no:02d}_{e["axis"]}_{direction}_{family}_대표36점.png'
    im=Image.open(SRC/f'{e["slug"]}_발표용.png').convert('RGB');w,h=im.size
    canvas=Image.new('RGB',(w,h+150),(23,25,31));canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,w,144),fill=(23,25,31))
    d.text((30,5),title,font=font(53),fill=(245,247,249))
    d.text((30,69),f'작성일 2026-10-15  ·  {family}  ·  대표 36점  ·  원본 사진 {len(e["files"])}장',font=font(31),fill=(208,220,232))
    d.text((30,110),'위: 실제 이동 방향·길이 (가로 ×1 / 세로 ×1)',font=font(28),fill=(139,250,131))
    d.text((30,h+8),'보는 순서  ① 연두 네모 = 시작   ② 보라→노랑 선 = 명령값 증가   ③ 노랑 화살촉 = 끝 위치',font=font(35),fill=(240,244,248))
    d.text((30,h+62),'비교할 것  위: 이동 방향·길이   /   아래: 미세한 세로 변화 (세로만 100배 확대)',font=font(34),fill=(218,226,236))
    d.text((30,h+109),'명령값 순으로 연결한 측정 경로이며, 실제 시간 순 궤적이 아닙니다.',font=font(27),fill=(178,189,202))
    canvas.save(OUT/filename)
    rows.append(dict(no=no,title=title,filename=filename,family=family,photos=len(e['files'])))
html='<meta charset="utf-8"><title>2026-10-15 QR 이동 그림 목록</title><style>body{font-family:Malgun Gothic;background:#17191f;color:#eee;margin:28px;line-height:1.7}a{color:#8bfa83}img{max-width:100%}td,th{border:1px solid #555;padding:10px}table{border-collapse:collapse}</style><h1>2026-10-15 · 새 QR 모서리 이동 그림</h1><p>날짜는 그림 작성일입니다. 대표 36점은 center의 품질과 공간 간격으로 고른 동일 점입니다.</p><p>먼저 위 패널에서 좌우 이동 방향과 길이를 확인하고, 아래 패널에서 미세한 세로 변화와 점별 경로 차이를 확인하세요. 아래는 세로 변위만 100배라 길이와 기울기를 실제값으로 읽으면 안 됩니다.</p><p>연두 네모: 시작 / 보라→노랑: 명령값 진행 / 노랑 화살촉: 끝. 서로 다른 그림에서 같은 점을 사용합니다.</p><table><tr><th>번호</th><th>내용 / 파일 열기</th><th>사진 수</th></tr>'
for r in rows:html+=f'<tr><td>{r["no"]:02d}</td><td><a href="{r["filename"]}">{r["title"]} · {r["family"]}</a></td><td>{r["photos"]}</td></tr>'
html+='</table><p>01·02: 0.05, 0.15, …, 1.95 계열에 2.00 끝점을 점선으로 추가. 03·04: 0.10, 0.20, …, 2.00. 05·06: 0.01, 0.02, …, 1.00. 파일명 명령값 순이며 촬영 시간 순이 아닙니다.</p>'
for r in rows:html+=f'<h2>{r["title"]}</h2><img src="{r["filename"]}">'
(OUT/'00_그림목록_보는방법.html').write_text(html,encoding='utf-8')
(OUT/'labels.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(P/'2026-10-15_이동6구간_라벨포함.zip','w',zipfile.ZIP_DEFLATED) as z:
    for r in rows:z.write(OUT/r['filename'],r['filename'])
    z.write(OUT/'00_그림목록_보는방법.html','00_그림목록_보는방법.html')
print('Labeled six figures and saved index/ZIP')
