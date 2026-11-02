"""Check report artifacts only. No image measurements or statistical tests."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import subprocess
import re
from urllib.parse import urlsplit, unquote

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding="utf-8"))

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.ids=[]; self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if 'href' in a:self.links.append(a['href'])
        if tag=='img': self.images.append(a['src'])

report_html=(OUT/'report.html').read_text(encoding='utf-8')
p=Links();p.feed(report_html)
assert len(p.ids)==len(set(p.ids))
for url in p.links+p.images:
    u=urlsplit(url)
    if u.scheme: continue
    if u.path: assert (OUT/unquote(u.path)).is_file(),url
    elif u.fragment: assert u.fragment in p.ids,url
prov=read(OUT/'provenance.json')
for item in prov['copied_evidence']:
    assert digest(ROOT/item['source'])==item['sha256'],item
    assert digest(OUT/item['copy'])==item['sha256'],item
source=read(ROOT/prov['source_results'])['records']
rows=read(OUT/'evidence/measurements_56.json')['records']
assert rows==[r for r in source if r['method_id'] in prov['methods']]
assert len(rows)==56
for name,folder in [('devernay.c','devernay_1.0'),('devernay.h','devernay_1.0'),('lsd.c','lsd_1.6'),('lsd.h','lsd_1.6')]:
    assert digest(OUT/'evidence/code'/folder/name)==prov['author_code_audit'][name]
for m in prov['methods']:
    for ref in ['R01','R02','R03','R04']:
        assert (OUT/'figures'/f'{m}_{ref}.png').is_file()

# A separate, headless browser profile avoids touching the user's existing browser.
edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
qa=OUT/'qa';qa.mkdir(exist_ok=True)
render=[]
if edge.is_file():
    profile=ROOT/'.tmp/six_methods_report_browser_20261030'
    for anchor in ['', 'E06', 'Q04']:
        screenshot=qa/(anchor or 'overview')
        screenshot=screenshot.with_suffix('.png')
        page=OUT/'report.html'
        if anchor:
            # Exact report markup/CSS in a shorter preview avoids headless offscreen paint timing.
            head=report_html.split('<body>')[0]
            head=head.replace('</head>', '<base href="'+OUT.as_uri()+'/"></head>')
            nav=re.search(r'<nav\b.*?</nav>',report_html,re.S).group()
            content=re.search(r'<section id="'+anchor+r'">.*?</section>',report_html,re.S).group()
            script=re.search(r'<script>.*?</script>',report_html,re.S).group()
            page=qa/f'preview_{anchor}.html'
            page.write_text(head+'<body><div class="layout">'+nav+'<main>'+content+'</main></div>'+script+'</body></html>',encoding='utf-8')
        args=[str(edge),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
              '--disable-background-networking','--disable-extensions','--hide-scrollbars',
              '--run-all-compositor-stages-before-draw',
              '--window-size=1600,1200','--virtual-time-budget=1500',
              '--user-data-dir='+str(profile),'--screenshot='+str(screenshot),
              page.as_uri()]
        result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        assert screenshot.is_file() and screenshot.stat().st_size>10000,(anchor,result.stderr[-1000:])
        render.append({'section':anchor or 'overview','screenshot':str(screenshot.relative_to(OUT)),
                       'isolated_exact_markup_preview':bool(anchor)})

result={'scope':'Report QA only; no detector runs, refits or statistical tests',
        'local_links_and_images_exist':True,'copied_evidence_hashes_match':len(prov['copied_evidence']),
        'original_measurement_records_unchanged':len(rows),'method_figures_present':24,'browser_renders':render}
(qa/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
