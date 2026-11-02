"""Check new report coverage and links only; do not run the measurement pipeline."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
from collections import Counter
import json

HERE=Path(__file__).resolve().parent

class Doc(HTMLParser):
    def __init__(self,text):
        super().__init__(); self.ids=[]; self.links=[]; self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])

def main():
    text=(HERE/'report.html').read_text(encoding='utf-8')
    d=Doc(text); errors=[]; cache={}; count=0
    for href in d.links:
        u=urlsplit(href)
        if u.scheme in ('http','https'):continue
        count+=1
        p=(HERE/unquote(u.path)).resolve() if u.path else HERE/'report.html'
        if not p.is_file():errors.append({'href':href,'reason':'missing file'}); continue
        if u.fragment and p.suffix=='.html':
            if p not in cache:cache[p]=Doc(p.read_text(encoding='utf-8')).ids
            if unquote(u.fragment) not in cache[p]: errors.append({'href':href,'reason':'missing anchor'})
    ids=set(d.ids)
    wanted=[f'E{i:02}' for i in range(1,10)]+[f'Q{i:02}' for i in range(1,13)]+[f'F{i:02}' for i in range(1,33)]
    for id in wanted:
        if id not in ids:errors.append({'id':id,'reason':'missing section'})
    duplicates=[k for k,v in Counter(d.ids).items() if v>1]
    if duplicates:errors.append({'reason':'duplicate ids','ids':duplicates})
    if '\ufffd' in text:errors.append({'reason':'replacement character in UTF-8 report'})
    findings=json.loads((HERE/'findings.json').read_text(encoding='utf-8'))
    result={'ok':not errors,'local_links_checked':count,'coverage_sections':len(wanted),
            'findings':len(findings),'code_evidence_locations':sum(len(f['code_evidence']) for f in findings),
            'errors':errors,'scope':'Report coverage/links only. No measurement or statistical test.'}
    (HERE/'report_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
