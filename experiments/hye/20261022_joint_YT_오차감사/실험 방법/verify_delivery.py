"""Final artifact links, snapshots and completed execution checks."""
from pathlib import Path
import json
from hashlib import sha256
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
H=Path(__file__).resolve().parent
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])

def main():
    p=Links();p.feed((H/'report.html').read_text(encoding='utf-8'));missing=[]
    for link in p.links:
        if link.startswith('#'):
            if link[1:] not in p.ids:missing.append(link)
        elif link.startswith('file:'):
            path=unquote(urlsplit(link).path).lstrip('/')
            if not Path(path).exists():missing.append(link)
        elif not link.startswith(('https://','http://')):
            if not (H/unquote(link)).exists():missing.append(link)
    snapshot=json.loads((H/'audit/dependency_snapshot.json').read_text(encoding='utf-8'))
    changed=[r['path'] for r in snapshot['files'] if sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']]
    assert not missing,missing
    assert not changed,changed
    initial=json.loads((H/'audit/dependency_snapshot_initial.json').read_text(encoding='utf-8'))
    initial_hash={r['path']:r['sha256'] for r in initial['files']}
    refreshed=[r['path'] for r in snapshot['files'] if r['sha256']!=initial_hash.get(r['path'])]
    assert all(Path(path).name=='ALGORITHM_RULES.md' for path in refreshed),refreshed
    smoke=json.loads((H/'smoke_test.json').read_text());assert len(smoke['records'])==2
    out=dict(html_links_checked=len(p.links),missing_links=missing,snapshot_dependency_hashes_checked=len(snapshot['files']),changed_dependencies=changed,initial_snapshot_preserved=True,documentation_snapshot_refresh=refreshed,refresh_reason='Shared algorithm rules gained staged-goal clarification during this turn; inference dependencies unchanged. No independent test had been frozen or run.',real_image_execution_checks=len(smoke['records']),independent_performance_test=False)
    (H/'delivery_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
