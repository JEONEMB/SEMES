from pathlib import Path
from html.parser import HTMLParser
import json,zipfile
import pandas as pd
P=Path(__file__).resolve().parent
class Check(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href') and not v.startswith('#'):
                assert (P/v).exists(),v
h=(P/'report.html').read_text(encoding='utf-8')
assert '검증 중' not in h
Check().feed(h)
v=json.loads((P/'remeasurement_validation.json').read_text())
assert v['images']==575 and v['max_feature_difference']==0 and v['failures']==0
s=pd.read_csv(P/'session_summary.csv')
assert s.images.sum()==575 and s.evaluated.sum()==571
assert len(pd.read_csv(P/'session_feature_statistics.csv'))==14*104
with zipfile.ZipFile(P/'session_eda_report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in P.iterdir():
        if f.is_file() and f.suffix in ('.html','.png','.csv','.json','.md','.py','.ps1','.txt'):
            z.write(f,f.name)
print('PASS: report resources, 575/571 counts, 14 x 104 feature statistics, remeasurement consistency, report ZIP')
