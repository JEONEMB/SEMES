from pathlib import Path
import hashlib, json, urllib.request, tarfile, io
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SRC = HERE / 'sources'
SRC.mkdir(exist_ok=True)
records = []
def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Codex-local-reproducibility-audit'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()
def save(url, name):
    data = get(url)
    path = SRC / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    records.append({'url': url, 'file': str(path.relative_to(HERE)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes':len(data)})
    print(name, len(data), flush=True)
    return data

commit = json.loads(get('https://api.github.com/repos/XiaoouPan/adaHuber/commits/master'))['sha']
for name in ['src/adaHuber.cpp','R/adaHuber.R','R/RcppExports.R','src/RcppExports.cpp','DESCRIPTION','NAMESPACE','README.md']:
    save(f'https://raw.githubusercontent.com/XiaoouPan/adaHuber/{commit}/{name}', 'adaHuber/' + name)
for url, name in [
 ('https://www3.stat.sinica.edu.tw/ss_newpaper/SS-2019-0045_na.pdf','wang2021_accepted.pdf'),
 ('https://www3.stat.sinica.edu.tw/preprint/supp/2019-0045_supp.pdf','wang2021_supplement.pdf'),
 ('https://arxiv.org/pdf/1706.06991','sun2020_preprint.pdf'),
]:
    try: save(url,name)
    except Exception as e: records.append({'url':url,'error':str(e)})

# Dependency only: actual Armadillo headers. No algorithm changes, no global installation.
arm_commit=json.loads(get('https://api.github.com/repos/RcppCore/RcppArmadillo/commits/master'))['sha']
data=save(f'https://codeload.github.com/RcppCore/RcppArmadillo/tar.gz/{arm_commit}','rcpparmadillo_source.tar.gz')
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
    for member in tf.getmembers():
        key='/inst/include/'
        if member.isfile() and key in member.name:
            relative=member.name.split(key,1)[1]
            # Only the standalone Armadillo header tree is needed.
            if relative == 'armadillo' or relative.startswith('armadillo_bits/') or relative.startswith('current/'):
                dest=HERE/'dependencies'/relative
                if HERE.resolve() not in dest.resolve().parents: raise RuntimeError('unsafe archive path')
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'adahuber_commit':commit,'armadillo_dependency_commit':arm_commit,'sources':records}
(HERE/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'adahuber_commit':commit,'dependency_commit':arm_commit}))
