"""Record immutable prior experiment files and development-only scope."""
from pathlib import Path
import hashlib,json,datetime
H=Path(__file__).resolve().parent
folders=['joint_yt_improved_20261022','evaluate_archives_20261022','joint_yt_generalization_20261022','joint_yt_disentangle_20261022']
files={}
for name in folders:
    for p in sorted((H.parent/name).rglob('*')):
        if p.is_file() and p.suffix in ['.py','.joblib','.csv','.json','.npz','.html']:
            files[str(p.relative_to(H.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
out=H/'preservation_before.json'
if out.exists():raise RuntimeError('Already initialized; preservation baseline must not be overwritten.')
out.write_text(json.dumps(dict(recorded=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files),indent=2),encoding='utf-8')
print('Preservation fingerprints:',len(files))
