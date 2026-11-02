"""Read-only evidence inventory; does not run image measurements or statistical tests."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
ROOTS = [ROOT / 'analysis' / n for n in (
    'reference_line_methods_108_20261030',
    'reference_line_methods_selection_20261030',
    'six_methods_explanation_20261030',
)]

def inventory():
    result = {}
    for base in ROOTS:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ('__pycache__', 'build', '.git', 'qa')):
                continue
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            result[p.relative_to(ROOT).as_posix()] = {'size': p.stat().st_size, 'sha256': h.hexdigest()}
    return result

if __name__ == '__main__':
    current = inventory()
    baseline = OUT / 'baseline_manifest.json'
    if '--verify' in sys.argv:
        old = json.loads(baseline.read_text(encoding='utf-8'))['files']
        diff = {p: {'before': old.get(p), 'after': current.get(p)}
                for p in old.keys() | current.keys() if old.get(p) != current.get(p)}
        payload = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                   'checked_files': len(current), 'unchanged': not diff, 'differences': diff}
        (OUT / 'preservation_check.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'checked_files': len(current), 'unchanged': not diff, 'differences': len(diff)}))
        sys.exit(bool(diff))
    if baseline.exists():
        raise SystemExit('Baseline exists; refusing to replace it.')
    baseline.write_text(json.dumps({'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'All files under three source roots except build, __pycache__, .git and qa directories',
        'files': current}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'preserved_files': len(current)}))
