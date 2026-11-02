"""Render the newly created audit report for layout inspection only."""
from pathlib import Path
import subprocess
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
edge=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
profile=ROOT/'.tmp/full_line_audit_browser_20261030'
shot=HERE/'report_preview.png'
command=[str(edge),'--headless','--disable-gpu','--no-first-run',
         '--disable-background-networking','--disable-features=msEdgeSidebarV2',
         '--user-data-dir='+str(profile),'--window-size=1440,1100',
         '--screenshot='+str(shot),'--virtual-time-budget=1800',
         (HERE/'report.html').as_uri()]
result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                      timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
(HERE/'preview_log.txt').write_bytes(result.stdout+result.stderr)
print(json.dumps({'exit_code':result.returncode,'screenshot':shot.exists(),
                  'bytes':shot.stat().st_size if shot.exists() else 0}))
raise SystemExit(0 if shot.exists() else 1)
