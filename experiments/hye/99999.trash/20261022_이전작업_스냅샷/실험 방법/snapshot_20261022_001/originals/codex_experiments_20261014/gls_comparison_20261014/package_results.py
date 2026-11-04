from pathlib import Path
import json, sys, zipfile
import numpy, scipy, sklearn

HERE=Path(__file__).resolve().parent
versions=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,scikit_learn=sklearn.__version__)
(HERE/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf8')
destination=HERE/'GLS_비교실험_기록.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
    for file in sorted(HERE.rglob('*')):
        relative=file.relative_to(HERE)
        if not file.is_file() or file==destination or any(s in {'__pycache__','raw_checks'} for s in relative.parts):continue
        if file.suffix in {'.pyc','.zip'}:continue
        archive.write(file,relative.as_posix())
print(json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,versions=versions),ensure_ascii=False))
