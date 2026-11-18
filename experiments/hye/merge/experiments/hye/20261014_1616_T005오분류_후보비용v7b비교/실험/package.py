from pathlib import Path
import zipfile,json
HERE=Path(__file__).resolve().parent
def main():
 out=HERE/'T005_v7b_비교실험_기록.zip'
 files=[p for p in HERE.rglob('*') if p.is_file() and p!=out and not any(q in {'measurements','__pycache__'} for q in p.relative_to(HERE).parts)]
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(HERE).as_posix())
  z.write(HERE.parent/'colleague_brief_20261015/동료공유용_단일파일.html','동료공유용_단일파일.html')
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 print(json.dumps(dict(path=str(out),files=len(files)+1,bytes=out.stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
