from pathlib import Path
import json,zipfile
HERE=Path(__file__).resolve().parent
def main():
 output=HERE/'픽셀잡음가중_GN_비교실험_기록.zip'
 excluded={'cache','raw_checks','__pycache__'}
 files=[p for p in HERE.rglob('*') if p.is_file() and p!=output and not any(part in excluded for part in p.relative_to(HERE).parts)]
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(HERE).as_posix())
 with zipfile.ZipFile(output) as z:assert z.testzip() is None
 print(json.dumps({'zip':str(output),'files':len(files),'bytes':output.stat().st_size},ensure_ascii=False))
if __name__=='__main__':main()
