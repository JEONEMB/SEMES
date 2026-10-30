import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from engine import Engine
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output');a=p.parse_args();e=Engine('compiled')
 with threadpool_limits(1):r=e.predict(Path(a.image).read_bytes())
 r['startup_seconds']=e.init_seconds;text=json.dumps(r,indent=2)
 if a.output:Path(a.output).write_text(text,encoding='utf-8')
 print(text)
