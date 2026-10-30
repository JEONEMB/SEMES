import argparse,json,time,urllib.request
from pathlib import Path
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);a=p.parse_args();raw=Path(a.image).read_bytes();start=time.perf_counter()
 request=urllib.request.Request('http://127.0.0.1:8766/predict',data=raw,headers={'Content-Type':'application/octet-stream'})
 result=json.load(urllib.request.urlopen(request,timeout=30));result['client_roundtrip_seconds']=time.perf_counter()-start;print(json.dumps(result,indent=2))
