"""Persistent image-only correction inference API, separate from Motion Studio."""
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import argparse,json,time,hashlib
from threadpoolctl import threadpool_limits
from engine import Engine,HERE

class Handler(BaseHTTPRequestHandler):
    def send(self,status,value,mime='application/json; charset=utf-8'):
        data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':
            self.send(200,dict(ready=True,points=630,variant='compiled_bounded_LM',warm_startup_seconds=self.server.engine.init_seconds,result_cache=False,model_sha256=hashlib.sha256((HERE/'assets/model.json').read_bytes()).hexdigest(),ids_sha256=hashlib.sha256(self.server.engine.extractor.ids.tobytes()).hexdigest()))
        elif self.path in ('/','/report.html') and (HERE/'report.html').exists():self.send(200,(HERE/'report.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/results/comparison.png' and (HERE/'results/comparison.png').exists():self.send(200,(HERE/'results/comparison.png').read_bytes(),'image/png')
        else:self.send(404,dict(error='Not found'))
    def do_POST(self):
        if self.path!='/predict':self.send(404,dict(error='Not found'));return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=32*1024**2:raise ValueError('Provide image bytes, at most 32 MB')
            raw=self.rfile.read(length);start=time.perf_counter();result=self.server.engine.predict(raw)
            result['service_processing_seconds']=time.perf_counter()-start;result['T_unit']='unconfirmed';result['scope']='single-axis correction estimate; camera/robot latency excluded';self.send(200,result)
        except ValueError as exc:self.send(422,dict(error=str(exc)))
        except Exception as exc:self.send(500,dict(error=str(exc)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    engine=Engine('compiled');limits=threadpool_limits(1)
    server=HTTPServer(('127.0.0.1',args.port),Handler);server.engine=engine
    print(f'630-point optimized engine ready: http://127.0.0.1:{args.port} | startup {engine.init_seconds:.3f}s',flush=True)
    server.serve_forever()
