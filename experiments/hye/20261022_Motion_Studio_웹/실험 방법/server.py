"""Local image-only motion workbench; existing experiments remain read-only."""
from pathlib import Path
import base64
import hashlib
import json
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs, unquote
import batch_jobs as batches
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'codex_experiments_20261014'
MODEL = BASE / 'center_patch_v2_20261021'
sys.path[:0] = [str(BASE / 'measurement')]
from structural import StructuralExtractor
from measure import ROOT, BOXES

CENTER = ROOT / 'center.bmp'
provenance = json.loads((HERE/'structural_assets/provenance.json').read_text())
if hashlib.sha256(CENTER.read_bytes()).hexdigest() != provenance['center_sha256']:
    raise ValueError('고정 center가 630점 보정 기준과 다릅니다.')

params = json.loads((MODEL / 'v1_edge630_model.json').read_text())

center_image = cv2.imdecode(np.fromfile(CENTER, np.uint8), 0)
extractor = StructuralExtractor(center_image)
MODEL_ID = "v1_edge630_boundary_fixed"


def jpeg(a):
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', a)[1]).decode()


def infer(raw):
    start = time.perf_counter()
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), 0)
    if a is None:
        raise ValueError('이미지를 읽을 수 없습니다. BMP, PNG, JPG 파일을 선택해 주세요.')
    if a.shape != center_image.shape:
        raise ValueError(f'기준 사진과 같은 {center_image.shape[1]} × {center_image.shape[0]} 원본 사진이 필요합니다. 자르거나 크기를 바꾸지 않은 사진을 사용해 주세요.')
    # Bytes only; no filename, label, or trial selects the model.
    r = extractor.extract(a)
    scores = np.asarray(r['coarse_score']).ravel()[:3]
    if not np.isfinite(r['D']).all() or len(scores) != 3 or np.min(scores) < .6:
        raise ValueError('기준 위치에서 QR 3개를 안정적으로 찾지 못했습니다. 같은 카메라·설치 환경에서 촬영한 사진을 사용해 주세요.')
    d = r['D'].astype(float)
    q = r['quality']
    p = r['points']
    u, v = (p[:, 0] - 1906) / 486, (p[:, 1] - 1070) / 122
    b = np.c_[u * 0 + 1, u, v, u*u, u*v, v*v]
    ok = r['valid']
    # Fixed 630 protocol: diagnostic failures remain in the unweighted fit.
    coefficients = np.linalg.lstsq(b, d, rcond=None)[0]
    spatial_fit = b @ coefficients
    x = coefficients.ravel() - np.array(params['zero_feature'])
    W = np.array(params['W'])
    choices = []
    all_candidates = []
    for key, (axis, origin, slope, curve) in params['branches'].items():
        z = W @ (x - np.array(origin))
        s, c = W @ np.array(slope), W @ np.array(curve)
        value = float(z @ s / (s @ s))
        for _ in range(8):
            j = s + 2 * value * c
            value += float(j @ (z - value*s - value*value*c) / max(j @ j, 1e-20))
        cost = float(np.sum((z-value*s-value*value*c)**2))
        eligible = True  # Original v1 edge630 predictor fits Y/T curves without a sign gate.
        model_vector = np.array(origin) + value * np.array(slope) + value*value*np.array(curve)
        candidate = dict(branch=key, axis=axis, value=value, cost=cost, eligible=bool(eligible),
                         origin=origin, slope=slope, curve=curve, fitted=model_vector.tolist(),
                         whitened_residual=(W @ (x-model_vector)).tolist())
        all_candidates.append(candidate)
        if not eligible:
            continue
        choices.append(candidate)
    choices.sort(key=lambda row: row['cost'])
    best = {k: choices[0][k] for k in ('branch', 'axis', 'value', 'cost')}
    selected = choices[0]
    if np.array_equal(a, center_image):
        best = dict(axis='CENTER', value=0., cost=0.)
    crop = (1450, 900, 2350, 1240)
    x0, y0, x1, y1 = crop
    reference = cv2.cvtColor(center_image[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    current = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    colors = [(255, 210, 0), (255, 70, 215), (0, 235, 160)]
    for point, delta, region, valid in zip(p, d, r['regions'], r['valid']):
        pos = tuple(np.rint(point - [x0, y0]).astype(int))
        color = colors[int(region)] if valid else (65, 65, 235)
        cv2.circle(reference, pos, 3, (15,15,25), -1)
        cv2.circle(reference, pos, 2, color, -1)
        end = tuple(np.rint(point + delta - [x0, y0]).astype(int))
        cv2.circle(current, end, 3, (15,15,25), -1)
        cv2.circle(current, end, 2, color, -1)
    vectors = np.full((340, 900, 3), (37, 29, 24), np.uint8)
    for point, delta, region, valid in zip(p, d, r['regions'], r['valid']):
        pos = tuple(np.rint(point - [x0, y0]).astype(int))
        end = tuple(np.rint(point - [x0, y0] + delta * 20).astype(int))
        cv2.arrowedLine(vectors, pos, end, colors[int(region)] if valid else (65, 65, 235), 2, tipLength=.3)
    diff = cv2.absdiff(center_image[y0:y1, x0:x1], a[y0:y1, x0:x1])
    ok = int(r['valid'].sum())
    other = min(row['cost'] for row in choices if row['axis'] != selected['axis'])
    diagnostics = dict(coefficients=coefficients.tolist(), feature=x.tolist(), spatial_fitted=spatial_fit.tolist(),
        spatial_rmse=float(np.sqrt(np.mean((d-spatial_fit)**2))),
        ids=r['ids'].tolist(), reasons=r['reasons'], measurement='blurred_grid_boundary', model=MODEL_ID,
        points=p.tolist(), displacements=d.tolist(), regions=r['regions'].tolist(), valid=r['valid'].tolist(),
        quality=q.tolist(), ncc=scores.tolist(), W=W.tolist(), candidates=sorted(all_candidates,key=lambda row:row['cost']),
        selected_branch=selected['branch'], other_axis_cost=other,
        cost_margin=(other-selected['cost'])/(other+selected['cost']+1e-12),
        identical_reference=bool(np.array_equal(a,center_image)),
        crops=[dict(reference=jpeg(center_image[y0:y1,x0:x1]), current=jpeg(a[y0:y1,x0:x1]), box=[x0,y0,x1,y1]) for x0,y0,x1,y1 in BOXES])
    return dict(**best, model=MODEL_ID, Y=best['value'] if best['axis']=='Y' else 0., T=best['value'] if best['axis']=='T' else 0.,
                valid=ok, total=len(p), rejected=len(p)-ok, seconds=round(time.perf_counter()-start, 2),
                quality_warning=ok / len(p) < .95, candidates=choices, trace=diagnostics,
                images=[jpeg(reference), jpeg(current), jpeg(vectors), jpeg(diff)],
                regions=[dict(name=name, dx=float(np.median(d[r['regions']==i, 0])), dy=float(np.median(d[r['regions']==i, 1]))) for i, name in enumerate(['왼쪽 QR', '가운데 QR', '오른쪽 QR'])])


class Handler(BaseHTTPRequestHandler):
    def send(self, code, content, mime):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path.startswith('/api/batch'):
            try:
                if parsed.path == '/api/batch/known':
                    value = dict(path=str(batches.KNOWN), archives=[dict(name=p.name, bytes=p.stat().st_size) for p in sorted(batches.KNOWN.glob('*.zip'))])
                elif parsed.path == '/api/batch/list':
                    value = [dict(id=p.parent.name) for p in sorted(batches.RUNS.glob('*/job.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:20]]
                else:
                    job = batches.get_job(query.get('id',[''])[0])
                    if parsed.path == '/api/batch/result':
                        index = int(query.get('index',['-1'])[0])
                        if index < 0 or index >= len(job['sources']): raise ValueError('잘못된 사진 번호')
                        path = batches.RUNS/job['id']/f'result_{index}.json'
                        if not path.exists(): raise ValueError('분석 결과가 아직 없습니다.')
                        self.send(200, path.read_bytes(), 'application/json'); return
                    else: value = batches.snapshot(job)
                self.send(200, json.dumps(value, ensure_ascii=False).encode(), 'application/json')
            except (ValueError, KeyError) as exc:
                self.send(400, json.dumps(dict(error=str(exc))).encode(), 'application/json')
            return
        if parsed.path in ('/batch','/batch.html'):
            self.send(200, (HERE/'batch.html').read_bytes(), 'text/html; charset=utf-8'); return
        if parsed.path in ('/batch.js','/batch.css'):
            self.send(200, (HERE/parsed.path[1:]).read_bytes(), 'text/javascript; charset=utf-8' if parsed.path.endswith('.js') else 'text/css; charset=utf-8'); return
        self.path = parsed.path
        if self.path in ('/', '/index.html'):
            self.send(200, (HERE/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path in ('/developer.js', '/developer.css'):
            self.send(200, (HERE/self.path[1:]).read_bytes(), 'text/javascript; charset=utf-8' if self.path.endswith('.js') else 'text/css; charset=utf-8')
        elif self.path == '/reference':
            self.send(200, cv2.imencode('.jpg', center_image)[1].tobytes(), 'image/jpeg')
        elif self.path == '/health':
            self.send(200, json.dumps(dict(ready=True, points=630, model=MODEL_ID, measurement='blurred_grid_boundary')).encode(), 'application/json')
        else:
            self.send(404, b'Not found', 'text/plain')

    def do_POST(self):
        parsed = urlsplit(self.path)
        if parsed.path.startswith('/api/batch/'):
            query = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/batch/upload':
                    job = batches.get_job(query.get('id',[''])[0])
                    batches.add_upload(job, self.rfile, int(self.headers.get('Content-Length','0')), query.get('name',['image.bmp'])[0], query.get('relative',['image.bmp'])[0])
                    value = dict(ok=True)
                else:
                    size = int(self.headers.get('Content-Length','0'))
                    if size > 65536: raise ValueError('요청이 너무 큽니다.')
                    data = json.loads(self.rfile.read(size) or b'{}')
                    if parsed.path == '/api/batch/create':
                        job = batches.new_job()
                        value = dict(id=job['id'])
                    elif parsed.path == '/api/batch/import-known':
                        job = batches.new_job();batches.start(job, known=True, workers=data.get('workers',2));value=dict(id=job['id'])
                    else:
                        job = batches.get_job(data.get('id',''))
                        if parsed.path == '/api/batch/start':batches.start(job, workers=data.get('workers',2))
                        elif parsed.path == '/api/batch/cancel':
                            with batches.LOCK:
                                if job['state'] in ('running','preparing'):
                                    job['stop']=True;job['state']='cancelling';batches.persist(job)
                        else:raise ValueError('알 수 없는 요청')
                        value = dict(id=job['id'])
                self.send(200, json.dumps(value).encode(), 'application/json')
            except (ValueError, KeyError) as exc:
                self.send(400, json.dumps(dict(error=str(exc))).encode(), 'application/json')
            return
        if self.path != '/api/predict':
            self.send(404, b'Not found', 'text/plain'); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 32*1024*1024:
                raise ValueError('32 MB 이하의 사진 한 장을 선택해 주세요.')
            result = infer(self.rfile.read(size))
            self.send(200, json.dumps(result, allow_nan=False).encode(), 'application/json')
        except (ValueError, cv2.error) as exc:
            message = str(exc) if isinstance(exc, ValueError) else '영상 추적에 실패했습니다. 동일한 촬영 환경의 원본 사진을 확인해 주세요.'
            self.send(422, json.dumps(dict(error=message)).encode(), 'application/json')
        except Exception:
            import traceback
            traceback.print_exc()
            self.send(500, json.dumps(dict(error='분석 중 오류가 발생했습니다. 서버 로그를 확인해 주세요.')).encode(), 'application/json')


if __name__ == '__main__':
    print('Motion Studio ready: http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
