"""Local image-only motion workbench; existing experiments remain read-only."""
from pathlib import Path
import base64
import json
import sys
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'codex_experiments_20261014'
MODEL = BASE / 'ppt_six_origin_20261021' / 'restored_model'
sys.path[:0] = [str(BASE / 'measurement')]
from spread import SpreadPairExtractor
from measure import ROOT, BOXES

CENTER = ROOT / 'center.bmp'
extractor = SpreadPairExtractor(CENTER, MODEL / 'features.npz')
params = json.loads((MODEL / 'model.json').read_text())
reference_displacement = np.zeros_like(extractor.points)
center_image = cv2.imdecode(np.fromfile(CENTER, np.uint8), 0)


def jpeg(a):
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', a)[1]).decode()


def infer(raw):
    start = time.perf_counter()
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), 0)
    if a is None:
        raise ValueError('이미지를 읽을 수 없습니다. BMP, PNG, JPG 파일을 선택해 주세요.')
    if a.shape != center_image.shape:
        raise ValueError(f'기준 사진과 같은 {center_image.shape[1]} × {center_image.shape[0]} 원본 사진이 필요합니다. 자르거나 크기를 바꾸지 않은 사진을 사용해 주세요.')
    # The HTTP body is only bytes. No filename or command label enters inference.
    with tempfile.TemporaryDirectory(prefix='qr_motion_', dir=HERE) as tmp:
        path = Path(tmp) / 'current.png'
        path.write_bytes(cv2.imencode('.png', a)[1].tobytes())
        r = extractor.extract(path)
    scores = np.asarray(r['coarse_score']).ravel()[:3]
    if not np.isfinite(r['D']).all() or len(scores) != 3 or np.min(scores) < .6:
        raise ValueError('기준 위치에서 QR 3개를 안정적으로 찾지 못했습니다. 같은 카메라·설치 환경에서 촬영한 사진을 사용해 주세요.')
    d = r['D'].astype(float) - reference_displacement
    q = r['quality']
    p = r['points']
    u, v = (p[:, 0] - 1906) / 486, (p[:, 1] - 1070) / 122
    b = np.c_[u * 0 + 1, u, v, u*u, u*v, v*v]
    ok = r['valid']
    if ok.mean() < .95 or any(np.sum(ok & (r['regions'] == i)) < 20 for i in range(3)):
        raise ValueError('유효 특징점이 부족해 판정을 보류합니다. 같은 촬영 환경의 원본 사진을 확인해 주세요.')
    coefficients = np.linalg.lstsq(b[ok], d[ok], rcond=None)[0]
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
        eligible = value * (1 if key.endswith('+') else -1) >= 0
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
        spatial_rmse=float(np.sqrt(np.mean((d[r['valid']]-spatial_fit[r['valid']])**2))),
        points=p.tolist(), displacements=d.tolist(), regions=r['regions'].tolist(), valid=r['valid'].tolist(),
        quality=q.tolist(), ncc=scores.tolist(), W=W.tolist(), candidates=sorted(all_candidates,key=lambda row:row['cost']),
        selected_branch=selected['branch'], other_axis_cost=other,
        cost_margin=(other-selected['cost'])/(other+selected['cost']+1e-12),
        identical_reference=bool(np.array_equal(a,center_image)),
        crops=[dict(reference=jpeg(center_image[y0:y1,x0:x1]), current=jpeg(a[y0:y1,x0:x1]), box=[x0,y0,x1,y1]) for x0,y0,x1,y1 in BOXES])
    return dict(**best, Y=best['value'] if best['axis']=='Y' else 0., T=best['value'] if best['axis']=='T' else 0.,
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
        if self.path in ('/', '/index.html'):
            self.send(200, (HERE/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path in ('/developer.js', '/developer.css'):
            self.send(200, (HERE/self.path[1:]).read_bytes(), 'text/javascript; charset=utf-8' if self.path.endswith('.js') else 'text/css; charset=utf-8')
        elif self.path == '/reference':
            self.send(200, cv2.imencode('.jpg', center_image)[1].tobytes(), 'image/jpeg')
        elif self.path == '/health':
            self.send(200, b'{"ready":true,"points":205}', 'application/json')
        else:
            self.send(404, b'Not found', 'text/plain')

    def do_POST(self):
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
    HTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
