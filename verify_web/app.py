"""확인 앱 웹 서버 (화면 1장 + JSON API + 실시간 영상 + 상태 스트림)."""
import json
import math
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from .algorithm import AlgorithmError
from .runner import RunError

ROOT = Path(__file__).resolve().parent
FONTS = ROOT.parent / 'demo_web' / 'static' / 'fonts'      # 데모 웹 글꼴을 읽기만 함
PLATE_CENTER = (1906.3, 1130.8)   # A507 센터 사진의 가운데 QR (알고리즘 좌표, SEMES 는 1907, 1065)
NO_FRAME_GIVE_UP_S = 3            # 카메라 프레임이 이만큼(초) 없으면 영상 응답을 끝낸다


def _rigid(src, dst):
    """src→dst 회전+이동 (최소제곱, 점 2개 이상). 반환: 점 변환 함수."""
    n = len(src)
    sx, sy = sum(p[0] for p in src) / n, sum(p[1] for p in src) / n
    dx, dy = sum(p[0] for p in dst) / n, sum(p[1] for p in dst) / n
    a = b = 0.0
    for (x1, y1), (x2, y2) in zip(src, dst):
        x1, y1, x2, y2 = x1 - sx, y1 - sy, x2 - dx, y2 - dy
        a += x1 * x2 + y1 * y2
        b += x1 * y2 - y1 * x2
    th = math.atan2(b, a)
    c, si = math.cos(th), math.sin(th)
    return lambda x, y: (c * (x - sx) - si * (y - sy) + dx, si * (x - sx) + c * (y - sy) + dy)


def make_overlay(cam, s):
    """센터 기준(목표, 점선)과 지금(실선) 모양을 화면 좌표로.
    QR: 기준 사진의 QR 테두리. 지금 모양 = 기준 테두리를 이번 측정의 QR 중심 3개에 맞춰 옮긴 것.
    나사: 기준/지금 각각 알고리즘이 찾은 나사 중심에 원 (나사는 QR 과 깊이가 달라 따로 그린다)."""
    vp = lambda x, y: [round(v, 1) for v in cam.view_point(x, y)]
    scale = cam.view_point(1000, 0)[0] - cam.view_point(0, 0)[0]     # 알고리즘 1000 px → 화면 px
    ov = dict(view_size=[1280, 720], target=None, now=None)
    ref = s.get('reference') or {}
    rois = {r['name']: r for r in ref.get('rois') or []}
    qr_names = sorted((n for n in rois if n.upper().startswith('QR')), key=lambda n: rois[n]['center'][0])
    screws = [n for n in rois if n not in qr_names]
    quads = ref.get('qr_alg')
    if not quads and qr_names:          # QR 테두리를 못 찾았으면 알고리즘 ROI 상자로
        quads = []
        for n in qr_names:
            x, y, w, h = rois[n]['rect']
            quads.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    if not quads and not screws:
        return ov

    def screw_circles(src):
        out = []
        for n in screws:
            if n in src:
                x, y, w, h = rois[n]['rect']
                out.append(dict(name=n, c=vp(*src[n]['center']), r=round(min(w, h) * 0.3 * scale / 1000, 1)))
        return out

    ov['target'] = dict(qr=[[vp(x, y) for x, y in q] for q in quads or []], screws=screw_circles(rois))
    now = {r['name']: r for r in s.get('rois_now') or []}
    if now:
        pairs = [n for n in qr_names if n in now]
        qr_now = []
        if quads and len(pairs) >= 2 and len(quads) == len(qr_names):
            f = _rigid([rois[n]['center'] for n in pairs], [now[n]['center'] for n in pairs])
            qr_now = [[vp(*f(x, y)) for x, y in q] for q in quads]
        ov['now'] = dict(qr=qr_now, screws=screw_circles(now),
                         dots=[vp(*now[n]['center']) for n in now])
    return ov


def create_app(runner, devices, demo):
    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    app.config['JSON_AS_ASCII'] = False

    def state():
        s = runner.snapshot()
        s['demo'] = demo.status()
        cam = devices.camera
        if cam:
            s['overlay'] = make_overlay(cam, s)
        return s

    def body():
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    @app.errorhandler(RunError)
    @app.errorhandler(ValueError)
    def run_error(exc):
        return jsonify(error=str(exc)), 409

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/fonts/<path:name>')
    def fonts(name):
        return send_from_directory(FONTS, name, max_age=86400)

    @app.get('/api/state')
    def api_state():
        return jsonify(state())

    @app.get('/api/events')
    def events():
        def stream():
            version = -1
            while True:
                s = state()
                version = s['version']
                yield f'data: {json.dumps(s, ensure_ascii=False)}\n\n'
                # 상태가 바뀌거나 1초마다 (위치·fps 갱신용)
                runner.wait_change(version, timeout=1.0)
        return Response(stream(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/video')
    def video():
        cam = devices.camera
        if cam is None:
            return jsonify(error='카메라 없음'), 503

        def stream():
            # 무언가를 계속 보내야 브라우저가 끊은 연결을 알아채고 작업 스레드를 돌려받는다.
            # 새 프레임이 없으면 1초마다 마지막 프레임을 다시 보내고, 프레임이 아예 없으면 끝낸다.
            cam.add_viewer(+1)
            try:
                seq, empty = 0, 0
                while devices.camera is cam:
                    seq, jpg = cam.wait_jpeg(seq, timeout=1.0)
                    if jpg is None:
                        empty += 1
                        if empty >= NO_FRAME_GIVE_UP_S:
                            return
                        continue
                    empty = 0
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode()
                           + b'\r\n\r\n' + jpg + b'\r\n')
            finally:
                cam.add_viewer(-1)
        return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={'Cache-Control': 'no-cache'})

    @app.post('/api/view')
    def view():
        b = body()
        zoom = float(b.get('zoom', 1.0))
        if not 1.0 <= zoom <= 8.0:
            raise ValueError('확대는 1~8 배')
        center = PLATE_CENTER if b.get('center') == 'plate' else None
        if devices.camera:
            devices.camera.set_view(zoom, center)
            if 'mode' in b:
                devices.camera.set_mode(b['mode'])
        return jsonify(ok=True)

    def offset_arg(v):
        if v is None or v == 'random':
            return v
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return float(v[0]), float(v[1])
        raise ValueError('offset 은 "random" 또는 [y_mm, t_deg]')

    @app.post('/api/home')
    def home():
        runner.home()
        return jsonify(ok=True)

    @app.post('/api/goto')
    def goto():
        runner.goto(offset_arg(body().get('offset', 'random')))
        return jsonify(ok=True)

    @app.post('/api/run')
    def run():
        runner.run(offset_arg(body().get('offset')))
        return jsonify(ok=True)

    @app.post('/api/step')
    def step():
        runner.step()
        return jsonify(ok=True)

    @app.post('/api/reference')
    def reference():
        runner.reference()
        return jsonify(ok=True)

    @app.post('/api/measure')
    def measure():
        runner.measure()
        return jsonify(ok=True)

    @app.post('/api/stop')
    def stop():
        runner.stop()
        return jsonify(ok=True)

    @app.post('/api/algorithm')
    def algorithm():
        algo_id = body().get('id')
        if not isinstance(algo_id, str):
            raise ValueError('id 가 필요함')
        try:
            runner.select_algorithm(algo_id)
        except AlgorithmError as exc:
            raise RunError(str(exc)) from None
        return jsonify(ok=True)

    @app.post('/api/center-here')
    def center_here():
        runner.set_center_here()
        return jsonify(ok=True)

    @app.post('/api/settings')
    def settings():
        if runner.busy():
            raise RunError('작업 중에는 설정을 바꿀 수 없습니다.')
        runner.settings.update(body())
        runner._set()       # 화면 갱신
        return jsonify(ok=True)

    @app.post('/api/handoff')
    def handoff():
        """`데모 웹으로` 버튼: 장치를 놓고 데모 웹을 띄운다."""
        if runner.busy():
            runner.stop()
            runner.wait_idle(15)
        devices.close()
        runner._set(phase='released', message='데모 웹으로 넘김')
        try:
            demo.start()
        except Exception as exc:
            devices.open()
            runner._set(phase='error', message=str(exc))
            return jsonify(error=str(exc)), 503
        return jsonify(ok=True, port=demo.port)

    @app.post('/api/reclaim')
    def reclaim():
        """확인 앱 화면을 열면: 데모 웹이 떠 있으면 끄고 장치를 다시 잡는다."""
        took = False
        if demo.running():
            runner._set(phase='released', message='데모 웹 종료 중…')
            demo.stop()
            took = True
        if not devices.ready:
            devices.close()
            ok = devices.open()
            runner._set(phase='idle' if ok else 'error',
                        message='장치를 다시 잡았습니다.' if ok else f'장치 연결 실패: {devices.error}')
            took = True
        return jsonify(ok=devices.ready, took=took, error=devices.error)

    return app
