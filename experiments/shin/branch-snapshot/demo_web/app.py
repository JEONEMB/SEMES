import json
import secrets
import threading
import time
from functools import wraps
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, session
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash

from .camera import Camera, PROFILES
from .rectification import reference_signature
from .controller import Controller
from .stage import make_stage
from .store import Problem, Store


def create_app(data_dir=None, mode='mock', camera_enabled=None, stage=None, camera_profile=None, transport='serial',
               home_stage=None):
    data_dir = Path(data_dir or Path(__file__).parent / 'instance')
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = Store(data_dir / 'demo.sqlite3')
    secret = store.meta('secret')
    if not secret:
        secret = secrets.token_hex(32)
        store.meta('secret', secret)
    app = Flask(__name__)
    app.config.update(SECRET_KEY=secret, MAX_CONTENT_LENGTH=256 * 1024,
                      SESSION_COOKIE_NAME='alignment_session', SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Strict', PERMANENT_SESSION_LIFETIME=8 * 3600)
    saved_camera = json.loads(store.meta('camera_config') or '{}')
    camera = Camera(saved_camera.get('enabled', False) if camera_enabled is None else camera_enabled,
                    camera_profile or saved_camera.get('profile', 'full'),
                    saved_camera.get('undistort', True))
    signature = reference_signature(camera.profile, camera.undistort)
    if store.meta('camera_reference') != signature:
        # Migrate old uncorrected references, CLI overrides, and updated calibration files.
        if store.scenarios():
            store.set_settings(store.settings())
        store.meta('camera_reference', signature)
    controller = Controller(store, stage or make_stage(mode, transport), camera, home_stage)
    sessions = {}
    attempts = {}
    auth_lock = threading.Lock()
    event_slots, camera_slots = threading.BoundedSemaphore(6), threading.BoundedSemaphore(4)
    app.extensions.update(store=store, controller=controller, camera=camera)

    def authenticated():
        sid = session.get('sid')
        with auth_lock:
            return sessions.get(sid, 0) > time.monotonic()

    def admin(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not authenticated():
                raise Problem('운영자 로그인이 필요합니다.', 401)
            return fn(*args, **kwargs)
        return wrapped

    def body():
        data = request.get_json()
        if not isinstance(data, dict):
            raise Problem('JSON 객체가 필요합니다.')
        return data

    def key():
        return request.headers.get('Idempotency-Key')

    @app.before_request
    def csrf():
        if 'csrf' not in session:
            session['csrf'] = secrets.token_hex(24)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            token = request.headers.get('X-CSRF-Token', '')
            if not secrets.compare_digest(token, session['csrf']):
                raise Problem('요청 토큰이 만료되었습니다. 새로고침하세요.', 403)

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        return response

    @app.errorhandler(Problem)
    def problem(exc):
        return jsonify(error=str(exc)), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.get('/')
    def index():
        return render_template('index.html', csrf=session['csrf'], operator=authenticated())

    @app.get('/login')
    def login_page():
        return render_template('login.html', csrf=session['csrf'], configured=bool(store.meta('password')))

    @app.get('/admin')
    def admin_page():
        if not authenticated():
            return redirect('/login')
        return render_template('admin.html', csrf=session['csrf'])

    @app.get('/api/session')
    def current_session():
        return jsonify(authenticated=authenticated(), csrf=session['csrf'])

    @app.post('/api/admin/login')
    def login():
        data = body()
        address = request.remote_addr
        with auth_lock:
            current = time.monotonic()
            for ip in list(attempts):
                if current - attempts[ip][0] > 60:
                    attempts.pop(ip)
            started, count = attempts.get(address, (current, 0))
            if count >= 8:
                raise Problem('로그인 시도가 많습니다. 1분 후 다시 시도하세요.', 429)
            attempts[address] = (started, count + 1)
        stored_hash = store.meta('password')
        password = data.get('password', '')
        if not isinstance(password, str) or len(password) > 512:
            raise Problem('비밀번호 형식이 올바르지 않습니다.')
        if not stored_hash or not check_password_hash(stored_hash, password):
            raise Problem('비밀번호를 확인하세요.', 401)
        with auth_lock:
            sessions.pop(session.get('sid'), None)
            for sid in list(sessions):
                if sessions[sid] <= time.monotonic():
                    sessions.pop(sid)
            session.clear()
            session['sid'] = secrets.token_hex(32)
            session['csrf'] = secrets.token_hex(24)
            session.permanent = True
            sessions[session['sid']] = time.monotonic() + 8 * 3600
            attempts.pop(address, None)
        return jsonify(ok=True)

    @app.post('/api/admin/logout')
    @admin
    def logout():
        with auth_lock:
            sessions.pop(session.get('sid'), None)
        session.clear()
        return jsonify(ok=True)

    @app.get('/api/state')
    def state():
        return jsonify(controller.public())

    @app.get('/api/events')
    def events():
        if not event_slots.acquire(blocking=False):
            raise Problem('상태 연결이 많습니다. 다른 시연 창을 닫아 주세요.', 503)

        def generate():
            try:
                while True:
                    state = controller.public()
                    yield f'id: {state["event_id"]}\ndata: {json.dumps(state, ensure_ascii=False)}\n\n'
                    with controller.condition:
                        controller.condition.wait(timeout=1)
            finally:
                event_slots.release()
        return Response(generate(), mimetype='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    @app.get('/camera/stream.mjpg')
    def stream():
        if not camera.health()['connected']:
            raise Problem('카메라 영상이 준비되지 않았습니다.', 503)
        if not camera_slots.acquire(blocking=False):
            raise Problem('카메라 연결이 많습니다. 다른 영상 창을 닫아 주세요.', 503)

        def generate():
            try:
                yield from camera.stream()
            finally:
                camera_slots.release()
        return Response(generate(), content_type='multipart/x-mixed-replace; boundary=FRAME')

    @app.post('/api/demo/random')
    @admin
    def random():
        return jsonify(controller.prepare(key()))

    @app.post('/api/demo/correct')
    @admin
    def correct():
        return jsonify(controller.correct(key()))

    @app.post('/api/demo/home')
    @admin
    def home():
        return jsonify(controller.home_y(key()))

    @app.post('/api/demo/stop')
    @admin
    def stop():
        return jsonify(controller.stop(key()))

    @app.get('/api/admin/state')
    @admin
    def admin_state():
        return jsonify(controller.admin_state())

    @app.post('/api/admin/probe')
    @admin
    def probe():
        return jsonify(controller.probe())

    @app.get('/api/admin/device')
    @admin
    def device():
        return jsonify(controller.admin_state())

    @app.put('/api/admin/device')
    @admin
    def configure_device():
        data = body()
        return jsonify(controller.configure_mode(data.get('mode'), data.get('transport', 'serial')))

    @app.post('/api/admin/y/<action>')
    @admin
    def manual_y(action):
        return jsonify(controller.manual_y(key(), action, body().get('y_um', 0)))

    @app.post('/api/admin/stm32')
    @admin
    def stm32_command():
        data = body()
        return jsonify(controller.console(key(), data.get('cmd'), data.get('y', 0), data.get('t', 0)))

    @app.get('/api/admin/camera')
    @admin
    def camera_config():
        return jsonify(current=camera.health(), profiles=PROFILES)

    @app.put('/api/admin/camera')
    @admin
    def configure_camera():
        data = body()
        undistort = data.get('undistort', camera.undistort)
        if (data.get('profile') not in PROFILES or type(data.get('enabled')) is not bool
                or type(undistort) is not bool):
            raise Problem('유효한 카메라 모드와 해상도를 선택하세요.')
        with controller.lock:
            controller.require_idle()
            signature = reference_signature(data['profile'], undistort)
            if signature != store.meta('camera_reference'):
                store.set_settings(store.settings())
            result = camera.configure(data['profile'], data['enabled'], undistort)
            store.meta('camera_config', json.dumps(dict(profile=data['profile'], enabled=data['enabled'], undistort=undistort)))
            store.meta('camera_reference', signature)
            return jsonify(result)

    @app.get('/api/admin/settings')
    @admin
    def settings():
        return jsonify(store.settings())

    @app.put('/api/admin/settings')
    @admin
    def save_settings():
        with controller.lock:
            return jsonify(store.set_settings(body()))

    @app.get('/api/admin/scenarios')
    @admin
    def scenarios():
        return jsonify(store.scenarios())

    @app.post('/api/admin/scenarios')
    @admin
    def create_scenario():
        with controller.lock:
            return jsonify(store.save(body())), 201

    @app.get('/api/admin/scenarios/<sid>')
    @admin
    def scenario(sid):
        return jsonify(store.scenario(sid))

    @app.put('/api/admin/scenarios/<sid>')
    @admin
    def update_scenario(sid):
        with controller.lock:
            return jsonify(store.save(body(), sid))

    @app.delete('/api/admin/scenarios/<sid>')
    @admin
    def delete_scenario(sid):
        with controller.lock:
            store.delete(sid)
        return jsonify(ok=True)

    @app.post('/api/admin/scenarios/<sid>/activate')
    @admin
    def activate(sid):
        with controller.lock:
            return jsonify(store.activate(sid, body().get('active'), controller.stage.mode))

    @app.post('/api/admin/scenarios/<sid>/test')
    @admin
    def test(sid):
        return jsonify(controller.prepare(key(), sid))

    @app.post('/api/admin/step')
    @admin
    def step():
        return jsonify(controller.correct(key(), single=True))

    @app.post('/api/admin/scenarios/<sid>/verify')
    @admin
    def verify(sid):
        data = body()
        with controller.lock:
            if data.get('confirmed') is not True:
                raise Problem('시험 결과 확인이 필요합니다.')
            if controller.stage.mode == 'hardware' and not camera.health()['connected']:
                raise Problem('실물 검증에는 실시간 카메라가 필요합니다.', 409)
            return jsonify(store.verify(sid, data.get('run_id', ''), controller.stage.mode))

    @app.get('/api/admin/runs')
    @admin
    def runs():
        return jsonify(store.runs())

    @app.get('/api/admin/export')
    @admin
    def export():
        return Response(json.dumps(dict(schema=1, settings=store.settings(), scenarios=store.scenarios()),
                                   ensure_ascii=False, indent=2), mimetype='application/json',
                        headers={'Content-Disposition': 'attachment; filename="alignment-scenarios.json"'})

    @app.post('/api/admin/import')
    @admin
    def import_data():
        from .store import validate_scenario
        data = body()
        rows = data.get('scenarios')
        if data.get('schema') != 1 or not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise Problem('버전 1 백업 파일과 1~100개 시나리오가 필요합니다.')
        # Import never silently replaces the physical reference or restores validation.
        with controller.lock:
            clean = [validate_scenario(row, store.settings()) for row in rows]
            created = [store.save(row) for row in clean]
        return jsonify(imported=len(created), message='현재 중앙 기준으로 초안을 가져왔습니다. 재검증하세요.')

    return app
