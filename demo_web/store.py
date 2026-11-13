"""SQLite storage. Every execution keeps an immutable scenario/reference snapshot."""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


class Problem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat()


def integer(value, label, low=-(2**31), high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        raise Problem(f'{label}: {low}~{high} 범위의 정수가 필요합니다.')
    return value


def validate_scenario(data, settings, limits=None):
    if not isinstance(data, dict):
        raise Problem('시나리오 형식이 올바르지 않습니다.')
    name = str(data.get('name', '')).strip()
    if not name or len(name) > 80:
        raise Problem('시나리오 이름은 1~80자로 입력하세요.')
    description = str(data.get('description', ''))
    if len(description) > 2000:
        raise Problem('설명은 2,000자 이하로 입력하세요.')
    result = dict(name=name, description=description,
                  order=integer(data.get('order', 0), '재생 순서', 0, 9999),
                  start_y_um=integer(data.get('start_y_um'), '시작 Y'),
                  start_t_mdeg=integer(data.get('start_t_mdeg'), '시작 T'), steps=[])
    steps = data.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 20:
        raise Problem('보정 단계는 1~20개여야 합니다.')
    y, t = result['start_y_um'], result['start_t_mdeg']
    positions = [(y, t)]
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise Problem('단계 형식이 올바르지 않습니다.')
        dy = integer(step.get('y_um'), f'{i}단계 Y')
        dt = integer(step.get('t_mdeg'), f'{i}단계 T')
        if dy == dt == 0:
            raise Problem(f'{i}단계: Y와 T가 모두 0인 보정은 저장할 수 없습니다.')
        delay = integer(step.get('settle_ms', 300), '안정화 시간(ms)', 0, 10000)
        result['steps'].append(dict(y_um=dy, t_mdeg=dt, settle_ms=delay))
        y = integer(y + dy, '예상 Y 위치')
        t = integer(t + dt, '예상 T 위치')
        positions.append((y, t))
    if limits:
        if any(not (limits[0] <= y <= limits[1] and limits[2] <= t <= limits[3])
               for y, t in positions):
            raise Problem('시작 또는 보정 경로가 장치 허용 범위를 벗어납니다.')
    if not at_center(positions[-1], settings):
        raise Problem('마지막 예상 위치가 중앙 기준의 허용 오차 밖입니다.')
    return result


def at_center(position, settings):
    return (abs(position[0] - settings['center_y_um']) <= settings['tolerance_y_um']
            and abs(position[1] - settings['center_t_mdeg']) <= settings['tolerance_t_mdeg'])


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scenarios (
              id TEXT PRIMARY KEY, body TEXT NOT NULL, revision INTEGER NOT NULL,
              active INTEGER NOT NULL DEFAULT 0, verified_mode TEXT, verified_run TEXT,
              reference_revision INTEGER, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, state TEXT NOT NULL,
              started TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, action TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        default = dict(center_y_um=0, center_t_mdeg=0, tolerance_y_um=10,
                       tolerance_t_mdeg=10, reference_note='', revision=1)
        self.db.execute('INSERT OR IGNORE INTO settings VALUES (1, ?)', (json.dumps(default),))
        self.db.commit()
        # A restart must not resurrect a motion or leave a test eligible for verification.
        with self.lock, self.db:
            for row in self.db.execute('SELECT id, state FROM runs').fetchall():
                state = json.loads(row['state'])
                if state['status'] in ('positioning', 'ready', 'correcting', 'stopping', 'manual_move', 'homing'):
                    state.update(status='interrupted', message='서버 재시작으로 중단됨. 장치 상태를 확인하세요.')
                    self.db.execute('UPDATE runs SET state=?, updated=? WHERE id=?',
                                    (json.dumps(state), now(), row['id']))

    def settings(self):
        with self.lock:
            return json.loads(self.db.execute('SELECT body FROM settings WHERE id=1').fetchone()[0])

    def set_settings(self, data):
        if not isinstance(data, dict):
            raise Problem('기준 설정 형식이 올바르지 않습니다.')
        with self.lock, self.db:
            old = self.settings()
            if data.get('revision') != old['revision']:
                raise Problem('기준 설정이 변경되었습니다. 새로고침하세요.', 409)
            new = {key: integer(data.get(key), key) for key in ('center_y_um', 'center_t_mdeg')}
            for key in ('tolerance_y_um', 'tolerance_t_mdeg'):
                new[key] = integer(data.get(key), key, 0, 100000)
            new['reference_note'] = str(data.get('reference_note', ''))[:2000]
            new['revision'] = old['revision'] + 1
            self.db.execute('UPDATE settings SET body=? WHERE id=1', (json.dumps(new),))
            self.db.execute('UPDATE scenarios SET active=0, verified_mode=NULL, verified_run=NULL')
            return new

    @staticmethod
    def scenario_row(row):
        return {**json.loads(row['body']), **{k: row[k] for k in
                ('id', 'revision', 'active', 'verified_mode', 'verified_run', 'reference_revision', 'updated')}}

    def scenarios(self):
        with self.lock:
            rows = [self.scenario_row(r) for r in self.db.execute('SELECT * FROM scenarios')]
            return sorted(rows, key=lambda s: (s['order'], s['name'], s['id']))

    def scenario(self, sid):
        with self.lock:
            row = self.db.execute('SELECT * FROM scenarios WHERE id=?', (sid,)).fetchone()
            if not row:
                raise Problem('시나리오를 찾을 수 없습니다.', 404)
            return self.scenario_row(row)

    def save(self, data, sid=None):
        with self.lock, self.db:
            clean = validate_scenario(data, self.settings())
            if sid:
                old = self.scenario(sid)
                if data.get('revision') != old['revision']:
                    raise Problem('다른 창에서 수정되었습니다. 다시 불러오세요.', 409)
                rev = old['revision'] + 1
                self.db.execute('''UPDATE scenarios SET body=?, revision=?, active=0,
                  verified_mode=NULL, verified_run=NULL, reference_revision=NULL, updated=? WHERE id=?''',
                                (json.dumps(clean), rev, now(), sid))
            else:
                sid = uuid.uuid4().hex
                self.db.execute('INSERT INTO scenarios(id,body,revision,updated) VALUES (?,?,1,?)',
                                (sid, json.dumps(clean), now()))
            return self.scenario(sid)

    def delete(self, sid):
        with self.lock, self.db:
            self.scenario(sid)
            self.db.execute('DELETE FROM scenarios WHERE id=?', (sid,))

    def activate(self, sid, active, mode):
        if type(active) is not bool:
            raise Problem('활성화 여부는 true/false여야 합니다.')
        with self.lock, self.db:
            s = self.scenario(sid)
            if active and (s['verified_mode'] != mode or
                           s['reference_revision'] != self.settings()['revision']):
                raise Problem('현재 모드와 중앙 기준으로 먼저 시험·검증하세요.', 409)
            self.db.execute('UPDATE scenarios SET active=? WHERE id=?', (int(active), sid))
            return self.scenario(sid)

    def verify(self, sid, run_id, mode):
        with self.lock, self.db:
            s = self.scenario(sid)
            run = self.run(run_id)
            snap = run['snapshot']
            if (not snap['test'] or snap['mode'] != mode or run['state']['status'] != 'complete'
                    or snap['scenario']['id'] != sid or snap['scenario']['revision'] != s['revision']
                    or snap['settings']['revision'] != self.settings()['revision']):
                raise Problem('현재 버전의 성공한 관리자 시험 실행이 필요합니다.', 409)
            self.db.execute('''UPDATE scenarios SET verified_mode=?, verified_run=?,
                reference_revision=? WHERE id=?''', (mode, run_id, self.settings()['revision'], sid))
            return self.scenario(sid)

    def save_run(self, run_id, snapshot, state):
        with self.lock, self.db:
            self.db.execute('''INSERT INTO runs VALUES (?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET state=excluded.state, updated=excluded.updated''',
                            (run_id, json.dumps(snapshot), json.dumps(state), now(), now()))

    def run(self, rid):
        with self.lock:
            row = self.db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone()
            if not row:
                raise Problem('실행 기록이 없습니다.', 404)
            return {**dict(row), 'snapshot': json.loads(row['snapshot']), 'state': json.loads(row['state'])}

    def runs(self):
        with self.lock:
            ids = self.db.execute('SELECT id FROM runs ORDER BY started DESC LIMIT 30').fetchall()
            return [self.run(r[0]) for r in ids]

    def claim_request(self, key, action):
        if not isinstance(key, str) or not 8 <= len(key) <= 100:
            raise Problem('유효한 Idempotency-Key가 필요합니다.')
        with self.lock, self.db:
            previous = self.db.execute('SELECT action FROM requests WHERE key=?', (key,)).fetchone()
            if previous:
                if previous[0] != action:
                    raise Problem('다른 요청에 사용된 중복 방지 키입니다.', 409)
                return False
            self.db.execute('INSERT INTO requests VALUES (?,?)', (key, action))
            return True

    def meta(self, key, value=None):
        with self.lock, self.db:
            if value is not None:
                self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, value))
            row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            return row[0] if row else None
