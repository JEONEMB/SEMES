'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(d);
  let state = null, events = null, videoRetry = null, settingsDirty = false, camOk = false;

  // ---------- API ----------
  async function api(path, body) {
    const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error) throw new Error(data.error || `요청 실패 (${r.status})`);
    return data;
  }
  function toast(msg, info) {
    const t = $('toast'); t.textContent = msg; t.className = 'toast' + (info ? ' info' : ''); t.hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, info ? 2500 : 6000);
  }
  const act = (id, path, body) => $(id).addEventListener('click', () => api(path, typeof body === 'function' ? body() : body).catch(e => toast(e.message)));

  // ---------- 영상 ----------
  function startVideo() {
    clearTimeout(videoRetry);
    const img = $('video');
    img.src = '/video?t=' + Date.now();
  }
  // 카메라가 정상일 때만 다시 붙는다 (안 되는 동안 요청을 쌓지 않도록). 살아나면 render 에서 다시 시작.
  const retryVideo = () => { clearTimeout(videoRetry); if (camOk) videoRetry = setTimeout(startVideo, 3000); };
  $('video').addEventListener('error', retryVideo);
  $('video').addEventListener('abort', retryVideo);
  $('video').addEventListener('load', () => { $('video-empty').hidden = true; });
  document.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', () => setView({zoom: +b.dataset.zoom}, b)));
  document.querySelectorAll('[data-center]').forEach(b => b.addEventListener('click', () => setView({center: b.dataset.center}, b)));
  const view = {zoom: 1, center: 'image', mode: 'normal'};
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setView({mode: b.dataset.mode}, b)));
  let showOutline = true;
  $('btn-outline').addEventListener('click', () => {
    showOutline = !showOutline; $('btn-outline').classList.toggle('on', showOutline);
    $('btn-outline').setAttribute('aria-pressed', showOutline); if (state) drawOverlay(state);
  });
  // 센터 기준 윤곽: 기준 사진의 QR 3개 테두리 + 가운데 십자 + 중심 잇는 선. 색 = 마지막 측정의 정렬 여부
  function drawOverlay(s) {
    const svg = $('ref-overlay'), badge = $('center-badge'), ov = s.overlay || {};
    const quads = ov.ref_qr, cls = s.aligned === true ? 'ok' : s.aligned === false ? 'bad' : 'unknown';
    badge.hidden = !quads;
    if (!quads) {
      svg.innerHTML = '';
      if ((s.camera || {}).view_mode === 'diff') { badge.hidden = false; badge.className = 'center-badge unknown'; badge.textContent = '차이 영상: 센터 기준을 먼저 잡으세요'; }
      return;
    }
    badge.className = 'center-badge ' + cls;
    badge.textContent = (s.camera || {}).view_mode === 'diff' ? '차이 영상 · ' : '';
    badge.textContent += cls === 'ok' ? '센터 OK' : cls === 'bad' ? '센터 벗어남' : (s.moving ? '이동 중' : '측정 전');
    if (!showOutline) { svg.innerHTML = ''; return; }
    const c = quads.map(q => [q.reduce((a, p) => a + p[0], 0) / 4, q.reduce((a, p) => a + p[1], 0) / 4]);
    const side = Math.hypot(quads[1][1][0] - quads[1][0][0], quads[1][1][1] - quads[1][0][1]);
    const [mx, my] = c[1], arm = side * 0.9;
    svg.innerHTML = quads.map(q => `<polygon class="${cls}" points="${q.map(p => p.join(',')).join(' ')}"/>`).join('') +
      `<line class="${cls}" x1="${c[0][0]}" y1="${c[0][1]}" x2="${c[2][0]}" y2="${c[2][1]}"/>` +
      `<line class="${cls}" x1="${mx - arm}" y1="${my}" x2="${mx + arm}" y2="${my}"/>` +
      `<line class="${cls}" x1="${mx}" y1="${my - arm}" x2="${mx}" y2="${my + arm}"/>`;
  }
  function setView(change, btn) {
    Object.assign(view, change);
    btn.parentElement.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === btn));
    api('/api/view', view).catch(e => toast(e.message));
  }

  // ---------- 렌더 ----------
  function cell(id, v, d, cls) { const c = $(id); c.textContent = fmt(v, d); c.className = cls || ''; }

  function render(s) {
    state = s;
    const st = s.settings, run = s.run, last = s.last || s.measure;
    // 상단 표시
    const dev = s.devices || {};
    const pStage = $('pill-stage');
    pStage.textContent = s.phase === 'released' ? '데모 웹 사용 중' :
      `스테이지 ${dev.stage_mode === 'mock' ? '모의' : dev.stage_mode === 'hardware' ? '실물 (' + dev.stage_transport + ')' : '없음'}`;
    pStage.className = 'pill' + (dev.stage_mode === 'mock' ? ' warn' : (!dev.ready || s.stage_error) ? ' bad' : '');
    const a = s.algorithm || {};
    $('pill-algo').textContent = `알고리즘 ${a.kind === 'mock' ? '모의' : a.detail || a.kind} · ${a.output === 'command' ? '명령값' : '오프셋'}`;
    $('pill-algo').className = 'pill' + (a.kind === 'mock' ? ' warn' : a.error ? ' bad' : '');
    const cam = s.camera || {};
    $('pill-camera').textContent = cam.ok ? `카메라 ${cam.kind === 'fake' ? '가짜' : ''} ${cam.fps} fps` : '카메라 없음';
    $('pill-camera').className = 'pill' + (cam.ok ? (cam.kind === 'fake' ? ' warn' : '') : ' bad');
    $('cam-info').textContent = cam.ok ? `${cam.fps} fps · ${cam.view_zoom}×` : '';
    const empty = !cam.ok;
    if (cam.ok && !camOk) { camOk = true; startVideo(); }       // 카메라가 살아남
    else if (!cam.ok && camOk) { camOk = false; clearTimeout(videoRetry); $('video').removeAttribute('src'); }
    $('video-empty').hidden = !empty;
    $('video-empty-msg').textContent = s.phase === 'released' ? '데모 웹이 카메라를 쓰는 중입니다.' : (cam.error || '카메라 연결을 기다리는 중');
    drawOverlay(s);
    if (s.overlay) {
      const [px, py] = s.overlay.plate, [vw, vh] = s.overlay.view_size;
      const m = $('plate-mark'); m.style.left = (px / vw * 100) + '%'; m.style.top = (py / vh * 100) + '%';
      m.hidden = px < 0 || py < 0 || px > vw || py > vh;
    }
    // 이동 표시
    const moving = !!s.moving || s.stage_state === 'moving' || s.stage_state === 'homing';
    const badge = $('motion-badge'); badge.dataset.moving = moving;
    badge.querySelector('span').textContent = moving ? (s.phase === 'homing' ? '원점 복귀 중' : '모터 이동 중') : '정지';
    // 회차
    const max = run ? run.max_trials : st.max_trials;
    $('trial').textContent = String(run ? run.trial : 0).padStart(2, '0');
    $('trial-max').textContent = max; $('btn-max').textContent = st.max_trials;
    $('trial-bar').max = max; $('trial-bar').value = run ? run.trial : 0;
    const res = $('result');
    if (run && run.result) {
      res.hidden = false; res.textContent = run.result_label;
      const truthOk = last && last.truth_within;
      res.className = 'result ' + (run.result === 'converged' ? (truthOk === false ? 'warn' : 'ok') : 'fail');
      if (run.result === 'converged' && truthOk === false) res.textContent += ' · 실제는 허용 밖!';
    } else res.hidden = true;
    // 단계
    document.querySelectorAll('#steps li').forEach(li => li.classList.toggle('on', li.dataset.phase === s.phase));
    $('phase-box').dataset.phase = s.phase;
    $('phase').textContent = s.phase_label + (s.job && s.phase === 'idle' ? '' : '');
    $('message').textContent = s.message || (s.stage_error ? '스테이지: ' + s.stage_error : '');
    // 값
    const tolY = st.tol_y_mm, tolT = st.tol_t_deg;
    if (last) {
      const ok = (v, tol) => v === null || v === undefined ? '' : Math.abs(v) <= tol ? 'good' : '';
      cell('inf-y', last.offset.y_mm, 3, ok(last.offset.y_mm, tolY));
      cell('inf-t', last.offset.t_deg, 3, st.enable_t ? ok(last.offset.t_deg, tolT) : '');
      cell('tru-y', last.truth && last.truth.y_mm, 3, last.truth ? ok(last.truth.y_mm, tolY) : '');
      cell('tru-t', last.truth && last.truth.t_deg, 3);
      cell('err-y', last.error && last.error.y_mm, 3, last.error && Math.abs(last.error.y_mm) > tolY ? 'bad' : '');
      cell('err-t', last.error && last.error.t_deg, 3, last.error && Math.abs(last.error.t_deg) > tolT ? 'bad' : '');
      cell('cmd-y', last.command.y_mm, 3); cell('cmd-t', st.enable_t ? last.command.t_deg : 0, 3);
      $('values-note').textContent = `${last.label} · 추론 ${last.infer_ms} ms · 흔들림 대기 ${last.settle_s} s` +
        (last.settled ? '' : ' (안정 판정 실패, 시간 초과)') + (st.enable_t ? '' : ' · Θ 명령은 꺼짐(0 전송)');
    }
    // 위치
    $('pos-y').textContent = s.position ? s.position.y_mm.toFixed(3) : '—';
    $('pos-t').textContent = s.position ? s.position.t_deg.toFixed(3) : '—';
    const f = s.flags;
    $('flags').innerHTML = f === null || f === undefined ? '' :
      [['Y 원점', 1, 'on'], ['Θ 원점', 2, 'on'], ['Y− 리밋', 4, 'alert'], ['Y+ 리밋', 8, 'alert']]
        .map(([n, bit, cls]) => `<span class="${f & bit ? cls : ''}">${n}</span>`).join('');
    $('center').textContent = `Y ${st.center_y_mm.toFixed(3)} mm · Θ ${st.center_t_deg.toFixed(3)}°`;
    $('tol').textContent = `±${tolY} mm` + (st.enable_t ? ` · ±${tolT}°` : '');
    const ref = s.reference;
    $('refinfo').textContent = ref ? `${new Date(ref.ts * 1000).toLocaleTimeString()} 등록 (학습 영점 대비 Y ${fmt(ref.zero_y_mm * 1000, 1)} µm, Θ ${fmt(ref.zero_t_deg, 4)}°)` : '없음 — 학습 영점 사용';
    // 버튼
    const ready = dev.ready && s.phase !== 'released';
    ['btn-home', 'btn-reference', 'btn-goto-random', 'btn-goto', 'btn-run', 'btn-step', 'btn-measure', 'btn-center']
      .forEach(id => { $(id).disabled = !ready || s.busy; });
    $('btn-stop').disabled = !ready;
    // 기록
    $('run-id').textContent = run ? `실행 ${run.id}` + (run.start_offset ? ` · 시작 센터+(${fmt(run.start_offset[0])} mm, ${fmt(run.start_offset[1])}°)` : '') : '';
    renderHistory(s.history || [], st);
    if (!settingsDirty) fillSettings(st);
  }

  function renderHistory(h, st) {
    const body = $('hist-body');
    if (!h.length) { body.innerHTML = '<tr><td colspan="11" class="muted">아직 기록이 없습니다.</td></tr>'; drawChart([], st); return; }
    body.innerHTML = h.slice().reverse().map((e, i) => {
      const verdict = e.within ? (e.truth_within === false ? '<span class="tag warn">추론만 OK</span>' : '<span class="tag ok">완료</span>') : '<span class="tag no">이동</span>';
      const t = e.truth || {}, er = e.error || {};
      return `<tr class="${i === 0 ? 'last' : ''}"><td>${e.label}</td><td>${fmt(e.offset.y_mm)}</td><td>${fmt(t.y_mm)}</td><td>${fmt(er.y_mm)}</td>` +
        `<td>${fmt(e.offset.t_deg)}</td><td>${fmt(t.t_deg)}</td><td>${fmt(er.t_deg)}</td><td>${fmt(e.command.y_mm)}</td>` +
        `<td>${e.infer_ms}</td><td>${e.settle_s}${e.settled ? '' : '!'}</td><td>${verdict}</td></tr>`;
    }).join('');
    drawChart(h, st);
  }

  function drawChart(h, st) {
    const svg = $('chart'), W = 320, H = 60;
    const vals = h.flatMap(e => [Math.abs(e.offset.y_mm), e.truth ? Math.abs(e.truth.y_mm) : 0]);
    const top = Math.max(st.tol_y_mm * 3, ...vals, 1e-6);
    const x = i => h.length < 2 ? W / 2 : 6 + i * (W - 12) / (h.length - 1);
    const y = v => H - 4 - Math.min(v, top) / top * (H - 8);
    const line = (get, color) => `<polyline fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke" points="${h.map((e, i) => `${x(i)},${y(get(e))}`).join(' ')}"/>`;
    svg.innerHTML = `<rect x="0" y="${y(st.tol_y_mm)}" width="${W}" height="${H - 4 - y(st.tol_y_mm)}" fill="#e8f6ee"/>` +
      (h.length ? line(e => Math.abs(e.offset.y_mm), '#0A3D91') + line(e => e.truth ? Math.abs(e.truth.y_mm) : 0, '#b06f0c') : '');
  }

  // ---------- 설정 ----------
  const form = $('settings-form');
  function fillSettings(st) {
    for (const el of form.elements) {
      if (!el.name || !(el.name in st)) continue;
      if (el.type === 'checkbox') el.checked = !!st[el.name]; else el.value = st[el.name];
    }
  }
  form.addEventListener('input', () => { settingsDirty = true; });
  form.addEventListener('submit', ev => {
    ev.preventDefault();
    const body = {};
    for (const el of form.elements) {
      if (!el.name) continue;
      body[el.name] = el.type === 'checkbox' ? el.checked : Number(el.value);
    }
    api('/api/settings', body).then(() => { settingsDirty = false; toast('설정 저장', true); }).catch(e => toast(e.message));
  });

  // ---------- 버튼 ----------
  const startOffset = () => [Number($('start-y').value), Number($('start-t').value)];
  act('btn-home', '/api/home');
  act('btn-goto-random', '/api/goto', {offset: 'random'});
  act('btn-goto', '/api/goto', () => ({offset: startOffset()}));
  act('btn-run', '/api/run', {});
  act('btn-step', '/api/step');
  act('btn-measure', '/api/measure');
  act('btn-reference', '/api/reference');
  act('btn-stop', '/api/stop');
  act('btn-center', '/api/center-here');

  // ---------- 데모 웹 넘김 / 되찾기 ----------
  function overlay(text) { $('overlay-text').textContent = text; $('overlay').hidden = !text; }

  $('btn-demo').addEventListener('click', async () => {
    overlay('데모 웹으로 전환 중…');
    if (events) events.close();
    $('video').removeAttribute('src');
    try {
      const r = await api('/api/handoff');
      location.href = `${location.protocol}//${location.hostname}:${r.port}/`;
    } catch (e) {
      overlay(''); toast(e.message); connect();
    }
  });

  function connect() {
    if (events) events.close();
    events = new EventSource('/api/events');
    events.onmessage = e => render(JSON.parse(e.data));
    events.onerror = () => { $('pill-stage').textContent = '서버 재연결 중'; };
    camOk = false;      // 첫 상태에서 카메라가 정상이면 영상 시작
  }

  async function boot() {
    overlay('장치 확인 중…');
    try {
      const r = await api('/api/reclaim');
      if (!r.ok) toast('장치 연결 실패: ' + (r.error || ''));
    } catch (e) { toast(e.message); }
    overlay('');
    connect();
  }
  // 뒤로 가기로 돌아왔을 때(bfcache)도 장치를 다시 잡는다
  window.addEventListener('pageshow', ev => { if (ev.persisted) boot(); });
  boot();
})();
