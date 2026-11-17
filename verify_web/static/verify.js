'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (v >= 0 ? '+' : '−') + Math.abs(Number(v)).toFixed(d);
  const esc = v => String(v);
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
  function startVideo() { clearTimeout(videoRetry); $('video').src = '/video?t=' + Date.now(); }
  // 카메라가 정상일 때만 다시 붙는다 (안 되는 동안 요청을 쌓지 않도록). 살아나면 render 에서 다시 시작.
  const retryVideo = () => { clearTimeout(videoRetry); if (camOk) videoRetry = setTimeout(startVideo, 3000); };
  $('video').addEventListener('error', retryVideo);
  $('video').addEventListener('abort', retryVideo);
  $('video').addEventListener('load', () => { $('video-empty').hidden = true; });
  const view = {zoom: 1, center: 'plate', mode: 'normal'};     // 확대는 QR(판) 중심, 1.7× 는 QR+나사 전체의 가운데
  document.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', () => setView({zoom: +b.dataset.zoom, center: b.dataset.center || 'plate'}, b)));
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setView({mode: b.dataset.mode}, b)));
  function setView(change, btn) {
    Object.assign(view, change);
    btn.parentElement.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === btn));
    api('/api/view', view).catch(e => toast(e.message));
  }

  // 목표(점선) = 센터 기준 사진의 QR 3개 + 나사 2개, 지금(실선) = 마지막 측정. 색 = 정렬 여부
  function drawOverlay(s) {
    const svg = $('ref-overlay'), badge = $('center-badge'), ov = s.overlay || {};
    const cls = s.aligned === true ? 'ok' : s.aligned === false ? 'bad' : 'unknown';
    const diff = (s.camera || {}).view_mode === 'diff';
    if (!ov.target) {
      svg.innerHTML = '';
      badge.hidden = false; badge.className = 'center-badge unknown';
      badge.textContent = '센터 기준 없음';
      return;
    }
    badge.hidden = false;
    badge.className = 'center-badge ' + cls;
    badge.textContent = (diff ? '차이 영상 · ' : '') +
      (cls === 'ok' ? '센터 OK' : cls === 'bad' ? '센터에서 벗어남' : (s.moving ? '이동 중…' : '측정 전'));
    const poly = (q, c) => `<polygon class="${c}" points="${q.map(p => p.join(',')).join(' ')}"/>`;
    const circ = (k, c) => `<circle class="${c}" cx="${k.c[0]}" cy="${k.c[1]}" r="${Math.max(k.r, 6)}"/>`;
    const t = ov.target;
    let out = t.qr.map(q => poly(q, 'target')).join('') + t.screws.map(k => circ(k, 'target')).join('');
    // 가운데 QR 중심 십자 (목표)
    if (t.qr.length === 3) {
      const q = t.qr[1], mx = q.reduce((a, p) => a + p[0], 0) / 4, my = q.reduce((a, p) => a + p[1], 0) / 4;
      const arm = Math.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]) * 0.25;
      out += `<line class="target" x1="${mx - arm}" y1="${my}" x2="${mx + arm}" y2="${my}"/><line class="target" x1="${mx}" y1="${my - arm}" x2="${mx}" y2="${my + arm}"/>`;
    }
    if (ov.now) {
      out += ov.now.qr.map(q => poly(q, 'now ' + cls)).join('') + ov.now.screws.map(k => circ(k, 'now ' + cls)).join('');
      out += ov.now.dots.map(p => `<circle class="dot ${cls}" cx="${p[0]}" cy="${p[1]}" r="3"/>`).join('');
      // 나사: 목표 → 지금 잇는 선 (어긋난 방향)
      t.screws.forEach(k => {
        const n = ov.now.screws.find(x => x.name === k.name);
        if (n) out += `<line class="link ${cls}" x1="${k.c[0]}" y1="${k.c[1]}" x2="${n.c[0]}" y2="${n.c[1]}"/>`;
      });
    }
    svg.innerHTML = out;
  }

  // ---------- Y·Θ 지도 ----------
  // 가운데(허용 범위)를 크게 보이게 대칭 로그 눈금: f(v) = sign·log(1+|v|/c)/log(1+R/c), c = 허용값
  function drawPlot(s) {
    const st = s.settings, svg = $('plot'), box = svg.getBoundingClientRect();
    // 실제 크기로 그려서 글자가 커지지 않게
    const W = Math.max(240, Math.round(box.width) || 400), H = Math.max(200, Math.round(box.height) || 300), L = 52, R = 14, T = 24, B = 36;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const cy = st.tol_y_mm, ct = st.tol_t_deg;
    const hist = s.history || [];
    const goal = s.moving ? moveGoal(s) : null;       // 지도는 카메라로 본 위치만 (스테이지 좌표는 백래시를 몰라 안 그림)
    const ys = [st.random_y_mm, ...hist.map(e => Math.abs(e.offset.y_mm)), goal ? Math.abs(goal[0]) : 0];
    const ts = [st.random_t_deg, ...hist.map(e => Math.abs(e.offset.t_deg)), goal ? Math.abs(goal[1]) : 0];
    const Ry = Math.max(...ys) * 1.08, Rt = Math.max(...ts) * 1.08;
    const f = (v, c, Rm) => Math.sign(v) * Math.log1p(Math.min(Math.abs(v), Rm) / c) / Math.log1p(Rm / c);
    const X = v => L + (f(v, cy, Ry) + 1) / 2 * (W - L - R);
    const Y = v => T + (1 - (f(v, ct, Rt) + 1) / 2) * (H - T - B);
    const tick = (v, d) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
    let g = '';
    const yt = [0, cy, 0.2, Ry / 1.08].filter((v, i, a) => v <= Ry && a.indexOf(v) === i && (i < 2 || v > cy * 2.5));
    const tt = [0, ct, 0.05, Rt / 1.08].filter((v, i, a) => v <= Rt && a.indexOf(v) === i && (i < 2 || v > ct * 2.5));
    for (const v of yt) for (const sg of v ? [-1, 1] : [1]) {
      const x = X(sg * v);
      g += `<line class="grid" x1="${x}" y1="${T}" x2="${x}" y2="${H - B}"/><text class="tk" x="${x}" y="${H - B + 15}" text-anchor="middle">${tick(sg * v, v < 0.1 ? 3 : 2)}</text>`;
    }
    for (const v of tt) for (const sg of v ? [-1, 1] : [1]) {
      const y = Y(sg * v);
      g += `<line class="grid" x1="${L}" y1="${y}" x2="${W - R}" y2="${y}"/><text class="tk" x="${L - 4}" y="${y + 3}" text-anchor="end">${tick(sg * v, v < 0.05 ? 3 : 2)}</text>`;
    }
    g += `<rect class="tolbox" x="${X(-cy)}" y="${Y(ct)}" width="${X(cy) - X(-cy)}" height="${Y(-ct) - Y(ct)}"/>`;
    g += `<line class="axis" x1="${X(0)}" y1="${T}" x2="${X(0)}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${Y(0)}" x2="${W - R}" y2="${Y(0)}"/>`;
    g += `<text class="ax-l" x="${W - R}" y="${H - 6}" text-anchor="end">Y (mm)</text><text class="ax-l" x="4" y="11">Θ (°)</text>`;
    // 카메라 판단 경로
    if (hist.length) {
      g += `<polyline class="cam-path" points="${hist.map(e => `${X(e.offset.y_mm)},${Y(e.offset.t_deg)}`).join(' ')}"/>`;
      hist.forEach((e, i) => {
        const x = X(e.offset.y_mm), y = Y(e.offset.t_deg), last = i === hist.length - 1;
        g += `<circle class="cam-pt${e.within ? ' ok' : ''}${last ? ' last' : ''}" cx="${x}" cy="${y}" r="${last ? 7 : 5.5}"/>` +
          `<text class="pt-n" x="${x}" y="${y + 3}" text-anchor="middle">${e.trial ? e.trial - 1 : ''}</text>`;
      });
    }
    if (goal && s.moving) g += `<g class="goal"><line x1="${X(goal[0]) - 6}" y1="${Y(goal[1]) - 6}" x2="${X(goal[0]) + 6}" y2="${Y(goal[1]) + 6}"/><line x1="${X(goal[0]) - 6}" y1="${Y(goal[1]) + 6}" x2="${X(goal[0]) + 6}" y2="${Y(goal[1]) - 6}"/></g>`;
    svg.innerHTML = g;
  }

  window.addEventListener('resize', () => { if (state) drawPlot(state); });
  // 지도 칸 크기가 바뀌면 (칸 높이를 남는 만큼 채우므로) 다시 그린다
  if (window.ResizeObserver) new ResizeObserver(() => { if (state) drawPlot(state); }).observe($('plot-box'));

  // ---------- 위치 ----------
  function stageOffset(s) {
    if (!s.position) return null;
    return [s.position.y_mm - s.settings.center_y_mm, s.position.t_deg - s.settings.center_t_deg];
  }
  function moveGoal(s) {          // 최종 목표 (접근점을 거치는 중이면 그다음 목표)
    if (!s.target || s.target.relative) return null;
    const g = s.target.final || s.target;
    return [g.y_mm - s.settings.center_y_mm, g.t_deg - s.settings.center_t_deg];
  }

  function lastSeen(s) {
    // 지금 자리에서 카메라로 잰 값 (움직이면 aligned 가 null 로 바뀌어 무효)
    if (s.moving || s.aligned === null || s.aligned === undefined) return null;
    const a = s.last, b = s.measure;
    const e = a && b ? (a.ts > b.ts ? a : b) : (a || b);
    if (e && (!s.reference || e.ts >= s.reference.ts)) return e;
    return s.reference ? {label: '센터 기준', offset: {y_mm: 0, t_deg: 0}, within: true} : null;
  }
  function renderPosition(s) {
    const st = s.settings, off = stageOffset(s), card = $('pos-card'), seen = lastSeen(s);
    const v2 = seen ? [seen.offset.y_mm, seen.offset.t_deg] : off;
    $('off-y').textContent = v2 ? fmt(v2[0]) : '—';
    $('off-t').textContent = v2 ? fmt(v2[1]) : '—';
    $('pos-label').textContent = seen ? `센터와의 차이 · ${seen.label}` : '센터와의 차이 · 스테이지 좌표';
    card.dataset.ok = !v2 ? 'unknown' : s.moving ? 'moving' : seen ? (seen.within ? 'ok' : 'bad') : 'unknown';
    $('pos-sub').textContent = '';
    const v = $('verdict');
    const run = s.run;
    v.hidden = !!s.moving;                // 이동 중에는 게이지·남은 시간이 대신 보여 준다
    if (s.moving) { v.className = 'verdict moving'; }
    else if (s.aligned === true) { v.textContent = `센터 OK · 허용 ±${st.tol_y_mm} mm, ±${st.tol_t_deg}° 안`; v.className = 'verdict ok'; }
    else if (s.aligned === false) { v.textContent = '센터에서 벗어남'; v.className = 'verdict bad'; }
    else { v.textContent = '카메라로 아직 안 봄'; v.className = 'verdict'; }
    if (run && run.result && !s.moving && s.aligned !== null) {
      v.textContent = run.result === 'converged' ? `✓ ${run.result_label}` : run.result_label;
    }
  }

  // ---------- 흐름 (1 → 2 → 3) ----------
  function flowStep(s) {
    return s.start && s.start.reached ? 2 : 1;      // 랜덤 위치에 있으면 다음은 보정
  }
  function renderFlow(s) {
    const next = s.busy ? 0 : flowStep(s), jobStep = {goto: 1, run: 2, step: 2}[s.job] || 0;
    document.querySelectorAll('.flow-btn[data-step], .step-card[data-step]').forEach(b => {
      const n = +b.dataset.step;
      b.classList.toggle('next', n === next);
      b.classList.toggle('active', n === jobStep);
    });
    const conv = !!(s.run && s.run.result === 'converged');
    const done = {1: !!(s.start && s.start.reached) || !!(s.run && s.run.start_offset), 2: conv, 3: conv && s.aligned === true};
    document.querySelectorAll('#flow-steps li').forEach(li => {
      const n = +li.dataset.step;
      li.className = n === jobStep ? 'on' : done[n] ? 'done' : n === next ? 'next' : '';
    });
    const st = s.settings;
    ['start-y', 'start-t'].forEach(id => { $(id).disabled = s.busy; });
    $('start-t').closest('label').hidden = !st.enable_t;
    document.querySelectorAll('[data-slow]').forEach(b => b.classList.toggle('on', String(st.demo_slow) === b.dataset.slow));
  }
  document.querySelectorAll('[data-slow]').forEach(b => b.addEventListener('click', () =>
    api('/api/settings', {demo_slow: b.dataset.slow === 'true'}).catch(e => toast(e.message))));

  // ---------- 축별 이동 표시 ----------
  // 이동 중에는 STM32 에 위치를 묻지 않으므로, 서버가 준 이동량÷속도(예상 시간)로 어느 축이 도는지,
  // 지금 어디쯤인지(예상값)를 보여 준다. 도착하면 다시 STM32·카메라 값으로 바뀐다.
  let clockSkew = 0;
  function axisInfo(ax) {
    const s = state, tg = s && s.target, a = tg && tg.axes;
    if (!s || !s.moving || !a) return null;
    const total = ax === 'y' ? a.y_s : a.t_s, d = ax === 'y' ? a.dy_mm : a.dt_deg;
    if (!total) return {on: false, d, left: 0, frac: 1};
    const el = Date.now() / 1000 + clockSkew - a.started;
    return {on: el < total + 0.3, d, left: Math.max(0, total - el), frac: Math.min(1, el / total)};
  }
  function motion(s) {
    // 이번 이동의 출발·목표·예상 위치 (모두 센터와의 차이). 접근점을 거치면 출발은 처음 자리, 목표는 최종 목표
    const tg = s && s.target, a = tg && tg.axes;
    if (!s || !s.moving || !tg || !a || s.phase === 'homing') return null;
    const c = [s.settings.center_y_mm, s.settings.center_t_deg], d = [a.dy_mm, a.dt_deg];
    const pos = s.position ? [s.position.y_mm, s.position.t_deg] : null;
    let end;
    if (tg.relative) { if (!pos) return null; end = [pos[0] + d[0], pos[1] + d[1]]; }
    else end = [tg.y_mm, tg.t_deg];
    const start = [end[0] - d[0], end[1] - d[1]];
    const origin = tg.origin ? [tg.origin.y_mm, tg.origin.t_deg] : start;
    const goal = tg.final ? [tg.final.y_mm, tg.final.t_deg] : end;
    const sp = s.speed || {}, v = [sp.y_mm_s, sp.t_deg_s];
    let left = 0, extra = 0;
    const axes = ['y', 't'].map((ax, i) => {
      const info = axisInfo(ax) || {on: false, frac: 1, left: 0};
      left = Math.max(left, info.left);
      if (tg.final && v[i]) extra = Math.max(extra, Math.abs(goal[i] - end[i]) / v[i]);
      const o = origin[i] - c[i], g = goal[i] - c[i], a0 = start[i] - c[i], a1 = end[i] - c[i];
      const est = a0 + d[i] * info.frac;
      // 진행률 = 지나온 거리 ÷ 전체 거리 (접근점을 거쳐 되돌아와도 줄지 않게, 접근점까지 + 접근점→목표)
      const [run, total] = tg.final
        ? [Math.abs(est - o), Math.abs(a1 - o) + Math.abs(g - a1)]
        : [Math.abs(a0 - o) + Math.abs(est - a0), Math.abs(a0 - o) + Math.abs(g - a0)];
      return {o, g, est, on: info.on, moves: total >= 0.0005,
              frac: total >= 0.0005 ? Math.min(1, run / total) : 1, done: !tg.final && info.frac >= 1};
    });
    return {y: axes[0], t: axes[1], left: left + extra};
  }

  function renderAxes() {
    const s = state; if (!s) return;
    const moving = !!s.moving || s.stage_state === 'moving' || s.stage_state === 'homing';
    const homing = s.phase === 'homing';
    const iy = homing ? {on: true} : axisInfo('y'), it = homing ? {on: false} : axisInfo('t');
    const names = [iy && iy.on ? 'Y' : '', it && it.on ? 'Θ' : ''].filter(Boolean);
    const badge = $('motion-badge'); badge.dataset.moving = moving;
    badge.querySelector('span').textContent = !moving ? '모터 정지' : homing ? 'Y 원점 복귀 중' :
      names.length ? `${names.join(' · ')} 이동 중` : '도착 중';
    $('ax-y').dataset.on = !!(moving && iy && iy.on); $('ax-t').dataset.on = !!(moving && it && it.on);
    $('ax-y').hidden = $('ax-t').hidden = !moving;
    renderMotion(s);
  }
  function renderMotion(s) {
    // 위치 카드: 이동 중이면 큰 숫자 = 예상 위치, 아래에 축마다 진행 막대, 남은 시간 하나
    const m = motion(s), card = $('pos-card');
    card.classList.toggle('in-motion', !!m);
    for (const [ax, unit] of [['y', ' mm'], ['t', '°']]) {
      // 큰 숫자(현재) 아래에 막대 한 줄: 채워진 만큼이 진행률, 오른쪽 끝이 목표
      const g = $('g-' + ax), mm = m && m[ax];
      g.hidden = !mm;
      if (!mm) continue;
      g.dataset.state = !mm.moves ? 'idle' : mm.done ? 'done' : 'on';
      g.querySelector('u').style.width = (mm.frac * 100).toFixed(1) + '%';
      g.querySelector('i').textContent = !mm.moves ? '그대로' : mm.done ? '도착' : `${Math.floor(mm.frac * 100)}%`;
      g.querySelector('em').textContent = `목표 ${fmt(mm.g)}${unit}`;
    }
    if (m) {
      $('off-y').textContent = fmt(m.y.est); $('off-t').textContent = fmt(m.t.est);
      $('pos-label').textContent = '센터와의 차이 · 이동 중 (예상)';
      $('pos-sub').textContent = m.left > 0.05 ? `⏱ ${m.left.toFixed(1)}초 남음` : '⏱ 도착 중';
    } else if (s.moving && s.phase === 'homing') {
      $('pos-sub').textContent = 'Y 원점 복귀 중…';
    }
  }
  setInterval(() => { if (state && state.moving) renderAxes(); }, 200);

  // ---------- 렌더 ----------
  function render(s) {
    state = s;
    const st = s.settings, run = s.run;
    const dev = s.devices || {};
    const a = s.algorithm || {};
    $('pill-algo').textContent = `알고리즘 ${a.kind === 'mock' ? '모의' : a.detail || a.kind}`;
    $('pill-algo').className = 'pill' + (a.kind === 'mock' ? ' warn' : a.error ? ' bad' : '');
    const cam = s.camera || {};
    $('cam-info').textContent = cam.ok ? `카메라 ${cam.kind === 'fake' ? '(가짜) ' : ''}${cam.fps} fps` : '카메라 없음';
    if (cam.ok && !camOk) { camOk = true; startVideo(); }       // 카메라가 살아남
    else if (!cam.ok && camOk) { camOk = false; clearTimeout(videoRetry); $('video').removeAttribute('src'); }
    $('video-empty').hidden = cam.ok;
    $('video-empty-msg').textContent = s.phase === 'released' ? '데모 웹이 카메라를 쓰는 중입니다.' : (cam.error || '카메라 연결을 기다리는 중');
    drawOverlay(s);
    clockSkew = s.now ? s.now - Date.now() / 1000 : 0;
    renderPosition(s);
    renderAxes();            // 위치 카드 위에 이동 중 예상값·게이지를 덮어쓴다
    drawPlot(s);
    renderFlow(s);
    renderAlgorithms(s);
    renderAxisSteps(s);

    // 진행
    // 보정 이동 횟수 (측정 n 번째 = 이동 n−1 번 뒤, 이동 중이면 그 이동까지 셈)
    $('trial').textContent = run ? Math.max(0, run.trial - 1) + (s.moving && s.phase === 'moving' ? 1 : 0) : 0;
    $('trial-max').textContent = run ? run.max_trials : st.max_trials;
    document.querySelectorAll('#steps li').forEach(li => li.classList.toggle('on', li.dataset.phase === s.phase));
    $('phase-box').dataset.phase = s.phase;
    $('phase').textContent = s.phase_label;
    $('message').textContent = s.message || (s.stage_error ? '스테이지: ' + s.stage_error : '');
    const res = $('result');
    if (run && run.result && run.result !== 'converged') {        // 성공은 위 판정 칸에 나오니 실패만
      res.hidden = false; res.textContent = run.result_label; res.className = 'result fail';
    } else res.hidden = true;

    // 관리 도구 정보
    $('center').textContent = `Y ${st.center_y_mm.toFixed(3)} mm · Θ ${st.center_t_deg.toFixed(3)}°`;
    $('tol').textContent = `±${st.tol_y_mm} mm` + (st.enable_t ? ` · ±${st.tol_t_deg}°` : '');
    const ref = s.reference;
    $('refinfo').textContent = ref ? `${new Date(ref.ts * 1000).toLocaleTimeString()} 등록 (학습 영점 대비 Y ${fmt(ref.zero_y_mm * 1000, 1)} µm, Θ ${fmt(ref.zero_t_deg, 4)}°)` : '없음';
    $('pos-raw').textContent = s.position ? `Y ${s.position.y_mm.toFixed(3)} mm · Θ ${s.position.t_deg.toFixed(3)}°` : '—';
    const f = s.flags;
    $('flags').innerHTML = f === null || f === undefined ? '' :
      [['Y 원점', 1, 'on'], ['Θ 원점', 2, 'on'], ['Y− 리밋', 4, 'alert'], ['Y+ 리밋', 8, 'alert']]
        .map(([n, bit, cls]) => `<span class="${f & bit ? cls : ''}">${n}</span>`).join('');

    // 버튼
    const ready = dev.ready && s.phase !== 'released';
    ['btn-home', 'btn-reference', 'btn-goto-random', 'btn-goto', 'btn-run', 'btn-step', 'btn-measure', 'btn-center']
      .forEach(id => { $(id).disabled = !ready || s.busy; });
    document.querySelectorAll('[data-slow]').forEach(b => { b.disabled = s.busy; });
    $('btn-stop').disabled = !ready || !s.busy;

    renderHistory(s.history || []);
    if (!settingsDirty) fillSettings(st);
  }

  const axisLabel = e => e.within || !e.axis ? '—' : e.axis === 'y_then_t' ? '<span class="tag ax">Y → Θ</span>' : '<span class="tag ax">Y·Θ 함께</span>';

  // ---------- 알고리즘 고르기 ----------
  function renderAlgorithms(s) {
    const list = s.algorithms, pick = $('algo-pick');
    pick.hidden = !list || list.length < 2;
    $('pill-algo').hidden = !pick.hidden;
    if (pick.hidden) return;
    const seg = $('algo-seg'), key = JSON.stringify(list) + s.busy;
    if (seg.dataset.key === key) return;
    seg.dataset.key = key;
    seg.innerHTML = list.map(a => `<button data-algo="${a.id}" class="${a.current ? 'on' : ''}" ${a.available && !s.busy ? '' : 'disabled'} ` +
      `title="${a.available ? (a.desc || '') : a.reason}"><b>${a.label}</b>${a.desc ? `<small>${a.desc}</small>` : ''}</button>`).join('');
  }
  $('algo-seg').addEventListener('click', ev => {
    const b = ev.target.closest('[data-algo]');
    if (!b || b.classList.contains('on')) return;
    api('/api/algorithm', {id: b.dataset.algo}).then(() => toast('알고리즘을 바꿨습니다', true)).catch(e => toast(e.message));
  });

  // ---------- Y → Θ 차례 보정 (한 번 측정한 보정량으로 Y 이동 → Θ 이동) ----------
  function renderAxisSteps(s) {
    const st = s.settings, seq = st.sequential_axes && st.enable_t, ol = $('axis-steps');
    document.querySelectorAll('[data-seq]').forEach(b => { b.classList.toggle('on', String(seq) === b.dataset.seq); b.disabled = s.busy || !st.enable_t; });
    ol.hidden = !seq;
    const last = s.run && s.history && s.history.length ? s.history[s.history.length - 1] : null;
    const stage = s.axis_stage;                       // 'y' | 't' | null (이번 보정 이동 중 어느 단계)
    const done = !stage && last && last.within;
    ol.querySelectorAll('li').forEach(li => {
      const ax = li.dataset.ax;
      li.className = done ? (ax === 'done' ? 'done on' : 'done') :
        stage === 'y' ? (ax === 'y' ? 'on' : '') :
        stage === 't' ? (ax === 't' ? 'on' : ax === 'y' ? 'done' : '') : '';
    });
    // 산출 보정량: 이번 측정에서 한 번 잰 두 축 값을 그대로 보여 주고, 지금 어느 축을 옮기는지 표시
    const calc = $('calc'), e = last;
    calc.hidden = !e || e.within;
    if (!calc.hidden) {
      const which = !seq ? 'Y·Θ 함께 이동' : stage === 'y' ? '① Y 이동 중 → 다음 Θ' : stage === 't' ? '② Θ 이동 중 (Y 끝)' : 'Y 먼저, 이어서 Θ';
      calc.innerHTML = `${e.trial}번째 보정 이동 · 산출 보정량 <b class="${stage === 't' ? 'dim' : ''}">Y ${fmt(e.command.y_mm)} mm</b> · ` +
        `<b class="${stage === 'y' ? 'dim' : ''}">Θ ${fmt(e.command.t_deg)}°</b><span>${which}</span>`;
    }
  }
  document.querySelectorAll('[data-seq]').forEach(b => b.addEventListener('click', () =>
    api('/api/settings', {sequential_axes: b.dataset.seq === 'true'}).catch(e => toast(e.message))));

  function renderHistory(h) {
    const body = $('hist-body');
    if (!h.length) { body.innerHTML = '<tr><td colspan="10" class="muted">아직 기록이 없습니다.</td></tr>'; return; }
    body.innerHTML = h.slice().reverse().map((e, i) => {
      const verdict = e.within ? (e.truth_within === false ? '<span class="tag warn">카메라만 OK</span>' : '<span class="tag ok">센터 OK</span>') : '<span class="tag no">보정 이동</span>';
      const t = e.truth || {};
      return `<tr class="${i === 0 ? 'last' : ''}"><td>${esc(e.label)}</td><td>${fmt(e.offset.y_mm)}</td><td>${fmt(e.offset.t_deg)}</td>` +
        `<td>${fmt(t.y_mm)}</td><td>${fmt(t.t_deg)}</td>` +
        `<td class="${e.axis === 't' ? 'dim' : ''}">${e.within ? '—' : fmt(e.command.y_mm)}</td><td class="${e.axis === 'y' ? 'dim' : ''}">${e.within ? '—' : fmt(e.command.t_deg)}</td>` +
        `<td>${axisLabel(e)}</td><td>${e.infer_ms}</td><td>${verdict}</td></tr>`;
    }).join('');
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
  $('goto-form').addEventListener('submit', ev => {        // Enter 로도 이동
    ev.preventDefault();
    const [y, t] = startOffset();
    if (!Number.isFinite(y) || !Number.isFinite(t)) { toast('Y, Θ 에 숫자를 넣으세요'); return; }
    api('/api/goto', {offset: [y, t]}).catch(e => toast(e.message));
  });
  act('btn-run', '/api/run', {});
  act('btn-step', '/api/step');
  act('btn-measure', '/api/measure');
  act('btn-reference', '/api/reference');
  act('btn-stop', '/api/stop');
  // 현재 위치를 센터로: 먼저 그 자리를 카메라로 재고, 센터와 많이 다르면 묻는다
  async function waitIdle() {
    for (let i = 0; i < 200; i++) {
      const s = await (await fetch('/api/state')).json();
      if (!s.busy) return s;
      await new Promise(r => setTimeout(r, 300));
    }
    throw new Error('센터 확인 측정이 끝나지 않습니다');
  }
  $('btn-center').addEventListener('click', async () => {
    try {
      await api('/api/center-check');
      const s = await waitIdle(), c = s.center_check;
      if (!c) throw new Error(s.message || '센터 확인 측정 실패');
      const where = c.basis === 'reference' ? '지금 센터 기준 사진' : '알고리즘의 센터(학습·고정 영점)';
      let force = false;
      if (!c.ok) {
        force = window.confirm(`카메라로 재 보니 이 자리는 ${where}에서\n  Y ${fmt(c.y_mm)} mm, Θ ${fmt(c.t_deg, 4)}°\n떨어져 있습니다 (기준 Y ±${c.limit.y_mm} mm, Θ ±${c.limit.t_deg}°).\n\n` +
          '실제 센터가 아니면 센터 기준·이동 위치가 모두 틀어집니다.\n그래도 여기를 센터로 저장할까요?');
        if (!force) { toast('센터를 바꾸지 않았습니다', true); return; }
      }
      await api('/api/center-here', {force});
      toast('센터를 저장했습니다', true);
    } catch (e) { toast(e.message); }
  });

  // ---------- 장치 되찾기 (화면을 열 때 데모 웹이 장치를 잡고 있으면 끄고 다시 잡음) ----------
  function overlay(text) { $('overlay-text').textContent = text; $('overlay').hidden = !text; }

  function connect() {
    if (events) events.close();
    events = new EventSource('/api/events');
    events.onmessage = e => render(JSON.parse(e.data));
    events.onerror = () => { if (!toast.timer || $('toast').hidden) toast('서버 연결이 끊겨 다시 연결하는 중입니다'); };
    camOk = false;      // 첫 상태에서 카메라가 정상이면 영상 시작
  }

  async function boot() {
    overlay('장치 확인 중…');
    try {
      const r = await api('/api/reclaim');
      if (!r.ok) toast('장치 연결 실패: ' + (r.error || ''));
    } catch (e) { toast(e.message); }
    overlay('');
    api('/api/view', view).catch(() => {});      // 처음부터 QR 중심
    connect();
  }
  // 뒤로 가기로 돌아왔을 때(bfcache)도 장치를 다시 잡는다
  window.addEventListener('pageshow', ev => { if (ev.persisted) boot(); });
  boot();
})();
