// Service tab. Plays one alignment session step by step.
// Data source today: window.SEMES_DEMO (demo/data.js, real Y/T estimates of two selectable 7-value models on scenario photos)
// and window.SEMES_SIM (demo/sim.js, every photo's real estimate, for the simulation input).
// Later: the same shapes from the C++ service API (/api/session ...).
(() => {
  const D = window.SEMES_DEMO;
  const SIM = window.SEMES_SIM;
  const TOL_Y = D.rules.tolerance_y, TOL_T = D.rules.tolerance_t;   // 한 장 허용오차
  const STEP_MS = 1600;
  const $ = (id) => document.getElementById(id);

  // model: 고른 모델(semes = SEMES 사진으로 학습한 채택 모듈, rig = 우리 장비 사진 전체로 학습). 시나리오는 모델마다 자기 사진으로 재생한다
  // source: sim = 입력한 (Y, T)에서 바로 시뮬레이션, scenario = 미리 만든 시나리오 재생
  const state = { model: D.default_model, sc: D.scenarios.findIndex((x) => x.model === D.default_model), revealed: 0, current: 0, view: "qr", timer: null,
                  source: SIM ? "sim" : "scenario", simSc: null, simIn: { y: 0.5, t: 0.3 }, simSeed: 1 };
  const scenario = () => (state.source === "sim" ? state.simSc : D.scenarios[state.sc]);
  const step = () => (state.current > 0 ? scenario().steps[state.current - 1] : null);
  const finished = () => state.revealed === scenario().steps.length;

  // ---------- formatting ----------
  const fmt = (v, d = 4) => (v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const stageName = () => "Y·T 추정";   // 2차 control 없이 값 7개 모듈 하나로 Y와 T를 함께 보정한다
  const isCenter = (s) => s && s.estimate.ok && Math.abs(s.estimate.y) <= TOL_Y && Math.abs(s.estimate.t) <= TOL_T;
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // ---------- simulation ----------
  // 입력한 자세에서 시작해 Y와 T를 함께 움직인다(규칙은 make_demo.py 와 같다). 매 단계 자세 근처(모델별 허용)에 실제 사진이 있으면
  // 그 사진의 실제 모델 추정값을 쓰고, 없으면 참 자세에 그 모델의 검증 오차(주 검증 rms, Y·T 독립 정규)를 더한 값을 쓴다(오차 모형).
  const rng = (seed) => () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const r6 = (v) => Math.round(v * 1e6) / 1e6;
  function simImage(model, ph) {
    const key = `sim:${ph.p}`;
    if (!D.images[key]) {
      D.images[key] = { key, path: ph.p, qr: ph.img, full: ph.img, crop: true, roi: SIM.models[model].roi, motion: null,
                        qr_boxes: ph.q.map(([x, y, w, h, ncc]) => ({ x, y, w, h, ncc })) };
    }
    return key;
  }
  function simulate(model, y0, t0, seed) {
    const M = SIM.models[model], R = SIM.rules, r = rng(seed);
    const near = (y, t) => {
      let best = null, bd = Infinity;
      for (const ph of M.photos) {
        const dy = Math.abs(ph.y - y), dt = Math.abs(ph.t - t);
        if (dy > M.tol_y || dt > M.tol_t) continue;
        const d = dy / M.tol_y + dt / M.tol_t;
        if (d < bd) { bd = d; best = ph; }
      }
      return best;
    };
    let pose = { y: y0, t: t0 }, result = null;
    const steps = [];
    for (let i = 0; i < R.max_iterations; i++) {
      const ph = near(pose.y, pose.t);
      const rec = { index: i + 1, stage: "joint", pose_before: { y: r6(pose.y), t: r6(pose.t) }, command: { dy: 0, dt: 0 }, moved: false };
      if (ph) {
        rec.frame = { image: simImage(model, ph), label: `실제 사진 (명령 Y ${fmt(ph.cy, 2)}, T ${fmt(ph.ct, 2)})`, pose: { y: ph.y, t: ph.t } };
        rec.source = "photo";
        rec.estimate = ph.ok ? { ok: true, y: ph.ey, t: ph.et } : { ok: false, error: ph.err };
      } else {
        rec.frame = { image: null, label: "근처에 실제 사진 없음", pose: null };
        rec.source = "error";
        rec.estimate = { ok: true, y: r6(pose.y + M.err.sd_y * gauss(r)), t: r6(pose.t + M.err.sd_t * gauss(r)) };
      }
      const how = rec.source === "photo" ? "실제 사진 추정" : "오차 모형 추정";
      if (!rec.estimate.ok) {
        rec.decision = `${how}. 측정 실패`;
        rec.next = "failed";
        steps.push(rec);
        result = { status: "failed", reason: rec.estimate.error || "측정 실패" };
        break;
      }
      const e = rec.estimate;
      if (Math.abs(e.y) <= R.tolerance_y && Math.abs(e.t) <= R.tolerance_t) {
        rec.decision = `${how}. Y ${Math.abs(e.y).toFixed(4)}, T ${Math.abs(e.t).toFixed(4)} 모두 기준 이내`;
        rec.next = "done";
      } else {
        const over = [["Y", e.y, R.tolerance_y], ["T", e.t, R.tolerance_t]].filter(([, v, k]) => Math.abs(v) > k).map(([n]) => n);
        rec.command = { dy: r6(-e.y), dt: r6(-e.t) };
        rec.moved = true;
        rec.decision = `${how}. ${over.join(", ")}가 기준 밖이라 Y와 T를 함께 이동`;
        rec.next = "joint";
      }
      pose = { y: pose.y + rec.command.dy, t: pose.t + rec.command.dt };
      rec.pose_after = { y: r6(pose.y), t: r6(pose.t) };
      steps.push(rec);
      if (rec.next === "done") { result = { status: "done", reason: "Y, T 기준 이내" }; break; }
    }
    if (!result) result = { status: "failed", reason: `최대 ${R.max_iterations}회를 넘었습니다` };
    Object.assign(result, { steps: steps.length, moves: steps.filter((s) => s.moved).length, final_pose: { y: r6(pose.y), t: r6(pose.t) },
                            photo_steps: steps.filter((s) => s.source === "photo").length });
    return { id: "SIM", model, category: "시뮬레이션", description: `시작 Y ${fmt(y0, 3)}, T ${fmt(t0, 3)}`, start: { y: y0, t: t0 }, steps, result };
  }
  function rebuildSim() {
    reset();
    state.simSc = simulate(state.model, state.simIn.y, state.simIn.t, state.simSeed);
  }

  // ---------- static parts ----------
  function renderModel() {
    const M = D.models[state.model];
    const pick = $("modelPick");
    pick.replaceChildren();
    Object.values(D.models).forEach((m) => {
      const b = el("button", m.id === state.model ? "is-on" : "", m.label);
      b.title = `${m.name}
${m.train}`;
      b.addEventListener("click", () => selectModel(m.id));
      pick.append(b);
    });
    $("jointModel").textContent = M.name;
    $("jointModel").title = `${M.path}
${M.train}`;
    const badge = $("modelBadge");
    badge.className = "badge " + (M.adopted ? "badge-adopted" : "badge-trial");
    badge.textContent = M.adopted ? "채택" : "실험";
    $("pageFoot").textContent = `학습 ${M.train}. ${M.note}. T 단위 ${M.t_unit}.  ${D.source}  생성 ${D.generated}`;
  }

  function renderInput() {
    document.querySelectorAll(".source-tabs button[data-source]").forEach((b) => b.classList.toggle("is-on", b.dataset.source === state.source));
    const sim = state.source === "sim";
    $("scenarioList").hidden = sim;
    $("simForm").hidden = !sim;
    if (sim) {
      const M = SIM.models[state.model], unit = D.models[state.model].t_unit;
      if (document.activeElement !== $("simY")) $("simY").value = state.simIn.y;
      if (document.activeElement !== $("simT")) $("simT").value = state.simIn.t;
      $("simTUnit").textContent = unit;
      const res = state.simSc.result;
      $("simNote").textContent = `범위 Y ${M.range.y[0]}~${M.range.y[1]} mm, T ${M.range.t[0]}~${M.range.t[1]} ${unit}. ` +
        `실제 사진 ${res.photo_steps} / ${res.steps}단계, 나머지는 오차 모형(σY ${M.err.sd_y.toFixed(4)}, σT ${M.err.sd_t.toFixed(4)})`;
    }
  }

  function renderScenarioList() {
    renderInput();
    if (state.source === "sim") return;
    const ul = $("scenarioList");
    ul.replaceChildren();
    D.scenarios.forEach((sc, i) => {
      if (sc.model !== state.model) return;
      const b = el("button", i === state.sc ? "is-on" : "");
      b.append(el("span", "sc-cat", sc.category));
      b.append(el("span", "sc-start", `Y ${fmt(sc.start.y, 2)}  T ${fmt(sc.start.t, 2)}`));
      b.title = `${sc.id}  ${sc.description}`;
      b.addEventListener("click", () => selectScenario(i));
      const li = el("li");
      li.append(b);
      ul.append(li);
    });
    // Keep the chosen scenario visible inside the short list.
    const on = ul.querySelector("button.is-on");
    if (on) ul.scrollTop = on.parentElement.offsetTop - ul.offsetTop - (ul.clientHeight - on.offsetHeight) / 2;
  }

  // ---------- viewer ----------
  function renderViewer() {
    const sc = scenario(), s = step();
    const imgKey = s ? s.frame.image : sc.steps[0].frame.image;
    const img = imgKey ? D.images[imgKey] : null;
    const noPhoto = $("noPhoto");
    noPhoto.hidden = !!img;
    $("frameImg").hidden = !img;
    if (img) $("frameImg").src = state.view === "qr" ? img.qr : img.full;
    $("framePath").textContent = img ? `data/${img.path}` : "사진 없음, 오차 모형으로 추정";
    $("frameSub").textContent = s ? `${s.index}단계  ${stageName(s.stage)} 입력` + (state.source === "sim" ? `  ${s.frame.label}` : "")
      : state.source === "sim" ? `실행 전  ${sc.steps[0].frame.label}` : "실행 전  시작 사진";

    // 시뮬레이션 미리보기는 QR 확대 영역만 있으므로 전체 보기에서도 같은 영역을 쓴다
    const box = state.view === "qr" || (img && img.crop) ? (img && img.roi) || D.roi : { x: 0, y: 0, w: D.frame.w, h: D.frame.h };
    const pass = s && isCenter(s), fail = s && !pass;
    const overlay = $("overlay");
    overlay.replaceChildren();
    (img ? img.qr_boxes : []).forEach((q, i) => {
      const d = el("div", "qr-box" + (pass ? " is-pass" : fail ? " is-fail" : ""));
      d.style.left = `${((q.x - box.x) / box.w) * 100}%`;
      d.style.top = `${((q.y - box.y) / box.h) * 100}%`;
      d.style.width = `${(q.w / box.w) * 100}%`;
      d.style.height = `${(q.h / box.h) * 100}%`;
      if (state.view === "qr") d.append(el("span", "qr-label", `QR${i + 1}  NCC ${q.ncc.toFixed(3)}`));
      overlay.append(d);
    });

    const tag = $("estimateTag");
    tag.replaceChildren();
    tag.className = "estimate-tag";
    if (s && s.estimate.ok) {
      tag.classList.add(pass ? "is-pass" : "is-fail");
      // 추정값은 통과해도 0으로 바꾸지 않고 그대로 보여 준다. 상태는 그 단계의 판정이다
      const label = pass ? "기준 이내" : "기준 밖";
      tag.append(el("span", "num", `Y ${fmt(s.estimate.y)}`), el("span", "num", `T ${fmt(s.estimate.t)}`), el("span", "state", label));
    } else {
      tag.append(el("span", "state", "추정 전"));
    }

    const chip = $("planChip");
    chip.replaceChildren();
    chip.className = "plan-chip";
    const res = sc.result;
    const last = state.current === sc.steps.length && finished();
    if (!s) {
      chip.classList.add("is-idle");
      chip.append(el("span", null, "실행을 누르면 Y와 T를 함께 추정합니다"));
    } else if (last && res.status === "done") {
      chip.classList.add("is-done");
      chip.append(el("span", null, "도착"), el("span", "sep"), el("span", null, `${res.steps}단계`), el("span", "sep"), el("span", null, `이동 ${res.moves}회`));
    } else if (last && res.status === "failed") {
      chip.classList.add("is-fail");
      chip.append(el("span", null, "중단"), el("span", "sep"), el("span", null, res.reason));
    } else if (s.moved) {
      chip.append(el("span", null, "동시 이동"), el("span", "sep"), el("span", null, `Y ${fmt(s.command.dy)}`), el("span", "sep"), el("span", null, `T ${fmt(s.command.dt)}`));
    }
  }

  // ---------- Y-T plane ----------
  function renderPlane() {
    const sc = scenario();
    const svg = $("plane");
    // Draw in the card's real pixel size so the graph fills the card instead of floating in it.
    const box = svg.getBoundingClientRect();
    const W = Math.max(240, Math.round(box.width) || 320), H = Math.max(160, Math.round(box.height) || 240), cx = W / 2, cy = H / 2;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const span = Math.max(0.15, ...sc.steps.flatMap((s) => [Math.abs(s.pose_before.y), Math.abs(s.pose_before.t)]), Math.abs(sc.start.y), Math.abs(sc.start.t)) * 1.15;
    const k = Math.min((W / 2 - 16) / span, (H / 2 - 14) / span);
    const X = (y) => cx + y * k, Y = (t) => cy - t * k;
    const ns = "http://www.w3.org/2000/svg";
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([a, v]) => n.setAttribute(a, v));
      if (text) n.textContent = text;
      svg.append(n);
      return n;
    };
    svg.replaceChildren();
    add("rect", { x: X(-TOL_Y), y: Y(TOL_T), width: 2 * TOL_Y * k, height: Math.max(2, 2 * TOL_T * k), fill: "#e6f4ee", stroke: "#b7dfcd", rx: 2 });
    add("line", { x1: 8, y1: cy, x2: W - 8, y2: cy, stroke: "#d5dbe5" });
    add("line", { x1: cx, y1: 6, x2: cx, y2: H - 6, stroke: "#d5dbe5" });
    add("text", { x: W - 8, y: cy - 8, "text-anchor": "end", "font-size": 13, "font-weight": 700, fill: "#8a94a6" }, "Y");
    add("text", { x: cx + 8, y: 16, "font-size": 13, "font-weight": 700, fill: "#8a94a6" }, "T");
    add("text", { x: X(span / 1.15), y: cy + 18, "text-anchor": "end", "font-size": 12, fill: "#8a94a6" }, fmt(span / 1.15, 2));

    const shown = sc.steps.slice(0, state.current);
    shown.forEach((s) => {
      if (s.moved) add("line", { x1: X(s.pose_before.y), y1: Y(s.pose_before.t), x2: X(s.pose_after.y), y2: Y(s.pose_after.t), stroke: s.stage === "joint" ? "#b45309" : "#1d4ed8", "stroke-width": 1.5, "stroke-dasharray": "4 3" });
    });
    shown.forEach((s, i) => {
      const cur = i === state.current - 1;
      add("circle", { cx: X(s.pose_before.y), cy: Y(s.pose_before.t), r: cur ? 5 : 3.5, fill: "#0b2a5b", opacity: cur ? 1 : 0.45 });
      if (s.estimate.ok) add("circle", { cx: X(s.estimate.y), cy: Y(s.estimate.t), r: cur ? 7 : 5, fill: "none", stroke: "#1d4ed8", "stroke-width": 1.5, opacity: cur ? 1 : 0.45 });
    });
    if (!shown.length) add("circle", { cx: X(sc.start.y), cy: Y(sc.start.t), r: 5, fill: "#0b2a5b" });
  }

  // ---------- feature-point motion components ----------
  function renderMotion() {
    const sc = scenario(), s = step();
    const key = s ? s.frame.image : sc.steps[0].frame.image;
    const m = key ? D.images[key].motion : null;
    const svg = $("motionField");
    if (!m) {
      svg.replaceChildren();
      $("motionComps").replaceChildren();
      $("motionScale").textContent = "시뮬레이션에서는 특징점 이동 성분을 계산하지 않습니다";
      $("motionNote").textContent = "";
      return;
    }
    const ns = "http://www.w3.org/2000/svg";
    const add = (tag, attrs) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      svg.append(n);
      return n;
    };
    // Points occupy the QR band only; frame the field on their bounding box.
    const xs = m.field.map((p) => p[0]), ys = m.field.map((p) => p[1]);
    const pad = 24, x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad;
    const w = Math.max(...xs) - Math.min(...xs) + 2 * pad, h = Math.max(...ys) - Math.min(...ys) + 2 * pad;
    svg.setAttribute("viewBox", `${x0} ${y0} ${w} ${h}`);
    svg.replaceChildren();
    // Translation dominates (Y and T both shift the image sideways), so it is drawn once as a bold
    // arrow. Per-point arrows show only the fitted rotation + scale part (magnified); what is left
    // after the fit is measurement noise and is reported as the residual instead of drawn.
    const ok = m.field.filter((p) => p[4]);
    const mx = ok.reduce((a, p) => a + p[0], 0) / ok.length, my = ok.reduce((a, p) => a + p[1], 0) / ok.length;
    const rot = m.rotation_mrad / 1000, scl = m.scale_ppm / 1e6;
    const part = m.field.map((p) => { const u = p[0] - mx, v = p[1] - my; return [p[0], p[1], -rot * v + scl * u, rot * u + scl * v, p[4]]; });
    const maxPart = Math.max(1e-12, ...part.map((p) => Math.hypot(p[2], p[3])));
    const k = 40 / maxPart;
    part.forEach((p, i) => {
      add("circle", { cx: p[0], cy: p[1], r: 3, fill: p[4] ? "#cfd6e2" : "#c62f3a" });
      if (i % 7 === 0 && maxPart > 1e-6) add("line", { x1: p[0], y1: p[1], x2: p[0] + p[2] * k, y2: p[1] + p[3] * k, stroke: "#1d4ed8", "stroke-width": 3, "stroke-linecap": "round", opacity: 0.7 });
    });
    const tLen = Math.hypot(m.tx, m.ty);
    if (tLen > 1e-3) {
      const cxm = x0 + w / 2, cym = y0 + h / 2, L = Math.min(w, h) * 0.8;
      const ex = cxm + (m.tx / tLen) * L / 2, ey = cym + (m.ty / tLen) * L / 2;
      const sx = cxm - (m.tx / tLen) * L / 2, sy = cym - (m.ty / tLen) * L / 2;
      add("line", { x1: sx, y1: sy, x2: ex, y2: ey, stroke: "#0b2a5b", "stroke-width": 9, "stroke-linecap": "round" });
      const a = Math.atan2(m.ty, m.tx), hl = 34;
      add("polygon", { points: `${ex + Math.cos(a) * 10},${ey + Math.sin(a) * 10} ${ex - Math.cos(a - 0.5) * hl},${ey - Math.sin(a - 0.5) * hl} ${ex - Math.cos(a + 0.5) * hl},${ey - Math.sin(a + 0.5) * hl}`, fill: "#0b2a5b" });
    }
    $("motionScale").textContent = tLen > 1e-3
      ? `굵은 화살표  평행이동 방향
파란 화살표  회전과 크기 성분 ×${k >= 100 ? Math.round(k).toLocaleString() : k.toFixed(0)}`
      : "기준 사진과 같습니다";

    const comps = $("motionComps");
    comps.replaceChildren();
    const row = (label, value, unit) => {
      const d = el("div", "comp");
      d.append(el("dt", null, label), el("dd", null, value), el("span", "comp-unit", unit));
      comps.append(d);
    };
    row("평행이동 X", fmt(m.tx, 3), "px");
    row("평행이동 Y", fmt(m.ty, 3), "px");
    row("회전", fmt(m.rotation_mrad, 3), "mrad");
    row("크기 변화", fmt(m.scale_ppm, 1), "ppm");
    $("motionNote").textContent = `기준 사진 대비  유효 ${m.valid} / ${m.points}점  잔차 ${m.residual_px.toFixed(3)} px  근사 측정`;
  }

  // ---------- timeline ----------
  function renderTimeline() {
    const sc = scenario();
    const ol = $("timeline");
    ol.replaceChildren();
    sc.steps.forEach((s, i) => {
      const shown = i < state.revealed;
      const lastDone = shown && i === sc.steps.length - 1 && finished();
      let cls = "tl-item";
      if (!shown) cls += " is-pending";
      else if (lastDone) cls += sc.result.status === "done" ? " st-done" : " st-fail";
      else cls += s.stage === "joint" ? " st-joint" : " st-control";
      if (state.current === i + 1) cls += " is-current";
      const li = el("li", cls);
      li.append(el("span", "tl-num", String(i + 1)));
      const body = el("div");
      if (!shown) {
        body.append(el("div", "tl-stage", "대기"));
      } else {
        body.append(el("div", "tl-stage", lastDone ? (sc.result.status === "done" ? `${stageName(s.stage)}  도착` : `${stageName(s.stage)}  중단`) : stageName(s.stage)));
        if (s.estimate.ok) body.append(el("div", "tl-est", `y ${fmt(s.estimate.y)}   t ${fmt(s.estimate.t)}`));
        body.append(el("div", "tl-decision", s.decision));
        if (s.moved) {
          body.append(el("div", "tl-cmd", `Y ${fmt(s.command.dy)}, T ${fmt(s.command.dt)} 이동`));
        }
        li.addEventListener("click", () => { stop(); state.current = i + 1; render(); });
      }
      li.append(body);
      ol.append(li);
    });
    $("stepCount").textContent = `${state.revealed} / ${sc.steps.length}`;
  }

  function renderControls() {
    $("runBtn").textContent = state.timer ? "일시 정지" : finished() ? "다시 실행" : "실행";
    $("stepBtn").disabled = finished();
  }

  function render() {
    renderModel();
    renderScenarioList();
    renderViewer();
    renderPlane();
    renderMotion();
    renderTimeline();
    renderControls();
  }

  // ---------- playback ----------
  function advance() {
    if (finished()) { stop(); return; }
    state.revealed += 1;
    state.current = state.revealed;
    if (finished()) stop();
    render();
  }
  function stop() {
    clearInterval(state.timer);
    state.timer = null;
  }
  function reset() {
    stop();
    state.revealed = 0;
    state.current = 0;
  }
  function selectScenario(i) {
    reset();
    state.sc = i;
    render();
  }
  function selectModel(m) {
    if (m === state.model) return;
    state.model = m;
    state.sc = D.scenarios.findIndex((x) => x.model === m);
    rebuildSim();
    reset();
    render();
  }
  function selectSource(src) {
    if (src === state.source) return;
    state.source = src;
    reset();
    render();
  }
  function setSimInput(y, t) {
    const M = SIM.models[state.model];
    const cl = (v, [a, b]) => Math.min(b, Math.max(a, v));
    if (!Number.isFinite(y) || !Number.isFinite(t)) return;
    state.simIn = { y: cl(y, M.range.y), t: cl(t, M.range.t) };
    state.simSeed = (state.simSeed * 1103515245 + 12345) & 0x7fffffff;
    rebuildSim();
    render();
  }
  document.querySelectorAll(".source-tabs button[data-source]").forEach((b) => b.addEventListener("click", () => selectSource(b.dataset.source)));
  if (SIM) {
    const readIn = () => setSimInput(parseFloat($("simY").value), parseFloat($("simT").value));
    $("simY").addEventListener("change", readIn);
    $("simT").addEventListener("change", readIn);
    $("simRandom").addEventListener("click", () => {
      const M = SIM.models[state.model], u = Math.random;
      const v = (a, b) => Math.round((a + (b - a) * u()) * 1000) / 1000;
      setSimInput(v(...M.range.y), v(...M.range.t));
      $("runBtn").click();
    });
  }

  $("runBtn").addEventListener("click", () => {
    if (state.timer) { stop(); render(); return; }
    if (finished()) reset();
    advance();
    if (!finished()) state.timer = setInterval(advance, STEP_MS);
    render();
  });
  $("stepBtn").addEventListener("click", () => { stop(); advance(); });
  $("resetBtn").addEventListener("click", () => { reset(); render(); });
  document.querySelectorAll(".segmented button[data-view]").forEach((b) =>
    b.addEventListener("click", () => {
      state.view = b.dataset.view;
      document.querySelectorAll(".segmented button[data-view]").forEach((x) => x.classList.toggle("is-on", x === b));
      renderViewer();
    })
  );

  // Deep link for review: #<scenario id>/<step>[/full]
  const [hashId, hashStep, hashView] = decodeURIComponent(location.hash.slice(1)).split("/");
  const hashIdx = D.scenarios.findIndex((sc) => sc.id === hashId);
  if (hashIdx >= 0) {
    state.source = "scenario";
    state.sc = hashIdx;
    state.model = scenario().model;
    state.revealed = state.current = Math.min(Number(hashStep) || 0, scenario().steps.length);
    if (hashView === "full") document.querySelector('.segmented button[data-view="full"]').click();
  }
  if (SIM) state.simSc = simulate(state.model, state.simIn.y, state.simIn.t, state.simSeed);
  else document.querySelector('.source-tabs button[data-source="sim"]').disabled = true;
  window.addEventListener("resize", renderPlane);
  render();
})();
