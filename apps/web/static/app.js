// Service tab. Plays one alignment session step by step.
// Data source today: window.SEMES_DEMO (demo/data.js, real joint/control results on scenario photos).
// Later: the same shapes from the C++ service API (/api/session ...).
(() => {
  const D = window.SEMES_DEMO;
  const TOL = D.rules.tolerance;
  const STEP_MS = 1600;
  const $ = (id) => document.getElementById(id);

  const state = { sc: 0, revealed: 0, current: 0, view: "qr", timer: null };
  const scenario = () => D.scenarios[state.sc];
  const step = () => (state.current > 0 ? scenario().steps[state.current - 1] : null);
  const finished = () => state.revealed === scenario().steps.length;

  // ---------- formatting ----------
  const fmt = (v, d = 4) => (v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const stageName = (s) => (s === "joint" ? "1차 joint" : "2차 control");
  const isCenter = (e) => e && e.ok && Math.abs(e.y) <= TOL && Math.abs(e.t) <= TOL;
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // ---------- static parts ----------
  $("jointModel").textContent = D.models.joint;
  $("controlModel").textContent = D.models.control;
  $("pageFoot").textContent = `데모 데이터  ${D.source}  생성 ${D.generated}`;

  function renderScenarioList() {
    const ul = $("scenarioList");
    ul.replaceChildren();
    D.scenarios.forEach((sc, i) => {
      const b = el("button", i === state.sc ? "is-on" : "");
      b.append(el("span", "sc-cat", sc.category));
      b.append(el("span", "sc-start", `Y ${fmt(sc.start.y, 2)}  T ${fmt(sc.start.t, 2)}`));
      b.append(el("span", "sc-id", sc.id));
      b.addEventListener("click", () => selectScenario(i));
      const li = el("li");
      li.append(b);
      ul.append(li);
    });
  }

  // ---------- viewer ----------
  function renderViewer() {
    const sc = scenario(), s = step();
    const imgKey = s ? s.frame.image : sc.steps[0].frame.image;
    const img = D.images[imgKey];
    $("frameImg").src = state.view === "qr" ? img.qr : img.full;
    $("framePath").textContent = `data/${img.path}`;
    $("frameSub").textContent = s ? `${s.index}단계  ${stageName(s.stage)} 입력` : "실행 전  시작 사진";

    const box = state.view === "qr" ? D.roi : { x: 0, y: 0, w: D.frame.w, h: D.frame.h };
    const pass = s && isCenter(s.estimate), fail = s && !pass;
    const overlay = $("overlay");
    overlay.replaceChildren();
    img.qr_boxes.forEach((q, i) => {
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
      if (pass) {
        tag.append(el("span", "num", "Y = 0"), el("span", "num", "T = 0"), el("span", "state", "센터"));
      } else {
        tag.append(el("span", "num", `Y ${fmt(s.estimate.y)}`), el("span", "num", `T ${fmt(s.estimate.t)}`), el("span", "state", "센터 아님"));
      }
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
      chip.append(el("span", null, "실행을 누르면 1차 joint부터 시작합니다"));
    } else if (last && res.status === "done") {
      chip.classList.add("is-done");
      chip.append(el("span", null, "도착"), el("span", "sep"), el("span", null, `${res.steps}단계`), el("span", "sep"), el("span", null, `이동 ${res.moves}회`));
    } else if (last && res.status === "failed") {
      chip.classList.add("is-fail");
      chip.append(el("span", null, "중단"), el("span", "sep"), el("span", null, res.reason));
    } else if (s.moved) {
      const axis = s.command.dt !== 0 ? "T" : "Y";
      const v = s.command.dt !== 0 ? s.command.dt : s.command.dy;
      chip.append(el("span", null, stageName(s.stage)), el("span", "sep"), el("span", null, `${axis} ${fmt(v)} 이동`));
    } else if (s.next === "control") {
      chip.append(el("span", null, "1차 완료"), el("span", "sep"), el("span", null, "2차 control로 전환"));
    } else if (s.next === "joint") {
      chip.append(el("span", null, "1차 joint로 복귀"));
    }
  }

  // ---------- status cards ----------
  function renderStatus() {
    const sc = scenario(), s = step();
    const e = s && s.estimate.ok ? s.estimate : null;
    $("estY").textContent = e ? fmt(e.y) : "-";
    $("estT").textContent = e ? fmt(e.t) : "-";
    const note = $("estNote");
    note.className = "stat-note";
    if (!s) note.textContent = "실행 전";
    else if (!e) { note.textContent = "측정 실패"; note.classList.add("is-fail"); }
    else if (isCenter(e)) { note.textContent = "두 축 모두 기준 0.09 이내"; note.classList.add("is-pass"); }
    else {
      const out = [Math.abs(e.y) > TOL ? "Y" : null, Math.abs(e.t) > TOL ? "T" : null].filter(Boolean).join(", ");
      note.textContent = `${out} 기준 0.09 밖`;
      note.classList.add("is-fail");
    }

    const main = $("resultMain"), list = $("resultList");
    main.className = "result-main";
    list.replaceChildren();
    const row = (k, v) => list.append(el("dt", null, k), el("dd", null, v));
    if (finished() && state.current === sc.steps.length) {
      const r = sc.result;
      main.textContent = r.status === "done" ? "도착" : "중단";
      main.classList.add(r.status === "done" ? "is-pass" : "is-fail");
      row("단계", `${r.steps}`);
      row("이동", `${r.moves}회`);
      row("최종 자세", `Y ${fmt(r.final_pose.y)}  T ${fmt(r.final_pose.t)}`);
      if (r.status !== "done") row("사유", r.reason);
    } else if (s) {
      main.textContent = "진행 중";
      row("현재", `${s.index}단계  ${stageName(s.stage)}`);
      row("판단", s.decision);
    } else {
      main.textContent = "대기";
      row("시나리오", sc.category);
      row("시작 자세", `Y ${fmt(sc.start.y, 2)}  T ${fmt(sc.start.t, 2)}`);
    }
  }

  // ---------- Y-T plane ----------
  function renderPlane() {
    const sc = scenario();
    const svg = $("plane");
    const W = 320, H = 200, cx = 160, cy = 100;
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
    add("rect", { x: X(-TOL), y: Y(TOL), width: 2 * TOL * k, height: 2 * TOL * k, fill: "#e6f4ee", stroke: "#b7dfcd", rx: 2 });
    add("line", { x1: 8, y1: cy, x2: W - 8, y2: cy, stroke: "#d5dbe5" });
    add("line", { x1: cx, y1: 6, x2: cx, y2: H - 6, stroke: "#d5dbe5" });
    add("text", { x: W - 8, y: cy - 6, "text-anchor": "end", "font-size": 10, fill: "#8a94a6" }, "Y");
    add("text", { x: cx + 6, y: 14, "font-size": 10, fill: "#8a94a6" }, "T");
    add("text", { x: X(span / 1.15), y: cy + 14, "text-anchor": "end", "font-size": 9, fill: "#8a94a6" }, fmt(span / 1.15, 2));

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
          const axis = s.command.dt !== 0 ? "T" : "Y";
          body.append(el("div", "tl-cmd", `${axis} ${fmt(s.command.dt !== 0 ? s.command.dt : s.command.dy)} 이동`));
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
    renderScenarioList();
    renderViewer();
    renderStatus();
    renderPlane();
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

  $("runBtn").addEventListener("click", () => {
    if (state.timer) { stop(); render(); return; }
    if (finished()) reset();
    advance();
    if (!finished()) state.timer = setInterval(advance, STEP_MS);
    render();
  });
  $("stepBtn").addEventListener("click", () => { stop(); advance(); });
  $("resetBtn").addEventListener("click", () => { reset(); render(); });
  document.querySelectorAll(".segmented button").forEach((b) =>
    b.addEventListener("click", () => {
      state.view = b.dataset.view;
      document.querySelectorAll(".segmented button").forEach((x) => x.classList.toggle("is-on", x === b));
      renderViewer();
    })
  );

  // Deep link for review: #<scenario id>/<step>[/full]
  const [hashId, hashStep, hashView] = decodeURIComponent(location.hash.slice(1)).split("/");
  const hashIdx = D.scenarios.findIndex((sc) => sc.id === hashId);
  if (hashIdx >= 0) {
    state.sc = hashIdx;
    state.revealed = state.current = Math.min(Number(hashStep) || 0, scenario().steps.length);
    if (hashView === "full") document.querySelector('.segmented button[data-view="full"]').click();
  }
  render();
})();
