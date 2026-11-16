// [verify_web_3d] 화면 맞춤 + 3D 칸 보기 단추
(() => {
  // 배치는 폭 1920 · 높이 975 (1920×1080 화면에서 브라우저 창 최대화) 기준으로 짰다.
  // 창 크기가 달라도 같은 모양으로 한 화면에 들어오게 페이지 전체 배율(zoom)을 맞춘다. 폭 1000 미만(휴대폰 등)은 원래 세로 배치
  const W = 1920 + 2 * 60, H = 990;   // 배치 폭 1920 + 좌우 여백 60씩. 높이 = 위 막대(알고리즘 단추 포함) + 위쪽 두 칸 + 여유
  function fit() {
    const sb = 18;   // 아래 기록 때문에 늘 생기는 세로 스크롤바 폭 (가로 스크롤이 생기지 않게 뺌)
    const z = innerWidth < 1000 ? 1 : Math.min((innerWidth - sb) / W, innerHeight / H);
    document.documentElement.style.zoom = Math.abs(z - 1) < 0.005 ? '' : String(z);
  }
  fit();
  addEventListener('resize', fit);

  // 보기 단추: 같은 출처의 3D 화면(iframe)에 보는 방향만 알림. 장비와는 상관없음
  const frame = document.getElementById('viewer3d-frame'), seg = document.getElementById('viewer3d-views');
  if (!frame || !seg) return;
  seg.addEventListener('click', ev => {
    const b = ev.target.closest('[data-view]');
    if (!b) return;
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    frame.contentWindow && frame.contentWindow.postMessage({ytView: b.dataset.view}, location.origin);
  });
  // 움직임 과장 (3D 화면이 "움직임 ×10 과장 · 실제 크기 아님" 을 띄움)
  const ex = document.getElementById('viewer3d-exag');
  if (ex) ex.addEventListener('click', () => {
    const on = ex.getAttribute('aria-pressed') !== 'true';
    ex.setAttribute('aria-pressed', String(on)); ex.classList.toggle('on', on);
    frame.contentWindow && frame.contentWindow.postMessage({ytExag: on ? 10 : 1}, location.origin);
  });
})();
