// [verify_web_3d] 카메라 칸 탭: 장비 카메라(기본) / 3D 시뮬레이터
//   3D 시뮬레이터 = 3D 장비 안의 가상 카메라 화면 + 오른쪽 아래 3D 장비 전체 작은 창 (Unity WebGL 빌드, /viewer/?mode=simcam)
//   처음 탭을 열 때 iframe 을 불러옴. 3D 는 스테이지 위치만 받아 그리고 장비에는 명령을 보내지 않음
(() => {
  const $ = id => document.getElementById(id);
  const tabs = document.querySelectorAll('[data-camtab]'), view = $('sim-view'), frame = $('sim-frame'), ex = $('sim-exag');
  if (!view || !frame) return;
  function show(tab) {
    const sim = tab === 'sim';
    tabs.forEach(b => { const on = b.dataset.camtab === tab; b.classList.toggle('on', on); b.setAttribute('aria-selected', String(on)); });
    view.hidden = !sim;
    document.querySelectorAll('.cam-only').forEach(e => { e.hidden = sim; });
    document.querySelectorAll('.sim-only').forEach(e => { e.hidden = !sim; });
    if (sim && !frame.getAttribute('src')) frame.setAttribute('src', '/viewer/?mode=simcam');
  }
  tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.camtab)));
  show('cam');
  // 움직임 과장 (3D 화면이 "움직임 ×10 과장 · 실제 크기 아님" 을 띄움)
  if (ex) ex.addEventListener('click', () => {
    const on = ex.getAttribute('aria-pressed') !== 'true';
    ex.setAttribute('aria-pressed', String(on)); ex.classList.toggle('on', on);
    frame.contentWindow && frame.contentWindow.postMessage({ytExag: on ? 10 : 1}, location.origin);
  });
})();
