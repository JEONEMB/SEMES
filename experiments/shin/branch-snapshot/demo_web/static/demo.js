'use strict';
(() => {
  const {$,api,text,number,mode,action}=UI;
  const operator=document.querySelector('.demo-shell').dataset.operator==='true';
  let current, pending=false, connected=false;
  function render(state) {
    current=state; mode(state);
    text('trial',String(state.trial).padStart(2,'0'));
    $('trial-progress').value=state.trial;
    text('y-value',number(state.y_um)); text('t-value',number(state.t_mdeg));
    text('hardware-label',state.mode==='mock'?'제어기 모의 연결 · T축 실물 비활성':state.controller_connected?'STM32 연결 · T축 비활성':'STM32 연결 대기');
    text('mode-description',state.mode==='mock'?'모의 명령으로 실행 중 · 실제 모터를 구동하지 않습니다':'저장된 시나리오 실행 · T축 실물 비활성');
    $('execution-state').dataset.state=state.status;
    const moving=['positioning','correcting','stopping','manual_move','homing'].includes(state.status);
    $('home').disabled=!operator||pending||!connected||moving||state.status==='ready';
    $('random').disabled=!operator||pending||!connected||moving||state.status==='ready';
    $('correct').disabled=!operator||pending||!connected||state.status!=='ready';
    $('stop').disabled=!operator||!connected||!['positioning','correcting','ready','error','manual_move','homing'].includes(state.status);
  }
  const events=new EventSource('/api/events');
  events.onmessage=e=>{connected=true;text('connection-status','● 서버 연결됨');render(JSON.parse(e.data));};
  events.onerror=()=>{connected=false;text('connection-status','○ 서버 재연결 중');if(current)render(current);};
  for(const [id,endpoint] of [['home','home'],['random','random'],['correct','correct'],['stop','stop']]) {
    $(id).onclick=async()=>{pending=true;if(current)render(current);await action(async()=>render(await api('/api/demo/'+endpoint,'POST',{},true)));pending=false;if(current)render(current);};
  }
  $('fullscreen').onclick=()=>action(async()=>{
    if(document.fullscreenElement) await document.exitFullscreen();else await document.documentElement.requestFullscreen();
  });
})();
