'use strict';
window.UI = (() => {
  const $ = id => document.getElementById(id);
  const statuses = {idle:'준비 필요',positioning:'시작 위치로 이동 중',ready:'보정 대기',correcting:'보정 중',
    complete:'보정 완료',max_trials:'최대 시도 도달',exhausted:'중앙 미도달',stopping:'정지 중',stopped:'중단',
    error:'오류 · 장치 확인 필요',interrupted:'재시작으로 중단',manual_move:'Y축 수동 이동 중',homing:'Y축 원점 복귀 중',home_done:'Y축 원점 복귀 완료',manual_done:'Y축 수동 시험 완료'};
  let toastTimer;
  function toast(message, error=false) {
    const el=$('toast'); el.textContent=message; el.hidden=false; el.classList.toggle('error',error);
    clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,error?8000:4000);
  }
  async function api(path, method='GET', data, command=false) {
    const headers={};
    if(method!=='GET') headers['X-CSRF-Token']=document.querySelector('meta[name="csrf-token"]').content;
    if(data!==undefined) headers['Content-Type']='application/json';
    if(command) headers['Idempotency-Key']=token();
    const response=await fetch(path,{method,headers,body:data===undefined?undefined:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok) throw new Error(result.error||'요청을 처리하지 못했습니다.');
    return result;
  }
  // crypto.randomUUID is unavailable on some LAN HTTP origins.
  function token() {return Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');}
  const number = v => (v/1000).toFixed(3);
  function units(value) {
    const n=Number(value), scaled=n*1000;
    if(value===''||!Number.isFinite(n)||Math.abs(scaled-Math.round(scaled))>1e-6) throw new Error('수치는 소수점 셋째 자리까지 입력하세요.');
    return Math.round(scaled);
  }
  function text(id, value) {if($(id)) $(id).textContent=value;}
  let cameraLive=false, cameraRetry=0, cameraGeneration=-1;
  function camera(state) {
    const live=state.camera.connected, img=$('camera-feed');
    if(!img) return;
    if(live && (!cameraLive||cameraGeneration!==state.camera.generation||(!img.getAttribute('src')&&Date.now()>cameraRetry))) {
      img.src='/camera/stream.mjpg?connection='+Date.now();
    }
    if(!live) img.removeAttribute('src');
    img.hidden=!live; $('camera-empty').hidden=live;
    cameraLive=live;
    cameraGeneration=state.camera.generation;
    img.onerror=()=>{img.hidden=true; $('camera-empty').hidden=false; img.removeAttribute('src');cameraRetry=Date.now()+3000;};
    text('camera-message',state.camera.message);
    text('camera-label',live?`LIVE · ${state.camera.width} × ${state.camera.height}${state.camera.undistort?' · 왜곡 보정':''}`:'OFFLINE');
    $('camera-dot')?.classList.toggle('online',live);
  }
  function mode(state) {
    text('mode',state.mode==='mock'?'모의 통신 모드':'실물 모드 · Y축');
    $('mode')?.classList.toggle('mock',state.mode==='mock');
    text('status-title',statuses[state.status]||state.status);
    text('status-message',state.message);
    camera(state);
  }
  async function action(fn) {try{return await fn();}catch(error){toast(error.message,true);return null;}}
  return {$,api,toast,number,units,text,mode,statuses,action};
})();
