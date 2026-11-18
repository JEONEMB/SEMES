'use strict';
(() => {
  const {$,api,toast,number,units,text,mode,action,statuses}=UI;
  let scenarios=[], selected=null, reference=null, state=null, steps=[], dirty=false, pending=false;
  let lastRun=null, refreshTimer=null, deviceInitialized=false, consoleRun=null;
  function consoleLog(lines){const out=$('console-output');out.textContent+=[].concat(lines).join('\n')+'\n';out.scrollTop=out.scrollHeight;}
  function consoleInt(id){const v=Number($(id).value);if($(id).value===''||!Number.isInteger(v))throw new Error('정수로 입력하세요. 단위는 rsctl과 같습니다.');return v;}
  function markDirty(){dirty=true;buttons();}
  function make(tag, textContent, className) {const el=document.createElement(tag);if(textContent!==undefined)el.textContent=textContent;if(className)el.className=className;return el;}
  function buttons() {
    const moving=state&&['positioning','correcting','stopping','manual_move','homing'].includes(state.status);
    const locked=pending||!state||moving||state.status==='ready';
    const device=state?.device_status;
    const deviceMoving=device&&[1,2].includes(device.state);
    $('device-apply').disabled=locked||deviceMoving;
    $('camera-apply').disabled=locked||deviceMoving;
    // HOME always goes to the real STM32 (kernel driver in mock mode), so it does not need a probe first.
    $('home-y').disabled=locked||deviceMoving;
    document.querySelectorAll('[data-cmd]').forEach(b=>{b.disabled=pending&&b.dataset.cmd!=='stop';});
    $('jog-minus').disabled=$('jog-plus').disabled=locked||!device||device.state!==0||!(device.flags&1)||!state.controller_connected;
    $('manual-stop').disabled=!state||!(moving||deviceMoving||['ready','error'].includes(state.status));
    const matching=state?.snapshot?.scenario.id===selected?.id && state?.snapshot?.scenario.revision===selected?.revision;
    $('test-prepare').disabled=pending||dirty||!selected||moving||state?.status==='ready';
    $('test-step').disabled=pending||!matching||state?.status!=='ready'||!state?.test;
    $('test-auto').disabled=pending||!matching||state?.status!=='ready';
    $('test-stop').disabled=!state||!['positioning','correcting','ready','error','manual_move','homing'].includes(state.status);
    $('verify').disabled=pending||dirty||!matching||!state?.test||state?.status!=='complete'||!$('verify-confirm').checked;
    $('activate').disabled=pending||dirty||!selected||selected.verified_mode!==state?.mode||selected.reference_revision!==reference?.revision;
    text('activate',selected?.active?'시연 비활성화':'③ 시연에 활성화');
    $('duplicate').disabled=!selected;$('delete-scenario').disabled=!selected;
    $('add-step').disabled=steps.length>=20;
  }
  function list() {
    text('scenario-count',scenarios.length);$('scenario-list').replaceChildren();
    if(!scenarios.length){$('scenario-list').append(make('p','저장된 시나리오가 없습니다. 새 시나리오 또는 모의 예제로 시작하세요.','empty-list'));return;}
    scenarios.forEach(s=>{
      const button=make('button',undefined,'scenario-item'+(s.id===selected?.id?' selected':''));
      button.append(make('span',String(s.order).padStart(2,'0'),'scenario-order'),make('strong',s.name));
      button.append(make('small',`${s.steps.length}단계 · ${s.active?'활성':s.verified_mode==='mock'?'모의 검증':s.verified_mode?'실물 검증':'초안'}`));
      button.onclick=()=>{if(!dirty||confirm('저장하지 않은 내용을 버리고 이동할까요?'))edit(s);};
      $('scenario-list').append(button);
    });
  }
  function edit(s=null) {
    selected=s;dirty=false;steps=s?structuredClone(s.steps):[{y_um:0,t_mdeg:0,settle_ms:300}];
    $('scenario-name').value=s?.name||'';$('scenario-description').value=s?.description||'';
    $('scenario-order').value=s?.order??0;$('start-y').value=number(s?.start_y_um??0);$('start-t').value=number(s?.start_t_mdeg??0);
    text('editor-heading',s?`시나리오 편집 · v${s.revision}`:'새 시나리오');
    text('verification-badge',s?.verified_mode==='mock'?'모의 검증':s?.verified_mode?'실물 검증':'초안');
    $('verify-confirm').checked=false;rows();list();buttons();
  }
  function readRows() {
    return [...$('steps-body').rows].map(row=>({y_um:units(row.querySelector('[data-field="y_um"]').value),
      t_mdeg:units(row.querySelector('[data-field="t_mdeg"]').value),settle_ms:Number(row.querySelector('[data-field="settle_ms"]').value)}));
  }
  function endpoint() {
    try {
      steps=readRows();let y=units($('start-y').value),t=units($('start-t').value);
      steps.forEach(s=>{y+=s.y_um;t+=s.t_mdeg;});
      text('endpoint',`최종 예상 위치  Y ${number(y)} mm / T ${number(t)}°`);
      $('endpoint').classList.toggle('invalid',!!reference&&(Math.abs(y-reference.center_y_um)>reference.tolerance_y_um||Math.abs(t-reference.center_t_mdeg)>reference.tolerance_t_mdeg));
    } catch(error){text('endpoint',error.message);$('endpoint').classList.add('invalid');}
  }
  function rows() {
    $('steps-body').replaceChildren();
    steps.forEach((step,index)=>{
      const row=make('tr');row.append(make('td',String(index+1).padStart(2,'0')));
      for(const field of ['y_um','t_mdeg','settle_ms']) {
        const td=make('td'),input=make('input');input.type='number';input.required=true;input.dataset.field=field;
        input.step=field==='settle_ms'?'1':'0.001';input.value=field==='settle_ms'?step[field]:number(step[field]);
        input.setAttribute('aria-label',`${index+1}단계 ${field==='y_um'?'Y 보정량':field==='t_mdeg'?'T 보정량':'대기 시간'}`);
        if(field==='settle_ms'){input.min='0';input.max='10000';}
        input.oninput=()=>{markDirty();endpoint();};td.append(input);row.append(td);
      }
      const controls=make('td',undefined,'row-tools');
      for(const [label,delta] of [['↑',-1],['↓',1],['×',0]]) {
        const button=make('button',label,'icon-button');button.type='button';button.setAttribute('aria-label',`${index+1}단계 ${delta<0?'위로':delta>0?'아래로':'삭제'}`);
        button.disabled=delta<0?index===0:delta>0?index===steps.length-1:steps.length===1;
        button.onclick=()=>action(()=>{steps=readRows();if(delta){[steps[index],steps[index+delta]]=[steps[index+delta],steps[index]];}else{steps.splice(index,1);}markDirty();rows();});controls.append(button);
      }
      row.append(controls);$('steps-body').append(row);
    });
    text('step-count',`${steps.length} / 20`);endpoint();buttons();
  }
  async function refreshScenarios(keep=true) {
    scenarios=await api('/api/admin/scenarios');
    if(keep&&selected){selected=scenarios.find(s=>s.id===selected.id)||null;}
    list();buttons();
  }
  async function history() {
    const runs=await api('/api/admin/runs');$('history-body').replaceChildren();
    if(!runs.length){const row=make('tr'),td=make('td','아직 실행 기록이 없습니다.','muted');td.colSpan=5;row.append(td);$('history-body').append(row);}
    runs.forEach(run=>{
      const row=make('tr');[new Date(run.started).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'}),run.snapshot.scenario.name,
        (run.snapshot.mode==='mock'?'모의':'실물')+(run.snapshot.kind==='manual_y'?' · 수동':run.snapshot.test?' · 시험':' · 시연'),run.state.trial,statuses[run.state.status]||run.state.status]
        .forEach(value=>row.append(make('td',value)));$('history-body').append(row);
    });
  }
  async function adminState() {
    state=await api('/api/admin/state');mode(state);
    if(!deviceInitialized){syncDeviceForms();deviceInitialized=true;}
    const device=state.device_status, limits=state.device_limits;
    text('device-y',device?number(device.position[0]):'—');
    text('device-home',device?(device.flags&1?'설정됨':'미설정'):'—');
    text('device-limit',limits?`${number(limits[0])} ~ ${number(limits[1])}`:'—');
    const deviceStates=['대기','이동 중','원점 복귀 중','오류'];
    text('device-message',state.device_error||(device?`${state.mode==='mock'?'모의 장치':state.transport==='serial'?'USB 직접 연결':'커널 드라이버'} · ${deviceStates[device.state]||'상태 확인 필요'}`:'장치 응답 확인 중'));
    const camera=state.camera;
    text('camera-settings-status',`${camera.width} × ${camera.height} · 목표 ${camera.fps} fps · 왜곡 보정 ${camera.undistort?'켜짐':'꺼짐'} · ${camera.message}`);
    text('test-trial',`${state.trial} / 20`);text('test-y',number(state.y_um));text('test-t',number(state.t_mdeg));
    text('hardware-note',state.mode==='mock'?'모의 통신 · T값은 시뮬레이션만 실행합니다.':'T축 모터 미장착 · T=0인 Y축 시나리오만 실행합니다.');
    text('verify-label',state.mode==='mock'?'모의 실행 결과를 확인했습니다.':'실제 영상에서 중앙 정렬을 확인했습니다.');
    if(lastRun!==state.run_id){$('verify-confirm').checked=false;lastRun=state.run_id;}
    $('run-log').replaceChildren();
    state.history.forEach(item=>{
      const details=item.event==='ack'?`${item.manual_action==='home'?'원점 복귀':item.manual_action==='jog'?'수동 이동':item.relative?'보정':'시작 이동'} ACK · Y ${number(item.y_um)} / T ${number(item.t_mdeg)}`:
        item.event==='done'?`DONE · 결과 ${item.result} · Y ${number(item.position[0])} / T ${number(item.position[1])}`:item.message;
      $('run-log').append(make('li',details));
    });
    if(consoleRun&&state.run_id===consoleRun&&!['homing','manual_move','stopping'].includes(state.status)){
      const done=[...state.history].reverse().find(item=>item.event==='done');
      if(done)consoleLog(`DONE ${['OK','STOPPED','LIMIT','FAULT'][done.result]??done.result} (cmd 0x${done.cmd_type.toString(16).padStart(2,'0')})  y=${done.position[0]} t=${done.position[1]}`);
      if(state.status==='error')consoleLog('오류: '+state.message);
      consoleRun=null;
    }
    buttons();
  }
  function syncDeviceForms(){
    $('device-mode').value=state.mode;
    $('device-transport').value=state.transport==='serial'?'serial':'driver';
    $('camera-enabled').value=String(state.camera.enabled);
    $('camera-profile').value=state.camera.profile;
    $('camera-undistort').value=String(state.camera.undistort);
  }
  async function command(fn) {pending=true;buttons();try{await action(fn);}finally{await action(adminState);pending=false;buttons();}}
  $('device-form').onsubmit=event=>{event.preventDefault();command(async()=>{
    try{await api('/api/admin/device','PUT',{mode:$('device-mode').value,transport:$('device-transport').value});}
    finally{await adminState();syncDeviceForms();}
  });};
  $('home-y').onclick=()=>command(()=>api('/api/admin/y/home','POST',{},true));
  for(const [id,sign] of [['jog-minus',-1],['jog-plus',1]])$(id).onclick=()=>command(()=>{
    const distance=units($('jog-distance').value);
    if(distance<1||distance>1000)throw new Error('한 번 이동할 거리는 0.001~1.000 mm입니다.');
    return api('/api/admin/y/jog','POST',{y_um:sign*distance},true);
  });
  document.querySelectorAll('[data-cmd]').forEach(button=>button.onclick=()=>command(async()=>{
    const cmd=button.dataset.cmd, data={cmd};
    if(cmd==='move-rel'||cmd==='move-abs'){data.y=consoleInt('console-y');data.t=consoleInt('console-t');}
    if(cmd==='speed'){data.y=consoleInt('console-vy');data.t=consoleInt('console-vt');}
    consoleLog(`$ rsctl ${cmd}${cmd==='home'?' 1':data.y!==undefined?` ${data.y} ${data.t}`:''}`);
    try{const result=await api('/api/admin/stm32','POST',data,true);consoleLog(result.output);if(result.run_id)consoleRun=result.run_id;}
    catch(error){consoleLog(error.message);throw error;}
  }));
  $('console-clear').onclick=()=>{$('console-output').textContent='';};
  $('manual-stop').onclick=()=>command(()=>api('/api/demo/stop','POST',{},true));
  $('camera-form').onsubmit=event=>{event.preventDefault();command(async()=>{
    await api('/api/admin/camera','PUT',{profile:$('camera-profile').value,enabled:$('camera-enabled').value==='true',undistort:$('camera-undistort').value==='true'});
    reference=await api('/api/admin/settings');text('reference-version','기준 v'+reference.revision);
    await refreshScenarios();endpoint();await adminState();syncDeviceForms();
    toast(state.camera.enabled&&!state.camera.connected?'카메라 설정 저장됨 · 연결 상태를 확인하세요.':'카메라 설정을 저장했습니다.');
  });};
  $('scenario-form').addEventListener('input',markDirty);
  $('start-y').addEventListener('input',endpoint);$('start-t').addEventListener('input',endpoint);
  $('add-step').onclick=()=>action(()=>{steps=readRows();if(steps.length<20)steps.push({y_um:0,t_mdeg:0,settle_ms:300});markDirty();rows();});
  $('new-scenario').onclick=()=>{if(!dirty||confirm('저장하지 않은 내용을 버릴까요?'))edit();};
  $('scenario-form').onsubmit=event=>{event.preventDefault();action(async()=>{
    const data={name:$('scenario-name').value,description:$('scenario-description').value,order:Number($('scenario-order').value),
      start_y_um:units($('start-y').value),start_t_mdeg:units($('start-t').value),steps:readRows(),revision:selected?.revision};
    const saved=await api('/api/admin/scenarios'+(selected?'/'+selected.id:''),selected?'PUT':'POST',data);
    await refreshScenarios(false);edit(saved);toast('초안을 저장했습니다. 시험 실행 후 검증해 주세요.');
  });};
  $('duplicate').onclick=()=>action(async()=>{const original=await api('/api/admin/scenarios/'+selected.id);const saved=await api('/api/admin/scenarios','POST',{...original,name:original.name.slice(0,73)+' (복제)'});await refreshScenarios(false);edit(saved);toast('저장된 버전을 초안으로 복제했습니다.');});
  $('delete-scenario').onclick=()=>{if(selected&&confirm(`“${selected.name}” 시나리오를 삭제할까요? 실행 이력은 남습니다.`))action(async()=>{await api('/api/admin/scenarios/'+selected.id,'DELETE');edit();await refreshScenarios();});};
  $('example').onclick=()=>{
    if(dirty&&!confirm('입력 중인 내용을 모의 예제로 바꿀까요?'))return;
    const c=reference||{center_y_um:0,center_t_mdeg:0};
    edit();$('scenario-name').value='모의 예제 · Y 5단계';$('scenario-description').value='화면 및 실행 흐름 검증용. 실제 장치용 수치가 아닙니다.';
    $('start-y').value=number(c.center_y_um+3000);$('start-t').value=number(c.center_t_mdeg);
    steps=[-1200,-800,-500,-300,-200].map(y_um=>({y_um,t_mdeg:0,settle_ms:350}));markDirty();rows();
  };
  $('settings-form').onsubmit=event=>{event.preventDefault();if(!confirm('중앙 기준을 변경하면 모든 시나리오의 검증·활성화가 해제됩니다. 저장할까요?'))return;action(async()=>{
    reference=await api('/api/admin/settings','PUT',{revision:reference.revision,center_y_um:units($('center-y').value),center_t_mdeg:units($('center-t').value),
      tolerance_y_um:units($('tolerance-y').value),tolerance_t_mdeg:units($('tolerance-t').value),reference_note:$('reference-note').value});
    text('reference-version','기준 v'+reference.revision);await refreshScenarios();endpoint();toast('중앙 기준을 저장했습니다. 시나리오를 다시 검증하세요.');
  });};
  $('test-prepare').onclick=()=>command(()=>api('/api/admin/scenarios/'+selected.id+'/test','POST',{},true));
  $('test-step').onclick=()=>command(()=>api('/api/admin/step','POST',{},true));
  $('test-auto').onclick=()=>command(()=>api('/api/demo/correct','POST',{},true));
  $('test-stop').onclick=()=>command(()=>api('/api/demo/stop','POST',{},true));
  $('probe').onclick=()=>command(()=>api('/api/admin/probe','POST',{}));
  $('verify-confirm').onchange=buttons;
  $('verify').onclick=()=>command(async()=>{await api('/api/admin/scenarios/'+selected.id+'/verify','POST',{confirmed:true,run_id:state.run_id});await refreshScenarios();text('verification-badge',state.mode==='mock'?'모의 검증':'실물 검증');toast('검증을 등록했습니다. 시연에 활성화할 수 있습니다.');});
  $('activate').onclick=()=>command(async()=>{await api('/api/admin/scenarios/'+selected.id+'/activate','POST',{active:!selected.active});await refreshScenarios();toast(selected.active?'시연 목록에 활성화했습니다.':'시연 목록에서 비활성화했습니다.');});
  $('refresh-history').onclick=()=>action(history);
  $('logout').onclick=()=>action(async()=>{await api('/api/admin/logout','POST',{});location.href='/';});
  $('import-file').onchange=event=>action(async()=>{
    const file=event.target.files[0];if(!file)return;
    if(file.size>256*1024)throw new Error('백업 파일은 256KB 이하여야 합니다.');
    const result=await api('/api/admin/import','POST',JSON.parse(await file.text()));await refreshScenarios();toast(result.message);event.target.value='';
  });
  window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
  action(async()=>{
    reference=await api('/api/admin/settings');
    for(const [id,key] of [['center-y','center_y_um'],['center-t','center_t_mdeg'],['tolerance-y','tolerance_y_um'],['tolerance-t','tolerance_t_mdeg']])$(id).value=number(reference[key]);
    $('reference-note').value=reference.reference_note;text('reference-version','기준 v'+reference.revision);
    await refreshScenarios();edit(scenarios[0]||null);await adminState();await history();
    const events=new EventSource('/api/events');
    events.onmessage=event=>{const next=JSON.parse(event.data);mode(next);clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>action(adminState),100);};
    events.onerror=()=>{text('status-message','서버 재연결 중');};
  });
})();
