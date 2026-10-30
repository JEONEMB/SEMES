import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root='../../../', dirs=['20261028_1754_clean','20261028_1805_clean_20261020','20261029_1215_clean_20261029'];
function csv(t){let lines=t.trim().split(/\r?\n/);const parse=l=>[...l.matchAll(/(?:^|,)(?:"((?:[^"]|"")*)"|([^,]*))/g)].map(m=>(m[1]??m[2]).replaceAll('""','"'));let h=parse(lines.shift());return lines.map(l=>Object.fromEntries(parse(l).map((v,i)=>[h[i],v])));}
const datasets=await Promise.all(dirs.map(async(d,i)=>(csv(await fs.readFile(root+'evals/'+d+'/predictions.csv','utf8'))).filter(r=>i||r.path.startsWith('clean_20261014/')&&r.condition!=='center')));
const q=(a,p)=>{let v=(a.length-1)*p,l=Math.floor(v);return a[l]+(a[Math.ceil(v)]-a[l])*(v-l)};
function stat(rows,axis){let a=rows.map(r=>+r['error_'+axis]).sort((a,b)=>a-b),q1=q(a,.25),med=q(a,.5),q3=q(a,.75),iqr=q3-q1,lo=q1-1.5*iqr,hi=q3+1.5*iqr;return {n:a.length,mean:a.reduce((s,v)=>s+v,0)/a.length,q1,med,q3,low:a.find(v=>v>=lo),high:a.findLast(v=>v<=hi),max:a.at(-1),out:a.filter(v=>v<lo||v>hi),worst:rows.reduce((a,b)=>+a['error_'+axis]>+b['error_'+axis]?a:b),sum:a.reduce((s,v)=>s+v,0)};}
const stats=datasets.map(ds=>Object.fromEntries(['joint','control'].map(m=>[m,Object.fromEntries(['y','t'].map(a=>[a,stat(ds.filter(r=>r.module===m),a)]))])));
await fs.writeFile('stats.json',JSON.stringify(stats,null,2));
console.log(JSON.stringify(stats.map(s=>Object.fromEntries(Object.entries(s).map(([m,axes])=>[m,Object.fromEntries(Object.entries(axes).map(([a,v])=>[a,{...v,out:v.out.length}]))]))),null,2));
const p=Presentation.create({slideSize:{width:1280,height:720}}), ink='#111827',blue='#0073E6',orange='#D67518',muted='#6B7280',font='Noto Sans KR';
function shape(s,g,x,y,w,h,fill,line=fill,lw=1){return s.shapes.add({geometry:g,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:lw}})}
function txt(s,t,x,y,w,h,size=22,bold=false,color=ink){let a=shape(s,'textbox',x,y,w,h,'none','none',0);a.text=t;a.text.style={fontFamily:font,fontSize:size,bold,color};a.text.insets=0;return a;}
function line(s,x,y,w,h,c=muted,lw=1){shape(s,'line',x,y,w,h,'none',c,lw)}
function slide(title,kicker){let s=p.slides.add();s.background.fill='#F7F7F9';txt(s,kicker,77,32,1126,28,15,false,muted);txt(s,title,77,78,1126,62,35,true);return s;}
function boxplot(s,axis,ss,x,y,w,h,max,ticks){txt(s,axis==='y'?'Y 절대오차 (mm)':'T 절대오차 (명령 단위)',x,y-38,w,30,24,true);let px=x+64,pw=w-85,top=y+12,ph=h-66,Y=v=>top+ph*(1-v/max);for(let v of ticks){let yy=Y(v);line(s,px,yy,pw,0,'#DDE1E7');txt(s,String(v),x,yy-10,57,23,15,false,muted)}
 for(let j=0;j<2;j++){let m=['joint','control'][j],a=ss[m][axis],xx=px+pw*(j?.73:.27),c=j?orange:blue;line(s,xx,Y(a.high),0,Y(a.low)-Y(a.high),c,2);line(s,xx-17,Y(a.high),34,0,c,2);line(s,xx-17,Y(a.low),34,0,c,2);shape(s,'rect',xx-33,Y(a.q3),66,Math.max(1,Y(a.q1)-Y(a.q3)),j?'#F9E1C9':'#D6E9FF',c,2);line(s,xx-33,Y(a.med),66,0,ink,2);shape(s,'diamond',xx-6,Y(a.mean)-6,12,12,c,c);a.out.forEach((v,k)=>shape(s,'ellipse',xx-3+((k%5)-2)*4,Y(v)-3,6,6,'none',c,1));txt(s,m,xx-65,top+ph+12,130,26,21,true,c);txt(s,'평균 '+a.mean.toFixed(4)+'   중앙값 '+a.med.toFixed(4),x+(j?w/2:0),y+h-10,w/2,28,16,false,c);}
}
let s=slide('평균 절대오차의 계산과 박스플롯 읽기','02 Y/T 분리  /  성능 표 상세');
txt(s,'사진별 오차 = |추정값 − 파일명 명령값|',77,169,1126,48,30,true,blue);
txt(s,'평균 = 사진별 절대오차 합 ÷ 사진 수 N\nY와 T는 각각 계산하며, 두 축 동시 통과는 두 오차가 모두 0.09 이하인 사진 수',77,235,1126,85,25);
const cy=stats[0].control.y,ct=stats[0].control.t;
txt(s,`1차 control Y  ${cy.sum.toFixed(6)} ÷ 280 = ${cy.mean.toFixed(6)} → 0.0005`,77,349,1126,42,27,true);
const onlyY=datasets[0].filter(r=>r.module==='control'&&r.condition==='Y');
txt(s,`Y 이동 80장만 평균하면 ${stat(onlyY,'y').mean.toFixed(6)} mm\n표의 280장 평균에는 T 이동 200장의 비활성 Y 오차 0도 포함`,77,404,1126,83,25);
txt(s,'박스: 가운데 50% (Q1~Q3)     검은 선: 중앙값     ◆: 평균\n수염: Q1−1.5×IQR ~ Q3+1.5×IQR 안의 관측 범위     ○: 그 밖의 관측값',77,527,1126,78,22);
txt(s,'이상치는 분포 기준이며 0.09 허용오차 초과와는 별개  /  1차 센터 제외, 2차 센터 3장 포함',77,646,1126,35,18,false,muted);
s.speakerNotes.text='출처: evals/'+dirs[0]+'/predictions.csv, '+dirs[1]+'/predictions.csv, '+dirs[2]+'/predictions.csv. 원래 표와 동일한 부분집합. 분위수는 선형 보간 (n−1)p. IQR=Q3−Q1. 기존 사진 재채점이며 독립 검증 아님. 박스플롯은 편집 가능한 PowerPoint 도형으로 구성.';
const ranges=[[[.06,[0,.02,.04,.06]],[.016,[0,.004,.008,.012,.016]]],[[1.8,[0,.6,1.2,1.8]],[2.1,[0,.5,1,1.5,2]]],[[2.5,[0,.5,1,1.5,2,2.5]],[.45,[0,.1,.2,.3,.4]]]];
for(let i=0;i<3;i++){let ss=stats[i],n=[280,60,234][i];s=slide(['1차 단일축 280장 오차 분포','2차 센터 포함 60장 오차 분포','3차 혼합 234장 오차 분포'][i],'02 Y/T 분리  /  사진별 절대오차');boxplot(s,'y',ss,77,192,542,260,...ranges[i][0]);boxplot(s,'t',ss,660,192,542,260,...ranges[i][1]);
 txt(s,'최대오차 사례  ·  정답 (Y, T) → 추정 (Y, T)',77,486,1126,32,23,true);
 for(let j=0;j<2;j++){let m=['joint','control'][j],a=ss[m].y,b=ss[m].t;function fmt(v){return (+v).toFixed(4)};let r=a.worst,rr=b.worst;txt(s,`${m}  Y 최대 ${a.max.toFixed(4)}: (${r.label_y}, ${r.label_t}) → (${fmt(r.est_y)}, ${fmt(r.est_t)})`,77,532+j*43,1126,32,21,j===1,j?orange:blue);txt(s,`T 최대 ${b.max.toFixed(4)}: 정답 T=${rr.label_t}, 추정 T=${fmt(rr.est_t)}   ·   IQR 밖 Y ${a.out.length}장 / T ${b.out.length}장`,77,620+j*0,1126,30,18,false,muted);if(j===0){/* detailed T cases retained in notes to keep slide readable */}}
 const notes=[`1차 control: Y 이동 80장, T 이동 200장. 두 축 오차 모두 0.09 이내.`,`2차 control: 혼합 입력에서 한 축만 출력. T=2인 12장 모두 est_t=0, T 오차=2. 이 12장이 전체 T 절대오차 합의 ${(24/ss.control.t.sum*100).toFixed(1)}%를 차지.`,`3차 control: 혼합 입력에서 한 축만 출력. Y 최대 사례는 T=0.43에서 Y로 판단하여 T=0을 출력. 원인 단정은 관측·모델 진단이 추가로 필요.`];
 txt(s,notes[i].replace(/\. 이 12장이.*$/,''),77,665,1126,35,17,false,muted);
 // Replace shared diagnostic line with one compact legend.
 for(const sh of [...s.shapes.items])if(String(sh.text).includes('IQR 밖'))s.shapes.deleteById(sh.id);
 txt(s,`◆ 평균  ·  검은 선 중앙값  ·  ○ 1.5×IQR 밖  /  선형축, 0부터 표시  /  N=${n} (모듈별)`,77,620,1126,30,18,false,muted);
 s.speakerNotes.text=notes[i]+'\n출처: evals/'+dirs[i]+'/predictions.csv\n전체 통계 및 최대오차 원본 행:\n'+JSON.stringify(ss,null,2);
}
s=slide('T 최대오차 사례와 평균을 키우는 구간','02 Y/T 분리  /  큰 오차의 실제 출력');
const values=[['자료 / 모듈','T 정답','T 추정','T 절대오차','사진명']];
for(let i=0;i<3;i++)for(const m of ['joint','control']){let a=stats[i][m].t,r=a.worst;values.push([`${i+1}차 ${m}`,r.label_t,(+r.est_t).toFixed(4),a.max.toFixed(4),r.path.split('/').at(-1)]);}
const tb=s.tables.add({rows:7,columns:5,left:77,top:166,width:1126,height:322,columnWidths:[180,130,140,160,516],values});
for(let r=0;r<7;r++)for(let c=0;c<5;c++){let cell=tb.getCell(r,c);cell.fill=r===0?ink:r%2?'#FFFFFF':'#EEF0F4';cell.text.style={fontFamily:font,fontSize:c===4?18:22,bold:r===0,color:r===0?'#FFFFFF':ink};}
tb.borders.assign({fill:'#D6DBE3',width:1,style:'solid'});
txt(s,`2차 control: T=2인 12장이 전체 T 오차 합의 ${(24/stats[1].control.t.sum*100).toFixed(1)}%\n평균 0.4491은 이 큰 오차군의 영향을 크게 받음 (중앙값 ${stats[1].control.t.med.toFixed(4)})`,77,521,1126,77,24,true);
txt(s,`3차 control Y: 중앙값 ${stats[2].control.y.med.toFixed(4)}, 평균 0.5424, 최대 2.3406\n최대 한 장뿐 아니라 분포의 폭도 함께 확인  /  혼합 입력에서 한 축만 출력하는 제약`,77,613,1126,70,22);
s.speakerNotes.text='출처: evals/ 각 차수 predictions.csv. 동일 최대값이 여러 장이면 대표 한 장을 표시. 전체 최대 행은 앞 세 슬라이드 노트 및 stats.json에 기록. 2차 T=2 12장의 오차 합=24. 평균, 중앙값과 IQR 기준 이상치는 허용오차 초과 여부와 다름.';
await (await PresentationFile.exportPptx(p)).save('additions.pptx');
for(let i=0;i<p.slides.items.length;i++)await fs.writeFile('new'+(i+1)+'.png',new Uint8Array(await (await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
