from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'build.mjs').read_text(encoding='utf-8')
s=s.replace("const TMP=P+'/ppt_build';const FINAL=P+'/ppt_output/2026-10-15_QR_특징점이동_설명.pptx';", "const TMP=P+'/ppt_build_all755';const FINAL=P+'/ppt_output/2026-10-15_QR_전체755점_이동설명.pptx';")
s=s.replace('본문에는 같은 대표 36점을 크게 표시합니다.', '모든 이동 그림에 755개 특징점 전체를 표시합니다.')
s=s.replace('대표점은 QR별 4분면에서 center 품질순으로 최대 3개씩 선택하며, 점간 간격 20픽셀을 적용했습니다. 선택에는 이동 사진의 정답이나 추적 성적을 사용하지 않았습니다.', '전체 755점의 ID와 center 좌표를 고정했습니다. center에서 불확실했던 147점도 포함하며, 점을 선별하거나 대체하지 않았습니다.')
a=s.index('for(let i=0;i<data.length;i++){')
b=s.index("{\n const s=slide('현재 그림으로 확인한 범위'",a)
loop=r'''
for(let i=0;i<data.length;i++){
 const e=data[i],sign=e.sign<0?'−':'+';
 const range=`${e.axis} ${sign}${Math.abs(e.start).toFixed(2)} 부터 ${sign}${Math.abs(e.end).toFixed(2)} 까지`;
 const family=e.axis==='T'?'0.01 간격, 100장':e.extra_endpoint?'0.05 시작 계열 20장과 별도 2.00 끝점':'0.10 시작 계열, 20장';
 for(const amp of [1,100]){
  const s=slide(range,`${String(i+1).padStart(2,'0')} 구간    ${family}    전체 755점`);
  txt(s,amp===1?'실제 변위   가로 ×1 / 세로 ×1':'확대 변위   가로 ×1 / 세로 ×100',48,170,1820,58,37,amp===1?C.green:C.yellow,true);
  await image(s,P+`/slide_panels_bold/${e.slug}_all755_dy${amp}.png`,48,235,1824,608);
  txt(s,amp===1?'볼 것: 전체 점의 이동 방향과 길이, 세 QR 사이의 차이':'볼 것: 작은 세로 변화와 점별 경로 차이. 굴곡에는 측정 잡음도 포함됩니다.',48,868,1824,62,33,C.white);
  txt(s,'연두 네모: 시작     보라에서 노랑으로: 명령값 진행     노랑 화살촉: 끝',48,940,1824,50,29,C.gray);
  const special=e.extra_endpoint?'1.95 다음의 2.00은 별도 끝점으로 점선 연결했습니다. ':'';
  txt(s,special+(amp===100?'세로 변위를 100배 확대했으므로 실제 길이와 기울기로 읽지 않습니다.':'명령값 크기순 측정 경로이며 실제 촬영 시간 순 궤적과는 구분합니다.'),48,993,1824,40,25,C.yellow);
  note(s,`전체 755점이며 대표점 선별은 하지 않았습니다. center에서 불확실했던 147점도 유지했습니다. 모든 사진은 하나의 center에서 직접 추적했습니다. ${e.files.join(', ')}. ${special} 가로 변위 배율 1, 세로 변위 배율 ${amp}. 원본 파일명은 표시 순서에만 사용했습니다.`,[`slide_panels_bold/${e.slug}_all755_dy${amp}.png`,'tracks.npz','fixed_features.csv']);
 }
}
'''
s=s[:a]+loop+s[b:]
a=s.index("{\n const s=slide('부록: 전체 755점 경로'")
b=s.index('await(await PresentationFile.exportPptx(p))',a)
s=s[:a]+s[b:]
assert 'representative36' not in s and '대표 36점' not in s
(P/'build_all.mjs').write_text(s,encoding='utf-8')
print('Prepared all-755 deck builder, 15 slides')
