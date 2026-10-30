from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'build.mjs').read_text(encoding='utf-8')
s=s.replace("const TMP=P+'/ppt_build';const FINAL=P+'/ppt_output/2026-10-15_QR_특징점이동_설명.pptx';", "const TMP=P+'/ppt_build_reference_four';const FINAL=P+'/ppt_output/2026-10-15_기존표현유지_새특징점755_4장.pptx';")
a=s.index("{\n const s=slide('QR 모서리 이동 시각화'")
b=s.index('await(await PresentationFile.exportPptx(p))',a)
body=r'''
for(const [axis,kind] of [['Y','paths'],['T','paths'],['Y','residual'],['T','residual']]){
 const isPath=kind==='paths';
 const s=slide(`${axis} 양의 방향 ${isPath?'이동 경로':'공통 이동을 뺀 미세 변형'}`,'기존 표현 유지, 새 특징점 전체 755개');
 await image(s,P+`/reference_style_four/${axis}_${kind}_755.png`,48,210,1824,607.17);
 txt(s,isPath?'흰 점: 시작 위치    보라에서 노랑으로: 명령값 진행    굵은 선: 2차 추세선':'하늘색 점: 시작 위치    화살표: 공통 이동을 뺀 끝값 변위',48,854,1824,75,32,C.gray);
 txt(s,isPath?'가로 변위 ×1, 세로 변위 ×100. 옅은 점은 실제 측정입니다.':'미세 변형 ×50. 색상과 화살표 길이는 잔여 변위의 크기를 나타냅니다.',48,938,1824,70,31,C.yellow);
 note(s,'원본 analysis_v7b/trial_tracks_ppt.py의 표현 의미를 유지하고 새 QR 모서리 755개로 교체했습니다. 좌표는 고정 center에서 직접 추적했으며 첫 사진 대비 변위는 표시를 위해서만 계산합니다. 모형 보정용 시행별 영점이 아닙니다. 원본과 동일한 근접 QR 영역을 표시하므로 끝 경로가 그림 경계를 넘어가면 잘릴 수 있습니다. 구조적 점 전체를 선택했고 수를 줄이지 않았습니다.',[`reference_style_four/${axis}_${kind}_755.png`,'reference_style_four/settings.json']);
}
'''
s=s[:a]+body+s[b:]
(P/'build_reference_four.mjs').write_text(s,encoding='utf-8')
