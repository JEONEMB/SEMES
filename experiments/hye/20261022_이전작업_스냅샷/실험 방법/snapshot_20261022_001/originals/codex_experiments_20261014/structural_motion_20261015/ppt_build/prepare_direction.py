from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'build.mjs').read_text(encoding='utf-8')
s=s.replace("const TMP=P+'/ppt_build';const FINAL=P+'/ppt_output/2026-10-15_QR_특징점이동_설명.pptx';", "const TMP=P+'/ppt_build_direction';const FINAL=P+'/ppt_output/2026-10-15_Y_T_양의방향_전체755점.pptx';")
a=s.index("{\n const s=slide('QR 모서리 이동 시각화'")
b=s.index('await(await PresentationFile.exportPptx(p))',a)
body=r'''
{
 const s=slide('Y와 T의 양의 방향','전체 755점의 이동 방향을 같은 길이의 굵은 화살표로 표시했습니다');
 txt(s,'Y 양의 방향',48,292,490,85,53,C.green,true);
 txt(s,'전체 755점',48,390,490,65,31,C.gray);
 await image(s,P+'/positive_direction_only/Y_positive_direction_all.png',590,153,1280,416);
 txt(s,'T 양의 방향',48,738,490,85,53,C.green,true);
 txt(s,'전체 755점',48,836,490,65,31,C.gray);
 await image(s,P+'/positive_direction_only/T_positive_direction_all.png',590,600,1280,416);
 note(s,'화살표는 같은 center 모서리 좌표에서 시작합니다. 각 양의 명령 구간의 마지막 사진 좌표에서 첫 사진 좌표를 뺀 순변위를 정규화했습니다. 모든 화살표 길이가 같으며 이동량은 표시하지 않습니다. 원본 영상 좌표의 방향을 그대로 사용하고 세로 성분을 확대하지 않았습니다. 전체 755점을 빠짐없이 표시합니다.',['positive_direction_only/method.json','positive_direction_only/directions.csv']);
}
for(const axis of ['Y','T']){
 const s=slide(`${axis} 양의 방향`,'전체 755점의 이동 방향');
 await image(s,P+`/positive_direction_only/${axis}_positive_direction_all.png`,48,230,1824,593);
 txt(s,'모든 화살표의 길이는 같습니다. 화살촉이 향하는 방향만 비교합니다.',48,885,1824,80,35,C.green);
 note(s,'이 그림은 이동 방향만 표현합니다. 화살표 길이와 굵기는 가독성을 위해 고정했습니다. 원본 영상에서 측정한 방향을 보존했으며 가로와 세로의 비율을 다르게 확대하지 않았습니다. center 불확실점을 포함한 전체 755개 점을 표시합니다. 이동량을 추정하거나 Y/T 판정 성능을 평가한 장표가 아닙니다.',[`positive_direction_only/${axis}_positive_direction_all.png`,'positive_direction_only/method.json']);
}
'''
s=s[:a]+body+s[b:]
(P/'build_direction.mjs').write_text(s,encoding='utf-8')
print('Prepared 3 direction-only slides')
