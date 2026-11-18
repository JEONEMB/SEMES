from pathlib import Path
import os,ast,json,csv,hashlib,html,re
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parent;B=R/'codex_experiments_20261014'
def link(p,label=None):
    p=Path(p);rel=os.path.relpath(p,H).replace('\\','/')
    return '<a href="'+html.escape(rel,quote=True)+'">'+html.escape(label or str(p.relative_to(R)))+'</a>'
skip={'node_modules','deps','.git','.uv-cache','__pycache__','.venv','.review_ppt',H.name}
files=[]
for d,dirs,fs in os.walk(R):
    dirs[:]=[x for x in dirs if x not in skip]
    for f in fs:
        p=Path(d)/f
        if p.suffix.lower() in {'.py','.ps1','.mjs','.csv','.json','.npz','.npy','.md','.html','.pptx'}:files.append(p)
inventory=[];codes=[];tables=[];errors=[]
for p in sorted(files):
    rec=dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,kind=p.suffix)
    if p.suffix in {'.py','.ps1','.mjs','.csv','.json','.md'} and p.stat().st_size<20_000_000:rec['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    inventory.append(rec)
    if p.suffix=='.py':
        try:
            txt=p.read_text(encoding='utf-8-sig');tree=ast.parse(txt)
            funcs=[f'{n.name}:{n.lineno}' for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            io=sorted(set(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)<220 and re.search(r'\.(csv|npz|npy|json|bmp|png|html|pptx)$',n.value)))
            codes.append(dict(path=rec['path'],description=(ast.get_docstring(tree) or '').split('\n')[0],definitions='; '.join(funcs),literal_file_references='; '.join(io),status='인덱스 생성 단계는 정적 소스 확인 / 실행 결과는 개별 결과표 참조'))
        except Exception as e:errors.append(dict(path=rec['path'],error=str(e)))
    if p.suffix=='.csv':
        try:
            with p.open(encoding='utf-8-sig',newline='') as f:
                rd=csv.reader(f);cols=next(rd);n=sum(1 for _ in rd)
            tables.append(dict(path=rec['path'],rows=n,columns=' | '.join(cols),summary_candidate=bool(re.search(r'summary|metrics|ablation|score|평가|결과|accuracy|comparison',p.name,re.I))))
        except Exception as e:errors.append(dict(path=rec['path'],error=str(e)))
pd.DataFrame(inventory).to_csv(H/'file_inventory.csv',index=False,encoding='utf-8-sig');pd.DataFrame(codes).to_csv(H/'code_index.csv',index=False,encoding='utf-8-sig');pd.DataFrame(tables).to_csv(H/'table_index.csv',index=False,encoding='utf-8-sig')
checks=[]
families=[('기존 48조합',B/'multiref_4fold_20261021'),('측정6종 96조합',B/'multiref_4fold_20261021/extended'),('축/시행 분류기 104조합',B/'classifier_4fold_20261021'),('QR별 공간식 64조합',B/'multiref_4fold_20261021/local_fields'),('센터 품질/간격 선별 64조합',B/'multiref_4fold_20261021/center_subsets')]
families.append(('GN 창크기/흐림 96조합',B/'patch_windows_20261021/evaluation'))
families.extend([('강건 픽셀 GN 64조합',B/'robust_gn_20261021/evaluation'),('강건 P12 공간식 128조합',B/'multiref_4fold_20261021/robust_spatial'),('후보별 Gaussian likelihood 81조합',B/'likelihood_4fold_20261021')])
families.extend([('전체630점 직접 모델 30조합',B/'fullfield_4fold_20261021'),('P12 전체점 혼합 30조합',B/'hybridfield_4fold_20261021')])
families.append(('후보별 독립 변형곡선 32조합',B/'branch_curves_4fold_20261021'))
families.append(('후보별 평활 스플라인 54조합',B/'spline_4fold_20261021'))
families.append(('나사2개 추가 별도비교 64조합',B/'multiref_4fold_20261021/screw_fallback'))
families.extend([('QR 구조영역 48조합',B/'multiref_4fold_20261021/structural_regions'),('현재/대칭 기울기 GN 64조합',B/'forward_gn_20261021/evaluation')])
for _,folder in families:
    if not (folder/'summary.csv').exists():continue
    splits=json.loads((folder/'splits.json').read_text());alltest=[]
    for s in splits:
        a=set(s['train']);b=set(s['test']);assert len(a)==210 and len(b)==70 and not a&b
        assert sum(n.startswith('Y') for n in a)==60 and sum(n.startswith('T') for n in a)==150
        assert sum(n.startswith('Y') for n in b)==20 and sum(n.startswith('T') for n in b)==50
        alltest+=s['test']
    assert len(alltest)==len(set(alltest))==280
    pred=pd.read_csv(folder/'predictions.csv');summary=pd.read_csv(folder/'summary.csv')
    for _,s in summary.iterrows():
        g=pred[(pred.config==s.config)&(pred.protocol==s.protocol)]
        ok=(g.axis==g.pred_axis)&(np.sign(g.value)==np.sign(g.pred));ey=abs(np.where(g.axis=='Y',g.value,0)-np.where(g.pred_axis=='Y',g.pred,0));et=abs(np.where(g.axis=='T',g.value,0)-np.where(g.pred_axis=='T',g.pred,0))
        assert len(g)==280 and g.file.nunique()==280
        assert ok[g.axis=='Y'].sum()==s.Y_correct and ok[g.axis=='T'].sum()==s.T_correct
        assert np.isclose(ey.max(),s.Y_max) and np.isclose(et.max(),s.T_max)
        assert (~(ok&(ey<=.09)&(et<=.09))).sum()==s.failures
    checks.append(dict(folder=str(folder.relative_to(R)),summary_rows_checked=len(summary),split_ratio='train60Y150T/test20Y50T',test_once=True,summary_recalculation=True))
(H/'audit.json').write_text(json.dumps(dict(checks=checks,scan_errors=errors,files=len(files),python_files=len(codes),csv_files=len(tables)),ensure_ascii=False,indent=2),encoding='utf-8')
css='<style>body{font:16px "Malgun Gothic",sans-serif;margin:32px;color:#18304a}h1,h2{color:#153b60}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #dce3eb;text-align:left;overflow-wrap:anywhere}th{background:#e9f0f8;position:sticky;top:0}details{margin:16px 0}summary{cursor:pointer;font-weight:bold}section{margin:35px 0}.scroll{overflow:auto;max-height:650px}input{width:90%;padding:12px}a{color:#086aa0}</style>'
body=['<meta charset="utf-8">'+css+'<h1>QR 이동 추정 전체 실험과 코드 기록</h1><p>확인일 2026-10-21. 최신 4-fold 조합은 사진별 예측에서 지표를 재계산했다. 과거 실험은 저장된 표와 소스 위치를 목록화했으며 전부 재실행한 것은 아니다. 동일 자료의 재대입·교차검증·독립 촬영을 섞어 해석하지 않는다.</p><p>공통 요구사항 원본: '+link(R/'ALGORITHM_RULES.md')+'</p><p>'+link(H/'file_inventory.csv','전체 파일 목록')+' · '+link(H/'code_index.csv','코드 함수·파일 참조 목록')+' · '+link(H/'table_index.csv','전체 결과표 목록')+' · '+link(H/'audit.json','검증 기록')+'</p>']
stages=[('① 특징점 초기화','실제 격자755개 중 대각선125개 제외630개. 현재 원점 후보는 학습한12차원 벡터이며 여러 기준 사진이 아님.','analysis_grid_corners_20261019/track_corners.py'),('② 대응점 측정','경계 모형 / LK·GN / 광도 잡음 가중 / P12 직접 정합 / ECC affine,H8 / 위상 상관을 서로 구분.','codex_experiments_20261014/multiref_4fold_20261021/measure_extra.py'),('③ 변위 계산','동일 ID의 현재 좌표−센터 좌표. trial별 재검출 그림은 과거 시각화용.','codex_experiments_20261014/center_patch_v2_20261021/run.py'),('④ 공간 요약','P12는 6개 기저×dx,dy 계수12개. 점12개나12색 그룹이 아님. 일반/품질가중 최소제곱.','codex_experiments_20261014/center_patch_v2_20261021/run.py'),('⑤ 축·부호·양 추정','단일/복수 원점, 선형/이차, GLS 상관 축소. 최소 비용 후보 선택.','codex_experiments_20261014/zero_experiment.py'),('⑥ 검증','6종 측정×2원점×2차수×4축소=96개 설정. 외부4-fold 내부3-fold. Y:T비율2:5.','codex_experiments_20261014/multiref_4fold_20261021/experiment.py'),('⑦ 일반화','현장시험 대기. 잡음12종 대응은 설계/조사 상태와 실측 상태 분리.','codex_experiments_20261014/multiref_4fold_20261021/research.html')]
body.append('<h2>단계별 구현 지도</h2><table><tr><th>단계</th><th>내용</th><th>코드/근거</th></tr>'+''.join('<tr><td>'+a+'</td><td>'+b+'</td><td>'+link(R/c)+'</td></tr>' for a,b,c in stages)+'</table>')
release=B/'release_screw_fallback_20261021'
if (release/'raw_summary.csv').exists():
    body.append('<h2>나사 추가 방식의 별도 통과 후보</h2><p>QR630점 + 나사2개 고정 설정은 원본280장 재실행에서 4-fold와 재대입 모두 통과했다. 나사추가 실험군의 내부 모델선택 결과는279/280이다. 아래 QR536 통과후보와 별개이며 새촬영 일반화는 미검증이다.</p>'+pd.read_csv(release/'raw_summary.csv').to_html(index=False)+'<p>'+link(release/'report.html','원본 사진 재실행·fold별 결과·실행 코드')+'</p>')
qrrelease=B/'release_qr536_20261021'
if (qrrelease/'raw_summary.csv').exists():
    body.append('<h2>최종 QR-only536 고정 통과 후보 v3</h2><p>대각선 접촉 제외630개 중 외곽격자94개를 추가 제외한536개. 센터 격자 좌표만으로 선택하며 같은ID를 추적한다. 원본280장 재실행에서 OOF4와 재대입 모두280/280, 두축 최대절대오차0.09 이내. 이는 사후 선택한 고정설정 성적이고 내부 모델선택은279/280으로 새촬영 일반화가 입증된 것은 아니다.</p>'+pd.read_csv(qrrelease/'raw_summary.csv').to_html(index=False)+'<p>'+link(qrrelease/'report.html','최종 방법·특징점 그림·원본 재실행·모델·코드')+'</p>')
body.append('<h2>추가 실험군 비교와 각 fold 결과</h2><p>최상위 고정 설정은 외부 OOF 결과를 본 사후 탐색 결과다. 내부 선택 결과와 구분한다. 기존48조합은96조합에 포함되므로 합산하지 않는다.</p>')
family_rows=[]
for family,folder in families:
    if not (folder/'summary.csv').exists():continue
    ss=pd.read_csv(folder/'summary.csv');oo=ss[(ss.protocol=='oof4')&(ss.config!='nested_selected')].sort_values(['failures','Y_max','T_max']);best=oo.iloc[0]
    for label,g in [('사후 최상위 OOF',best),('같은 설정 재대입',ss[(ss.protocol=='resub280')&(ss.config==best.config)].iloc[0]),('학습 내부 선택 OOF',ss[ss.config=='nested_selected'].iloc[0])]:
        family_rows.append(dict(family=family,evaluation=label,config=g.config,Y=f'{int(g.Y_correct)}/{int(g.Y_n)}',T=f'{int(g.T_correct)}/{int(g.T_n)}',failures=int(g.failures),Y_max=g.Y_max,T_max=g.T_max))
    ff=pd.read_csv(folder/'fold_results.csv');body.append('<details><summary>'+family+' — 전체 조합·fold 표</summary><p>'+link(folder/'summary.csv','전체 요약')+' · '+link(folder/'predictions.csv','전체 사진 예측')+' · '+link(folder/'configs.json','조합 설정')+'</p><h3>최상위 고정 설정과 내부 선택의 fold별 결과</h3>'+ff[ff.config.isin([best.config,'nested_selected'])].to_html(index=False)+'<h3>전체 조합 결과</h3><div class="scroll">'+ss.to_html(index=False)+'</div></details>')
pd.DataFrame(family_rows).to_csv(H/'family_results.csv',index=False,encoding='utf-8-sig');body.append(pd.DataFrame(family_rows).to_html(index=False))
ext=B/'multiref_4fold_20261021/extended';s=pd.read_csv(ext/'summary.csv')
body.append('<h2>최신 96개 조합 결과</h2><p>oof4=각 사진을 학습에서 제외하고 얻은280개 예측. resub280=전280장 학습 후 같은280장 재대입. nested_selected=외부 시험을 보지 않고 학습 내부에서 설정 선택. mode inferred_session은 학습시 시행그룹 절편을 추정하되 추론에는 시행번호를 입력하지 않음. shrinkage=1은 대각 가중, .001은 거의 전체 공분산.</p><p>'+link(ext/'configs.json','설정 ID 해석')+' · '+link(ext/'predictions.csv','사진별 예측')+' · '+link(ext/'fold_results.csv','fold별 결과')+' · '+link(ext/'failures.csv','실패 목록')+' · '+link(ext/'nested_selection.json','내부 선택 기록')+'</p><div class="scroll">'+s.to_html(index=False)+'</div>')
body.append('<h2>재현 순서와 주의점</h2><p>기존 측정 캐시가 있으므로 최신 비교는 experiment.py --extended로 재현한다. 측정부터 다시 할 경우 center_patch_v2_20261021/run.py → multiref_4fold_20261021/measure_extra.py → experiment.py --extended 순서다. 스크립트는 자기 출력 폴더를 덮어쓸 수 있으므로 새로운 실험은 출력 경로를 바꾼 복사본에서 실행한다. 과거 코드를 목록화를 위해 일괄 재실행하지 않았다. 추가로 실행한 신규 실험은 위 실험군 비교에 따로 기록했다.</p><pre>$env:PYTHONPATH="'+str(B/'deps')+'"\n$env:OPENBLAS_NUM_THREADS="1"\n$env:OMP_NUM_THREADS="1"\n&amp; "C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" "'+str(B/'multiref_4fold_20261021/experiment.py')+'" --extended</pre><p>고정 QR ROI·3840×2160 카메라 배치·원본 BMP 경로·numpy/scipy/pandas/OpenCV/sklearn 의존. 새 설치 자동 QR 초기화가 완성된 배포 시스템으로 해석하지 않는다. 저장 cNN_model.json은 전체280장 학습 모델이며 fold별 모델은 저장되지 않아 재생성이 필요하다.</p>')
body.append('<h2>확인된 한계와 미완료</h2><ul><li>96개 설정 중 사후 최상위 성적과 내부 선택 성적은 다르다. 과거부터 반복해 본 동일 데이터이므로 중첩평가만으로 완전한 독립 일반화가 증명되지 않는다.</li><li>복수 원점은 촬영 시행과 연관된 절편이다. 새 시행에서도 유효한지는 미검증.</li><li>최신 실험 공분산 자유도는 코드에서 len(R)-16으로 고정되어 있다. 공통 배율이므로 동일 모델 내 후보 비용 순위에는 영향을 주지 않지만 절대 비용을 확률로 해석하지 않는다.</li><li>모든 가능한 알고리즘을 실행한 것이 아니다. HALCON은 문서 조사이며 SDK 검증 결과가 없다.</li><li>오차 정답은 파일명의 명령값이다. 별도 계측한 실제 자세가 아니다.</li><li>정적 소스 참조는 연결 단서이며 실제 실행 로그나 완전한 의존성 증명은 아니다.</li></ul>')
body.append('<h2>추가 실험 근거와 다음 조사</h2><p><a href="https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html">LDA 공식 문서</a> · <a href="https://sklearn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">학습 데이터 표준화 공식 문서</a> · <a href="https://publications.ri.cmu.edu/lucas-kanade-20-years-on-a-unifying-framework-part-2">Baker 등 가중 L2와 강건 영상 정합</a>. 분류기104조합은 구현·평가 완료. 픽셀 단위 강건 정합(Huber/Cauchy) 64설정, 강건 P12 128설정, 후보별 공분산과 사전확률 81설정을 추가 평가했다. 동일한 수학적 모형으로 귀결되는 설정도 있어 설정 수를 서로 다른 알고리즘 수로 해석하지 않는다.</p>')
body.append('<h2>전체 폴더별 결과와 코드</h2><input id="q" placeholder="폴더명 / 방법 / 파일명 검색" oninput="document.querySelectorAll(\'.entry\').forEach(x=>x.style.display=x.textContent.toLowerCase().includes(this.value.toLowerCase())?\'\':\'none\')">')
groups={}
for p in files:
    rel=p.relative_to(R);key='/'.join(rel.parts[:2]) if rel.parts[0]=='codex_experiments_20261014' and len(rel.parts)>2 else rel.parts[0] if len(rel.parts)>1 else 'root'
    groups.setdefault(key,[]).append(p)
for key,paths in sorted(groups.items()):
    body.append('<details class="entry"><summary>'+html.escape(key)+' — '+str(len(paths))+'개 파일</summary>')
    cs=[x for x in codes if R/x['path'] in paths]
    if cs:body.append('<h3>코드</h3><table><tr><th>파일</th><th>설명 / 함수:행</th></tr>'+''.join('<tr><td>'+link(R/c['path'])+'</td><td>'+html.escape(c['description']+' / '+c['definitions'])+'</td></tr>' for c in cs)+'</table>')
    ts=[x for x in tables if R/x['path'] in paths]
    for t in ts:
        body.append('<details><summary>'+html.escape(Path(t['path']).name)+' ('+str(t['rows'])+'행)</summary><p>'+link(R/t['path'],'원본 CSV')+'</p><p>'+html.escape(t['columns'])+'</p>')
        if t['summary_candidate'] and t['rows']<=2000:
            try:body.append('<div class="scroll">'+pd.read_csv(R/t['path']).to_html(index=False)+'</div>')
            except Exception:pass
        else:body.append('<p>사진별 측정/예측 또는 대형 표: 원본 파일에서 전체 확인.</p>')
        body.append('</details>')
    other=[p for p in paths if p.suffix not in {'.py','.csv'}]
    body.append('<details><summary>모델·캐시·문서·발표자료</summary><ul>'+''.join('<li>'+link(p)+'</li>' for p in sorted(other))+'</ul></details></details>')
body.append('<h2>검증 기록</h2><pre>'+html.escape(json.dumps(checks,ensure_ascii=False,indent=2))+'</pre>')
(H/'index.html').write_text(''.join(body),encoding='utf-8')
bad=[]
for href in re.findall(r'href="([^"]+)"',''.join(body)):
    href=html.unescape(href)
    if not href.startswith(('http:','https:','#')) and not (H/href).exists():bad.append(href)
assert not bad,bad
(H/'link_check.json').write_text(json.dumps(dict(local_links_checked=len(re.findall(r'href="([^"]+)"',''.join(body))),missing=bad),indent=2))
print(json.dumps(dict(files=len(files),codes=len(codes),tables=len(tables),errors=errors,checks=checks),ensure_ascii=False),flush=True)


