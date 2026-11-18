"""Descriptive extension results, fixed cohorts, excluded counts and native parity."""
import base64
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'gaussian_window_sensitivity_20261101'
NS=np.array([11]+list(range(21,62,2)))
STARTS=[11,5,21]
DTYPE=[('ref','U3'),('target','U20'),('x','i4'),('coarse_y','i4'),('peak','f8'),
       ('n','i4'),('initial_sigma','i4'),('A','f8'),('mu','f8'),('sigma','f8'),
       ('sse','f8'),('iterations','i4'),('converged','i4'),('finite','i4'),
       ('legacy_selected','i4'),('status','U30')]
def dist(a):
    a=np.asarray(a,dtype=float)
    assert np.isfinite(a).all()
    return {'count':int(a.size),'median':float(np.median(a)) if a.size else None,
            'p95':float(np.quantile(a,.95)) if a.size else None,'max':float(a.max()) if a.size else None}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v): return '—' if v is None else f'{v:.6g}'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def embed(name,cap):
    return '<figure><img src="data:image/png;base64,'+base64.b64encode((HERE/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'

def main():
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    unchanged={p:sha(p)==v for p,v in provenance['unchanged_source_hashes'].items()}
    assert all(unchanged.values())
    data=np.genfromtxt(HERE/'fits.csv',delimiter=',',skip_header=1,dtype=DTYPE,encoding='utf-8')
    assert data.size==6104*22*3,data.size
    assert set(data['target'])=={'lower_structure'}
    old=np.load(OLD/'paired_arrays.npz')
    profiles=json.loads((HERE/'profiles.json').read_text())
    assert len(profiles)==6104
    old_profiles=json.loads((OLD/'profiles.json').read_text())
    old_pmap={(p['ref'],p['target'],p['x']):p for p in old_profiles}
    for p in profiles:
        op=old_pmap[(p['ref'],p['target'],p['x'])]
        assert p['coarse_y']==op['coarse_y']
        assert p['sobel_samples_m30_p30'][20:41]==op['sobel_samples_m10_p10']
    summary={'scope':provenance['design']['scope'],'attempts':int(data.size),'groups':[],
             'source_files_unchanged':unchanged,'sample_counts':NS.tolist(),
             'acceptance_threshold':None,'image_count':4,
             'note':'All position differences are setting sensitivity, not ground-truth error or robot theta error. Legacy +/-2px acceptance is not validated accuracy.'}
    archive={}
    for ref in ['R01','R02','R03','R04']:
        key=ref+'_lower_structure'
        group=data[data['ref']==ref]
        xs=np.unique(group['x'])
        assert len(xs)==1526
        mats={name:np.empty((len(xs),len(NS),3),dtype=float) for name in
              ['mu','A','sigma','sse','iterations','finite','converged','legacy_selected']}
        for ni,n in enumerate(NS):
            for si,start in enumerate(STARTS):
                rows=np.sort(group[(group['n']==n)&(group['initial_sigma']==start)],order='x')
                assert np.array_equal(rows['x'],xs)
                for name,mat in mats.items(): mat[:,ni,si]=rows[name]
        assert np.array_equal(xs,old[key+'_xs'])
        for n in [11,21]:
            ni=int(np.flatnonzero(NS==n)[0]); oi=(n-3)//2
            for name,mat in mats.items():
                assert np.array_equal(mat[:,ni,:],old[key+'_'+name][:,oi,:],equal_nan=True),(ref,n,name)
        selected=mats['legacy_selected'][:,:,0].astype(bool)
        raw=(mats['finite'][:,:,0]>0)&(mats['converged'][:,:,0]>0)&(mats['A'][:,:,0]>0)
        mu=mats['mu'][:,:,0]
        prior=old[key+'_common']
        common=prior&selected[:,1:].all(axis=1)
        g={'ref':ref,'visible_columns':len(xs),'prior_common_count':int(prior.sum()),
           'extended_fixed_common_count':int(common.sum()),
           'native_11_21_all_fields_all_starts_exact_parity':True,'windows':[],'spans':{},'initialization':[]}
        for ni,n in enumerate(NS):
            rows=group[(group['n']==n)&(group['initial_sigma']==11)]
            sts,ct=np.unique(rows['status'],return_counts=True)
            w={'n':int(n),'selected':int(selected[:,ni].sum()),'selected_percent':float(100*selected[:,ni].mean()),
               'prior_cohort_selected':int((prior&selected[:,ni]).sum()),'statuses':dict(zip(sts,map(int,ct)))}
            if ni>=2:
                pair=prior&selected[:,ni]&selected[:,ni-1]
                raw_pair=prior&raw[:,ni]&raw[:,ni-1]
                w['adjacent_fixed_common_px']=dist(abs(mu[common,ni]-mu[common,ni-1]))
                w['adjacent_prior_pair_selected_px']=dist(abs(mu[pair,ni]-mu[pair,ni-1]))
                w['adjacent_prior_raw_converged_px']=dist(abs(mu[raw_pair,ni]-mu[raw_pair,ni-1]))
            g['windows'].append(w)
            smask=mats['legacy_selected'][:,ni,:].all(axis=1)
            g['initialization'].append({'n':int(n),'three_start_span_selected_px':dist(np.ptp(mats['mu'][smask,ni,:],axis=1)),
              'selected_status_disagreements':int(np.any(mats['legacy_selected'][:,ni,:]!=mats['legacy_selected'][:,ni,0,None],axis=1).sum())})
        for a,b in [(21,31),(31,41),(41,51),(51,61),(21,61)]:
            m=(NS>=a)&(NS<=b)
            g['spans'][f'{a}_{b}']=dist(np.ptp(mu[common][:,m],axis=1))
        # Diagnostic counts only: other strong responses are not ground-truth edges.
        pp=[p for p in profiles if p['ref']==ref]
        more_pos=more_neg=0
        for p in pp:
            v=np.array(p['sobel_samples_m30_p30']); outside=np.r_[v[:20],v[41:]]
            more_pos+=int(outside.max()>p['peak']); more_neg+=int((-outside).max()>p['peak'])
        g['outer_sample_diagnostics']={'outside_original_pm10_stronger_positive_count':more_pos,
            'outside_original_pm10_stronger_negative_magnitude_count':more_neg,
            'meaning':'Raw signal diagnostic, not a physical-edge classification or automatic exclusion rule.'}
        summary['groups'].append(g)
        archive[key+'_xs']=xs; archive[key+'_common']=common; archive[key+'_prior_common']=prior
        for name,mat in mats.items(): archive[key+'_'+name]=mat
    assert sum(g['prior_common_count'] for g in summary['groups'])==2176
    np.savez_compressed(HERE/'paired_arrays.npz',**archive)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'savefig.dpi':150})
    colors=['#2458bd','#cf7014','#14866d','#9e479e']
    fig,axes=plt.subplots(2,2,figsize=(12,8.3),layout='constrained')
    for g,color in zip(summary['groups'],colors):
        ws=g['windows'][2:]
        for ax,stat in zip(axes[0],['median','p95']):
            ax.plot([w['n'] for w in ws],[w['adjacent_fixed_common_px'][stat] for w in ws],'-o',ms=3,color=color,label=g['ref'])
        axes[1,0].plot(NS[1:],[w['selected_percent'] for w in g['windows'][1:]],'-o',ms=3,color=color,label=g['ref'])
        axes[1,1].plot(NS[1:],[w['three_start_span_selected_px']['max'] for w in g['initialization'][1:]],'-o',ms=3,color=color,label=g['ref'])
    for ax,title in zip(axes[0],['표본 2개 증가 시 위치 변화 · 중앙값','표본 2개 증가 시 위치 변화 · P95']):
        ax.set_title(title); ax.set_ylabel('|μ(N) - μ(N-2)| [px]'); ax.set_ylim(bottom=0)
    axes[1,0].set_title('전체 위치 중 기존 조건으로 채택되는 비율'); axes[1,0].set_ylabel('채택 비율 [%]'); axes[1,0].set_ylim(0,100)
    axes[1,1].set_title('계산 초기값 3종에 따른 위치 차이 · 최대'); axes[1,1].set_ylabel('μ 최대-최소 [px]'); axes[1,1].set_ylim(bottom=0)
    for ax in axes.flat:
        ax.set_xlabel('표본 개수 N'); ax.set_xticks([21,25,31,35,41,45,51,55,61]); ax.grid(alpha=.2); ax.legend(ncol=2)
    cc=sum(g['extended_fixed_common_count'] for g in summary['groups'])
    fig.suptitle(f'하단 ±30px까지 확장 · 위쪽 비교는 같은 {cc:,}개 위치를 계속 사용',fontsize=15)
    fig.savefig(HERE/'extended_sensitivity.png'); plt.close(fig)

    # Prior conversation example, unchanged spatial location, displayed without selecting best window.
    key='R01_lower_structure'; x=1316
    ix=int(np.flatnonzero(archive[key+'_xs']==x)[0])
    p=next(p for p in profiles if p['ref']=='R01' and p['x']==x)
    cy=p['coarse_y']; v=np.array(p['sobel_samples_m30_p30']); offsets=np.arange(-30,31)
    source=Path(provenance['design']['inputs']['references'][0]['path'])
    photo=np.array(Image.open(source).convert('RGB'))
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[.75,1.6]},layout='constrained')
    ax=axes[0]; x0,x1=x-45,x+46; y0,y1=cy-40,cy+41
    ax.imshow(photo[y0:y1,x0:x1],extent=[x0-.5,x1-.5,y1-.5,y0-.5],interpolation='nearest')
    for n,col,xpos in [(21,'#e0a000',x-20),(61,'#168573',x+20)]:
        hw=(n-1)//2; ax.plot([xpos,xpos],[cy-hw,cy+hw],color=col,lw=3)
        ax.plot([xpos-4,xpos+4],[cy-hw,cy-hw],color=col,lw=2); ax.plot([xpos-4,xpos+4],[cy+hw,cy+hw],color=col,lw=2)
        ax.text(xpos,cy+hw+5,f'{n}개',color=col,ha='center',weight='bold')
    ax.plot([x,x],[cy-30,cy+30],':',color='#167bb1'); ax.plot(x,cy,'o',mfc='none',mec='#ffdd00',mew=2,ms=8)
    ax.set_title('이전 설명과 같은 위치의 원본 사진'); ax.set_xticks([]); ax.set_ylabel('원본 세로 좌표 y [px]')
    ax=axes[1]; ax.plot(offsets,v,'o-',ms=3,color='#666',lw=.7,label='실제 Sobel 표본')
    example=[]
    for n,col in [(21,'#e0a000'),(31,'#2458bd'),(41,'#b34a99'),(61,'#168573')]:
        ni=int(np.flatnonzero(NS==n)[0]); hw=(n-1)//2
        mu=archive[key+'_mu'][ix,ni,0]; A=archive[key+'_A'][ix,ni,0]; sig=archive[key+'_sigma'][ix,ni,0]
        u=np.linspace(-hw,hw,401); m=mu-cy
        ax.plot(u,A*np.exp(-(u-m)**2/(2*(sig+1e-12)**2)),color=col,lw=1.8,label=f'{n}개: μ={mu:.4f}px')
        ax.axvline(m,color=col,ls=':',alpha=.6)
        example.append({'n':n,'mu':float(mu),'sigma':float(sig),'legacy_selected':bool(archive[key+'_legacy_selected'][ix,ni,0])})
    ax.axvspan(-30,-10,color='#168573',alpha=.06); ax.axvspan(10,30,color='#168573',alpha=.06)
    ax.set_title('옅은 녹색: 기존 ±10px 밖에서 추가로 읽는 부분'); ax.set_xlabel('최초 최대점에서 위아래 거리 [px]'); ax.set_ylabel('세로 밝기 변화량'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle('R01 하단 x=1316px · 사진과 최초 최대점은 고정',fontsize=14)
    fig.savefig(HERE/'extended_profile_example.png'); plt.close(fig)
    (HERE/'example_provenance.json').write_text(json.dumps({'ref':'R01','x':x,'coarse_y':cy,'selection':'Same illustrative point as previous explanation, not a worst-case or best-case selection.','windows':example},indent=2),encoding='utf-8')

    rows=[]
    for ni,n in enumerate(NS[2:],start=2):
        ws=[g['windows'][ni] for g in summary['groups']]
        rows.append([f'{n-2}→{n}',f'{min(w["adjacent_fixed_common_px"]["median"] for w in ws):.6f}~{max(w["adjacent_fixed_common_px"]["median"] for w in ws):.6f}',
          f'{min(w["adjacent_fixed_common_px"]["p95"] for w in ws):.6f}~{max(w["adjacent_fixed_common_px"]["p95"] for w in ws):.6f}',
          fmt(max(w['adjacent_fixed_common_px']['max'] for w in ws)),
          f'{min(w["selected_percent"] for w in ws):.2f}~{max(w["selected_percent"] for w in ws):.2f}'])
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>하단 Gaussian 표본 범위 확장</title><style>
body{font-family:'Malgun Gothic',sans-serif;background:#f3f5f8;color:#263445;line-height:1.7;margin:0}main{max-width:1120px;padding:32px 24px 70px;margin:auto}h1{font-size:29px}h2{margin-top:36px}figure{margin:20px 0;background:white}img{width:100%;display:block}figcaption{padding:12px;font-size:14px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#e5ecf4}.scroll{overflow:auto}.note{padding:16px 22px;background:white;border-left:5px solid #2458bd}a{color:#2458bd}details{background:white;margin:12px 0;padding:12px}</style><main>
<p>2026-11-01 · E01 · 하단 구조물만 · 기준 사진 4장</p><h1>주변 표본을 23~61개로 넓혔을 때</h1>
<div class="note">기존 21개(±10px)에서 61개(±30px)까지 확장했다. 61개는 이번 탐색 범위의 끝이며 최적값이나 정확도 기준이 아니다. 실제 경계 정답, 최종 직선, θ 보정 성능은 평가하지 않았다.</div>
<p>최초 경계 선택, Sobel, Gaussian 모델, 계산 초기값, 수렴 조건, 기존 ±2px 채택 조건은 그대로 유지했다. 기본 초기 σ는 모든 표본 수에서 11로 고정하고, 5·21은 별도 초기값 민감도 계열로 비교했다. 세 결과 중 좋은 값만 고르지 않았다.</p>
<p>하단 6,104곳 × 22개 표본 수 × 3개 초기값 = 402,864회 계산. 11·21개는 이전 결과와 좌표·모든 피팅 값·상태가 정확히 같았고, 기존 ±10px의 Sobel 표본도 전부 일치했다.</p>
'''
    page+='<h2>이번 결과의 의미</h2><p><b>표본 범위를 넓히자 대부분의 점에서 위치 변화가 줄었다.</b> 51~61개 범위에서 같은 점의 최대 위치와 최소 위치 차이의 P95는 사진별 약 0.000038~0.000043px였다. 그러나 R03의 일부 점은 같은 범위에서도 최대 약 1.155px 차이가 남았다. 초기값 3종 비교에서도 최대 약 2.274px 차이와 채택 여부가 달라지는 72곳이 남았다. 따라서 51~61개는 추가 검토할 후보 구간이며 최적값이나 정확도 검증 완료를 의미하지 않는다.</p><p>0 또는 매우 작은 위치 변화는 현재 계산 조건에서 출력이 같아졌다는 뜻이다. 실제 경계 오차가 0이라는 뜻이 아니며, 현재 수렴 조건의 영향을 분리해 검증한 것도 아니다.</p>'
    page+='<h2>동일 위치끼리 비교하기 위한 범위</h2>'+table(['사진','전체 위치','기존 5~21개 공통 채택','기존 공통 중 21~61개도 모두 채택'],[[g['ref'],g['visible_columns'],g['prior_common_count'],g['extended_fixed_common_count']] for g in summary['groups']])
    page+=f'<p>아래 주 비교는 마지막 열의 같은 {cc:,}곳만 사용한다. 조건을 통과한 위치만 포함하므로 전체 6,104곳으로 일반화하지 않는다. 모든 시도와 제외 이유는 fits.csv에 보존했고, 이전 2,176곳 안에서 각 인접 조건이 둘 다 채택된 경우 및 ±2px 밖의 수렴 결과도 summary.json에 따로 남겼다. 기존 ±2px 기준 자체는 검증된 정확도 기준이 아니다.</p>'
    page+=embed('extended_sensitivity.png','P95는 위치 변화의 95백분위수이며 신뢰구간이 아니다. 초기값 차이와 채택 비율도 함께 확인해야 한다.')
    page+='<h2>표본을 2개씩 늘릴 때 경계점 이동</h2>'+table(['표본 변경','중앙값 범위 [px]','P95 범위 [px]','최대 [px]','전체 채택 비율 [%]'],rows)
    page+='<h2>넓은 구간 안에서 누적된 위치 변화</h2>'+table(['비교 구간','사진','중앙값 [px]','P95 [px]','최대 [px]'],[[span,g['ref'],fmt(g['spans'][span]['median']),fmt(g['spans'][span]['p95']),fmt(g['spans'][span]['max'])] for span in ['21_31','31_41','41_51','51_61','21_61'] for g in summary['groups']])
    page+=embed('extended_profile_example.png','이전 설명에서 사용한 한 위치를 그대로 확대했다. 추가 표본의 밝기 변화와 피팅 곡선을 보여준다. 전체를 대표하는 정확도 또는 최악 사례를 의미하지 않는다.')
    page+='<h2>다른 밝기 변화가 포함되는지에 대한 진단</h2><p>원래 ±10px 밖에서 최초 최대점보다 더 큰 양의 응답이나 더 강한 음의 응답이 있는 위치를 세었다. 반사·다른 경계·잡음 중 무엇인지는 이 값만으로 확정할 수 없으며 새 제외 규칙으로 사용하지 않았다.</p>'+table(['사진','더 큰 양의 응답 위치','더 강한 음의 응답 위치'],[[g['ref'],g['outer_sample_diagnostics']['outside_original_pm10_stronger_positive_count'],g['outer_sample_diagnostics']['outside_original_pm10_stronger_negative_magnitude_count']] for g in summary['groups']])
    page+='<h2>조건별 제외 내역</h2>'
    for g in summary['groups']:
        page+='<details><summary>'+g['ref']+'</summary>'+table(['N','채택/1526','기존 공통 중 채택','상태별 개수'],[[w['n'],w['selected'],w['prior_cohort_selected'],json.dumps(w['statuses'],ensure_ascii=False)] for w in g['windows']])+'</details>'
    page+='<h2>기록과 근거</h2><p><a href="summary.json">수치 요약</a> · <a href="fits.csv">모든 피팅 시도</a> · <a href="provenance.json">조건·해시</a> · <a href="verification.json">재현 점검</a> · <a href="../gaussian_window_sensitivity_20261101/report.html">기존 3~21개 비교</a></p><p>모델 근거: <a href="https://www.annals-csis.org/Volume_2/pliks/136.pdf">Fabijańska (2014), Gaussian-Based Approach to Subpixel Detection of Blurred and Unsharp Edges</a>. 이번 23~61개 범위는 원 논문이 권장한 설정이 아니라 사용자가 요청한 추가 민감도 탐색이다. 실제 구현의 Gy 고정 방향 등 기존 변형은 이번에 변경하거나 검증하지 않았다.</p></main></html>'
    (HERE/'report.html').write_text(page,encoding='utf-8')
    verification={'attempts':int(data.size),'columns':len(profiles),'images':4,
        'all_old_11_21_values_and_statuses_exact_parity':True,
        'all_old_pm10_sobel_samples_and_coarse_positions_exact_parity':True,
        'source_hashes_unchanged':all(unchanged.values()),'prior_common_count':2176,
        'extended_fixed_common_count':cc,'no_production_setting_changes':True,'no_line_or_theta_fitting':True}
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    for g in summary['groups']:
        print(g['ref'],'common',g['extended_fixed_common_count'],'/',g['prior_common_count'])
        for n in [23,31,41,51,61]:
            w=next(w for w in g['windows'] if w['n']==n)
            print(n,'selected',w['selected'],'adj',w['adjacent_fixed_common_px'])
        print('spans',g['spans'],'outer',g['outer_sample_diagnostics'])
    print('Verification',verification)

if __name__=='__main__': main()
