"""Render the measured line-angle evidence and a Korean report."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = {
    'L_top': '좌 QR 윗변', 'L_bottom': '좌 QR 아랫변',
    'C_top': '중 QR 윗변', 'C_bottom': '중 QR 아랫변',
    'R_top': '우 QR 윗변', 'R_bottom': '우 QR 아랫변',
    'rail_UL_edge': '왼쪽 위 구조물 경계', 'rail_UL_ridge': '왼쪽 위 구조물 밝은 띠',
    'rail_UR_edge': '오른쪽 위 구조물 경계', 'rail_UR_ridge': '오른쪽 위 구조물 밝은 띠',
    'rail_BL_edge': '왼쪽 아래 구조물 경계', 'rail_BR_edge': '오른쪽 아래 구조물 경계',
}
COLORS = {2: '#2878B5', 3: '#D55E00', 4: '#2878B5', 5: '#D55E00', 6: '#009E73', 7: '#A45AAB'}


def save(fig, stem):
    fig.savefig(HERE/(stem+'.png'), dpi=160, bbox_inches='tight')
    fig.savefig(HERE/(stem+'.svg'), bbox_inches='tight')
    plt.close(fig)


def response(ax, feature, rows, results, groups, relative=False):
    for group in groups:
        rr = [r for r in rows if r.get('group') == group]
        q = np.array([r['label'] for r in rr])
        phi = np.array([r['features'][feature]['angle_deg'] for r in rr])
        fit = results['features'][feature]['first_fits'][str(group)]
        b, a = fit['intercept_image_deg'], fit['slope_image_deg_per_label_unit']
        reference = b if relative else 0
        ax.scatter(q, phi-reference, s=10, color=COLORS[group], alpha=.65, label=f'{group}번')
        span = np.array([min(q), max(q)])
        ax.plot(span, b+a*span-reference, color=COLORS[group], lw=1.4)
    ax.set_title(NAMES[feature], fontsize=12)
    ax.set_xlabel('Y 라벨 (mm)' if groups[0] == 2 else 'θ 라벨 (°)')
    ax.set_ylabel('φ - 구간별 절편 (영상 °)' if relative else '영상 선 각도 φ (°)')
    ax.grid(alpha=.2)
    ax.ticklabel_format(useOffset=False, style='plain')


def main():
    data = json.loads((HERE/'line_measurements.json').read_text(encoding='utf-8'))
    results = json.loads((HERE/'hypothesis_results.json').read_text(encoding='utf-8'))
    rows = data['measurements']
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 10, 'svg.fonttype': 'none'})
    for axis, groups, relative in [('theta', [4, 5], False), ('y', [2, 3, 6, 7], True)]:
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), layout='constrained')
        for ax, feature in zip(axes.flat, NAMES):
            response(ax, feature, rows, results, groups, relative)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=len(groups), frameon=False)
        subtitle = '각 선·각 촬영 구간을 따로 적합. θ 증가 시 선마다 반응 방향·크기가 다름.' if axis == 'theta' else '구간 간 영점 차이를 섞지 않도록 각 구간 절편을 뺌. Y에 대한 변화는 남겨 둠.'
        fig.suptitle(('1차 θ 단독 이동: 선 기울기 반응' if axis == 'theta' else '1차 Y 단독 이동: 선 기울기 변화')+'\n'+subtitle, fontsize=16)
        save(fig, 'first_'+axis+'_all_lines')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for col, feature in enumerate(['rail_UL_edge', 'rail_UR_edge']):
        response(axes[0, col], feature, rows, results, [4, 5])
        axes[0, col].legend(frameon=False)
        ax = axes[1, col]
        for theta in [.01, .02, .03]:
            rr = [r for r in rows if r['dataset'] == 'second' and r['Y'] == .05 and r['Theta'] == theta]
            vals = [r['features'][feature]['angle_deg'] for r in rr]
            ax.plot([theta, theta], [min(vals), max(vals)], color='#2878B5', lw=2)
            ax.scatter(np.array([theta]*3)+np.array([-.0006, 0, .0006]), vals, s=50, color='#2878B5', zorder=3)
        ax.set_xticks([.01, .02, .03])
        ax.set_xlabel('θ 라벨 (°); Y=0.05mm 고정')
        ax.set_ylabel('영상 선 각도 φ (°)')
        ax.set_title('2차 최소 간격: 각 조건 3장 실측값')
        ax.grid(alpha=.2); ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    fig.suptitle('큰 θ 변화에 대한 반응과 0.01° 구분은 별개의 확인 항목\n아래 세로선은 관측 최솟값~최댓값이며 신뢰구간이 아님', fontsize=16)
    save(fig, 'theta_and_minimum_step')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    for ax, feature in zip(axes, ['rail_UL_edge', 'rail_UR_edge']):
        predictions = results['features'][feature]['variant_checks']['angle_deg']['second_condition_cv']['predictions']
        ax.axhspan(-.005, .005, color='#009E73', alpha=.18, label='목표 ±0.005°')
        for y, color in zip([.05, .1, .5, 1], ['#2878B5', '#D55E00', '#009E73', '#A45AAB']):
            rr = [r for r in predictions if r['Y'] == y]
            ax.scatter([r['Theta'] for r in rr], [r['error_deg'] for r in rr], s=30, color=color, alpha=.8, label=f'Y={y}mm')
        ax.axhline(0, color='gray', lw=.8)
        ax.set_xlabel('제외한 조건의 θ 라벨 (°)')
        ax.set_ylabel('추정 θ - 라벨 θ (°)')
        ax.set_title(NAMES[feature]); ax.grid(alpha=.2); ax.legend(fontsize=9, frameon=False)
    fig.suptitle('2차 조건별 교차검증: 같은 조건 3장을 함께 제외\n한 선의 기울기로 θ만 추정하며 Y는 입력하지 않음. 19조건·57장.', fontsize=16)
    save(fig, 'second_heldout_theta_errors')

    write_report(data, results)
    provenance = {'measurement_sha256': hashlib.sha256((HERE/'line_measurements.json').read_bytes()).hexdigest(),
                  'results_sha256': hashlib.sha256((HERE/'hypothesis_results.json').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'figures': ['first_theta_all_lines', 'first_y_all_lines', 'theta_and_minimum_step', 'second_heldout_theta_errors'],
                  'scope': 'Saved measured values plotted; no added registration. Error bars in minimum-step panels are observed 3-image ranges.'}
    (HERE/'figure_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved report.txt and four PNG/SVG figures')


def write_report(data, results):
    lines = [
        '선 기울기를 이용한 θ 선행 보정 가설 검토', '',
        '결론: 일부 외부 구조물의 기울기는 θ에 뚜렷하게 반응한다.',
        '그러나 라벨 영점의 선 기울기는 0°(영상 수평)가 아니며, 이번 선별 기울기→θ 선형 추정은 ±0.005° 목표를 충족하지 못했다.',
        '따라서 수평 맞추기→Y 이동이라는 원안은 현재 근거로 채택하지 않는다. θ 먼저/Y 나중이라는 제어 순서 전체의 불가능성을 뜻하지 않는다.', '',
        '범위: 1차 281장(기준 포함), 2차 60장. QR 윗변·아랫변 6개와 멘토 표시 외부 구조물 선 6개.',
        '4번은 이전 영점 동등성 분석에서만 제외했다. 이번에는 θ 양/음 방향 반응 확인을 위해 4·5번을 각각 계산했다.',
        'QR별/촬영 구간별 계수를 평균하지 않았다. 선을 조합한 다중 특징 제어나 실제 장비 이동은 수행하지 않았다.',
        '1차와 2차 절대 기준은 합치지 않았다. 1차 기울기 전달 검토에서는 2차 center1의 각도를 자체 원점으로 사용했다.', '',
        '측정 방법',
        '원본 픽셀에서 선을 측정. 기존 아핀 정합의 위치 이동량은 탐색 영역을 따라가는 데만 사용하며 이미지를 변형하지 않음.',
        '부호 있는 세로 밝기 변화로 경계를 찾고, 각 열의 11개 밝기에 연속적인 흐린 계단 함수(B+A*정규CDF)를 적합하여 서브픽셀 위치 계산.',
        '경계점에 Huber 직선 적합. φ=atan(dy/dx); 영상에서 오른쪽으로 내려갈수록 양수.',
        '초기 이차 피크 보간법은 알려진 합성 직선에서 픽셀 위상 편향이 발견되어 교체함. 초기 결과는 별도 initial_quadratic 파일로 보존.',
        'Gaussian sigma=1px, 30회 수치 반복 등은 측정기 설정이며 정확도 합격 기준이 아님. L2 직선, 선 중앙 80%, sigma=2도 민감도 확인.',
        '수치 시험은 알려진 각도의 이상적인 흐린 경계에 대한 검증이며 실제 카메라 정확도를 입증하지 않음.', '',
        '1차 θ 변화에 대한 반응 (영상 각도/로봇 θ 단위 구분)',
        '선 | 촬영 | a(영상 °/θ °) | b(θ=0의 영상 °) | R² | 잔차 RMS(영상 °) | 잔차 RMS/|a|(θ °)',
    ]
    for f, r in results['features'].items():
        for g in ['4', '5']:
            z = r['first_fits'][g]
            lines.append(f"{NAMES[f]} | {g} | {z['slope_image_deg_per_label_unit']:.7f} | {z['intercept_image_deg']:.7f} | {z['r2']:.6f} | {z['rms_image_deg']:.7f} | {z['rms_equivalent_label_unit']:.7f}")
    lines += ['', 'Y만 바꿔도 기울기가 변하는가?', '선 | 촬영 | 기울기 변화율(영상 °/Y mm) | 영상 각도 관측 범위(°)']
    for f, r in results['features'].items():
        for g in ['2', '3', '6', '7']:
            z = r['first_fits'][g]
            lines.append(f"{NAMES[f]} | {g} | {z['slope_image_deg_per_label_unit']:.7f} | {z['phi_min_max']}")
    lines += ['', '수평 목표 확인',
              'θ=0에서 예상하는 영상 각도 b와, θ ±0.005° 주변의 근사 HAC 불확실성 범위에 영상 0°가 포함되는지 계산.',
              'HAC는 점별 근사 민감도 계산. 정밀한 동시 95% 보장이나 신규 영상 예측구간이 아님.',
              '선 | 촬영 | θ=0 예상 영상 각도 | θ±0.005° 주변 영상각도 범위 | 수평 0° 포함']
    for f, r in results['features'].items():
        for z in r['horizontal_target_check']:
            lines.append(f"{NAMES[f]} | {z['first_group']} | {z['image_angle_at_nominal_theta0']:.7f} | {z['image_angle_envelope_near_theta0_approx_hac']} | {z['horizontal_0_in_envelope']}")
    lines += ['', '2차 교차검증: 한 조건의 3장 모두 제외한 뒤 나머지 18조건 54장으로 선별 φ=b+aθ 적합.',
              '제외한 3장의 θ=(φ-b)/a를 계산. Y는 입력하지 않음. 19조건을 순환하여 57장 전체 평가.',
              '이 표의 값은 라벨 대비 오차이며 독립적으로 실측한 실제 장비 오차가 아님.',
              '선 | MAE(θ °) | RMS(θ °) | 최대 절댓값(θ °) | ±0.005° 안의 사진 수/57']
    for f, r in results['features'].items():
        z = r['variant_checks']['angle_deg']['second_condition_cv']['metrics']
        lines.append(f"{NAMES[f]} | {z['mae_deg']:.7f} | {z['rmse_deg']:.7f} | {z['max_abs_deg']:.7f} | {z['within_0_005_count']}/57")
    lines += ['', '민감도 확인: 같은 교차검증을 측정 설정별로 반복 (결과를 보고 최적 설정의 검증 성능을 보장하는 것은 아님)',
              '선 | 설정 | RMS(θ °) | 최대 절댓값(θ °)']
    for f, r in results['features'].items():
        for v, z in r['variant_checks'].items():
            m = z['second_condition_cv']['metrics']
            lines.append(f"{NAMES[f]} | {v} | {m['rmse_deg']:.7f} | {m['max_abs_deg']:.7f}")
    lines += ['', '주요 외부 구조물: 동일 라벨 반복 편차와 Y 조건 차이',
              '아래 θ 환산값은 4·5번의 기울기를 각각 사용한 두 결과. 촬영 구간 평균이 아님.',
              '3장 관측 범위는 모집단 노이즈 한계/신뢰구간이 아니다. Y 조건별 평균 차이 역시 물리적 원인을 분리한 결과가 아니다.']
    for f in ['rail_UL_edge', 'rail_UR_edge']:
        r = results['features'][f]
        m = max(r['second_repeats'], key=lambda z: z['range_image_deg'])
        lines.append(f"{NAMES[f]}: 동일 라벨 최대 범위 {m['range_image_deg']:.7f} 영상 °, 조건 {m['condition']}; θ환산 {m['equivalent_theta_range_by_first_group']}")
        for z in r['second_fixed_theta_y_effect']:
            lines.append(f"  θ={z['theta']}, Y=0.05/0.1/0.5/1.0의 조건별 3장 평균 간 범위: {z['range_of_condition_means_image_deg']:.7f} 영상 °; θ환산 {z['equivalent_theta_range_by_first_group']}")
        for z in r['second_minimum_steps']:
            lines.append(f"  Y=.05, θ {z['theta_low']}→{z['theta_high']}: 3장 범위 겹침={z['observed_ranges_overlap']}, 평균 각도 변화={z['mean_angle_difference_deg']:.7f} 영상 °, θ환산 {z['equivalent_theta_step_by_first_group']}")
    lines += ['', '해석 범위',
              '현재 자료는 미세 기울기를 안정적으로 θ ±0.005° 수준으로 역산하는 단일 선 방법을 지지하지 않는다.',
              '실제 θ 반응과 영상 측정/조명/모델의 오차가 섞여 있다. 미세 기계오류 원인 분석은 하지 않았다.',
              '수평 0° 대신 목표 자세의 실제 선 기울기를 기준으로 삼아야 한다는 근거는 확인되지만, 그 방법의 목표 정확도까지 충족한 것은 아니다.',
              'Y 보정 후 잔여 θ 확인, 20회 이내 폐루프 수렴, 다른 특징/다중 선 조합은 검증하지 않았다.',
              '', '재현 명령',
              'python analysis/theta_line_hypothesis/measure_lines.py',
              'python analysis/theta_line_hypothesis/verify_measurement.py',
              'python analysis/theta_line_hypothesis/analyze_lines.py',
              'python analysis/theta_line_hypothesis/present_results.py',
              '', '원본 BMP 341개 SHA256을 기존 목록과 대조. 원본 및 이전 영점 동등성 결과 변경 없음.', '']
    (HERE/'report.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
