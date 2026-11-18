# -*- coding: utf-8 -*-
"""
roi_setup.py — 처음 설치할 때 한 번 : 센터(목표 자세) 사진으로 ROI 박스를 맞추고 영점을 텍스트로 기록

  python roi_setup.py <센터 사진> [--config alignment_text.yml] [--out alignment_text.yml] [--overlay roi_check.png] [--repeat 사진2 사진3 ...]

1. 설정의 ROI(대략 좌표)에서 출발해 QR 3개 중심 · 배경 홀 4개 중심을 찾는다 (박스 ± 40 px 안, 더 멀면 ① 대략 위치가 옮김)
2. 찾은 중심으로 박스 좌표를 다시 적는다 (정수 px) → 다음부터는 이 좌표만 씀
3. 같은 사진의 s, d 를 영점(zero_s_px, zero_d_px)으로 적는다. --repeat 로 센터를 여러 장 주면 평균 (권장 3장)
4. 박스·찾은 중심을 그린 확인용 그림을 저장 → 사람이 눈으로 한 번 확인
검사 : QR 3개 간격 차이 > 2 px, 홀 중심이 박스 가장자리 20 px 안, 센터 여러 장의 s 흔들림 > 0.3 px 이면 경고
"""
import argparse, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import absmeas, text_config

def main():
    p = argparse.ArgumentParser(); p.add_argument('center'); p.add_argument('--config', default='alignment_text.yml'); p.add_argument('--out')
    p.add_argument('--overlay', default='roi_check.png'); p.add_argument('--repeat', nargs='*', default=[])
    a = p.parse_args()
    cfg = text_config.load(a.config) if os.path.exists(a.config) else dict(absmeas.CFG, zero_s=np.nan, zero_d=np.nan, model_file='model_text_s1d2.yml')
    r = absmeas.measure(a.center, cfg); warn = []
    pitch = np.diff(r['qx']); 
    if abs(pitch[0] - pitch[1]) > 2: warn.append('QR 간격 차이 %.2f px' % abs(pitch[0] - pitch[1]))
    newq = [dict(q, cx=int(round(x)), cy=int(round(y))) for q, x, y in zip(cfg['qr'], r['qx'], r['qy'])]
    newb = []
    for q, x, y in zip(cfg['bg'], r['bx'], r['by']):
        if abs(x - q['cx']) > q['hw'] + cfg['bg_search_px'] - 20 or abs(y - q['cy']) > q['hw'] + cfg['bg_search_px'] - 20: warn.append('%s 중심이 박스 가장자리 근처' % q['name'])
        newb.append(dict(q, cx=int(round(x)), cy=int(round(y))))
    S = [r['s_abs']]; Dd = [r['d_abs']]
    for f in a.repeat:
        rr = absmeas.measure(f, cfg); S.append(rr['s_abs']); Dd.append(rr['d_abs'])
    if len(S) > 1 and np.ptp(S) > 0.3: warn.append('센터 여러 장의 s 차이 %.3f px (로봇이 같은 자리에 섰는지 확인)' % np.ptp(S))
    cfg.update(qr=newq, bg=newb, zero_s=float(np.mean(S)), zero_d=float(np.mean(Dd)))
    out = a.out or a.config
    text_config.save(cfg, out, '(%s, %d장 평균)' % (os.path.basename(a.center), len(S)))
    im = cv2.cvtColor(cv2.imread(a.center, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    for q, x, y in zip(newq, r['qx'], r['qy']):
        cv2.rectangle(im, (q['cx'] - q['hw'], q['cy'] - q['hw']), (q['cx'] + q['hw'], q['cy'] + q['hw']), (0, 200, 0), 3); cv2.drawMarker(im, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 60, 3)
    for q, x, y in zip(newb, r['bx'], r['by']):
        cv2.rectangle(im, (q['cx'] - q['hw'], q['cy'] - q['hw']), (q['cx'] + q['hw'], q['cy'] + q['hw']), (255, 120, 0), 3); cv2.drawMarker(im, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.putText(im, q['name'], (q['cx'] - q['hw'], q['cy'] - q['hw'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 120, 0), 3)
    cv2.imwrite(a.overlay, cv2.resize(im, (1920, 1080)))
    print('저장 :', out, ' 확인 그림 :', a.overlay)
    print('영점 s = %.4f px, d = %.4f px (%d장)' % (cfg['zero_s'], cfg['zero_d'], len(S)))
    for w in warn: print('경고 :', w)
    return 1 if warn else 0

if __name__ == '__main__': sys.exit(main())
