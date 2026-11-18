# -*- coding: utf-8 -*-
"""
text_align.py — 기준 사진 없이 텍스트 설정만으로 사진 한 장의 Y·Θ 추정 (C++ 이식 전 확인용)
  python text_align.py <사진...> [--config alignment_text.yml]
출력 : 사진마다 y_mm, t_deg (로봇 보정 이동 = (−y, −t)), 그리고 측정한 s, d
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import absmeas, text_config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'setup_package'))
from pose_model_tool import load_model, invert

def main():
    p = argparse.ArgumentParser(); p.add_argument('images', nargs='+'); p.add_argument('--config', default='alignment_text.yml'); a = p.parse_args()
    cfg = text_config.load(a.config)
    M = load_model(os.path.join(os.path.dirname(os.path.abspath(a.config)), cfg['model_file']))
    for f in a.images:
        try:
            r = absmeas.measure(f, cfg); s = r['s_abs'] - cfg['zero_s']; d = r['d_abs'] - cfg['zero_d']
            y, t = invert(M, s, d); print('%s\ty=%+.4f mm\tt=%+.4f deg\t(s=%.3f d=%.3f px)' % (os.path.basename(f), y, t, s, d))
        except Exception as e: print('%s\t실패: %s' % (os.path.basename(f), e))

if __name__ == '__main__': main()
