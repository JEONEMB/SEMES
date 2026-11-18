# -*- coding: utf-8 -*-
"""텍스트 설정(alignment_text.yml, OpenCV FileStorage 형식) 읽기·쓰기 — C++ 에서도 cv::FileStorage 로 그대로 읽을 수 있는 형식"""
import cv2, os

def load(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): raise RuntimeError('설정 파일을 열 수 없음: ' + path)
    def rois(name):
        n = fs.getNode(name); out = []
        for i in range(n.size()):
            e = n.at(i); out.append(dict(name=e.getNode('name').string(), cx=int(e.getNode('cx_px').real()), cy=int(e.getNode('cy_px').real()), hw=int(e.getNode('hw_px').real())))
        return out
    g = lambda k, d: (fs.getNode(k).real() if not fs.getNode(k).empty() else d)
    cfg = dict(qr=rois('qr_rois'), bg=rois('background_rois'), qr_search_px=int(g('qr_search_px', 40)), bg_search_px=int(g('bg_search_px', 40)),
               coarse_range_px=int(g('coarse_range_px', 350)), coarse_max_spread_px=float(g('coarse_max_spread_px', 15)),
               bg_method=fs.getNode('bg_method').string() or 'radial',
               zero_s=g('zero_s_px', float('nan')), zero_d=g('zero_d_px', float('nan')),
               model_file=fs.getNode('model_file').string(), path=path)
    fs.release(); return cfg

def save(cfg, path, zero_note=''):
    L = ['%YAML:1.0', '---',
         '# 기준 사진 없이 텍스트만으로 측정하는 설정 — ROI 는 모두 사진 절대 좌표 (px, 3840×2160 보정 프레임)',
         '# roi_setup.py 가 센터 사진 한 장에서 박스를 다시 맞추고 영점(zero_*)을 적는다. 이후 알고리즘은 사진 파일을 쓰지 않음.',
         'image_width_px: 3840', 'image_height_px: 2160',
         '# ① 대략 위치 : QR 사이 어두운 틈 2곳을 찾아 아래 박스를 통째로 옮김 (±coarse_range_px 안)',
         'coarse_range_px: %d' % cfg.get('coarse_range_px', 350), 'coarse_max_spread_px: %g' % cfg.get('coarse_max_spread_px', 15),
         '# ② QR : 밝은 사각형 4변을 행·열마다 서브픽셀로 잡아 직선 맞춤 → 중심 (박스 ± qr_search_px 안)',
         'qr_search_px: %d' % cfg.get('qr_search_px', 40), 'qr_rois:']
    L += ['   - { name: %s, cx_px: %d, cy_px: %d, hw_px: %d }' % (q['name'], q['cx'], q['cy'], q['hw']) for q in cfg['qr']]
    L += ['# ③ 배경(뒷벽 홀) : 어두운 링의 방사 대칭 중심 (Parthasarathy 2012), 좌우·상하 대칭 4곳 평균',
          'bg_method: "%s"' % cfg.get('bg_method', 'radial'), 'bg_search_px: %d' % cfg.get('bg_search_px', 40), 'background_rois:']
    L += ['   - { name: %s, cx_px: %d, cy_px: %d, hw_px: %d }' % (q['name'], q['cx'], q['cy'], q['hw']) for q in cfg['bg']]
    L += ['# ④ 영점 : 목표 자세(센터)에서 잰 s, d (px). s = QR 3개 중심 x 평균, d = s − 배경 4곳 x 평균. %s' % zero_note,
          'zero_s_px: %.6f' % cfg['zero_s'], 'zero_d_px: %.6f' % cfg['zero_d'],
          '# ⑤ 보정식 (s − zero_s, d − zero_d ↔ Y·Θ). 형식은 config/model_*.yml 과 같음', 'model_file: "%s"' % cfg.get('model_file', 'model_text_s1d2.yml'), '']
    open(path, 'w', encoding='utf-8', newline='\r\n').write('\n'.join(L))
