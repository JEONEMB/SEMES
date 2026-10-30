"""어제 결과표(17번)의 데이터, 특징점 조합, 교차검증 분할, 알고리즘 C1~C5를 그대로 불러온다 (grid_repro.py의 앞부분 실행)."""
import os
_src = open(os.path.join(os.path.dirname(__file__), 'v1_data', 'grid_repro.py'), encoding='utf-8').read()
_src = _src.split('rows = []')[0].replace('os.path.abspath(__file__)', repr(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v1_data', 'grid_repro.py')))
exec(_src)
