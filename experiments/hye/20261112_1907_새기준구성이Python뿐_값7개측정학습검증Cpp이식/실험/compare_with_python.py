"""C++ 도구가 잰 값 7개(+나사 세로 4)를 Python 분석 값과 대조한다. 검증용 스크립트다.

실행: python "<이 폴더>/실험/compare_with_python.py" <C++ features7.tsv> <python_features7.tsv>
"""
import csv, sys
import numpy as np
c = {int(r["id"]): r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t")}
p = {int(r["id"]): r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8"), delimiter="\t")}
assert set(c) == set(p), "사진 목록이 다르다"
cols = [k for k in next(iter(p.values())) if k != "id"]
print(f"사진 {len(c)}장\n값\t최대 차(px)\tRMS 차(px)\t값의 표준편차")
for k in cols:
    a = np.array([float(c[i][k]) for i in sorted(c)]); b = np.array([float(p[i][k]) for i in sorted(c)]); d = a - b
    print(f"{k}\t{abs(d).max():.2e}\t{np.sqrt((d ** 2).mean()):.2e}\t{b.std():.3f}")
