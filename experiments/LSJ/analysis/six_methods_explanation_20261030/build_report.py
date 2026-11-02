"""Describe existing measurements; do not run detectors, refit lines or test hypotheses.

All numeric examples and plots are read from the saved 2026-10-30 outputs.
The Gaussian curve is evaluated with saved coefficients only, without fitting.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SRC = ROOT / "analysis/reference_line_methods_108_20261030"
METHODS = ["E01", "E02", "E05", "E06", "Q02", "Q04"]
NAMES = {"E01": "가우시안 곡선으로 경계점 찾기", "E02": "Canny–Devernay로 구조물 경계점 찾기",
         "E05": "Zernike 모멘트로 경계점 찾기", "E06": "PAE로 경계점 찾기",
         "Q02": "Canny–Devernay로 QR 윗변 경계점 찾기", "Q04": "LSD로 QR 윗변 선분 찾기"}
TARGETS = {"upper_structure": "상단 구조물", "lower_structure": "하단 구조물",
           "qr_L": "왼쪽 QR", "qr_C": "중앙 QR", "qr_R": "오른쪽 QR"}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

all_data = read(SRC / "results/line_measurements.json")
records = [r for r in all_data["records"] if r["method_id"] in METHODS]
manifest = read(SRC / "configs/inputs_manifest.json")
config = read(SRC / "configs/methods.json")
assert len(records) == 56 and all(r["status"] == "ok" for r in records)

def row(m, ref="R01", target=None):
    target = target or ("upper_structure" if m.startswith("E") else "qr_L")
    return next(r for r in records if r["method_id"] == m and r["reference_id"] == ref and r["target_id"] == target)

def inter(m, ref="R01", target=None):
    return read(SRC / "results" / row(m, ref, target)["intermediate_json"])["intermediates"]

e1 = inter("E01")["profile_examples"][0]
e5 = inter("E05")["window_examples"][0]
e6 = inter("E06")["window_examples"][0]
assert e1["x"] == e5["x"] == e6["x"] == 24
assert e1["state"] == e5["state"] == e6["state"] == "selected"

# Copy evidence rather than modifying any original output.
copied = []
def copy_file(source, relative):
    dest = OUT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    copied.append({"source": str(source.relative_to(ROOT)), "copy": relative, "sha256": sha(source)})

for name in ["inputs_manifest.json", "targets.json", "methods.json", "method_sources.json"]:
    copy_file(SRC / "configs" / name, "evidence/configs/" + name)
for name in ["methods_points.cpp", "methods_segments.cpp", "line_fit.cpp", "io_utils.cpp", "main.cpp"]:
    copy_file(SRC / "cpp/src" / name, "evidence/code/" + name)
for folder, names in [("devernay_1.0", ["devernay.c", "devernay.h"]), ("lsd_1.6", ["lsd.c", "lsd.h"])]:
    for name in names:
        copy_file(SRC / "third_party" / folder / name, "evidence/code/" + folder + "/" + name)
for r in records:
    copy_file(SRC / "results" / r["intermediate_json"], "evidence/" + r["intermediate_json"])
copy_file(SRC / "figures/targets_R01_overview.png", "figures/targets_R01_overview.png")
for m in METHODS:
    for ref in ["R01", "R02", "R03", "R04"]:
        copy_file(SRC / "figures/methods" / f"{m}_{ref}.png", f"figures/{m}_{ref}.png")
write_json(OUT / "evidence/measurements_56.json", {"note": "원본 기록을 그대로 발췌. 측정·재피팅·통계 검정 미실행.", "records": records})

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
fig_prov = []

def save_plot(fig, name, description, measurement_ids):
    fig.savefig(OUT / "figures" / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    fig_prov.append({"file": "figures/" + name, "description": description,
                     "measurement_ids": measurement_ids, "new_measurement": False})

# A simple profile: saved samples and a saved fitted model, no optimization.
fig, ax = plt.subplots(figsize=(10.4, 4.1), constrained_layout=True)
ys = np.asarray(e1["y_samples"])
grid = np.linspace(ys.min(), ys.max(), 600)
curve = e1["A"] * np.exp(-(grid-e1["mu"])**2/(2*e1["sigma"]**2))
ax.plot(ys, e1["sobel_samples"], "o", color="#1565c0", label="저장된 Sobel 응답 11개")
ax.plot(grid, curve, color="#e07b16", lw=2, label="저장된 A·μ·σ로 그린 곡선")
ax.axvline(707, color="#8995a5", ls="--", label="처음 찾은 정수 위치: 707 px")
ax.axvline(e1["mu"], color="#c12845", ls="-", alpha=.8, label=f"곡선 꼭대기 μ: {e1['mu']:.4f} px")
ax.set(xlabel="위아래 영상 좌표 y (px)", ylabel="밝기 변화 응답 (Sobel 단위)",
       title="E01 | 한 열(x=24)에서 경계점 하나를 찾는 과정 — R01 상단")
ax.grid(alpha=.15)
ax.legend(fontsize=10, loc="upper right")
save_plot(fig, "E01_one_point.png", "저장된 프로파일과 가우시안 파라미터를 표시. 새 피팅 없음.", [row("E01")["measurement_id"]])

# Show the actual local patches and already-stored subpixel positions.
for m, ex, arrkey, coarsekey, name in [
    ("E05", e5, "window", "coarse_y", "E05_one_point.png"),
    ("E06", e6, "window_rows_top_to_bottom", "edge_pixel_y", "E06_one_point.png")]:
    values = np.asarray(ex[arrkey]); h, w = values.shape
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    # Coordinates below are local to the stored coarse point, in original pixels.
    ax.imshow(values, cmap="gray", vmin=110, vmax=180,
              extent=(-(w//2)-.5, w//2+.5, h//2+.5, -(h//2)-.5))
    for j in range(h):
        for i in range(w):
            ax.text(i-w//2, j-h//2, str(int(values[j,i])), ha="center", va="center",
                    color="white" if values[j,i] < 148 else "#1b3455", fontsize=10)
    ax.scatter([0], [0], s=150, facecolors="none", edgecolors="#168ee0", linewidth=2)
    if m == "E05":
        dx, dy = ex["u"]-ex["x"], ex["v"]-ex[coarsekey]
    else:
        dx, dy = 0, ex["a"]
    ax.scatter([dx], [dy], marker="x", s=160, color="#d42646", linewidth=2)
    ax.annotate("", xy=(dx,dy), xytext=(0,0), arrowprops={"arrowstyle":"->", "color":"#d42646", "lw":1.7})
    ax.set_xticks(range(-(w//2),w//2+1)); ax.set_yticks(range(-(h//2),h//2+1))
    ax.set(xlabel="처음 찾은 점에서 가로 거리 (px)", ylabel="처음 찾은 점에서 세로 거리 (px)",
           title=f"{m} | 실제 {w}×{h} 밝기 표본 — R01 상단 x=24\n파란 원: 정수 위치 (24, 707) / 빨간 ×: 추정 경계점")
    save_plot(fig, name, "저장된 원본 밝기 패치와 저장된 추정점만 표시.", [row(m)["measurement_id"]])

fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
r = row("E01"); it = inter("E01"); pts = np.asarray(it["fit_points"])
ax.scatter(pts[:,0], pts[:,1], s=3, color="#237bb8", alpha=.55, label="좌우에서 찾은 경계점")
x = np.array([24,3816]); ax.plot(x,r["slope_dy_dx"]*x+r["intercept_y_px"], color="#c12845", lw=1.8, label="좌우 점 전체로 추정한 직선 하나")
ax.axvspan(1370,2450,color="#e7e9ee",alpha=.8,label="중앙 가림 구간: 측정점 없음")
ax.invert_yaxis(); ax.set(xlabel="영상 x (px)",ylabel="영상 y (px)",title="좌우를 따로 피팅하지 않음 — R01 · E01 · 상단\n읽기 쉽게 세로축을 확대했으므로, 그림의 시각적 각도가 실제 각도는 아님")
ax.legend(fontsize=9,loc="lower left"); ax.grid(alpha=.15)
save_plot(fig,"one_line_both_sides.png","저장된 모든 피팅 입력점과 저장된 직선. 세로축 확대; 재피팅 없음.",[r["measurement_id"]])

def link(url, label):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'

def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in rr)+'</tr>' for rr in rows)+'</tbody></table></div>'

def steps(items):
    return '<ol class="steps">'+''.join('<li>'+item+'</li>' for item in items)+'</ol>'

def figure(path, caption):
    with Image.open(OUT/path) as img:
        width,height=img.size
    return f'<figure><a href="{path}" target="_blank"><img src="{path}" width="{width}" height="{height}" alt="{escape(caption, quote=True)}" loading="lazy"></a><figcaption>{caption} 그림을 클릭하면 원본 크기로 볼 수 있습니다.</figcaption></figure>'

def note(s, cls="note"):
    return '<div class="'+cls+'">'+s+'</div>'

def mathbox(s):
    return '<div class="formula">'+s+'</div>'

def details(title, body):
    return '<details><summary>'+title+'</summary>'+body+'</details>'

refs = {
 "S01": ("https://annals-csis.org/proceedings/2014/pliks/136.pdf", "Fabijańska (2014), Gaussian-Based Approach to Subpixel Detection of Blurred and Unsharp Edges"),
 "S02": ("https://www.ipol.im/pub/art/2017/216/", "Grompone von Gioi & Randall (2017), A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm"),
 "S04": ("https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf", "Grompone von Gioi 등 (2012), LSD: a Line Segment Detector"),
 "S05": ("https://pmc.ncbi.nlm.nih.gov/articles/PMC8321028/", "Renshaw & Christian (2020), Subpixel Localization of Isolated Edges and Streaks in Digital Images"),
 "S06": ("https://doi.org/10.1016/j.imavis.2012.10.005", "Trujillo-Pino 등 (2013), Accurate subpixel edge location based on partial area effect"),
 "S06M": ("https://github.com/juanse77/EdgeLocator/blob/master/Method.md", "EdgeLocator 저자 측 구현 설명, Method.md"),
 "H": ("https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html", "OpenCV 공식 문서: fitLine · DIST_HUBER")
}
def cite(k, label=None):
    u,t=refs[k]
    return link(u,label or t)

def output_table(m, all_refs=False):
    rr=[x for x in records if x["method_id"]==m and (all_refs or x["reference_id"]=="R01")]
    return table(["영상", "대상", "피팅 입력점", "최종 거리 기준 통과점", "기울기 m (px/px)", "영상 선 각도 φ (°)"],
                 [[x["reference_id"], TARGETS[x["target_id"]], x["diagnostics"]["fit_input_points"], x["diagnostics"]["fit_inlier_points"],
                   f'{x["slope_dy_dx"]:.9f}',f'{x["image_angle_deg"]:.6f}'] for x in rr])

def gallery(m):
    labels={"R01":"1차 center.bmp","R02":"2차 center1.bmp","R03":"2차 center2.bmp","R04":"2차 center3.bmp"}
    buttons=''.join(f'<button type="button" data-ref="{ref}" class="{"active" if ref=="R01" else ""}" aria-pressed="{"true" if ref=="R01" else "false"}">{ref} · {lab}</button>' for ref,lab in labels.items())
    caption=("왼쪽·오른쪽 확대 화면은 관찰용입니다. 최종 직선은 두 쪽의 점을 합쳐 추정했습니다." if m.startswith("E") else
             "세 행은 왼쪽·중앙·오른쪽 QR입니다. 각 QR의 상단 테두리에 직선 하나를 추정했습니다.")
    return '<div class="gallery" data-method="'+m+'"><div class="switches">'+buttons+'</div>'+figure(f"figures/{m}_R01.png",f"{m} 기존 측정 상세 그림. {caption}")+ '</div>'

parts=[]
def section(anchor,title,body,subtitle=""):
    parts.append(f'<section id="{anchor}"><h2>{title}</h2>'+ (f'<p class="lead">{subtitle}</p>' if subtitle else '')+body+'</section>')

section("start","먼저 볼 내용",note("<b>핵심 차이는 ‘경계점 또는 선분을 어디서 어떻게 찾는가’입니다.</b> 그 뒤에는 여섯 방법 모두 같은 직선 피팅 코드를 사용했습니다. 다만 Q04의 피팅 입력점은 영상에서 직접 찾은 경계점이 아니라, LSD가 이미 만든 선분 위에 다시 찍은 점입니다.")+
 table(["방법", "점 또는 선분을 만드는 방식", "최종 측정 대상", "구현 근거의 상태"],[
 ["E01", "세로 밝기 변화에 가우시안 곡선 → 꼭대기 위치", "상단·하단 구조물 각 1선", "논문의 세부 위치 계산 적용 / 앞뒤 과정 변경"],
 ["E02", "Canny 경계 → Devernay의 소수점 위치 보정", "상단·하단 구조물 각 1선", "저자 검출 코드 유지 / 대상 선택 규칙 추가"],
 ["E05", "7×7 밝기 분포의 모멘트 → 위치 보정", "상단·하단 구조물 각 1선", "논문의 ramp-edge 식 구현 / 흐림 폭 가정"],
 ["E06", "3×9 밝기 합 → 픽셀 안의 경계 위치", "상단·하단 구조물 각 1선", "PAE 기본식 부분 구현 / 창·밝기 추정 변경"],
 ["Q02", "Canny–Devernay 경계점 중 QR 윗변 선택", "QR 3개 각각의 윗변", "저자 검출 코드 유지 / 대상 선택 규칙 추가"],
 ["Q04", "LSD 선분 → 윗변 선분 선택 → 선분 위 표본", "QR 3개 각각의 윗변", "저자 검출 코드 유지 / 선분 선택·통합 추가"]])+
 '<p>아래 설명은 실제 C++ 코드, 설정 파일, 저장된 중간값과 그림을 대조해 작성했습니다. 새 검출·재피팅·통계 검정은 실행하지 않았습니다. 숫자는 기존 기록의 발췌이며, 설명용 그래프도 저장된 값으로 그렸습니다.</p>'+
 '<p><b>읽는 순서:</b> 먼저 측정 영역과 공통 피팅을 보고, 각 방법의 ‘실제 처리 순서 → 실제 점 하나 → 학술 근거와 자체 설정’을 읽으면 됩니다. 긴 계산식과 전체 기록은 펼쳐서 확인하도록 두었습니다.</p>'+
 '<p class="small">작성일 2026-10-30 · 검토 대상 E01/E02/E05/E06/Q02/Q04 · 기존 결과 56건. 이 문서는 E05의 재채택이나 기존 후보 목록 변경을 뜻하지 않습니다.</p>')

section("targets","1. 이미지에서 어디를 측정했나",
 '<p>측정 입력은 <b>1차 center.bmp 1장과 2차 center1·2·3.bmp 3장</b>입니다. 모두 3840×2160 원본을 회색조로 읽었습니다. 측정 좌표에 맞추려고 이미지를 회전·정합·확대하지 않았습니다. 방법별 평활 처리는 아래에 따로 설명합니다.</p>'+
 figure("figures/targets_R01_overview.png","R01에서 정의한 측정 영역. 중앙 가림 부분과 QR 모서리를 제외한 영역을 네 기준 영상에 동일하게 적용했습니다.")+
 table(["대상", "실제로 선택한 경계", "사용하지 않은 부분"],[
 ["상단 구조물", "밝기가 위쪽의 어두움에서 아래쪽의 밝음으로 바뀌는 경계 하나. 좌우 점을 한데 모아 직선 1개.", "중앙 커넥터·장착부로 가려진 부분. 구조물의 모든 경계를 쓰지는 않음."],
 ["하단 구조물", "같은 밝기 방향의 경계 하나. 좌우 점을 한데 모아 직선 1개.", "중앙 하단 장착물·반사부. 위·아래 여러 줄을 평균하지 않음."],
 ["QR 3개", "각 QR 바깥 밝은 사각 테두리의 윗변. 왼쪽·중앙·오른쪽 각 1개, 총 3선.", "아랫변·옆변·내부 검은 무늬·파인더·둥근 모서리는 최종 직선에서 제외."]])+
 note("<b>‘좌우 가시 영역 사용’은 좌우 직선 두 개의 기울기를 평균했다는 뜻이 아닙니다.</b> 중앙은 가려져 점이 없고, 양쪽에서 보이는 점을 함께 사용해 직선 하나를 만들었습니다. 또한 QR은 두 개만 쓴 것이 아니라 세 개 모두 썼습니다.")+
 details("정확한 픽셀 범위와 선택 이유",table(["대상", "x 범위", "y 탐색 띠", "제외 x 범위"],[
 ["상단", "24~3816", "692~736", "1370~2450"], ["하단", "24~3816", "1845~1905", "1560~2300"],
 ["QR 왼쪽", "1560~1725", "958~978", "없음"], ["QR 중앙", "1817~1984", "958~978", "없음"], ["QR 오른쪽", "2077~2243", "958~978", "없음"]])+
 '<p>위 범위는 양 끝을 포함합니다. QR 검출기에는 각 QR 전체 ROI에 8px 여백을 더한 영상을 주지만, 최종 선택은 표의 윗변 범위로 제한합니다. E02의 검출용 여백은 16px입니다. E01/E05/E06은 미분·주변 창 계산에 필요한 여백을 별도로 읽습니다.</p><p>범위의 근거는 R01 영상에서 대상을 식별한 위치와 기존 영역 설정입니다. 논문이 이 좌표를 정해준 것은 아닙니다. 설정 설명상 네 장의 결과를 본 뒤 좌표를 조정한 것은 아니지만, 이 범위가 다른 이동 영상에도 적절하다는 검증과는 별개입니다.</p>'+link("evidence/configs/targets.json","실제 targets.json 확인")))

section("fit","2. 모인 점으로 어떻게 직선을 만들었나",
 '<p><b>여섯 방법의 공통 후처리는 Huber 가중 직교 직선 피팅입니다.</b> 쉽게 말하면, 점들과의 거리가 작아지는 선을 찾되, 멀리 튀는 점이 직선을 지나치게 끌고 가지 못하도록 영향력을 줄였습니다. 위아래 거리만 최소화하는 일반적인 단순선형회귀와 달리 <b>점에서 선까지의 수직 최단거리</b>를 사용했습니다.</p>'+
 steps([
 '<b>첫 선을 찾습니다.</b> 처음에는 모든 입력점의 비중을 같게 두고, 점 분포를 가장 잘 따라가는 방향을 계산합니다.',
 '<b>멀리 떨어진 점의 비중을 줄입니다.</b> 현재 선까지 거리가 1.345px 이하면 비중 1, 더 멀면 1.345÷거리로 낮춥니다. 낮춘 비중으로 선을 다시 계산합니다.',
 '<b>이 계산을 반복합니다.</b> 최대 100회 또는 설정한 수렴 조건에서 멈춥니다. 이는 계산 종료 조건이며 정렬 성공 기준이 아닙니다.',
 '<b>큰 이탈점을 제외하고 다시 맞춥니다.</b> 구조물은 선에서 2px, QR은 1px 밖의 점을 제외하는 단계와 재피팅을 2회 수행합니다. 최소 점 수는 각각 50개·20개입니다.',
 '<b>최종 선의 기울기와 각도를 저장합니다.</b> 마지막 선에 대한 거리 기준 통과 여부도 다시 기록합니다. 표의 ‘최종 거리 기준 통과점 수’는 이 기록을 가리킵니다.'
 ])+
 figure("figures/one_line_both_sides.png","실제 저장된 E01 상단 입력점과 직선. 중앙 회색 부분에는 경계점이 없습니다. 가려진 구간을 지나는 빨간 선은 양쪽 점에서 연장된 추정선입니다.")+
 mathbox("영상 직선: y = m·x + b<br>기울기: m = Δy / Δx<br>영상 선 각도: φ = atan(m) × 180 / π")+
 '<p>x·y·b의 단위는 px, m의 단위는 px/px이므로 무차원, φ의 단위는 도(°)입니다. 영상에서는 y가 아래로 증가하므로 오른쪽으로 내려가는 선의 φ가 양수입니다. <b>여기서 φ는 이미지 안 직선의 각도이며, 기계의 θ값을 직접 측정한 결과는 아닙니다.</b></p>'+
 '<p>Huber로 이탈점의 영향력을 낮추는 방식과 C=1.345 상수는 '+cite("H")+'에 근거가 있습니다. 현재 코드는 cv::fitLine을 호출한 것이 아니라 같은 가중 원리를 배정밀도 C++로 직접 구현했습니다. 추가 제외 거리 2/1px, 최소 점 수, 반복 횟수는 이 프로젝트가 정한 설정입니다. 이 영상에서 적절한지는 별도 검증되지 않았습니다.</p>'+
 details("수식과 종료 설정까지 확인",mathbox("dᵢ = −(xᵢ−pₓ)·vᵧ + (yᵢ−pᵧ)·vₓ, 단 vₓ²+vᵧ²=1<br>wᵢ = 1 (|dᵢ|≤C), 또는 C/|dᵢ| (|dᵢ|>C)<br>C = 1.345 px")+
 '<p>가중 중심을 구하고 2×2 가중 산포 행렬에서 선의 방향을 계산하는 것을 반복합니다. 코드의 수렴 조건은 연속 각도 변화 &lt;10⁻¹³ rad이면서 중심의 x·y 변화가 각각 &lt;10⁻⁹px입니다. 작은 수렴값을 썼다고 그만큼 정확히 측정했다는 뜻은 아닙니다.</p>'+link("evidence/code/line_fit.cpp","line_fit.cpp — huber_line, robust_fit 확인"))+
 note("<b>RMS를 읽을 때:</b> 저장된 피팅 RMS는 ‘선에 사용한 점이 그 선 주위에 얼마나 퍼져 있는가’입니다. 실제 경계의 정답 좌표와 비교한 오차가 아닙니다. Q04는 입력점의 생성 방식이 달라 특히 별도로 읽어야 합니다."))

def method_section(m, body, basis_rows, settings_rows, code_location, extra_math=""):
    body += '<h3>학술 근거를 따른 부분과 달라진 부분</h3>'+table(["구분","실제 적용 내용"],basis_rows)
    body += '<h3>설정값은 무엇을 뜻하나</h3>'+table(["설정","현재 값","근거·확인 상태"],settings_rows)
    if extra_math:
        body += details("계산식 자세히 보기",extra_math)
    body += '<h3>기존 측정 결과를 따라가 보기</h3><p>아래는 R01의 저장값입니다. 입력점 수는 공통 피팅 함수에 넘긴 개수이고, 마지막 수는 최종 선과의 거리 기준을 통과한 개수입니다. 소수점 표시는 기존 수치를 옮긴 것으로 정확도 보증 자릿수가 아닙니다.</p>'+output_table(m)
    body += gallery(m)
    body += '<p class="small">상세 그림: 파랑은 후보, 주황은 선택점·선분, 보라 ×는 제외점, 빨강은 최종선입니다. 잔차 그래프는 세로축을 확대하므로 모양만 보고 실제 각도를 읽지 마세요. Q04의 지지 픽셀 진단값은 주 측정값과 별도입니다.</p>'
    body += details("기준 영상 4장 전체 기록 보기",output_table(m,True))
    body += '<p class="source-code">코드 위치: '+code_location+'</p>'
    section(m,m+' · '+NAMES[m],body)

method_section("E01",
 '<p class="lead">경계 주변에서 밝기가 가장 빠르게 바뀌는 위치를, 매끄러운 종 모양 곡선의 꼭대기로 찾습니다.</p>'+
 '<p>여기서 가우시안 곡선을 맞추는 대상은 원래 밝기 자체가 아니라 <b>위아래 밝기 변화량(Sobel 응답)</b>입니다. 흐릿한 경계에서는 밝기가 여러 픽셀에 걸쳐 바뀌므로, 응답의 봉우리가 픽셀 중간에 있을 수도 있다고 보는 것입니다.</p>'+
 steps([
 '구조물의 정해진 탐색 띠 안을 x 방향 2px 간격으로 훑습니다. 중앙 가림 구간은 건너뜁니다.',
 '3×3 세로 Sobel로 밝기 변화량을 구하고, 각 열에서 어두움→밝음 변화가 가장 큰 정수 y를 찾습니다.',
 '그 y의 위아래 5px, 총 11개 응답을 가져옵니다. 이 표본에 가우시안 곡선을 맞춰 높이 A, 꼭대기 위치 μ, 폭 σ를 구합니다. 수치 최적화에는 Nelder–Mead를 사용합니다.',
 '수렴하지 않았거나, 곡선 높이가 양수가 아니거나, 꼭대기가 최초 y에서 2px보다 멀면 제외합니다. 통과하면 <b>(그 열의 x, μ)</b>가 경계점 하나입니다.',
 '좌우에서 얻은 경계점 전체에 앞의 Huber 피팅을 적용해 구조물 한 개의 직선 하나를 얻습니다. 상단과 하단은 각각 처리합니다.'
 ])+
 figure("figures/E01_one_point.png","실제 R01 상단의 x=24 열. 점은 저장된 응답이고 곡선은 저장된 파라미터로 다시 그렸습니다. 새로 최적화한 결과가 아닙니다.")+
 f'<p class="example"><b>실제 점 하나:</b> 처음 위치 y=707 → μ={e1["mu"]:.6f}. 따라서 경계점은 <b>(24, {e1["mu"]:.6f})px</b>입니다. 이 점 하나에서 선의 기울기를 구한 것이 아니라 같은 방식으로 모은 많은 점으로 선을 맞췄습니다.</p>',
 [
 ["논문에서 가져온 핵심", cite("S01")+'의 III.B: 경사 응답에 가우시안 곡선을 맞추고 중심 μ로 서브픽셀 위치를 얻는 방식. 초기 A=최댓값, μ=중심, σ=표본 구간 크기 규칙도 참조했습니다.'],
 ["앞단을 바꾼 부분", '논문의 초기 경계 검출은 Sobel 크기·ISODATA 임계값·골격화입니다. 현재는 <b>세로 Sobel + 지정 띠 안 각 열 최댓값</b>으로 대체했습니다.'],
 ["뒷단을 바꾼 부분", '논문 III.C의 곡선 연결은 cubic spline입니다. 현재 목표가 긴 직선이므로 공통 Huber 직선 피팅을 붙였습니다. 따라서 논문 전체 절차를 그대로 재현한 것은 아닙니다.'],
 ["확인되지 않은 부분", '세로 방향 고정, 11개 표본, 제외 임계값이 이 경계에서 얼마나 적절한지는 검증되지 않았습니다. 모델식에 출처가 있다는 것과 현재 조합이 검증됐다는 것은 구분해야 합니다.']
 ],[
 ["표본 창", "위아래 5px → 11개", "기존 측정 코드와 표본 수를 맞춘 선택. 해당 영상의 최적값 검증 없음."],
 ["최소 Sobel 봉우리", "8", "약한 후보를 제외하려는 자체 설정. Sobel 응답 단위이며 원본 밝기 8과 같지 않음."],
 ["허용 위치 보정", "최초 y에서 ±2px", "다른 경계로 넘어가는 것을 막으려는 자체 제한."],
 ["최적화 종료", "최대 800회 / 상대 목적함수 차이 10⁻¹⁰", "계산용 설정. 영상 각도의 정확도 기준이 아님."],
 ["직선 피팅", "공통 Huber + 거리 2px", "2절의 공통 설정 적용."]
 ],link("evidence/code/methods_points.cpp","methods_points.cpp")+' : method_profile_gaussian, 104행부터',
 mathbox('g(y) = A·exp(−(y−μ)²/(2σ²))<br>목적함수 = Σ[측정 Sobel 응답 − g(y)]²')+
 '<p>이 목적함수가 작아지도록 A·μ·σ를 찾습니다. 계산 중 σ의 절댓값을 사용합니다. ‘곡선 피팅으로 점 찾기’와 이후 ‘여러 점에 직선 피팅’은 서로 다른 두 단계입니다.</p>')

devernay_math=(mathbox('δ = ½·(a−c)/(a−2b+c)<br>보정 위치 = 정수 경계 위치 + δ')+
 '<p>a·b·c는 봉우리 주변 세 위치의 경사 크기입니다. 세 점에 맞는 이차곡선의 꼭대기를 계산합니다. 저자 코드는 주된 경사 축 방향으로 보정하며 보정은 ±0.5px 안입니다. 이후 프로젝트의 위치·밝기 방향 필터가 별도로 적용됩니다.</p>')
method_section("E02",
 '<p class="lead">먼저 경계 후보들을 찾고, 밝기 변화 봉우리 주변 세 값을 이용해 위치를 소수점까지 보정합니다.</p>'+
 steps([
 '구조물의 탐색 영역에 주변 16px를 더해 Canny–Devernay 저자 코드에 넣습니다. σ=1 Gaussian 평활로 밝기 변동을 완화합니다.',
 '밝기 변화가 주변보다 큰 지점을 남기고, 강한 경계에 이어진 약한 경계도 연결합니다. 강한 경계 임계값은 4, 약한 경계 임계값은 2입니다.',
 '경사 봉우리 주변 세 값을 이차곡선으로 해석해, 정수 경계 위치를 소수점 좌표로 보정합니다. 이 과정은 수정하지 않은 저자 코드가 수행합니다.',
 '반환된 점 중 지정 띠·가로 범위 안에 있고 중앙 가림 구간 밖인 점만 봅니다. 위→아래 밝아지는 방향이며 세로 밝기 변화가 가로 변화보다 큰 점을 남깁니다.',
 '2px 폭의 x 묶음마다 밝기 변화가 가장 강한 점 하나를 고릅니다. 반올림한 좌표는 묶음 선택에만 쓰고, 피팅에는 <b>원래 서브픽셀 좌표</b>를 넣습니다.',
 '좌우의 선택점을 합쳐 공통 Huber 피팅으로 구조물별 직선 하나를 구합니다.'
 ])+
 '<p class="example"><b>R01 상단의 실제 흐름:</b> 주변 영역에서 검출한 점 16,594개 → y 탐색 띠 안 후보 9,768개 → 위치·방향·강도 선택 후 피팅 입력 1,357개 → 최종 거리 기준 통과 1,357개. 각 숫자가 가리키는 처리 단계가 다릅니다.</p>',
 [
 ["학술 근거", cite("S02")+'의 Canny 검출 및 Devernay 서브픽셀 위치 보정. 공식 배포본 1.0의 devernay.c/h를 사용했습니다.'],
 ["원본 유지", '저자 코드 devernay.c/h는 보관된 공식 배포 ZIP과 바이트 단위로 일치합니다. 이 확인은 검출 핵심 코드에 대한 것이며 프로젝트 전체 절차가 원본과 같다는 의미는 아닙니다.'],
 ["프로젝트가 추가한 부분", '구조물 영역 자르기, 밝기 방향·가로세로 경사 조건, 2px 묶음 최강점 선택, Huber 피팅 및 거리 제외를 추가했습니다.'],
 ["설정 근거의 한계", '약한 하단 경계도 잡으려 임계값을 낮췄다는 목적은 설정 파일에 있습니다. 4/2라는 숫자가 적절하다는 비교 검증은 없습니다.']
 ],[
 ["Gaussian 평활", "σ=1px", "프로젝트가 선택한 평활 세기. 원본 라이브러리가 받는 매개변수."],
 ["경계 임계값", "강함 4 / 약함 2", "구조물의 약한 경계를 포함하려는 자체 설정."],
 ["가로 묶음", "2px마다 최강점 1개", "한 위치의 여러 경계 중 하나를 고르는 자체 규칙."],
 ["방향 선택", "gy>0, |gy|>|gx|", "목표가 어두움→밝음의 거의 수평 경계라는 대상 정의에서 추가."],
 ["직선 피팅", "공통 Huber + 거리 2px", "저자 경계 연결 결과에 프로젝트의 직선 피팅을 추가."]
 ],link("evidence/code/methods_points.cpp","methods_points.cpp")+' : method_canny_devernay, 184행부터',devernay_math)

method_section("E05",
 '<p class="lead">경계 주변 작은 사각형의 밝기 분포를 몇 개의 숫자로 요약한 뒤, 경계가 어느 방향으로 얼마나 치우쳐 있는지 계산합니다.</p>'+
 '<p><b>모멘트</b>는 픽셀마다 다른 가중치를 곱해 더한 값입니다. 여기서는 좌우·상하의 밝기 배치와 중심에서 떨어진 밝기 배치를 나타내는 세 가중합을 씁니다. 경계가 창 중앙에서 움직이면 이 값들이 달라진다는 모델에서 위치를 역산합니다.</p>'+
 steps([
 'σ=1로 평활한 영상에서, 각 열의 위→아래 밝기 변화가 가장 큰 정수 y를 먼저 찾습니다. x는 2px 간격이며 중앙 가림 구간은 건너뜁니다.',
 '그 위치를 중심으로 <b>평활 전 원본 밝기 7×7</b>을 가져오고, 패치 평균 밝기를 뺍니다.',
 '7×7 픽셀에 세 가지 가중치 표를 곱해 더하여 A11의 두 성분과 A20을 구합니다. 가중치 표는 Zernike 식의 픽셀 면적 적분을 수치 계산해 만들었습니다.',
 '모멘트에서 경계에 수직인 방향 ψ와 위치 이동량을 구합니다. 이때 경계가 흐려지는 폭을 σ=1.5px라고 가정합니다. <b>실제로 측정한 흐림 값은 아닙니다.</b>',
 '계산된 방향으로 최초 점을 이동시킵니다. 방향이 예상과 30° 넘게 다르거나, 계산식의 실수 해가 없거나, 이동량이 1.5px보다 크면 제외합니다.',
 '남은 2차원 경계점들을 공통 Huber 피팅에 넣어 직선을 얻습니다. 각 패치의 ψ를 평균해 최종 기울기를 만든 것이 아닙니다.'
 ])+
 figure("figures/E05_one_point.png","R01 상단 x=24에서 사용한 실제 7×7 밝기. 숫자는 원본 밝기이고, 계산할 때에는 이 패치의 평균을 뺍니다.")+
 f'<p class="example"><b>실제 점 하나:</b> (24, 707) → <b>({e5["u"]:.6f}, {e5["v"]:.6f})px</b>. 저장된 ψ는 {e5["psi_deg"]:.6f}°이며, 이는 경계에 <b>수직인 국소 방향</b>입니다. 최종 구조물 선 각도 약 0.272°와 같은 종류의 값이 아닙니다.</p>',
 [
 ["학술 근거", cite("S05")+'의 흐릿한 경계(ramp edge) 위치 추정식. 방향 식 (31), 흐림 폭 관계 (33), 위치 식 (25), ramp 위치 식 (41) 및 비율 식 (42)를 참조했습니다.'],
 ["수치 구현", 'Zernike 가중치의 면적 적분을 픽셀당 64×64 표본으로 근사했습니다. 적분을 코드로 계산한 방식이며 새로운 경계 이론을 만든 것은 아닙니다. 이 근사의 오차 검증은 하지 않았습니다.'],
 ["프로젝트 선택·추가", '7×7 크기 선택, 초기 열별 봉우리 검출, 패치 평균 제거, 위치·방향 제한, 공통 직선 피팅을 사용했습니다. 논문은 다른 초기 검출기와 5×5·7×7 등의 창을 허용하므로, 이런 선택 모두를 공식 변조라고 볼 수는 없습니다.'],
 ["핵심 미확인 가정", '<b>실제 흐림 σ=1.5px 가정</b>이 위치 계산식에 직접 들어갑니다. 논문 식을 사용했더라도 이 가정이 해당 영상에서 맞는지는 확인되지 않았습니다. 경계용 부분만 구현했으며 논문의 streak 기능은 사용하지 않았습니다.']
 ],[
 ["초기 평활 / 최소 경사", "σ=1 / 0.5", "최초 정수 위치를 찾기 위한 자체 설정."],
 ["지역 창 / 적분", "7×7 / 픽셀당 64×64", "허용 가능한 창 크기의 선택 + 수치 적분 해상도. 해당 값의 최적성·근사 오차 검증 없음."],
 ["모델의 흐림 폭", "σ=1.5px, k=1.8", "k=1.8은 참조 논문의 edge 계수. σ=1.5는 계측 없이 설정."],
 ["후보 제외", "보정 ≤1.5px / 방향 편차 ≤30°", "잘못된 후보 제거를 위한 자체 설정."],
 ["직선 피팅", "공통 Huber + 거리 2px", "2절의 공통 설정 적용."]
 ],link("evidence/code/methods_points.cpp","methods_points.cpp")+' : zernike_masks 및 method_zernike, 266·297행부터',
 mathbox('ψ = atan2(Im A11, Re A11)<br>A11′ = Re A11·cosψ + Im A11·sinψ<br>Q = A20 / A11′<br>w̄ = 2·k·σ / N, N=7<br>ℓ = [1−w̄²−√((w̄²−1)²−2w̄²Q)] / w̄²<br>점 = (x₀, y₀) + (N·ℓ/2)·(cosψ, sinψ)')+
 '<p>σ=1인 앞단 평활과 σ=1.5인 모델 흐림 가정은 역할이 다른 값입니다. 앞단에서 영상을 평활했다고 카메라 영상의 실제 흐림 폭이 측정되는 것은 아닙니다.</p>')

method_section("E06",
 '<p class="lead">경계가 한 픽셀을 가르면, 그 픽셀의 밝기에는 양쪽 면적의 비율이 섞인다는 생각으로 위치를 계산합니다.</p>'+
 '<p>예를 들어 어두운 쪽과 밝은 쪽이 한 픽셀을 절반씩 차지하면 중간 밝기가 관측된다는 모델입니다. 실제 구현은 경계 주변 세 열의 밝기 합으로, 픽셀 안에서 경계가 지나가는 위치를 계산합니다.</p>'+
 steps([
 '원본 밝기의 중앙차분으로 위아래·좌우 변화를 구합니다. 위→아래 밝아지고 세로 변화가 더 크며, 위아래 이웃보다 경사가 큰 후보를 찾습니다. 각 2px 간격 열에서 가장 강한 후보 하나를 고릅니다.',
 '정수 후보를 중심으로 가로 3열, 세로 9행의 원본 밝기를 가져옵니다.',
 '맨 윗행 세 픽셀의 평균을 어두운 쪽 밝기 A, 맨 아랫행 평균을 밝은 쪽 밝기 B로 둡니다. 세 열 각각의 밝기 합 S_L·S_M·S_R도 계산합니다.',
 '이 작은 창 안 경계를 y_local=a+b_local·x_local+c·x_local²로 놓고 a·b_local·c를 계산합니다. <b>창 중앙 x_local=0에서 경계 위치가 a</b>이므로, 점은 (원래 x, 최초 y+a)가 됩니다.',
 'B−A가 3보다 작거나 |a|가 1.5px보다 크면 제외합니다. 남은 점을 공통 Huber 피팅에 넣어 구조물별 직선 하나를 구합니다.'
 ])+
 figure("figures/E06_one_point.png","R01 상단 x=24에서 사용한 3×9 밝기. 맨 윗행에서 A, 맨 아랫행에서 B를 잡고 세 열의 합으로 점 위치를 계산했습니다.")+
 f'<p class="example"><b>실제 점 하나:</b> A={e6["A"]:.0f}, B={e6["B"]:.0f}, 세 열 합은 {e6["S_L"]:.0f}·{e6["S_M"]:.0f}·{e6["S_R"]:.0f}입니다. a={e6["a"]:.6f}px이므로 (24, 707) → <b>(24, {707+e6["a"]:.6f})px</b>가 됩니다.</p>'+
 note('여기서 계산한 b_local은 <b>3×9 창 안의 국소 곡선 계수</b>입니다. 보고서의 최종 직선 기울기 m은 많은 경계점을 모아 별도로 계산합니다.'),
 [
 ["이론의 출처", cite("S06")+'의 부분 면적 효과(PAE). 이번 코드 대조에서 직접 확인한 계산 설명은 '+cite("S06M")+'입니다.'],
 ["실제로 참조한 부분", 'Method.md의 기본 2차 경계 모델, 열별 밝기 합으로 a·b·c를 구하는 관계를 자체 C++로 작성했습니다. 저자 전체 구현을 그대로 가져온 방식은 아닙니다.'],
 ["변경한 부분", '<b>설명 예시의 3×5 창을 3×9로 확대</b>했습니다. A/B도 설명의 모서리 쪽 픽셀 선택 대신 <b>창의 맨 윗행·아랫행 세 픽셀 평균</b>으로 정했습니다. 높이 변경은 식의 창 높이에 반영됐지만, 현재 영상에서 더 적절하다는 근거가 확보된 것은 아닙니다.'],
 ["구현 범위·확인 한계", '평활된 영상용 보정식, floating window, 수직 경계 분기는 사용하지 않았습니다. <b>2013년 논문 전문을 직접 대조한 상태가 아니므로 논문 전체의 충실한 재현이라고 표현할 수 없습니다.</b> 기본형의 부분 구현입니다.']
 ],[
 ["밝기 창", "3×9", "주변을 더 넓게 읽도록 바꾼 자체 선택. 3×5 대비 타당성 검증 없음."],
 ["A·B 추정", "위·아래 끝행 각각 3개 평균", "국소 밝기 대표값을 정하는 자체 변경. 끝행이 실제로 균일한 면인지는 별도 확인 필요."],
 ["최소 경사 / 밝기차", "1 / B−A≥3", "잡음·약한 후보 제외용 자체 설정."],
 ["위치 보정 제한", "|a|≤1.5px", "멀리 이탈하는 후보 제외용 자체 설정."],
 ["직선 피팅", "공통 Huber + 거리 2px", "2절의 공통 설정 적용."]
 ],link("evidence/code/methods_points.cpp","methods_points.cpp")+' : method_pae, 392행부터',
 mathbox('H = 9, D = 2(A−B)<br>c = (S_L+S_R−2S_M)/D<br>b_local = (S_R−S_L)/D<br>a = [2S_M−(A+B)H]/D − c/12<br>경계점 = (x₀, y₀+a)')+
 '<p>이 계산은 작은 창 안에 어두운 쪽과 밝은 쪽의 대표 밝기를 둘 수 있고, 경계가 2차 곡선으로 근사된다는 모델을 전제로 합니다. A/B를 잘못 잡으면 이 식에 넣을 숫자부터 달라집니다.</p>')

method_section("Q02",
 '<p class="lead">E02와 같은 경계 검출기를 사용하지만, QR 세 개 각각의 흰 바깥 테두리 윗변만 골라 직선으로 만듭니다.</p>'+
 steps([
 '왼쪽·중앙·오른쪽 QR 각각의 전체 ROI에 8px 여백을 더합니다. 이 단계에서는 QR 내부 무늬도 검출 대상 영상에 포함됩니다.',
 'Canny–Devernay를 실행합니다. 평활 σ=1, 강한 경계 임계값 30, 약한 경계 임계값 15로, 구조물용 E02와 임계값이 다릅니다.',
 '저자 코드가 경계의 정수 후보를 구하고, 봉우리 주변 세 값으로 서브픽셀 위치를 보정합니다.',
 '이 중 y=958~978과 QR별 x 범위 안에서, 위가 어둡고 아래가 밝으며 세로 변화가 큰 점만 남깁니다. 따라서 내부 무늬를 포함한 검출 결과에서 <b>흰 테두리의 윗변만 선택</b>됩니다.',
 'x를 1px 폭으로 묶어 가장 강한 점 하나씩 남깁니다. 각 QR의 점들을 따로 공통 Huber 피팅에 넣어 <b>QR별 직선 1개, 총 3개</b>를 구합니다.'
 ])+
 '<p class="example"><b>R01 왼쪽 QR의 실제 흐름:</b> QR 주변 영역에서 검출된 점 4,726개 → y 띠 안 후보 207개 → 피팅 입력 166개 → 최종 거리 기준 통과 166개. 중앙과 오른쪽도 각각 피팅했습니다.</p>'+
 '<p>QR 디코딩이나 파인더 중심·코너 검출로 구한 선이 아닙니다. QR의 밝은 바깥 테두리를 일반적인 밝기 경계로 취급했습니다.</p>',
 [
 ["학술·구현 근거", cite("S02")+'의 공식 Devernay 1.0 코드. 검출 핵심은 E02와 같습니다.'],
 ["프로젝트가 추가한 부분", 'QR별 ROI, 상단 띠, 모서리를 피한 x 범위, 밝기 방향 필터, 1px 묶음 최강점 선택과 공통 Huber 피팅입니다.'],
 ["E02와 다른 부분", '임계값 4/2→30/15, 가로 묶음 2px→1px, 최종 제외 거리 2px→1px, 최소 점 수 50→20으로 설정했습니다. 이 차이는 대상별 설정이며 논문이 자동 결정한 값이 아닙니다.'],
 ["확인되지 않은 부분", '이 임계값·영역에서 특정 경계를 일관되게 잡는지에 대한 별도 타당성 검증은 없습니다. 네 장에서 선이 산출됐다는 사실까지만 기존 기록으로 확인됩니다.']
 ],[
 ["평활 / 경계 임계값", "σ=1 / 강함 30, 약함 15", "QR용 자체 설정. 저자 코드의 입력값."],
 ["영역", "각 QR 전체 ROI+8px → 상단 띠 선택", "최종적으로 윗변만 사용. 아랫변과 옆변은 사용하지 않음."],
 ["가로 묶음", "1px마다 최강점 1개", "짧은 윗변에서 점을 더 촘촘히 유지하려는 선택."],
 ["직선 피팅", "공통 Huber / 거리 1px / 최소 20개", "E02보다 엄격한 거리 설정. 해당 숫자의 적합성 검증 없음."]
 ],link("evidence/code/methods_points.cpp","methods_points.cpp")+' : method_canny_devernay, 184행부터. QR 설정으로 호출.',devernay_math)

method_section("Q04",
 '<p class="lead">영상에서 짧은 선분들을 먼저 검출하고, 그중 QR 윗변에 해당하는 선분을 골라 최종 직선을 만듭니다.</p>'+
 '<p>LSD는 밝기 변화의 방향이 비슷한 주변 픽셀을 묶고, 길쭉한 영역에 선분을 맞춥니다. 무작위 방향에서도 이런 모양이 우연히 나올 가능성을 고려해 선분을 걸러냅니다. 이 검출기의 출력부터 이미 선분 양 끝점 좌표가 존재합니다.</p>'+
 steps([
 '각 QR의 전체 ROI+8px 영상에 LSD 1.6 저자 코드를 실행합니다. scale=1.0을 사용하여 원본 크기를 유지하며, 이 설정에서는 저자 코드의 Gaussian 평활·재표본화 분기도 건너뜁니다.',
 '검출된 선분 중 영상 각도 절댓값이 5° 이내이고, 중점 y가 윗변 띠 안이며, 지정 x 범위 안의 가시 길이가 10px 이상인 선분을 고릅니다.',
 '선분의 양쪽으로 수직 거리 2px 떨어진 밝기를 비교합니다. 아래쪽이 위쪽보다 평균 20 이상 밝은 선분만 남깁니다.',
 '남은 선분을 같이 설명하는 직선 가설들을 비교합니다. 가설에서 양 끝점이 모두 1px 이내인 선분들의 길이 합이 가장 큰 가설을 고르고, 그에 속하는 선분들을 선택합니다.',
 '<b>선택된 선분 위에 x 간격 1px로 점을 다시 찍습니다.</b> 이렇게 만든 표본을 공통 Huber 피팅에 넣어 최종 직선 하나를 얻습니다. QR 3개에 각각 수행합니다.'
 ])+
 '<p class="example"><b>R01 왼쪽 QR의 실제 흐름:</b> LSD 선분 187개 → 위치·방향·길이·밝기 필터 통과 1개 → 선택 선분 1개 → 선분 위 표본 167개 → 최종 각도 0.265242°. <b>167개는 영상에서 독립적으로 찾은 경계점 수가 아닙니다.</b></p>'+
 note('<b>이 방법에서 RMS가 거의 0일 수 있는 이유:</b> 선분 하나 위에 찍은 점은 처음부터 한 직선 위에 있습니다. 그 점에 다시 선을 맞추면 오차가 거의 0인 것은 당연합니다. 그래서 기존 코드도 선분 1개인 경우 주 RMS를 비워 두었습니다. LSD의 원래 지지 픽셀로 계산한 별도 진단 RMS·재피팅 각도는 주 측정값과 구분해야 합니다.'),
 [
 ["학술 근거", cite("S04")+'의 방향 영역 확장, 영역의 선분화, a-contrario/NFA 검출 기준. 공식 LSD 1.6 C 코드의 LineSegmentDetection을 호출했습니다.'],
 ["원본 유지·설정 변경", 'lsd.c/h는 보관된 공식 1.6 ZIP과 바이트 단위로 일치합니다. 저자 기본 설정에서 scale만 0.8→1.0으로 바꿨습니다. <b>이 값에서는 Gaussian 평활·재표본화도 실행되지 않습니다.</b> 즉 코드 원본은 유지했지만 기본 전처리와 같은 실행 조건은 아닙니다.'],
 ["프로젝트가 추가한 부분", 'QR 영역 입력, 윗변·길이·각도·밝기 필터, 선분 가설의 길이 합 선택, 선분 위 재표본화와 Huber 통합입니다. 이 전체 후처리가 LSD 논문의 원래 알고리즘은 아닙니다.'],
 ["통계 기준의 의미", 'LSD의 NFA는 검출 단계에서 우연한 선분을 제한하기 위한 모델 기준입니다. 네 기준 영상의 기울기 동등성이나 θ 추정 정확도를 검증한 결과가 아닙니다. 또한 논문은 잘라낸 입력 영역과 전체 영상의 검출 결과가 달라질 수 있음을 보여줍니다.']
 ],[
 ["검출 크기", "scale=1.0", "저자 기본 0.8에서 변경. 원본 해상도 유지가 목적이며 성능 향상 검증은 아님."],
 ["나머지 LSD 설정", "σ_scale=.6 / quant=2 / ang_th=22.5° / log_eps=0 / density=.7 / bins=1024", "저자 표준 기본값을 유지. σ_scale는 저장되어 있지만 scale=1에서는 평활에 사용되지 않음. ang_th는 내부 방향 허용값이며 아래 5° 필터와 다름."],
 ["대상 선분 선택", "|각도|≤5° / 가시 길이≥10px / 밝기차≥20", "QR 윗변을 고르기 위한 자체 규칙."],
 ["선분 통합", "양 끝점 거리≤1px / 길이 합 최대", "여러 선분을 같은 윗변으로 묶기 위한 자체 규칙."],
 ["재표본 / 최종 피팅", "x 간격 1px / Huber 거리 1px", "선분을 표본점으로 바꾸어 통합. 직접 경계점 기반 방법과 입력의 성격이 다름."]
 ],link("evidence/code/methods_segments.cpp","methods_segments.cpp")+' : select_and_fit 51행부터, method_lsd 213행부터. '+link("evidence/code/lsd_1.6/lsd.c","lsd.c")+' : scale 분기 2068행부터.',
 '<p>후처리의 가설 후보는 각 선분의 직선과 두 선분 중점을 잇는 직선입니다. 길이가 긴 순으로 최대 300개 후보 선분에서 가설을 만들고, 조건에 맞는 선분의 가시 길이 합을 점수로 씁니다. 마지막 표본 생성은 잘린 선분의 가시 x 구간을 1px씩 따라갑니다. 이 점들은 영상 밝기를 다시 읽어 위치를 재검출한 결과가 아닙니다.</p>')

section("read-results","3. 이 보고서로 판단할 수 있는 내용",
 table(["확인한 사항", "현재 의미"],[
 ["여섯 방법 모두 기준 영상 네 장에서 선 산출", "E01/E02/E05/E06은 각각 상·하단 8건, Q02/Q04는 각각 QR 3개×4장 12건. 총 56건의 저장 기록이 존재함."],
 ["점 검출과 직선 피팅을 구분", "E01/E02/E05/E06/Q02는 영상에서 찾은 경계점, Q04는 검출 선분 위 재표본이 피팅 입력임."],
 ["이론·원본 코드의 출처가 있는 부분", "가우시안 모델, Devernay, Zernike ramp 식, PAE 기본형, LSD, Huber 가중 원리."],
 ["추가 설정의 타당성", "목적이 설명된 설정은 있으나, 그 숫자·변형이 해당 영상에 적절하다는 검증은 별도 문제임."],
 ["이번 보고서에 없는 결론", "기울기 균일성 통계 검정, Y·θ 상관 검정, 실제 경계 정답 대비 정확도, ±0.005° 도달 여부, 방법의 우열 판정."]])+
 '<p><b>방법별로 구분해야 할 지점:</b> E01은 초기 경계 검출과 마지막 연결 방식이 바뀌었고, E02/Q02는 원본 검출기 뒤의 선택 규칙이 추가됐습니다. E05는 흐림 폭 가정이 위치 식에 직접 들어갑니다. E06은 창과 A/B 추정이 변경됐으며 원논문 전문 대조가 완료되지 않았습니다. Q04는 원본 LSD 뒤에서 선분을 고르고 다시 통합하는 과정이 추가됐습니다.</p>'+
 '<p>따라서 현재 결과는 <b>“논문 이름 하나의 성능”보다 “그 이론·코드에 이번 영역과 설정을 결합한 구현의 출력”</b>으로 읽는 것이 정확합니다. 이 문서는 그 결합 과정을 공개해 다음 판단에 사용할 수 있게 정리한 것입니다.</p>')

source_rows=[
 ["E01 / S01",cite("S01"),"III.A 초기 검출, III.B Gaussian 피팅, III.C 곡선 연결을 대조. 직접 구현한 세부 위치 계산과 전후단 변경을 구분."],
 ["E02·Q02 / S02",cite("S02"),"IPOL 공식 설명과 배포된 저자 C 코드 1.0 대조. 서브픽셀 보정·연결·hysteresis 확인. 논문 전문 전체를 대조했다고 주장하지 않음."],
 ["E05 / S05",cite("S05"),"공개 전문의 모멘트·ramp-edge 식 (25),(31),(33),(41),(42) 대조. streak 부분은 사용하지 않음."],
 ["E06 / S06",cite("S06")+'<br>'+cite("S06M"),"직접 확인한 계산 근거는 저자 측 Method.md의 기본 2차 PAE 설명. 원논문 전문 미대조. 기본형의 일부를 자체 구현."],
 ["Q04 / S04",cite("S04")+'<br>'+link("https://www.ipol.im/pub/art/2012/gjmr-lsd/","논문·공식 코드 배포 페이지"),"공개 전문의 선분 영역/사각형 모델과 NFA, 공식 LSD 1.6 코드를 대조. ROI 차이에 대한 Fig.19·20도 참고."],
 ["공통 피팅 / H",cite("H"),"공식 fitLine/DIST_HUBER 문서의 가중 원리와 C=1.345. 현재 프로젝트는 자체 IRLS 구현이며 cv::fitLine 직접 호출이 아님."]
]
section("sources","부록 A. 학술 자료와 코드 근거",table(["적용 방법","자료","실제로 확인한 범위"],source_rows)+
 '<p>PAE 원문을 직접 확인하지 못한 범위는 그대로 명시했습니다. 출처의 식을 적용한 사실만으로 현재의 창 크기·ROI·임계값까지 학술적으로 검증됐다고 서술하지 않았습니다.</p>'+
 '<p>재현 근거 파일: '+link("evidence/configs/method_sources.json","기존 method_sources.json")+' · '+link("evidence/configs/methods.json","실제 설정값 전체")+' · '+link("evidence/measurements_56.json","56건 원본 측정 기록 발췌")+' · '+link("provenance.json","파일 출처·해시·그림 생성 근거")+'</p>')

input_table=table(["ID","입력 파일","구분","원본 경로"],[[x["reference_id"],x["file"],x["label"],'<span class="path">'+escape(x["path"])+"</span>"] for x in manifest["references"]])
section("evidence","부록 B. 입력과 저장 기록",input_table+
 '<p>입력 SHA-256은 기존 측정 명세와 각 기록의 값을 보존했습니다. 이번 설명 보고서 작성에서 원본 BMP의 내용이나 측정 설정을 변경하지 않았습니다.</p>'+
 '<p>기존 코드·설정·중간 기록을 이 보고서 폴더의 evidence에 복사했습니다. 보고서 그림의 계산 예시는 R01 상단 x=24에서 이미 저장되어 있던 값입니다. E01 곡선은 저장된 계수를 평가해 그렸고, E05/E06 패치는 저장된 밝기 행렬을 표시했습니다. 새로운 경계점이나 직선 계수를 추정한 것은 아닙니다.</p>'+
 '<p>'+link("build_report.py","보고서 생성 코드")+' · '+link("report.txt","동일 내용 텍스트판")+'</p>')

nav=[("start","한눈에 보기"),("targets","측정 영역"),("fit","공통 직선 피팅")]+[(m,m+" "+NAMES[m]) for m in METHODS]+[("read-results","결과 읽는 법"),("sources","학술 근거"),("evidence","입력·기록")]
css='''
:root{--ink:#162f45;--blue:#1565ad;--line:#dbe4ed;--bg:#f3f6fa;--muted:#526677}
*{box-sizing:border-box}html{scroll-padding-top:20px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 "Malgun Gothic","Segoe UI",sans-serif;word-break:keep-all;overflow-wrap:anywhere}
a{color:#1262a3;text-underline-offset:3px}header{background:#142f45;color:white;padding:45px max(28px,calc((100vw - 1390px)/2));border-bottom:5px solid #47b0c0}header .eyebrow{color:#92d6e3;font-size:13px;letter-spacing:2px}header h1{font-size:32px;line-height:1.4;margin:10px 0}header p{margin:5px 0;color:#d1dee8}
.layout{max-width:1450px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:28px;padding:28px}nav{position:sticky;top:18px;align-self:start;max-height:94vh;overflow:auto;font-size:13px}nav a{display:block;text-decoration:none;padding:8px 12px;border-bottom:1px solid #e1e7ee;border-left:3px solid transparent}nav a:hover{background:#e2edf7;border-left-color:#1686b6}nav strong{display:block;padding:5px 12px}
main{min-width:0}section{background:white;border:1px solid var(--line);border-radius:12px;padding:30px 34px;margin-bottom:24px;box-shadow:0 3px 14px #14364b05}h2{font-size:25px;line-height:1.5;margin:0 0 18px;border-bottom:2px solid #dbe8f2;padding-bottom:13px}h3{font-size:18px;margin:30px 0 12px}p{margin:14px 0}.lead{font-size:19px;font-weight:600;color:#245778}.small{font-size:13px;color:var(--muted)}
.note{border-left:4px solid #3394ba;background:#eef7fb;padding:17px 20px;margin:20px 0}.example{border-left:4px solid #e8a13e;background:#fff8e9;padding:17px 20px}.formula{background:#152f43;color:#f0f8ff;border-radius:8px;font-family:"Consolas","Malgun Gothic",monospace;padding:18px 22px;margin:15px 0;line-height:2;overflow:auto}
.tablewrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#e9f0f6;text-align:left;font-weight:700}td,th{border:1px solid #dce5ee;padding:11px 13px;vertical-align:top}td:first-child{font-weight:600;min-width:90px}tbody tr:nth-child(even){background:#fafcfe}.steps{padding-left:26px}.steps li{padding:4px 0 9px 6px}.steps li::marker{color:#1565ad;font-weight:bold}
figure{margin:20px 0}figure img{display:block;width:100%;height:auto;border:1px solid #dae2e9;border-radius:6px}figcaption{font-size:13px;color:#526677;margin:9px 0;line-height:1.7}details{border:1px solid #d6e2ec;border-radius:7px;padding:13px 17px;margin:18px 0}summary{cursor:pointer;font-weight:700;color:#235473}details[open] summary{margin-bottom:15px}.switches{display:flex;gap:7px;flex-wrap:wrap;margin-top:22px}.switches button{background:#eff3f7;color:#365671;border:1px solid #c8d8e4;border-radius:6px;padding:8px 12px;cursor:pointer;font:13px "Malgun Gothic",sans-serif}.switches button.active{background:#176b9c;color:white;border-color:#176b9c}.source-code{font-size:13px;background:#f3f6f9;padding:12px}.path{font-family:Consolas,"Malgun Gothic",monospace;font-size:12px;word-break:break-all}footer{max-width:1050px;margin:0 auto 36px;padding:12px 28px;color:#647789;font-size:12px}
@media(max-width:1000px){.layout{display:block;padding:15px}nav{position:static;max-height:none;margin-bottom:20px;display:flex;flex-wrap:wrap;gap:5px}nav strong{display:none}nav a{border:1px solid #d9e3ec;border-radius:5px;padding:6px 10px}section{padding:22px}header{padding:28px}header h1{font-size:25px}}
@media print{body{background:white;font-size:10pt}.layout{display:block;padding:0}nav,.switches,footer{display:none}header{padding:15px;color:#142f45;background:white;border-bottom:2px solid #142f45}header p,header .eyebrow{color:#526677}header h1{font-size:23pt}section{border:0;box-shadow:none;border-radius:0;padding:15px 0;break-before:page}section:first-child{break-before:auto}h2,h3{break-after:avoid}figure,table,.note,.example{break-inside:avoid}details{display:block}.formula{color:#142f45;background:#edf2f5}a{color:#142f45}figure img{max-height:180mm;object-fit:contain}td,th{padding:5px;font-size:9pt}}
'''
script='''document.querySelectorAll('.gallery').forEach(g=>{g.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{const path='figures/'+g.dataset.method+'_'+b.dataset.ref+'.png';g.querySelector('img').src=path;g.querySelector('img').alt=g.dataset.method+' '+b.textContent+' 저장된 상세 측정 그림';g.querySelector('figure a').href=path;g.querySelectorAll('button').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});}));});'''
body=''.join(parts)
html='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>6가지 직선 측정 방법 — 점 추출·직선 피팅·학술 근거</title><style>'+css+'</style></head><body><header><div class="eyebrow">MEASUREMENT METHOD REVIEW · 2026.10.30</div><h1>6가지 방법은 실제로 어떻게 직선을 측정했나</h1><p>E01 · E02 · E05 · E06 · Q02 · Q04</p><p>점과 선분의 선택 → 직선 추정 → 학술 근거와 자체 설정</p></header><div class="layout"><nav aria-label="보고서 목차"><strong>보고서 목차</strong>'+''.join(link('#'+k,t) for k,t in nav)+'</nav><main>'+body+'</main></div><footer>기존 코드·설정·저장 결과 설명 보고서. 원본 측정 재실행 및 통계 검정 없음.</footer><script>'+script+'</script></body></html>'
(OUT/"report.html").write_text(html,encoding="utf-8")

class TextReport(HTMLParser):
    def __init__(self):
        super().__init__();self.out=[];self.href=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag in ('h2','h3','p','li','tr','summary','figure','div'):self.out.append('\n')
        if tag=='li':self.out.append('• ')
        if tag=='br':self.out.append('\n')
        if tag in ('td','th'):self.out.append(' | ')
        if tag=='a':self.href=attrs.get('href')
        if tag=='img':self.out.append('[그림: '+attrs.get('src','')+']\n')
    def handle_endtag(self,tag):
        if tag in ('h2','h3','p','li','tr','summary','figure','div'):self.out.append('\n')
        if tag=='a' and self.href:self.out.append(' ('+self.href+')');self.href=None
    def handle_data(self,s):self.out.append(s)
p=TextReport();p.feed(body)
import re
text=re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',''.join(p.out)).strip()
(OUT/'report.txt').write_text('6가지 직선 측정 방법 설명 보고서\n2026-10-30\n\n'+text+'\n',encoding='utf-8')

provenance={"report_date":"2026-10-30","scope":"기존 6개 방법 설명; 재측정·재피팅·통계 검정 없음",
 "source_results":str((SRC/'results/line_measurements.json').relative_to(ROOT)),"source_results_sha256":sha(SRC/'results/line_measurements.json'),
 "methods":METHODS,"record_count":len(records),"input_manifest":manifest,"copied_evidence":copied,
 "new_figures_from_saved_values":fig_prov,"sources":{k:{"url":u,"title":t} for k,(u,t) in refs.items()},
 "author_code_audit":{"devernay.c":"bfcf9061b05cd7867999f38dfb2199f9b656ee5b695bd0c3197b18ff8001458f",
 "devernay.h":"880478fdfa7793d73adeab0a5513f2115748342af49e434ab49ba60f582d2510",
 "lsd.c":"f4b0c6194724e9d2cec01503fc2f4b7120f2e7d1fa570de6e67b4193a6b6723f",
 "lsd.h":"02edcbb6e7b19dfd7117ae1a13b9b5810c74e1cf69ea5492eafe7ccc65026eaa",
 "note":"현재 검토에서 확인한 파일과 보관된 공식 배포 ZIP의 동일성. 최신 온라인 ZIP 재다운로드 여부를 뜻하지 않음."}}
write_json(OUT/'provenance.json',provenance)
print(json.dumps({"report":str(OUT/'report.html'),"text":str(OUT/'report.txt'),"records":len(records),"copied_evidence":len(copied),"new_figures":len(fig_prov),"text_characters":len(text)},ensure_ascii=False))
