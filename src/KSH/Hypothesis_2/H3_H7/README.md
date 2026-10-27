# Hypothesis_3 / H3_H7 — QR 만으로 Y·Θ 분리 (H3 물리 제약, H6 서브픽셀 치우침, H7 세션 보정)

VS 프로젝트 `QR_Image` 에서 실행하던 코드를 정리해 옮긴 폴더입니다 (2026-10-27 정리, 이후 QR_Image 는 운동 모델 데모로 교체).

## 결론 요약

| 가설 | 내용 | 결과 |
|---|---|---|
| H3 | 카메라·플레이트 물리 모델이 허용하는 움직임만으로 (Y, Θ) 추정 | **기각** — 1차 Θ 스윕 0.0179° (41/100), 기존 Gabor 와 같은 수준. 완전 물리 모델은 매개변수가 서로 얽혀 물리값이 나오지 않음 |
| H6 | 대응점 서브픽셀 위치 치우침(pixel-locking) 보정 | **기각** — 설명력 9~14 %, 보정 후 개선 없음 |
| H7 | 같은 날 기준 + 기지 자세 몇 장으로 오프셋·배율 재보정 | **보조 기능으로 채택** — 날짜 간 차이 제거 (배경 시차 0.0276° → 0.0065°), QR 단독은 보정 후에도 15/33 |

상세 수치·분석 : [`results/H3_H7_결과.md`](results/H3_H7_결과.md)

## 폴더

| 위치 | 내용 |
|---|---|
| `code/` | C++ (OpenCV 3.4.15) — `main.cpp` 와 dataset, qr_track(LK 대응점), field_model(H3 선형화), phys_model(H3 완전 물리), bias_h6, session_h7, analysis, report_plot |
| `results/` | C++ 실행 결과 : 요약 txt·png, 세트별 CSV, 모델 yml, 결과 보고서 |
| `python_prototype/` | 처음 파이썬 시제품 코드와 결과 (C++ 로 옮기기 전) |
| `project_backup/` | 당시 `QR_Image.vcxproj(.filters)` — `.h3_h7` 은 이 코드를 빌드하던 설정, `.before_h3cpp` 는 그 직전 설정 |

## 다시 실행하려면

1. `code/` 의 파일을 VS 프로젝트의 `h3_h7\code\` 에 두고, `project_backup/QR_Image.vcxproj.h3_h7` 을 `QR_Image.vcxproj` 로 사용
2. 프로젝트 폴더에 `img/` 와, 비교용 `legacy/center_check/results/parallax_set*.csv`(→ `Hypothesis_2/Center_check/results/center/`), `legacy/h2_ecc/results/*/h2_set*_results.csv`(→ `Hypothesis_2/ECC_1st/results/`) 를 배치
3. 실행 : `QR_Image.exe` (전체) / `QR_Image.exe img --no-phys` / `--retrack`
