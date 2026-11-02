# third_party 출처·버전·라이선스·변경 내역

다운로드: 2026-10-30. 원본 압축/파일은 `_downloads/`, 수정 전 사본은 `_pristine/`에 그대로 둡니다. 원본 SHA-256은 `logs/` 환경 기록과 아래 표에 있습니다.

| 폴더 | 사용 방법 | 출처 | 버전/commit | 라이선스 | 이번 변경 |
|---|---|---|---|---|---|
| `lsd_1.6/` | E04·Q04 | https://www.ipol.im/pub/art/2012/gjmr-lsd/lsd_1.6.zip | lsd 1.6 | AGPL-3.0-or-later | 없음 (lsd.c, lsd.h 원본 그대로) |
| `devernay_1.0/` | E02·Q02 | https://www.ipol.im/pub/art/2017/216/devernay_1.0.zip | 1.0 (2017-10-10) | AGPL-3.0-or-later | 없음 (devernay.c, devernay.h 원본 그대로) |
| `ED_Lib/` | E07·Q07 | https://github.com/CihanTopal/ED_Lib | 69b8d081bd6d28192d816ec0ed02aff9186d73c1 | MIT | `EDLines.h`에 `getLineSegments()` 접근자 추가 (`ED_Lib/CHANGES.diff`) |
| `ELSED/` | E09·Q09 | https://github.com/iago-suarez/ELSED | 1878213b2f5f06a9261d8b1838f53d48e5fd128d | Apache-2.0 | 반환 선분별 픽셀 보관 `validSegmentPixels`와 접근자 추가 (`ELSED/CHANGES.diff`). Python 바인딩·pybind11 미사용 |
| `fld_port/` | E08·Q08 | opencv_contrib 3.4 `modules/ximgproc/src/fast_line_detector.cpp` | 파일 최종 변경 8a33d68eac40487e7857c47aecd85222d642969b (branch head ae9a95ecdd8b4014a45b38c5576adf73c5d96f35) | 3-clause BSD | 아래 참조 (`fld_port/CHANGES.diff`) |
| `nlohmann_json/` | 설정·결과 JSON 입출력 | https://github.com/nlohmann/json/releases/tag/v3.11.3 | v3.11.3 | MIT | 없음 |

## FLD 이식 변경 (빌드 연결 vs 알고리즘)

- 빌드 연결 부분만 변경: `precomp.hpp` → 공개 `opencv2/imgproc.hpp`, `cv::ximgproc` 네임스페이스 → `lsj_fld`, 추상 클래스 `FastLineDetector` 상속·`CV_OVERRIDE`·`CV_EXPORTS createFastLineDetector`·`CV_INSTRUMENT_REGION()` 제거.
- 알고리즘 계산(`lineDetection`, `extractSegments`, `getPointChain`, `additionalOperationsOnSegment`, `mergeSegments`)은 그대로입니다.
- 추가: `extractSegments`가 각 선분의 `fitLine`에 쓴 체인 화소(`l_points`)를 함께 반환하는 출력 인자, 이를 노출하는 `detectWithSupport()`. `do_merge=false`에서만 선분·지지 목록이 정렬됩니다.

## 사용하지 않은 코드

- PAE: https://github.com/juanse77/EdgeLocator (commit 537db170ce64a3cc7e27c26381cebcb507c1f1ec). 저장소에 라이선스 표시가 없어 코드를 가져오지 않았습니다. 방법 설명 문서(Method.md)만 읽고, 식은 직접 다시 유도해 `cpp/src/methods_points.cpp::method_pae`에 구현했습니다. 압축 원본은 `_downloads/`에 참고용으로만 있습니다.
- OpenCV 내장 `createLineSegmentDetector`: 사용하지 않았습니다(3.4.15 내장 구현 동작 가정 금지 요구).

## AGPL 주의

LSD와 Devernay는 AGPL-3.0-or-later입니다. 두 코드를 링크한 `line_methods.exe`를 배포하거나 네트워크 서비스로 제공하면 AGPL 의무가 생깁니다. 이번 사용은 로컬 분석 실행에 한정했습니다. 팀 저장소·제어기에 넣으려면 별도 라이선스 검토가 필요합니다.
