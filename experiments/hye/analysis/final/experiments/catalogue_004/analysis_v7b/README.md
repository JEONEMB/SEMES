# analysis_v7b 실험 출처

## 문제

작은 이동과 원점·시행 불일치

## 조사

zero_cal, zero_ms, consistency, bias, twostage 등

## 해결 방법

각 파일의 원점·시행 사용 조건을 독립 보존

## 유지한 부분

원본 결과와 시행별 추적 자료

## 바꾼 부분

원점 보정·다단계 판정·다중 촬영 비교

## 다른 실험과의 구분

단일 center 방식과 복수 절편 후보 방식을 합치지 않음

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-14T15:55:49.565+09:00. 관측된 최종 파일 수정: 2026-10-15T13:35:27.218+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### bias.py

- 원본: `analysis_v7b/bias.py`
- 생성 KST: 2026-10-14T16:16:33.300+09:00
- 수정 KST: 2026-10-14T16:29:18.137+09:00
- SHA256: `94f06d0bb2a46acfcbf102f82d625b3fabb406dfa0144f6d853273cd7c506c71`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### build_deck.js

- 원본: `analysis_v7b/build_deck.js`
- 생성 KST: 2026-10-14T17:06:52.582+09:00
- 수정 KST: 2026-10-14T17:07:56.238+09:00
- SHA256: `dbfca6e7031591dd53f8cb3393cafe28c05ddd2955739fa0519bcf1250447650`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T17:06:49+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\651881fc-8a33-4815-8366-ab4d211e211a.jsonl`의 1347행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### consistency.py

- 원본: `analysis_v7b/consistency.py`
- 생성 KST: 2026-10-14T16:06:23.093+09:00
- 수정 KST: 2026-10-14T16:29:18.141+09:00
- SHA256: `e25bc2f3e999b4fb99658d7c3df224b706da9b1492667cafe6c504a9b80aa146`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### exp.py

- 원본: `analysis_v7b/exp.py`
- 생성 KST: 2026-10-14T15:59:56.883+09:00
- 수정 KST: 2026-10-14T16:16:33.036+09:00
- SHA256: `eca6e1c018b29168b72aee084657dba281935dbe794be75c7e4b773a188b52ce`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### figs.py

- 원본: `analysis_v7b/figs.py`
- 생성 KST: 2026-10-14T16:01:36.106+09:00
- 수정 KST: 2026-10-14T16:29:18.146+09:00
- SHA256: `ebeb0b488e6b3f2e30075c49b960787d57c5f59528e5b18bfb7280a96bd77f45`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### fig_wrong.py

- 원본: `analysis_v7b/fig_wrong.py`
- 생성 KST: 2026-10-14T16:04:54.406+09:00
- 수정 KST: 2026-10-14T16:29:18.150+09:00
- SHA256: `9641a19c9683c6a066e2129097a73aed5eb1c7da7cf9c5df6cea68cc893fb6bb`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### final.py

- 원본: `analysis_v7b/final.py`
- 생성 KST: 2026-10-14T16:00:50.348+09:00
- 수정 KST: 2026-10-14T16:29:18.152+09:00
- SHA256: `78ba35fbc9ac09ef0da3d6640dca9fe17e7cbf13044ba37d2bee72498dfa3a9e`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### lib.py

- 원본: `analysis_v7b/lib.py`
- 생성 KST: 2026-10-14T15:59:56.743+09:00
- 수정 KST: 2026-10-14T16:29:18.133+09:00
- SHA256: `f715e9a8e375b9fba57a6fc6c82c40061c3f92ef02da38c319fa12d8c585efc1`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### multishot.py

- 원본: `analysis_v7b/multishot.py`
- 생성 KST: 2026-10-14T16:08:32.672+09:00
- 수정 KST: 2026-10-14T16:29:20.870+09:00
- SHA256: `9bcd5046abd7efe86bdaf4ceae310731345f87bdcb8ae18e66d92fbc8675869e`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### ppt_figs.py

- 원본: `analysis_v7b/ppt_figs.py`
- 생성 KST: 2026-10-14T17:03:14.776+09:00
- 수정 KST: 2026-10-14T17:04:09.232+09:00
- SHA256: `1aea80207892b82df4e3ddf05283670f3d761f7becb364c0e025fa073de8bebd`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T17:03:13+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\651881fc-8a33-4815-8366-ab4d211e211a.jsonl`의 1285행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### trial_tracks.py

- 원본: `analysis_v7b/trial_tracks.py`
- 생성 KST: 2026-10-15T13:26:30.297+09:00
- 수정 KST: 2026-10-15T13:26:30.297+09:00
- SHA256: `61fda63001fc6d4045c1e5bea4a6634925f5dc34f30ff0cee56a8668dab7b2a3`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-15T13:26:09+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\6c200343-efe4-4cc2-9d91-9cc8d85f63d9.jsonl`의 105행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### trial_tracks_ppt.py

- 원본: `analysis_v7b/trial_tracks_ppt.py`
- 생성 KST: 2026-10-15T13:34:33.263+09:00
- 수정 KST: 2026-10-15T13:35:21.225+09:00
- SHA256: `935f12369af3c79b7ac820384fc34efb94cff81d607519d2339ef41104fd99c0`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-15T13:29:43+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\6c200343-efe4-4cc2-9d91-9cc8d85f63d9.jsonl`의 163행.
- 2026-10-15T13:34:31+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\6c200343-efe4-4cc2-9d91-9cc8d85f63d9.jsonl`의 350행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### ts33.py

- 원본: `analysis_v7b/ts33.py`
- 생성 KST: 2026-10-14T17:35:42.527+09:00
- 수정 KST: 2026-10-14T17:35:42.527+09:00
- SHA256: `f0f97960ea8db2466901d81a5e31002fe3ba4f23d9e7620f0a43eda326b970e6`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T17:35:42+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\8e4b3608-063f-4e2e-8384-d5e2a57b0a42.jsonl`의 123행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### ts33b.py

- 원본: `analysis_v7b/ts33b.py`
- 생성 KST: 2026-10-14T17:36:34.862+09:00
- 수정 KST: 2026-10-14T17:36:34.910+09:00
- SHA256: `272a1a3e6f0b204bdc6d1220645e4c92fa925f9cddbc1382f059d69bce78234d`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### twostage.py

- 원본: `analysis_v7b/twostage.py`
- 생성 KST: 2026-10-14T16:17:27.167+09:00
- 수정 KST: 2026-10-14T16:29:18.159+09:00
- SHA256: `6a63e1f88d375cd88c735f49eebae75fa620c245a6e555db43dec382bf7002a8`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### zero_cal.py

- 원본: `analysis_v7b/zero_cal.py`
- 생성 KST: 2026-10-14T16:36:20.086+09:00
- 수정 KST: 2026-10-14T16:36:20.086+09:00
- SHA256: `2be17e60cb5fad6b3d9b77d901622120f8f32bd0f6bce3d6cc8f3e922c997743`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T16:36:18+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\651881fc-8a33-4815-8366-ab4d211e211a.jsonl`의 1138행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### zero_ms.py

- 원본: `analysis_v7b/zero_ms.py`
- 생성 KST: 2026-10-14T16:38:52.361+09:00
- 수정 KST: 2026-10-14T16:38:52.361+09:00
- SHA256: `d2805e00ba4d91ff492f47337a91135bf1d6522451dc6bca64df4a1d1e5ad530`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T16:38:10+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\651881fc-8a33-4815-8366-ab4d211e211a.jsonl`의 1155행.
- 2026-10-14T16:38:50+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\651881fc-8a33-4815-8366-ab4d211e211a.jsonl`의 1174행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

## 보존된 결과와 문서

- `analysis_v7b/consistency_pred.csv` (2026-10-14T16:06:27.909+09:00)
- `analysis_v7b/out/00_v7b결과리포트.md` (2026-10-14T16:45:40.123+09:00)
- `analysis_v7b/out/00_v7_계획.md` (2026-10-14T15:55:49.565+09:00)
- `analysis_v7b/out/54_v7_최종평가.csv` (2026-10-14T16:01:02.533+09:00)
- `analysis_v7b/out/55_v7_사진별예측.csv` (2026-10-14T16:01:02.556+09:00)
- `analysis_v7b/out/60_v7_증폭_일관성.csv` (2026-10-14T16:06:27.903+09:00)
- `analysis_v7b/out/61_v7_여러장_시뮬레이션.csv` (2026-10-14T16:15:18.595+09:00)
- `analysis_v7b/out/62_v7_0근처편향.csv` (2026-10-14T16:16:58.344+09:00)
- `analysis_v7b/out/63_v7_2단계_사진별예측.csv` (2026-10-14T16:17:50.404+09:00)
- `analysis_v7b/out/63_v7_2단계_평가.csv` (2026-10-14T16:17:50.379+09:00)
- `analysis_v7b/out/64_v7_여러장_시뮬레이션.csv` (2026-10-14T16:32:41.925+09:00)
- `analysis_v7b/out/65_v7_0점맞추기.csv` (2026-10-14T16:37:03.844+09:00)
- `analysis_v7b/out/66_v7_0점맞추기_여러장.csv` (2026-10-14T16:44:49.509+09:00)
- `analysis_v7b/out/70_2단계_33px_평가.csv` (2026-10-14T17:36:17.391+09:00)
- `analysis_v7b/out/71_2단계_33px_반복40.csv` (2026-10-14T17:36:17.445+09:00)
- `analysis_v7b/out/72_2단계_LOC민감도.csv` (2026-10-14T17:38:42.393+09:00)
- `analysis_v7b/out/ppt_fig/sizes.json` (2026-10-14T17:04:54.479+09:00)
- `analysis_v7b/package-lock.json` (2026-10-14T17:05:04.526+09:00)
- `analysis_v7b/package.json` (2026-10-14T17:05:04.514+09:00)
- `analysis_v7b/S1.csv` (2026-10-14T15:58:19.296+09:00)
- `analysis_v7b/S3.csv` (2026-10-14T15:58:31.822+09:00)
- `analysis_v7b/S4.csv` (2026-10-14T15:59:04.949+09:00)
- `analysis_v7b/S4b.csv` (2026-10-14T16:00:07.579+09:00)
- `analysis_v7b/S4c.csv` (2026-10-14T15:59:34.423+09:00)
- `analysis_v7b/S5.csv` (2026-10-14T16:00:30.350+09:00)
