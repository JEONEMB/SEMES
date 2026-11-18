# analysis_v8 실험 출처

## 문제

특징점 측정 잡음의 원인

## 조사

강체 검사, 흐림·조명·위상·합성 잡음·CRLB·원본 쌍 차이

## 해결 방법

r1~r13의 원인별 진단

## 유지한 부분

원본 영상과 기존 잡음 진단 결과

## 바꾼 부분

합성 영상에 넣는 원인과 비교 통계

## 다른 실험과의 구분

실제 자세 오차와 영상 측정 잡음의 설명을 구분

## 시간과 작성 흔적

관측된 최초 파일 생성: 2026-10-14T16:54:47.008+09:00. 관측된 최종 파일 수정: 2026-10-14T17:16:18.734+09:00. 파일 복사·재생성·덮어쓰기로 실제 작업 시간과 다를 수 있다.

### build_ppt_v8.js

- 원본: `analysis_v8/build_ppt_v8.js`
- 생성 KST: 2026-10-14T17:16:18.734+09:00
- 수정 KST: 2026-10-14T17:16:18.734+09:00
- SHA256: `75df645b80bc3f5dd5b1765dcf9e7f10ecef872750996b09e0ace25aad172502`
- 관측된 작업자: Claude
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

- 2026-10-14T17:14:39+09:00 Claude Write 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\6f5a3e93-8ea9-4341-b58f-87dfc6a97d4f.jsonl`의 525행.
- 2026-10-14T17:16:13+09:00 Claude Edit 호출. 로그 `C:/Users/SSAFY/.claude/projects/C--Users-SSAFY-Desktop-hwlee-projects-2026-10-SEMES\6f5a3e93-8ea9-4341-b58f-87dfc6a97d4f.jsonl`의 595행.

위 기록은 쓰기 도구 호출의 흔적이다. 현재 파일의 모든 줄을 해당 작업자가 썼거나 호출이 성공했다고 단정하지 않는다.

### r10_zoom.py

- 원본: `analysis_v8/r10_zoom.py`
- 생성 KST: 2026-10-14T17:02:01.370+09:00
- 수정 KST: 2026-10-14T17:02:01.475+09:00
- SHA256: `467bb64b0b2be972486665bd2eb7878ad28a9f4348819d9904005ff8189b66cf`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r11_noise_model.py

- 원본: `analysis_v8/r11_noise_model.py`
- 생성 KST: 2026-10-14T17:02:39.542+09:00
- 수정 KST: 2026-10-14T17:02:39.643+09:00
- SHA256: `5ad3b310275df50444c5a0dfe3350189dfcdaf7e6396b101803d44a0929fbb54`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r12_synth2.py

- 원본: `analysis_v8/r12_synth2.py`
- 생성 KST: 2026-10-14T17:03:31.865+09:00
- 수정 KST: 2026-10-14T17:03:31.952+09:00
- SHA256: `00f4c5f919976701cc45e20d39ae7d2a6167557edbf0637f1f5d21b001d68e74`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r13_crlb.py

- 원본: `analysis_v8/r13_crlb.py`
- 생성 KST: 2026-10-14T17:04:44.045+09:00
- 수정 KST: 2026-10-14T17:04:44.165+09:00
- SHA256: `27274578db041adfb3a58ac59630f8aa2ac9d1b8214df5a4d719625b9b24673d`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r1_rigid.py

- 원본: `analysis_v8/r1_rigid.py`
- 생성 KST: 2026-10-14T16:54:47.008+09:00
- 수정 KST: 2026-10-14T16:54:47.091+09:00
- SHA256: `5aef6a313262744c7e336c1bf4f8f576fa37bb3773a32612cc57ecb2b4721c72`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r2_blur.py

- 원본: `analysis_v8/r2_blur.py`
- 생성 KST: 2026-10-14T16:55:15.043+09:00
- 수정 KST: 2026-10-14T16:55:15.148+09:00
- SHA256: `9d256c13d4220f9a1c527a8b4dff6131df2773489ad0b05d3a5bddc8d20000b3`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r3_synth.py

- 원본: `analysis_v8/r3_synth.py`
- 생성 KST: 2026-10-14T16:56:29.934+09:00
- 수정 KST: 2026-10-14T16:56:30.037+09:00
- SHA256: `87a330e663dffa46a1b64067f579670552592e310d3cdf9bd16491b196270863`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r4_struct.py

- 원본: `analysis_v8/r4_struct.py`
- 생성 KST: 2026-10-14T16:58:16.014+09:00
- 수정 KST: 2026-10-14T16:58:16.107+09:00
- SHA256: `68d60a79614d2ce7dbeeb458c07559d181d67edeeb40e02bdd297141b40120d8`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r5_phase.py

- 원본: `analysis_v8/r5_phase.py`
- 생성 KST: 2026-10-14T16:58:51.117+09:00
- 수정 KST: 2026-10-14T16:58:51.223+09:00
- SHA256: `a6dc8124252c95c353fb0e31296d28b258104766021a3b028cf0a123c5185a98`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r6_pairdiff.py

- 원본: `analysis_v8/r6_pairdiff.py`
- 생성 KST: 2026-10-14T16:59:23.852+09:00
- 수정 KST: 2026-10-14T16:59:23.951+09:00
- SHA256: `90f4a8062d06e5f0f1e4997065a2d437b34f3a09d39c23ead95f6d0db3d95ad6`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r7_viewdep.py

- 원본: `analysis_v8/r7_viewdep.py`
- 생성 KST: 2026-10-14T17:00:06.811+09:00
- 수정 KST: 2026-10-14T17:00:06.943+09:00
- SHA256: `e73b22e6a2e4c9a1054e6f318b95042c800a973ea45e7f7f8e973c27bb496d79`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r8_pairdiff2.py

- 원본: `analysis_v8/r8_pairdiff2.py`
- 생성 KST: 2026-10-14T17:00:51.968+09:00
- 수정 KST: 2026-10-14T17:00:52.072+09:00
- SHA256: `c0b421ff95cf903e0131d2af94998828db18d4ea31255289faebaf44f0dcfcc7`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

### r9_isp.py

- 원본: `analysis_v8/r9_isp.py`
- 생성 KST: 2026-10-14T17:01:36.277+09:00
- 수정 KST: 2026-10-14T17:01:36.386+09:00
- SHA256: `805e1bed500099c8c1617e7b107709db4c0b0c11d303cd8f1553e9ccf500cc8b`
- 관측된 작업자: unknown
- 이식 상태: `pending_individual_cpp_port`

원본 설명과 시작 부분은 `source_evidence.json`에 보존했다.

## 보존된 결과와 문서

- `analysis_v8/r12_synth2.csv` (2026-10-14T17:04:10.322+09:00)
- `analysis_v8/r1_rigid.csv` (2026-10-14T16:54:52.838+09:00)
- `analysis_v8/r2_blur.csv` (2026-10-14T16:55:35.319+09:00)
- `analysis_v8/r3_synth.csv` (2026-10-14T16:57:23.632+09:00)
- `analysis_v8/r6_pairdiff.csv` (2026-10-14T16:59:29.661+09:00)
- `analysis_v8/r7_viewdep.csv` (2026-10-14T17:00:30.185+09:00)
- `analysis_v8/r8_pairdiff2.csv` (2026-10-14T17:00:56.826+09:00)
