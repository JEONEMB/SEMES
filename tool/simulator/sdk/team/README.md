# sdk\team — 팀원 알고리즘 소스를 넣는 곳

폴더 하나 = 플러그인 하나. 폴더 이름이 DLL 이름이 된다 (영문·숫자·밑줄 권장).

```
sdk\team\hye\
├─ include\alignment.hpp        팀 공통 헤더 : Alignment_t alignment(cv::Mat &src);
├─ include\...                  그 밖의 헤더 (하위 폴더 그대로)
├─ src\*.cpp                    구현 — 폴더 안의 모든 .cpp 를 컴파일
├─ third_party\*.cpp            (있으면 같이 컴파일)
├─ assets\                      (선택) 모델·기준 사진 → plugin.cmake 로 DLL 옆에 복사
├─ sim_hooks.cpp                (선택) 초기화·기준 사진 교체 연결
└─ plugin.cmake                 (선택) 추가 라이브러리·복사 규칙
```

- 자동 제외 : `main.cpp`, `build\ results\ test(s)\ tool(s)\ examples\` 폴더
- 빌드 : 시뮬레이터 루트의 `build_plugin.bat` → `plugins\<이름>\<이름>.dll`
- 래퍼 `sdk\team_alignment_plugin.cpp` 가 `alignment()` 를 부르므로 **팀원 코드는 고치지 않는다**

## sim_hooks.cpp 가 필요한 경우

| 팀원 코드 | 할 일 |
|---|---|
| `alignment()` 만 있고 첫 호출 때 알아서 초기화 | 필요 없음 |
| `initialize_joint(assets)` 같은 초기화 함수가 따로 있음 | `examples\joint_style\sim_hooks.cpp` 복사 → 함수 이름 맞춤 |
| 기준 사진 교체 함수가 있음 | `sim_team_set_reference` 에서 부름 (예 `examples\ksh\sim_hooks.cpp`) |
| Windows 라이브러리(bcrypt 등)를 씀 | `examples\joint_style\plugin.cmake` 복사 |

## 확인한 예 (2026-11-03)

- `examples\ksh` : 내장 추정기와 같은 코드 → 플러그인으로 돌린 결과가 내장과 같음 (같은 seed 다회 시험 일치)
- 이혜원 `00000_final_Y_T_joint` (팀 저장소 `feature/hye/experiments-pa-pt`) : `include\`, `src\`, `third_party\`, `assets\` + `joint_style` 연결 파일로 빌드·실행 확인
