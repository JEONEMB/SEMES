# 영점 맞춤 시뮬레이터 키트

로봇이 **사진 → 추정(Y, Θ) → 보정 이동** 을 반복하며 20회 안에 영점을 맞추는지,
하드웨어 없이 실제 사진으로 만든 가상 장면에서 시험한다. 내 알고리즘을 DLL 로 넣어 같은 조건에서 비교할 수 있다.

## 1. 바로 실행

1. 압축을 아무 폴더에 푼다 (경로에 한글이 있어도 됨)
2. `SimulatorMFC.exe` 더블 클릭 → [1 단계] 또는 [자동 실행]
   - 콘솔 : `Simulator.exe --run 1.2 -0.6`, `Simulator.exe --monte 50`
3. 결과는 `results\` (화면 [결과 폴더])

가상 장면(`world\`)이 들어 있어 **사진 데이터 없이도** 시험이 된다.
Visual Studio 가 없는 PC 에서도 실행된다 (VC++ 실행 파일 포함).

## 2. 사진 데이터 넣기 (합성 검증 · 장면 다시 만들기 때만)

팀 공통 데이터를 그대로 `data\img\` 에 복사한다 → `data\img\README.md` 참고

```
data\img\1st\center.bmp, Y\, Theta\
data\img\2nd\기준 위치\, Y+a T+b\ …
data\img\3rd\…
```

## 3. 내 알고리즘 넣기 (플러그인)

| 방법 | 언제 | 할 일 |
|---|---|---|
| A. 소스 넣기 | 팀 공통 `Alignment_t alignment(cv::Mat &src)` 가 있을 때 | `sdk\team\<이름>\` 에 `include\`, `src\` 를 복사 → `build_plugin.bat` |
| B. 모듈 DLL | 이미 `semes_module_*` 함수를 내보내는 DLL 이 있을 때 | `plugins\<이름>\<이름>.dll` + `assets\` 복사 |

- 초기화 함수(예 `initialize_joint`)가 따로 있으면 `sdk\examples\joint_style\sim_hooks.cpp` 를 `sdk\team\<이름>\` 에 복사해 이름만 맞춘다
- 만든 DLL 은 화면 윗줄 **[추정기]** 상자에서 고른다 (콘솔 : `--estimator <이름>`)
- 자세한 설명 : `sdk\team\README.md`, `plugins\README.md`

> 먼저 **[합성 검증]** 으로 내 알고리즘이 실제 사진과 합성 사진에서 같은 값을 내는지 확인할 것.
> 합성 사진이 내 알고리즘이 보는 특징을 잘 재현하지 못하면 시뮬레이터 결과를 믿을 수 없다.

## 4. 설정 (`config\simulator.yml`)

| 묶음 | 주요 값 |
|---|---|
| `robot` | 반복 정밀도 σ Y 7.53 µm, Θ 0.00198° (실측), 배율 오차 0.2 % (가정), 한 번 최대 Y 2 mm / Θ 1° (장비 제원) |
| `control` | 허용 Y ±45 µm, Θ ±0.005° (과제의 절반), 이동 한도 20회, 멈춤 = 허용 × 0.5 |
| `trial` | 시작 범위 Y ±2 mm, Θ ±1°, 다회 시험 횟수, seed |
| `estimator` | `type: ksh` 또는 `plugin`, `plugin: <이름>` |
| `display` | `ui_font_pt` 화면 글자 크기, `plot_font_scale` 그림 글자 배율 |

경로는 모두 이 폴더 기준. 시뮬레이터는 실행할 때 `config\simulator.yml` 이 있는 폴더를 스스로 찾는다.

## 5. 문의 · 문서

- 개선 내역과 검증 결과 : `docs\`
- 만든 사람 : 김성현 (알고리즘 ksh, 시뮬레이터)
