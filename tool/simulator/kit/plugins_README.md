# plugins — 추정기 DLL

```
plugins\
├─ <이름>\<이름>.dll      ← 플러그인 하나 = 폴더 하나
│  └─ assets\             (선택) 그 알고리즘이 쓰는 모델·기준 사진 등
└─ …
```

시뮬레이터 화면 윗줄 **[추정기]** 상자에 자동으로 나타난다. 콘솔은 `Simulator.exe --estimator <이름>`,
목록은 `Simulator.exe --list-plugins`. 기본값으로 쓰려면 `config\simulator.yml` 의 `estimator: { type: "plugin", plugin: "<이름>" }`.

## 읽을 수 있는 DLL 두 가지

| 경계 | 내보내는 함수 | 만드는 법 |
|---|---|---|
| 시뮬레이터 플러그인 | `sim_plugin_api_version`, `sim_plugin_name`, `sim_plugin_init`, `sim_plugin_estimate` (+ 선택 `sim_plugin_set_reference`) | `sdk\team\<이름>\` 에 소스를 넣고 `build_plugin.bat` (`sdk\sim_plugin.h`) |
| 팀 저장소 모듈 DLL | `semes_module_api_version`, `semes_module_initialize`, `semes_module_estimate` (`include\module_plugin.h`) | 이미 만든 DLL 을 그대로 복사. 자산 폴더 = `estimator.plugin_config` → 없으면 DLL 옆 `assets\` |

## 기준(티칭) 사진

- `sim_plugin_set_reference` 를 구현하면 시뮬레이터가 시험마다 기준 사진(center + 새 촬영 잡음)을 넘긴다
- 구현하지 않으면 알고리즘 자체의 기준 사진(예 `assets\center.bmp`)을 쓴다. 시뮬레이터 기준 사진이 실제 center 사진이므로 조건은 거의 같다
- 다만 [합성 검증]에서 2차 사진은 center1 을 기준으로 해야 하므로, 교체를 지원하지 않으면 2차 묶음 결과는 기준 차이(약 0.5 px)만큼 어긋날 수 있다

## 실패 처리

- `alignment()` 가 NaN 을 돌려주거나 예외를 던지면 '측정 실패' 로 기록되고 그 시험은 실패로 끝난다
- DLL 이 opencv_world3415.dll 외의 DLL 을 쓰면 같은 폴더(`plugins\<이름>\`)에 함께 둔다
