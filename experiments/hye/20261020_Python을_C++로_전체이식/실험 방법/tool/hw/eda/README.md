# final의 C++ EDA 실행

`final` 디렉터리에서 실행한다. 원본 데이터는 `../hw-analysis/data/hw/bmp`에서 읽고, 새 결과는 매번 `final/results/eda_<UTC 실행시각>/<묶음>/`에 저장한다. 기존 결과를 덮어쓰지 않는다.

```powershell
.\build\tool\hw\eda\Release\hw_eda.exe 20261014
# 위 실행이 출력한 실제 summary.yml 경로를 지정한다.
.\build\tool\hw\eda\Release\hw_eda.exe 20261020 --ref 20261014 --ref-summary "results/eda_<위 실행의 UTC 시각>/20261014/summary.yml"
```

`--ref-summary`는 기준 수치를 읽을 파일을 명시한다. 상대경로는 실행 작업 디렉터리를 기준으로 해석하고 절대경로도 허용한다. `--ref`는 표시할 기준 묶음 이름이며, 생략하면 summary 파일의 부모 디렉터리 이름을 사용한다. 옵션 값 누락·기준 파일 부재는 사진 측정 전에 오류로 종료한다.

호환성을 위해 `--ref 20261014`만 지정하면 이전 위치인 `../hw-analysis/docs/validation-runs/hw/eda/20261014/summary.yml`을 읽는다. 새 실행 결과를 기준으로 비교할 때는 반드시 `--ref-summary`를 지정한다. 해석 메모는 여전히 `../hw-analysis/docs/hw/eda/sets`에서 읽는다. 따라서 `hw-analysis` 전체를 옮겨도 되는 상태는 아니다.

[ORIGINAL_README.md](ORIGINAL_README.md)는 이전 프로젝트의 설명을 보존한 문서다. 현재 출력 경로와 새 기준 결과 연결 방법은 이 문서를 따른다.
