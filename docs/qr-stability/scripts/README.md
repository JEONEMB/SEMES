# 검증 재현

검증 대상 BMP는 저장소 `data/SSAFY Test Image Set`에 별도로 준비합니다. 보고서에 원본 BMP를 복제하지 않습니다. CSV와 그래프에는 파일명 라벨, 검출 결과, 영상 좌표만 포함됩니다.

## 집계와 그래프 재생성

Python 3 + numpy, pandas, matplotlib, Pillow를 설치한 환경에서 저장소 루트 기준:

```powershell
python docs/qr-stability/scripts/analyze_stability.py docs/qr-stability/data/final
```

기본 출력은 지정한 입력 폴더입니다. 제공된 PNG는 보고서 images 폴더에 보관합니다. Windows에서 맑은 고딕을 사용하며 다른 OS에서는 사용 가능한 한글 폰트를 설정합니다.

## 전수 검출 재실행

1. `list_images.py`로 BMP 목록을 생성합니다.
2. `evaluate.cpp`와 `src/alignment.cpp`, `src/qr_features.cpp`, `src/qrcodegen.cpp`를 프로젝트와 같은 C++17/OpenCV 환경에서 함께 컴파일합니다. `src/main.cpp`는 포함하지 않습니다. 헤더 경로는 `include`와 OpenCV include 디렉터리, 링크 라이브러리는 프로젝트와 동일합니다.
3. 수집기를 `evaluate.exe images.txt output_directory`로 실행합니다. 출력 폴더는 미리 생성합니다.
4. 수집된 `frames.csv`, `qr_status.csv`, `points.csv`를 분석 스크립트에 전달합니다.

```powershell
python docs/qr-stability/scripts/list_images.py "data/SSAFY Test Image Set" images.txt
.\evaluate.exe images.txt evaluation-output
python docs/qr-stability/scripts/analyze_stability.py evaluation-output
```

CSV의 경로는 원본 촬영 데이터를 식별하는 용도입니다. 경로가 다른 환경에서도 이미 수집된 CSV로 그래프 재생성은 가능합니다. 원시 입력/출력과 소스 해시는 manifest.json을 참고합니다.

## CPU 병렬 처리

수집기는 `cv::setNumThreads(1)`로 내부 경쟁을 제한합니다. 아래 실행기는 독립 프로세스 네 개로 입력 목록을 나누며, 합칠 때 각 프로세스의 로컬 frame 번호를 전체 목록의 번호로 복구합니다.

```powershell
python docs/qr-stability/scripts/parallel_evaluate.py evaluate.exe images.txt evaluation-output --workers 4
```

프로세스별 로그는 worker_* 폴더, 총 수행 시간은 parallel_run.json에 남습니다. GPU 없이 동작합니다. 프로세스 수는 시스템 메모리·CPU 상황에 맞춰 조절합니다.
