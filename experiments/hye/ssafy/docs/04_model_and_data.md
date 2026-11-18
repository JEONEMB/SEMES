# 모델과 데이터

## 활성 모델

모델 식별자는 ssafy_wide_ncc_combinations_L2_1959_20261116이다. data/model/model_info.json에 출처와 정책을 기록했다.

| 파일 | 역할 |
|---|---|
| center.bmp | 기준 장비 사진 |
| reference.json | QR 3개, 나사 4개의 위치와 학습 범위 |
| coefficients/screws_조합.tsv | 16개 검출 조합별 Y, T 선형식 |
| checks/screws_조합.tsv | 나사 2개 이상인 11개 조합의 일치 점검식 |
| model_manifest.tsv | 활성 파일 29개의 SHA-256 |
| fit_report.txt | 학습 사진 재대입 결과 |
| model_info.json | 모델 설명과 출처 |

초기화는 model_manifest.tsv와 실제 기준 자료, 계수, 점검식의 해시를 비교한다. 파일 일부만 다른 버전으로 바꾸면 초기화가 실패한다. 해시 목록 자체의 배포 신뢰성은 별도 관리 대상이다.

## 학습 기준

SSAFY의 2026-11-12 사진 1959장과 기존 L2 보정 라벨을 사용했다. 2026-11-13 역순 사진은 이번 계수 학습에 사용하지 않았다. 새 QR 및 나사 측정기로 1959장 전부를 측정한 뒤 6회 Huber 반복과 ridge 계수 1e-7을 사용하는 기존 조합 학습 코드로 계수를 만들었다.

기존 L2 라벨은 영상 관측과 장비 명령을 사용해 보정한 라벨이다. 실제 판 자세를 별도 계측한 독립 정답은 아니다. SEMES의 모델 계수를 복사하지 않았다.

Y는 mm, T는 도 단위다. 부호는 SSAFY 장비 기준이며 SEMES와 같은 방향이라고 가정하지 않는다. 학습 범위는 Y -2.02214~2.04031 mm, T -1.02257~0.98128도다. 이 범위는 경고 기준이며 하드 제한이나 출력 잘라내기 기준이 아니다.

## 모델 재생성

```powershell
./build/Release/train_combinations.exe data/model training.tsv data/results/new_model
```

training.tsv는 id, path, y, t 머리줄을 가진 탭 구분 표다. 현재 도구는 원래 학습 자료 1959장 재현용이며 행 수가 다르면 중단한다. 모든 사진에서 QR과 나사 4개를 확보해야 하며 실패 사진을 조용히 제외하지 않는다.

새 모델에는 기준 사진과 좌표를 유지하고 16개 계산식, 11개 점검식, SHA-256 목록을 생성한다. 기존 출력 폴더를 덮어쓰지 않는다. 새 모델의 정확도, 실패 목록, 조합별 동작을 확인한 후 배포한다.

## 함께 배포할 것

운영에는 alignment 라이브러리, 공개 헤더, 모델 폴더 전체, 일치하는 OpenCV 런타임이 필요하다. 촬영 원본, data/results의 평가 원자료, build 산출물은 Git에 포함하지 않는다. legacy와 data/evidence는 과거 재현 자료이며 현재 운영 모델이 아니다.
