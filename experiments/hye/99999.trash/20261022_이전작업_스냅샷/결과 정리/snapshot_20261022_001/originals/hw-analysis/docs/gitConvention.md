# Git 컨벤션

## 브랜치 전략

- `main`: 검증된 안정 버전을 관리합니다.
- `develop`: 기능을 통합하고 검증하는 개발 버전을 관리합니다.
- 작업 브랜치는 `develop`에서 생성하고, 작업 완료 후 PR로 `develop`에 병합합니다.
- 통합 검증이 끝난 `develop`은 PR로 `main`에 병합합니다.

```text
feature/* -> develop -> main
```

`fix`, `refactor`, `test`, `docs`, `chore` 작업 브랜치도 동일한 흐름을 따릅니다. `release`, `hotfix` 브랜치는 미리 만들지 않고, 별도 배포 관리나 긴급 수정 절차가 필요할 때 팀 합의로 확장합니다.

## 브랜치 네이밍

기본 형식은 다음과 같습니다. 작업 설명은 짧은 영문 소문자로 작성하고 단어는 하이픈(`-`)으로 구분합니다.

```text
<type>/<short-description>
```

| 타입 | 용도 | 예시 |
| --- | --- | --- |
| `feature` | 새로운 기능 | `feature/qr-detection` |
| `fix` | 버그 수정 | `fix/opencv-build` |
| `refactor` | 기능 변경 없는 코드 구조 개선 | `refactor/image-loader` |
| `test` | 테스트 추가 및 수정 | `test/jacobian` |
| `docs` | 문서 작성 및 수정 | `docs/setup-guide` |
| `chore` | 기타 설정 및 정리 | `chore/gitignore` |

## 커밋 메시지

```text
<type>: <description>
```

설명은 변경 내용을 알 수 있도록 한국어로 간결하게 작성합니다. 브랜치의 기능 타입은 `feature`, 커밋의 기능 타입은 `feat`를 사용합니다.

| 타입 | 의미 | 예시 |
| --- | --- | --- |
| `feat` | 새로운 기능 | `feat: QR 코드 검출 기능 추가` |
| `fix` | 버그 수정 | `fix: QR 모서리 좌표 계산 오류 수정` |
| `refactor` | 기능 변경 없는 코드 구조 개선 | `refactor: 이미지 로딩 로직 분리` |
| `test` | 테스트 추가 및 수정 | `test: QR 특징점 추출 테스트 추가` |
| `docs` | 문서 작성 및 수정 | `docs: Windows 개발환경 설정 방법 추가` |
| `build` | CMake, 라이브러리, 빌드 설정 변경 | `build: OpenCV 링크 설정 추가` |
| `chore` | 기타 설정 및 정리 | `chore: gitignore에 이미지 데이터셋 추가` |

## 커밋 단위

**하나의 커밋에는 하나의 의미 있는 변경만 포함합니다.** 기능 구현, 빌드 설정, 테스트, 문서 변경은 의미에 따라 나눕니다.

```text
build: OpenCV 링크 설정 추가
feat: QR 코드 검출 기능 추가
test: QR 코드 검출 테스트 추가
docs: QR 검출 방법 문서화
```

## PR 규칙

- `main`, `develop`에 직접 push하지 않고 PR을 통해 병합합니다.
- PR 제목은 커밋 메시지와 동일한 `<type>: <description>` 형식을 사용합니다.
- PR 본문에는 변경 사항과 테스트 방법·결과를 간단히 기록합니다.
- 테스트를 수행하지 못했다면 그 이유를 기록합니다.

PR 제목 예시: `feat: QR 특징점 추출 기능 추가`

PR 본문 예시:

```markdown
## 변경 사항
- QRDetector 및 QR 모서리 좌표 추출 기능 추가
- cornerSubPix 기반 좌표 보정 추가

## 테스트
- Y=0.00 이미지 검출 확인
- Y=±2.00 데이터 검출 확인
```

