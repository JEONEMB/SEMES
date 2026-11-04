clean_20261014

C++ 원본 정리 도구: tool/01_prepare_clean.cpp
입력: data/raw/zip_20261014.zip
BMP 281장 / 동봉 TXT 0개

BMP/TXT 바이트와 촬영별 폴더명을 그대로 보존했습니다.
공통 최상위 포장 폴더 한 개만 제거했습니다. 사진 가공/제외/중복 제거는 없습니다.

_metadata/manifest.csv: 사진별 해시/크기/파일명 명령/동봉 TXT 원시값
_metadata/file_inventory.csv: 원본 ZIP 멤버와 전체 파일의 해시/CRC
_metadata/provenance.json: 원본 ZIP/분할 조각/검증 내역

파일명의 명령과 TXT의 Y_Actual/T_Actual은 서로 다른 필드입니다. 좌표계/원점/단위는 변환하지 않습니다.
파일명/TXT/정답 명령은 영상 추론 입력이 아닙니다.
기준 영상은 clean_20261014/center.bmp입니다. clean_20261020/기준 위치/center1~3.bmp로 자동 교체하지 않습니다.
이 자료는 이미 개발에서 사용한 자료이며 독립 신규 시험 자료가 아닙니다.
