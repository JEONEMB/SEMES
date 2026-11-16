# Unity 시뮬레이터 '알고리즘 확인' 화면용 실행기 (임시, 작동 시험용)
#   cmake -S . -B build -A x64 -DOpenCV_DIR="C:/opencv/build"
#   cmake --build build --config Release
#   → build/Release/align_worker.exe 경로를 Unity '알고리즘 확인 > 설정 > 실행기 경로' 에 넣음
#     설정 파일은 ksh_alignment_1101/config/alignment.yml (기준 사진 경로 reference_path 확인 필요)
# 상태: OpenCV 3.4.15 미설치로 아직 빌드 안 함
cmake_minimum_required(VERSION 3.15)
project(yt_align_worker LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ksh_alignment_1101)   # 김성현 11-01 버전 (ksh_alignment_api 라이브러리)
add_executable(align_worker align_worker.cpp)
target_link_libraries(align_worker PRIVATE ksh_alignment_api)
