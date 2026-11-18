# 김성현 Y·Θ 정렬 알고리즘 (ksh) — 한 프로젝트로 빌드
#   C++17, OpenCV 3.4.15 (vc15), Windows x64 / 작업 폴더 = 저장소 루트 (config/alignment.yml 을 상대 경로로 찾음)
#   cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR="C:/opencv/build"
#   cmake --build build --config Release   →   .\build\Release\alignment.exe
#
#   include/            alignment.hpp (제출 형식) · ksh/*.hpp (모듈 선언)
#   src/                alignment.cpp (진입점 alignment()) · main.cpp (예제) · modules/*.cpp (모듈 구현)
#   tool/ksh/           align_train.cpp (모델 학습) · align_eval.cpp (정확도 평가)
#   config/             alignment.yml (설정) · model_*.yml (학습된 보정식)
#   experiments/KSH/    실험 기록 (빌드에 포함하지 않음)
cmake_minimum_required(VERSION 3.15)

project(S15P31A507_KSH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KSH_OPENCV_VERSION "3.4" CACHE STRING "찾을 OpenCV 버전 (팀 기준 3.4)")
find_package(OpenCV ${KSH_OPENCV_VERSION} REQUIRED)

# ---- 함수 단위 모듈 : include/ksh/*.hpp 선언, src/modules/*.cpp 구현
file(GLOB KSH_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/modules/*.cpp")
add_library(ksh_align STATIC ${KSH_SOURCES})
target_include_directories(ksh_align PUBLIC "${PROJECT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ksh_align PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(ksh_align PUBLIC /utf-8 /EHsc)
    target_compile_definitions(ksh_align PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
elseif(UNIX)
    target_compile_definitions(ksh_align PUBLIC __error_t_defined=1)   # 스켈레톤 typedef double error_t 와 glibc 충돌 방지
endif()

# ---- 제출 형식 진입점 alignment() (ksh 모듈을 감싼 얇은 층)
add_library(ksh_alignment_api STATIC src/alignment.cpp)
target_link_libraries(ksh_alignment_api PUBLIC ksh_align)

# ---- 실행 파일 : 예제 · 모델 학습 · 정확도 평가 (각자 main 을 가지므로 분리)
add_executable(alignment src/main.cpp)
target_link_libraries(alignment PRIVATE ksh_alignment_api)
add_executable(align_train tool/ksh/align_train.cpp)
target_link_libraries(align_train PRIVATE ksh_align)
add_executable(align_eval tool/ksh/align_eval.cpp)
target_link_libraries(align_eval PRIVATE ksh_align)
set(KSH_APPS alignment align_train align_eval)

# ---- 선택 : 통합 정렬기 (ksh + 팀 고정 파이프라인, 실험 기록) — 기본 끔
option(KSH_BUILD_INTEGRATED "experiments/KSH/Integrated 도 빌드 (legacy/team_pipeline 필요)" OFF)
if(KSH_BUILD_INTEGRATED)
    add_subdirectory(experiments/KSH/Integrated)
endif()

# ---- OpenCV DLL 을 실행 파일 옆으로 (Windows)
if(WIN32)
    foreach(app ${KSH_APPS})
        add_custom_command(TARGET ${app} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
                "$<TARGET_FILE_DIR:${app}>")
    endforeach()
endif()
