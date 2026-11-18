# Alignment (김성현) — ksh 모듈 라이브러리 + 예제·학습·평가 실행 파일
#   팀 기준 : C++17, OpenCV 3.4.15 (vc15), Windows x64
#   cmake -S . -B build -A x64 -DOpenCV_DIR="C:/opencv/build"  →  cmake --build build --config Release
#   실행 파일은 작업 폴더 = 이 폴더에서 실행 (config/alignment.yml 을 상대 경로로 찾음)
cmake_minimum_required(VERSION 3.15)
project(KSH_Alignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KSH_OPENCV_VERSION "3.4" CACHE STRING "찾을 OpenCV 버전 (팀 기준 3.4)")
find_package(OpenCV ${KSH_OPENCV_VERSION} REQUIRED)

# ---- 함수 단위 모듈 (공유 단위) : ksh/include/ksh/*.hpp 선언, ksh/src/*.cpp 구현
file(GLOB KSH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ksh/src/*.cpp")
add_library(ksh_align STATIC ${KSH_SOURCES})
target_include_directories(ksh_align PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ksh/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ksh_align PUBLIC ${OpenCV_LIBS})

# ---- 제출 형식 진입점 alignment() (ksh 모듈을 감싼 얇은 층)
add_library(ksh_alignment_api STATIC alignment.cpp)
target_include_directories(ksh_alignment_api PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ksh_alignment_api PUBLIC ksh_align)

if(MSVC)
    target_compile_options(ksh_align PUBLIC /utf-8 /EHsc)
    target_compile_definitions(ksh_align PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
elseif(UNIX)
    # 스켈레톤의 typedef double error_t 가 glibc 의 error_t 와 겹치지 않도록
    target_compile_definitions(ksh_align PUBLIC __error_t_defined=1)
endif()

# ---- 실행 파일 : 예제(main) · 모델 학습 · 정확도 평가 (각자 main 을 가지므로 분리)
add_executable(alignment main.cpp)
target_link_libraries(alignment PRIVATE ksh_alignment_api)
add_executable(align_train tools/align_train.cpp)
target_link_libraries(align_train PRIVATE ksh_align)
add_executable(align_eval tools/align_eval.cpp)
target_link_libraries(align_eval PRIVATE ksh_align)

# OpenCV DLL 을 실행 파일 옆으로 (Windows)
if(WIN32)
    foreach(app alignment align_train align_eval)
        add_custom_command(TARGET ${app} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
                "$<TARGET_FILE_DIR:${app}>")
    endforeach()
endif()
