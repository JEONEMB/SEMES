cmake_minimum_required(VERSION 3.15)
project(verify_align_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 저장소 루트 CMakeLists.txt 와 같은 OpenCV 3.4 (Pi: /usr/local)
find_package(OpenCV 3.4 REQUIRED)
find_package(Threads REQUIRED)

set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# 알고리즘 고르기
#   -DALGO_DIR=/path/to/Code   김성현 A507 버전처럼 src/alignment.cpp + src/modules/*.cpp + include/ 구조.
#                              alignment_init / alignment_set_reference 를 쓴다 (영점 등록 가능).
#   (없으면) 저장소 src/alignment.cpp + include/, 또는 -DALGO_SOURCES=...;... -DALGO_INCLUDES=...
set(ALGO_DIR "" CACHE PATH "A507 알고리즘 Code 폴더 (src/, include/, config/)")
set(ALGO_SOURCES ${REPO_ROOT}/src/alignment.cpp CACHE STRING "alignment() 구현 소스 (; 로 구분)")
set(ALGO_INCLUDES ${REPO_ROOT}/include CACHE STRING "alignment.hpp 가 있는 폴더 (; 로 구분)")
#   -DHYE_DIR=/path -DHYE_MODULE=/path/experiments/hye/00000_final_Y_T_ssafy
#                              이혜원 값 7개 모듈. HYE_DIR 은 저장소 형태(include/alignment.hpp, src/alignment.cpp,
#                              src/alignment_diagnostics.cpp), HYE_MODULE 은 모듈 폴더(src/, include/, third_party/).
#                              실행할 때 --config 로 모듈의 assets 폴더를 준다. 영점은 자료에 고정 (영점 등록 없음).
set(HYE_DIR "" CACHE PATH "이혜원 알고리즘 저장소 형태 폴더")
set(HYE_MODULE "" CACHE PATH "이혜원 모듈 폴더 (예: .../experiments/hye/00000_final_Y_T_ssafy)")

if(HYE_MODULE)
    file(GLOB HYE_SOURCES ${HYE_MODULE}/src/*.cpp ${HYE_MODULE}/third_party/*.cpp)
    set(USE_SOURCES ${HYE_DIR}/src/alignment.cpp ${HYE_DIR}/src/alignment_diagnostics.cpp ${HYE_SOURCES})
    set(USE_INCLUDES ${HYE_DIR}/include ${HYE_MODULE}/include)
    message(STATUS "알고리즘: ${HYE_MODULE} (initialize_joint, 고정 영점)")
elseif(ALGO_DIR)
    file(GLOB ALGO_MODULES ${ALGO_DIR}/src/modules/*.cpp)
    set(USE_SOURCES ${ALGO_DIR}/src/alignment.cpp ${ALGO_MODULES})
    set(USE_INCLUDES ${ALGO_DIR}/include)
    message(STATUS "알고리즘: ${ALGO_DIR} (영점 등록 사용)")
else()
    set(USE_SOURCES ${ALGO_SOURCES})
    set(USE_INCLUDES ${ALGO_INCLUDES})
    message(STATUS "알고리즘: ${ALGO_SOURCES}")
endif()

add_executable(align_worker align_worker.cpp ${USE_SOURCES})
target_include_directories(align_worker PRIVATE ${USE_INCLUDES} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(align_worker PRIVATE ${OpenCV_LIBS} Threads::Threads)
if(HYE_MODULE)
    # 모듈 CMakeLists.txt 와 같은 정의 (tracker 이름공간 바꿈 포함)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_JOINT=1 NOMINMAX _USE_MATH_DEFINES
                               legacy_control=final00_tracking)
elseif(ALGO_DIR)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_REFERENCE=1)
endif()
# 저장소 팀 헤더의 error_t 가 glibc 와 겹치는 문제 우회 (linux_prelude.hpp 설명 참고).
# 김성현 A507 헤더는 스스로 같은 방법으로 피하므로 같이 켜 둬도 된다.
target_compile_options(align_worker PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/linux_prelude.hpp)
