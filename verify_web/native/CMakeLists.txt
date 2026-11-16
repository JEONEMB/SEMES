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
#   -DQR_RIG_DIR=/path/experiments/shin/qr_only_multiple_exe/native
#                              신윤석 QR 전용 Rig 모델 (qr::Cancellation). --config = models/rig-cancellation.xml.gz.
#                              센터 기준 사진(ALGR)으로 모델을 만든다 (기준 사진 대비 Y·Θ).
set(QR_RIG_DIR "" CACHE PATH "신윤석 qr_only_multiple_exe/native 폴더")
#   -DQR_POSE_DIR=/path/experiments/shin/ssafy_qr_pose
#                              신윤석 QR 전용 단일 프레임 추정 (src/alignment.cpp 의 qr_pose::estimate, include/qr_pose.hpp).
#                              실행할 때 --config 로 config/qr_pose*.yml 을 준다 (없으면 빌드 때 경로의 config/qr_pose.yml).
set(QR_POSE_DIR "" CACHE PATH "신윤석 ssafy_qr_pose 폴더")

if(QR_RIG_DIR)
    # 라이브러리 소스 전부 (실행 파일용 main·replay·rig_device 는 뺌)
    file(GLOB USE_SOURCES ${QR_RIG_DIR}/src/*.cpp)
    list(FILTER USE_SOURCES EXCLUDE REGEX "/(main|replay|rig_device)\\.cpp$")
    set(USE_INCLUDES ${QR_RIG_DIR}/include ${REPO_ROOT}/include)
    message(STATUS "알고리즘: ${QR_RIG_DIR} (qr::Cancellation, 센터 기준 사진 대비)")
elseif(QR_POSE_DIR)
    set(USE_SOURCES ${QR_POSE_DIR}/src/alignment.cpp)
    set(USE_INCLUDES ${QR_POSE_DIR}/include ${REPO_ROOT}/include)
    message(STATUS "알고리즘: ${QR_POSE_DIR} (qr_pose::estimate, 고정 영점)")
elseif(HYE_MODULE)
    file(GLOB HYE_SOURCES ${HYE_MODULE}/src/*.cpp ${HYE_MODULE}/third_party/*.cpp)
    # 모듈 CMakeLists.txt 처럼: tracker·spline 은 기록용, huber_fit 은 학습 도구용이라 빼고 빌드
    list(FILTER HYE_SOURCES EXCLUDE REGEX "/(tracker|spline|huber_fit)\\.cpp$")
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
if(QR_RIG_DIR)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_QR_RIG=1)
    if(EXISTS ${QR_RIG_DIR}/src/joint_visual_alignment.cpp)
        # 비전 상대 보정 모델 (qr::JointVisualModel, 979c1d75~). --config 의 kind 가 qr_affine_joint 면 이것을 쓴다
        target_compile_definitions(align_worker PRIVATE QR_RIG_JOINT=1)
    endif()
elseif(QR_POSE_DIR)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_QR_POSE=1
                               QR_POSE_DEFAULT_MODEL="${QR_POSE_DIR}/config/qr_pose.yml")
elseif(HYE_MODULE)
    # 모듈 CMakeLists.txt 와 같은 정의 (tracker 이름공간 바꿈 포함)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_JOINT=1 NOMINMAX _USE_MATH_DEFINES
                               legacy_control=final00_tracking)
    if(EXISTS ${HYE_MODULE}/include/joint_reference.hpp)
        # 영점 업데이트(set_joint_zero) 가 있는 버전 (2026-11-13 16e5d32e~) → ALGR 지원
        target_compile_definitions(align_worker PRIVATE ALGO_JOINT_ZERO=1)
    endif()
elseif(ALGO_DIR)
    target_compile_definitions(align_worker PRIVATE ALGO_HAS_REFERENCE=1)
endif()
# 저장소 팀 헤더의 error_t 가 glibc 와 겹치는 문제 우회 (linux_prelude.hpp 설명 참고).
# 김성현 A507 헤더는 스스로 같은 방법으로 피하므로 같이 켜 둬도 된다.
target_compile_options(align_worker PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/linux_prelude.hpp)
