# 빌드 대상: 다른 프로그램에서 쓸 측정 기능과 직접 실행할 개발 도구
# 장비 연동: alignment. 개발과 검증: tools
cmake_minimum_required(VERSION 3.15)
project(plate_alignment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)   # 검증 기준: OpenCV 3.4.15

# 현재 체크아웃의 금지 문자를 발견하면 빌드 전에 중단한다.
if(WIN32)
    add_custom_target(check_text_style
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_text_style.ps1"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Checking text style rules")
endif()
include(CTest)

# add_program: 실행 파일 생성과 공통 설정 적용
#   입력: name (실행 파일명), source (소스 파일), 함께 쓸 기능 목록
#   출력: 실행 파일 생성 규칙과 Windows 컴파일 설정
function(add_program name source)
    add_executable(${name} ${source})
    if(TARGET check_text_style)
        add_dependencies(${name} check_text_style)
    endif()
    target_link_libraries(${name} PRIVATE ${ARGN})
    if(WIN32 AND TARGET opencv_world)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:opencv_world>" "$<TARGET_FILE_DIR:${name}>"
            COMMENT "Copying the selected OpenCV runtime beside the executable")
    endif()
    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8 /EHsc)
        target_compile_definitions(${name} PRIVATE NOMINMAX)
    endif()
endfunction()

# 측정 라이브러리: include/alignment.hpp 의 함수들 (src/)
add_library(alignment STATIC
    src/alignment.cpp
    src/common/reference.cpp
    src/steps/01_find_qr.cpp
    src/steps/02_find_screws.cpp
    src/steps/03_compute_yt.cpp
    src/steps/04_check_result.cpp
    src/setup/settings.cpp
    src/setup/find_new_reference.cpp)
target_include_directories(alignment PUBLIC include ${OpenCV_INCLUDE_DIRS} PRIVATE src)
target_link_libraries(alignment PUBLIC ${OpenCV_LIBS})
if(TARGET check_text_style)
    add_dependencies(alignment check_text_style)
endif()
if(BUILD_TESTING)
    add_program(safety_tests tools/tests/safety.cpp alignment)
    add_test(NAME safety_contract COMMAND safety_tests "${CMAKE_CURRENT_SOURCE_DIR}/data/model" "${CMAKE_CURRENT_SOURCE_DIR}/data/sample/sample_1.png")
endif()
if(MSVC)
    target_compile_options(alignment PRIVATE /utf-8 /EHsc)
    target_compile_definitions(alignment PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# tools/try: 실행 예제, 빌드 확인, 문서 그림
foreach(name measure_one show_steps check_photos check_setup)
    add_program(${name} tools/try/${name}.cpp alignment)
    target_include_directories(${name} PRIVATE src)   # show_steps의 단계별 함수 접근용
endforeach()

# tools/common: 학습, 평가 도구가 같이 쓰는 코드
add_library(tools_common STATIC
    tools/common/photo_log.cpp
    tools/common/photo_files.cpp
    tools/common/data_preprocessing.cpp
    tools/common/tables.cpp
    tools/common/console.cpp
    tools/common/progress.cpp
    tools/common/coefficient_fit.cpp
    tools/common/dataset.cpp)
target_include_directories(tools_common PUBLIC tools src)   # 학습과 측정의 자료형 및 계산 함수 공유
target_link_libraries(tools_common PUBLIC alignment)
target_link_libraries(measure_one PRIVATE tools_common)
if(MSVC)
    target_compile_options(tools_common PRIVATE /utf-8 /EHsc)
    target_compile_definitions(tools_common PRIVATE NOMINMAX)
endif()

# tools/train: 사진과 정답으로 이동량 계산식 생성
foreach(name train prepare_dataset fit_coefficients)
    add_program(${name} tools/train/${name}.cpp tools_common)
endforeach()

# tools/evaluate: 성능 평가 도구
foreach(name evaluate_accuracy measure_speed score_photos)
    add_program(${name} tools/evaluate/${name}.cpp tools_common)
endforeach()
target_sources(score_photos PRIVATE tools/evaluate/photo_picker.cpp)
if(WIN32)
    target_sources(score_photos PRIVATE tools/evaluate/score_photos.manifest)
    target_link_libraries(score_photos PRIVATE ole32 shell32 comctl32 uuid)
endif()
add_program(compare_far_pose_methods tools/evaluate/compare_far_pose_methods.cpp)   # 식 모양 비교 실험. 표만 읽음

# tools/environment: 환경 변화 합성과 측정 반응 확인
add_library(scene_effects STATIC tools/environment/scene_effects.cpp)
target_include_directories(scene_effects PUBLIC tools/environment ${OpenCV_INCLUDE_DIRS})
target_link_libraries(scene_effects PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(scene_effects PRIVATE /utf-8 /EHsc)
endif()
foreach(name make_environment_images make_site_images check_new_installation)
    add_program(${name} tools/environment/${name}.cpp alignment scene_effects)
endforeach()
add_program(check_qr_blocking tools/environment/check_qr_blocking.cpp tools_common scene_effects)   # 실제 사진 목록(tables) 사용

if(WIN32)
    add_program(make_figures tools/try/make_figures.cpp tools_common scene_effects gdi32)
endif()

# 모든 콘솔 도구에서 같은 한글 출력 설정을 사용한다.
foreach(name check_new_installation check_qr_blocking make_environment_images make_site_images compare_far_pose_methods safety_tests check_photos check_setup make_figures show_steps)
    if(TARGET ${name})
        target_link_libraries(${name} PRIVATE tools_common)
    endif()
endforeach()

if(BUILD_TESTING)
    add_program(runtime_tests tools/tests/runtime.cpp alignment)
    target_include_directories(runtime_tests PRIVATE src)
    add_test(NAME runtime_contract COMMAND runtime_tests "${CMAKE_CURRENT_SOURCE_DIR}/data/model")
endif()

add_program(evaluate_runtime_photos tools/evaluate/runtime_photos.cpp alignment)

if(BUILD_TESTING)
    add_program(screw_quality_tests tools/tests/screw_quality.cpp alignment)
    target_include_directories(screw_quality_tests PRIVATE src)
    add_test(NAME screw_quality COMMAND screw_quality_tests "${CMAKE_CURRENT_SOURCE_DIR}/data/model")
    add_program(area_fallback_tests tools/tests/area_fallback.cpp alignment)
    target_include_directories(area_fallback_tests PRIVATE src)
    target_compile_definitions(area_fallback_tests PRIVATE HAS_AREA)
    add_test(NAME area_fallback COMMAND area_fallback_tests "${CMAKE_CURRENT_SOURCE_DIR}/data/model")
endif()
