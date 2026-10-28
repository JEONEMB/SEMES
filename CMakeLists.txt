cmake_minimum_required(VERSION 3.15)

project(S15P31A507 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

enable_testing()

# 사용할 1차·2차 모듈 경로 (SEMES_JOINT_DIR, SEMES_CONTROL_DIR)
include("${PROJECT_SOURCE_DIR}/experiments/selected_modules.cmake")
set(SEMES_JOINT_ASSETS   "${SEMES_JOINT_DIR}/assets")
set(SEMES_CONTROL_ASSETS "${SEMES_CONTROL_DIR}/assets")

# 두 모듈이 함께 쓰는 진단 저장소
add_library(semes_diagnostics STATIC src/alignment_diagnostics.cpp)
target_include_directories(semes_diagnostics PUBLIC ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semes_diagnostics PUBLIC ${OpenCV_LIBS})

add_subdirectory("${SEMES_JOINT_DIR}" joint)        # semes_joint
add_subdirectory("${SEMES_CONTROL_DIR}" control)    # semes_control

# 기존 alignment() 호환 진입점
add_library(semes_alignment STATIC src/alignment.cpp)
target_compile_definitions(semes_alignment PRIVATE SEMES_HAS_CONTROL)
target_link_libraries(semes_alignment PUBLIC semes_joint semes_control)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_definitions(main PRIVATE
    SEMES_JOINT_ASSETS="${SEMES_JOINT_ASSETS}"
    SEMES_CONTROL_ASSETS="${SEMES_CONTROL_ASSETS}"
)

target_link_libraries(main PRIVATE
    semes_alignment
)

add_executable(alignment_checks tests/alignment_checks.cpp)
target_link_libraries(alignment_checks PRIVATE semes_alignment)
add_executable(evaluate_alignment tool/02_evaluate_alignment.cpp)
target_link_libraries(evaluate_alignment PRIVATE semes_joint)

foreach(lib semes_diagnostics semes_alignment)
    if(MSVC)
        target_compile_options(${lib} PRIVATE /utf-8 /EHsc)
        target_compile_definitions(${lib} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    endif()
endforeach()

# OpenCV DLL을 실행파일 옆으로 자동 복사
# Debug   -> opencv_world3415d.dll
# Release -> opencv_world3415.dll
foreach(app main alignment_checks evaluate_alignment)
    if(MSVC)
        target_compile_options(${app} PRIVATE /utf-8 /EHsc)
        target_compile_definitions(${app} PRIVATE NOMINMAX)
    endif()
    add_custom_command(TARGET ${app} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:${app}>"
    )
endforeach()

get_filename_component(joint_assets_abs   "${SEMES_JOINT_ASSETS}"   ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
get_filename_component(control_assets_abs "${SEMES_CONTROL_ASSETS}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
add_test(NAME alignment_module_checks COMMAND alignment_checks
    "${joint_assets_abs}"
    "${control_assets_abs}"
    "${PROJECT_SOURCE_DIR}/data/clean/clean_20261014")
