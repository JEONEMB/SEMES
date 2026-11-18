cmake_minimum_required(VERSION 3.15)

project(stage_capture LANGUAGES CXX)

# stage_camera 헤더가 libcamera(std::span)를 따라 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 촬영 계획 (카메라·스테이지 없이 PC 에서도 빌드·시험)
add_library(capture_plan STATIC src/plan.cpp)
target_include_directories(capture_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
add_executable(plan_test tests/plan_test.cpp)
target_link_libraries(plan_test PRIVATE capture_plan)
add_test(NAME plan_test COMMAND plan_test)

# 촬영 도구 (라즈베리파이: 스테이지 드라이버 + libcamera)
set(STAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../stage)
set(CAMERA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../camera)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "리눅스가 아니라 capture_dataset 은 빌드하지 않음 (plan_test 만)")
elseif(NOT EXISTS ${CAMERA_DIR}/CMakeLists.txt)
    message(WARNING "hardware/camera 가 없어 capture_dataset 은 빌드하지 않음 (feat/lye/camera-calib 머지 필요)")
else()
    add_subdirectory(${STAGE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/stage)
    add_subdirectory(${CAMERA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/camera)
    add_executable(capture_dataset tools/capture_dataset.cpp)
    target_link_libraries(capture_dataset PRIVATE capture_plan stage_api stage_camera)
    target_compile_options(capture_dataset PRIVATE -Wall -Wextra)
endif()
