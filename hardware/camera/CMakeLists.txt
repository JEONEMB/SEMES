cmake_minimum_required(VERSION 3.15)

project(stage_camera LANGUAGES CXX)

# libcamera 헤더가 std::span 을 써서 C++20 이 필요하다
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV 3.4.15 (Pi: /usr/local 소스 빌드)
find_package(OpenCV 3.4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
find_package(Threads REQUIRED)

# 카메라 없이 쓰는 부분 (사진 → SEMES 규격 변환). libcamera 불필요
add_library(stage_rectifier STATIC src/rectifier.cpp)
target_include_directories(stage_rectifier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(stage_rectifier PUBLIC ${OpenCV_LIBS})

# 촬영 (libcamera)
add_library(stage_camera STATIC src/camera.cpp src/libcamera_source.cpp)
target_link_libraries(stage_camera PUBLIC stage_rectifier PRIVATE PkgConfig::LIBCAMERA Threads::Threads)
target_compile_options(stage_camera PRIVATE -Wall -Wextra)

add_executable(cam_capture tools/cam_capture.cpp)
target_link_libraries(cam_capture PRIVATE stage_camera)
