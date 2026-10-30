# 증강 검증용 실험 도구. 저장소 루트 빌드와 분리해서 이 폴더에서 따로 구성한다.
#   cmake -S tool -B tool/build -G "Visual Studio 18 2026" -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
#   cmake --build tool/build --config Release --target dump_features
cmake_minimum_required(VERSION 3.15)
project(yt_augment_tools LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)

get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
include("${REPO}/experiments/selected_modules.cmake")
add_library(semes_diagnostics STATIC "${REPO}/src/alignment_diagnostics.cpp")
target_include_directories(semes_diagnostics PUBLIC "${REPO}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semes_diagnostics PUBLIC ${OpenCV_LIBS})
add_subdirectory("${REPO}/${SEMES_JOINT_DIR}" joint)
add_subdirectory("${REPO}/${SEMES_CONTROL_DIR}" control)
target_include_directories(semes_joint PUBLIC "${REPO}/include")
target_include_directories(semes_control PUBLIC "${REPO}/include")

add_executable(dump_features dump_features.cpp)
target_include_directories(dump_features PRIVATE "${REPO}/${SEMES_JOINT_DIR}/include" "${REPO}/${SEMES_CONTROL_DIR}/include")
target_link_libraries(dump_features PRIVATE semes_joint semes_control)
if(MSVC)
  foreach(t semes_diagnostics dump_features)
    target_compile_options(${t} PRIVATE /utf-8 /EHsc)
    target_compile_definitions(${t} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  endforeach()
endif()
add_custom_command(TARGET dump_features POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415.dll" "$<TARGET_FILE_DIR:dump_features>")
