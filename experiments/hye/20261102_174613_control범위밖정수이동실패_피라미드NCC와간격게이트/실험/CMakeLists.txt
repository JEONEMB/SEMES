cmake_minimum_required(VERSION 3.15)
project(a17_out_of_range_shift LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(CONTROL "${REPO}/experiments/hye/00000_final_Y_T control")
# 정밀화 단계는 채택 모듈의 spline.cpp 를 그대로 링크한다.
# tracker.cpp 는 링크하지 않는다. 정수 이동 단계를 바꾼 복사본(src/tracker_variant.hpp)을 쓰기 때문이다.
# 그 복사본이 원본과 같은지는 S_A 가 A0 캐시의 유효점 수를 재현하는지로 확인한다.
add_executable(a17 src/main.cpp "${CONTROL}/src/spline.cpp" "${CONTROL}/src/tracker.cpp")
target_include_directories(a17 PRIVATE src "${REPO}/include" "${CONTROL}/include" "${REPO}/experiments/hye/00001_common/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(a17 PRIVATE ${OpenCV_LIBS} bcrypt)
target_compile_definitions(a17 PRIVATE REPO_DIR="${REPO}" EXPERIMENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../결과")
if(MSVC)
 target_compile_options(a17 PRIVATE /utf-8 /EHsc /O2)
 target_compile_definitions(a17 PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
endif()
add_custom_command(TARGET a17 POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:a17>")
