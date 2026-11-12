cmake_minimum_required(VERSION 3.15)
project(v2tool LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(CONTROL "${REPO}/experiments/hye/00000_final_Y_T control")
set(FINAL04 "${REPO}/experiments/hye/00004_final_Y_T")
# 536점 추적은 control 모듈, 바와 나사 초기 위치는 00004 모듈의 소스를 그대로 링크한다. 새로 쓴 것은 src/ 뿐이다.
add_executable(v2tool
    src/main.cpp src/screw_gradient.cpp src/huber_fit.cpp
    "${CONTROL}/src/tracker.cpp" "${CONTROL}/src/spline.cpp"
    "${FINAL04}/src/observer.cpp" "${FINAL04}/src/line_profile.cpp" "${FINAL04}/src/pin_observer.cpp"
    "${FINAL04}/src/model.cpp" "${FINAL04}/third_party/ecc_compat.cpp"
    "${REPO}/src/alignment_diagnostics.cpp")
target_include_directories(v2tool PRIVATE src "${REPO}/include" "${CONTROL}/include" "${FINAL04}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(v2tool PRIVATE ${OpenCV_LIBS} bcrypt)
if(MSVC)
    target_compile_options(v2tool PRIVATE /utf-8 /EHsc /O2 /openmp)
    target_compile_definitions(v2tool PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
endif()
add_custom_command(TARGET v2tool POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:v2tool>")
