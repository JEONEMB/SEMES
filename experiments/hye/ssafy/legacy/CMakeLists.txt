cmake_minimum_required(VERSION 3.15)
project(ssafy LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4.15 EXACT REQUIRED)
add_library(semes_diagnostics STATIC src/alignment_diagnostics.cpp)
target_include_directories(semes_diagnostics PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semes_diagnostics PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(semes_diagnostics PRIVATE /utf-8 /EHsc)
    target_compile_definitions(semes_diagnostics PRIVATE NOMINMAX)
endif()
option(SEMES_ALLOW_UNVALIDATED_OPENCV "Allow other OpenCV versions for compatibility audits only" OFF)
if(NOT OpenCV_VERSION VERSION_EQUAL "3.4.15")
    if(NOT SEMES_ALLOW_UNVALIDATED_OPENCV)
        message(FATAL_ERROR "final00 은 OpenCV 3.4.15 에서 검증했다. 다른 버전은 진단용으로만 SEMES_ALLOW_UNVALIDATED_OPENCV=ON 을 준다.")
    endif()
    message(WARNING "검증하지 않은 OpenCV 버전: ${OpenCV_VERSION}. 기준 성적을 그대로 쓰지 않는다")
endif()

# QR 값은 assets/site.json 의 qr_method 로 고른다: track536 = 536점 추적(src/tracker.cpp, src/spline.cpp, 20261028_1327_final_Y_T control 복사본, SEMES), ncc = qr_ncc.cpp(QR 3개 NCC + 포물선, 우리 장비). QR, 바 관측과 나사 초기 위치(observer, line_profile, pin_observer, model, ecc_compat)는
# 20261111_0944_final_Y_T 의 소스를 복사한 것이다. 새로 쓴 것은 joint.cpp, estimator.cpp, screw_gradient.cpp, huber_fit.cpp 다.
add_library(semes_joint STATIC
    src/joint.cpp src/estimator.cpp src/screw_gradient.cpp src/site.cpp src/sha256.cpp src/reference.cpp src/qr_ncc.cpp
    src/tracker.cpp src/spline.cpp
    src/observer.cpp src/line_profile.cpp src/pin_observer.cpp src/model.cpp
    third_party/ecc_compat.cpp)
target_include_directories(semes_joint PUBLIC "${PROJECT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS} PRIVATE include)
target_link_libraries(semes_joint PUBLIC semes_diagnostics ${OpenCV_LIBS})
target_compile_features(semes_joint PUBLIC cxx_std_17)
# 2차 모듈(semes_control)도 같은 tracker 소스를 쓴다. 한 실행 파일에 함께 링크될 때 기호가 겹치지 않게 이 모듈 안에서만 이름공간을 바꾼다.
set(FINAL00_DEFINITIONS NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES legacy_control=final00_tracking)
target_compile_definitions(semes_joint PRIVATE ${FINAL00_DEFINITIONS})
if(MSVC)
    target_compile_options(semes_joint PRIVATE /utf-8 /EHsc)
endif()

# 측정, 학습, 평가 도구
add_executable(final00_tool tools/final00_tool.cpp src/huber_fit.cpp)
target_include_directories(final00_tool PRIVATE include)
target_link_libraries(final00_tool PRIVATE semes_joint)
target_compile_definitions(final00_tool PRIVATE ${FINAL00_DEFINITIONS})
if(MSVC)
    target_compile_options(final00_tool PRIVATE /utf-8 /EHsc /openmp)
    add_custom_command(TARGET final00_tool POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:final00_tool>")
endif()

add_executable(measure_one tools/measure_one.cpp)
target_link_libraries(measure_one PRIVATE semes_joint)
if(MSVC)
    target_compile_options(measure_one PRIVATE /utf-8 /EHsc)
    target_compile_definitions(measure_one PRIVATE NOMINMAX)
    add_custom_command(TARGET measure_one POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:measure_one>")
endif()

add_executable(evaluate_photos tools/evaluate_photos.cpp)
target_link_libraries(evaluate_photos PRIVATE semes_joint)
if(MSVC)
    target_compile_options(evaluate_photos PRIVATE /utf-8 /EHsc)
    target_compile_definitions(evaluate_photos PRIVATE NOMINMAX)
    add_custom_command(TARGET evaluate_photos POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:evaluate_photos>")
endif()
