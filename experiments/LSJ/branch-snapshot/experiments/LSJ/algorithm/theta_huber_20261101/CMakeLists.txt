cmake_minimum_required(VERSION 3.15)
project(theta_huber_alignment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)

add_library(alignment_core STATIC src/edge_points.cpp src/huber_line.cpp
    src/theta_decoder.cpp src/theta_pipeline.cpp src/alignment.cpp src/utf8_io.cpp)
target_include_directories(alignment_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(alignment_core PUBLIC ${OpenCV_LIBS})
add_executable(theta_alignment src/main.cpp)
target_link_libraries(theta_alignment PRIVATE alignment_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(theta_tests tests/theta_tests.cpp)
    target_link_libraries(theta_tests PRIVATE alignment_core)
    add_test(NAME theta_components COMMAND theta_tests)
    add_executable(theta_parity_probe tests/parity_probe.cpp)
    target_link_libraries(theta_parity_probe PRIVATE alignment_core)
endif()

foreach(target alignment_core theta_alignment theta_tests theta_parity_probe)
    if(TARGET ${target} AND MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /W4)
    endif()
endforeach()
# Existing team OpenCV 3.4.15 Windows layout; no environment replacement.
if(WIN32)
    foreach(target theta_alignment theta_tests theta_parity_probe)
        if(TARGET ${target})
            add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
                "$<TARGET_FILE_DIR:${target}>")
        endif()
    endforeach()
endif()
