cmake_minimum_required(VERSION 3.15)
project(prepare_clean_tool LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "This tool uses Windows tar.exe and BCrypt.")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)

add_executable(prepare_clean 01_prepare_clean.cpp)
set_target_properties(prepare_clean PROPERTIES OUTPUT_NAME "01_prepare_clean")
target_include_directories(prepare_clean PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(prepare_clean PRIVATE ${OpenCV_LIBS} bcrypt)
target_compile_definitions(prepare_clean PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN
    PREPARE_CLEAN_SOURCE_DIR=L"${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
    target_compile_options(prepare_clean PRIVATE /utf-8 /W4)
endif()

# 새 자료 EDA 측정과 분석 (--target 20261020 / 20261029)
add_executable(eda 04_eda.cpp)
set_target_properties(eda PROPERTIES OUTPUT_NAME "04_eda")
target_include_directories(eda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(eda PRIVATE ${OpenCV_LIBS})
target_compile_definitions(eda PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
    target_compile_options(eda PRIVATE /utf-8 /W4)
endif()

# Same OpenCV binary/configuration as the team's main application.
add_custom_command(TARGET prepare_clean POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:prepare_clean>")
