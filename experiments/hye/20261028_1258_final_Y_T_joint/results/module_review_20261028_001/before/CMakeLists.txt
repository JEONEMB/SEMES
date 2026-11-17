cmake_minimum_required(VERSION 3.15)

project(S15P31A507 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

enable_testing()
add_subdirectory("experiments/00000_final_Y_T_joint")
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main PRIVATE
    semes_alignment
)
if(MSVC)
    target_compile_options(main PRIVATE /utf-8 /EHsc)
endif()

# OpenCV DLL을 실행파일 옆으로 자동 복사
# Debug   -> opencv_world3415d.dll
# Release -> opencv_world3415.dll
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:main>"
)
