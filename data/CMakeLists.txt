# data-automation.cpp 빌드용 (본 프로그램 main 과 따로 빌드)
cmake_minimum_required(VERSION 3.15)
project(data_automation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

add_executable(data-automation data-automation.cpp)
target_link_libraries(data-automation PRIVATE ${OpenCV_LIBS})
# 소스 파일을 UTF-8 로 읽음 (한글 주석, 한글 메시지)
target_compile_options(data-automation PRIVATE /utf-8)

# OpenCV DLL을 실행파일 옆으로 자동 복사
add_custom_command(TARGET data-automation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:data-automation>"
)
