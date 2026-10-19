cmake_minimum_required(VERSION 3.15)
project(data_ingest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

add_executable(data_ingest main.cpp)
target_link_libraries(data_ingest PRIVATE ${OpenCV_LIBS} bcrypt)
# 소스와 콘솔 출력 모두 UTF-8 (한글 메시지, 한글 경로)
target_compile_options(data_ingest PRIVATE /utf-8)

# OpenCV DLL을 실행파일 옆으로 자동 복사
add_custom_command(TARGET data_ingest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:data_ingest>"
)
