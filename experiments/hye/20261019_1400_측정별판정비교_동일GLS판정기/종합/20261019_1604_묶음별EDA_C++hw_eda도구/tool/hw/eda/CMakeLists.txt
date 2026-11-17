# tool/hw/eda: 데이터 묶음 EDA 분석 + 발표자료(pptx) 자동 생성 (본 프로그램 main 과 따로 빌드)
cmake_minimum_required(VERSION 3.15)
project(hw_eda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

file(GLOB HW_EDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_executable(hw_eda ${HW_EDA_SOURCES})
target_link_libraries(hw_eda PRIVATE ${OpenCV_LIBS})
# 소스를 UTF-8 로 읽음 (한글 주석, 한글 출력), 경고는 /W3
target_compile_options(hw_eda PRIVATE /utf-8 /W3)
# windows.h 의 min/max 매크로가 std::min/std::max 와 부딪히지 않게
target_compile_definitions(hw_eda PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# OpenCV DLL을 실행파일 옆으로 자동 복사
add_custom_command(TARGET hw_eda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:hw_eda>"
)
