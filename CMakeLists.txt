cmake_minimum_required(VERSION 3.15)

project(S15P31A507 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

# src 바로 아래 파일만 (src/KSH, src/Integrated 의 .cpp 와 main() 이 섞이지 않도록 재귀 검색을 쓰지 않음)
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

add_executable(main ${PROJECT_SOURCES})
target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
)
if(UNIX)
    target_compile_definitions(main PRIVATE __error_t_defined=1)   # alignment.hpp 의 error_t 와 glibc 충돌 방지
endif()

# OpenCV DLL을 실행파일 옆으로 자동 복사
# Debug   -> opencv_world3415d.dll
# Release -> opencv_world3415.dll
if(WIN32)
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:main>"
)
endif()

# 통합 정렬 : 김성현 ksh 추정 + 신윤석 판 검출·QR 검증 (src/Integrated/README.md)
add_subdirectory(src/Integrated)
