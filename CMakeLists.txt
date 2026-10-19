cmake_minimum_required(VERSION 3.15)

project(S15P31A507 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 3.4.x
find_package(OpenCV 3.4 REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

add_executable(main ${PROJECT_SOURCES})
target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
)

# OpenCV DLL을 실행파일 옆으로 자동 복사
# Debug   -> opencv_world3415d.dll
# Release -> opencv_world3415.dll
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:main>"
)
# Offline direct-registration collector; original alignment API remains unchanged.
set(YTHETA_SOURCES ${PROJECT_SOURCES})
list(FILTER YTHETA_SOURCES EXCLUDE REGEX "/main.cpp$")
add_executable(ytheta_collect tool/ytheta_collect.cpp ${YTHETA_SOURCES})
target_include_directories(ytheta_collect PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ytheta_collect PRIVATE ${OpenCV_LIBS})
add_custom_command(TARGET ytheta_collect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
        "$<TARGET_FILE_DIR:ytheta_collect>")
