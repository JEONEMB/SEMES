cmake_minimum_required(VERSION 3.15)

project(stage_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 드라이버와 같은 헤더(ioctl 정의, 프로토콜 enum)를 쓴다
set(RS_DRIVER_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../driver/include)

add_library(stage_api STATIC src/stage.cpp)
target_include_directories(stage_api
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${RS_DRIVER_INCLUDE} ${RS_DRIVER_INCLUDE}/uapi)
target_compile_options(stage_api PRIVATE -Wall -Wextra)

add_executable(stagectl tools/stagectl.cpp)
target_link_libraries(stagectl PRIVATE stage_api)
target_compile_options(stagectl PRIVATE -Wall -Wextra)
