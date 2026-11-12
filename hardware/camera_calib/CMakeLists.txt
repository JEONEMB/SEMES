cmake_minimum_required(VERSION 3.15)

project(camera_calib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV 3.4.x (Pi: /usr/local 소스 빌드, Windows: OpenCV_DIR 지정)
find_package(OpenCV 3.4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

add_executable(camera_calib calibrate.cpp)
target_link_libraries(camera_calib PRIVATE ${OpenCV_LIBS})
