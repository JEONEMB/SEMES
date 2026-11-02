cmake_minimum_required(VERSION 3.15)
project(gaussian_window_sensitivity LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV 3.4 REQUIRED)
set(BASE "${CMAKE_CURRENT_SOURCE_DIR}/../reference_line_methods_108_20261030")
add_executable(gaussian_window_sensitivity experiment.cpp)
target_include_directories(gaussian_window_sensitivity PRIVATE ${OpenCV_INCLUDE_DIRS} "${BASE}/third_party/nlohmann_json/include")
target_link_libraries(gaussian_window_sensitivity PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(gaussian_window_sensitivity PRIVATE /utf-8 /W4)
  target_compile_definitions(gaussian_window_sensitivity PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()
add_custom_command(TARGET gaussian_window_sensitivity POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
  "$<TARGET_FILE_DIR:gaussian_window_sensitivity>")
