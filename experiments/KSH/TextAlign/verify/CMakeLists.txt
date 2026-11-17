cmake_minimum_required(VERSION 3.10)
project(alignment_text CXX)

# OpenCV 3.4.15 (교보재와 같음) 또는 4.x
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/utf-8)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Threads REQUIRED)

file(GLOB KSH_SRC src/modules/*.cpp)
add_library(ksh STATIC ${KSH_SRC} src/alignment.cpp)
target_include_directories(ksh PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ksh PUBLIC ${OpenCV_LIBS} Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(ksh PUBLIC stdc++fs)
endif()

add_executable(alignment src/main.cpp)
target_link_libraries(alignment ksh)

add_executable(align_train tools/align_train.cpp)
target_link_libraries(align_train ksh)

# 단위 테스트 (합성 영상 · 보정식 · 설정 파일) : ctest 또는 ./align_tests
enable_testing()
add_executable(align_tests tests/test_alignment.cpp)
target_link_libraries(align_tests ksh)
add_test(NAME unit COMMAND align_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# 실사진 점검 (사진이 있을 때 수동 실행) : ./align_photo_check <센터 사진> [사진 ...]
add_executable(align_photo_check tests/test_real_photos.cpp)
target_link_libraries(align_photo_check ksh)
