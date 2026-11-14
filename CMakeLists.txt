cmake_minimum_required(VERSION 3.15)
project(semes_alignment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)   # OpenCV 3.4.15 에서 확인했다

# 라이브러리: include/alignment.hpp 의 함수들
add_library(alignment STATIC
    src/alignment.cpp
    src/common/reference.cpp
    src/steps/01_find_qr.cpp
    src/steps/02_find_screws.cpp
    src/steps/03_compute_yt.cpp
    src/steps/04_check_result.cpp
    src/setup/settings.cpp
    src/setup/find_new_reference.cpp)
target_include_directories(alignment PUBLIC include ${OpenCV_INCLUDE_DIRS} PRIVATE src)
target_link_libraries(alignment PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(alignment PRIVATE /utf-8 /EHsc)
    target_compile_definitions(alignment PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# 예제와 확인 프로그램
foreach(program examples/measure_one examples/show_steps tests/check_photos tests/check_setup tool/make_figures)
    get_filename_component(name ${program} NAME)
    add_executable(${name} ${program}.cpp)
    target_link_libraries(${name} PRIVATE alignment)
    target_include_directories(${name} PRIVATE src)   # show_steps, make_figures 는 단계 함수를 직접 부른다
    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8 /EHsc)
    endif()
endforeach()
