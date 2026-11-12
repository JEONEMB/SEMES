cmake_minimum_required(VERSION 3.15)
project(verify_align_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 저장소 루트 CMakeLists.txt 와 같은 OpenCV 3.4 (Pi: /usr/local)
find_package(OpenCV 3.4 REQUIRED)

set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
# 알고리즘이 바뀌면 소스·헤더 목록만 바꾼다. 예:
#   cmake -S . -B build -DALGO_SOURCES="/path/a.cpp;/path/b.cpp" -DALGO_INCLUDES="/path/include"
set(ALGO_SOURCES ${REPO_ROOT}/src/alignment.cpp CACHE STRING "alignment() 구현 소스 (; 로 구분)")
set(ALGO_INCLUDES ${REPO_ROOT}/include CACHE STRING "alignment.hpp 가 있는 폴더 (; 로 구분)")

add_executable(align_worker align_worker.cpp ${ALGO_SOURCES})
target_include_directories(align_worker PRIVATE ${ALGO_INCLUDES} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(align_worker PRIVATE ${OpenCV_LIBS})
# 팀 헤더의 error_t 가 glibc 와 겹치는 문제 우회 (linux_prelude.hpp 설명 참고)
target_compile_options(align_worker PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/linux_prelude.hpp)
