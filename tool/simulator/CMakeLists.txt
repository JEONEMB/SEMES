# Simulator — 영점 맞춤 시뮬레이터 (공용 코어 + 콘솔 앱 + 플러그인 SDK)
#   팀 기준 : C++17, OpenCV 3.4.15 (vc15), Windows x64. Linux 에서도 빌드됨 (확인용)
#   cmake -S . -B build -A x64 -DSIM_OPENCV_ROOT="C:/opencv/build"  →  cmake --build build --config Release
#   (또는 -DOpenCV_DIR=... 로 find_package 사용)
#   MFC 화면 앱은 mfc/SimulatorMFC.vcxproj (Visual Studio) — 배포 키트는 build_kit.bat 이 둘 다 만듦
cmake_minimum_required(VERSION 3.15)
project(ZeroPointSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(WIN32 AND SIM_OPENCV_ROOT)
    # Windows 배포판 OpenCV 를 직접 지정 (OpenCVConfig.cmake 가 새 Visual Studio 버전을 모를 때도 동작)
    set(OpenCV_INCLUDE_DIRS "${SIM_OPENCV_ROOT}/include")
    set(OpenCV_LIBS optimized "${SIM_OPENCV_ROOT}/x64/vc15/lib/opencv_world3415.lib" debug "${SIM_OPENCV_ROOT}/x64/vc15/lib/opencv_world3415d.lib")
    set(OpenCV_DIR "${SIM_OPENCV_ROOT}")
    set(OpenCV_VERSION "3.4.15 (SIM_OPENCV_ROOT)")
    set(OpenCV_FOUND TRUE)
else()
    if(NOT OpenCV_DIR AND DEFINED ENV{OPENCV_DIR})
        set(OpenCV_DIR "$ENV{OPENCV_DIR}")
    endif()
    find_package(OpenCV REQUIRED)
endif()
message(STATUS "OpenCV ${OpenCV_VERSION} : ${OpenCV_DIR}")

# 내장 추정기(ksh 정렬 알고리즘) 위치
#  ① 저장소 구조 : PJT/tool/simulator → 저장소 루트(PJT) 의 include/ · src/alignment.cpp · src/modules/*.cpp
#  ② 예전 구조   : QR_Image/Simulator → ../Alignment 의 alignment.cpp · ksh/src/*.cpp · ksh/include
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/modules")
    set(SIM_ALIGNMENT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
    set(SIM_ALIGNMENT_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../Alignment")
endif()
set(SIM_ALIGNMENT_DIR "${SIM_ALIGNMENT_DEFAULT}" CACHE PATH "내장 추정기(ksh 정렬 알고리즘) 폴더 — 저장소 루트 또는 예전 Alignment 폴더")
if(EXISTS "${SIM_ALIGNMENT_DIR}/src/modules")
    file(GLOB SIM_KSH_SOURCES CONFIGURE_DEPENDS "${SIM_ALIGNMENT_DIR}/src/modules/*.cpp")
    set(SIM_KSH_ENTRY "${SIM_ALIGNMENT_DIR}/src/alignment.cpp")
    set(SIM_KSH_INCLUDES "${SIM_ALIGNMENT_DIR}/include")
elseif(EXISTS "${SIM_ALIGNMENT_DIR}/alignment.cpp")
    file(GLOB SIM_KSH_SOURCES CONFIGURE_DEPENDS "${SIM_ALIGNMENT_DIR}/ksh/src/*.cpp")
    set(SIM_KSH_ENTRY "${SIM_ALIGNMENT_DIR}/alignment.cpp")
    set(SIM_KSH_INCLUDES "${SIM_ALIGNMENT_DIR}" "${SIM_ALIGNMENT_DIR}/ksh/include")
else()
    message(FATAL_ERROR "정렬 알고리즘 소스가 없음 : ${SIM_ALIGNMENT_DIR} (-DSIM_ALIGNMENT_DIR=... 로 지정)")
endif()
message(STATUS "내장 추정기 소스 : ${SIM_ALIGNMENT_DIR}")

# 공통 컴파일 설정
function(sim_common target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc /W3)
        target_compile_definitions(${target} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall)
        # 팀 공통 헤더의 typedef double error_t 가 glibc 의 error_t 와 겹치지 않도록
        target_compile_definitions(${target} PRIVATE __error_t_defined=1)
    endif()
endfunction()

# ---- 내장 추정기 : ksh 모듈 + 진입점 alignment()
add_library(sim_ksh STATIC ${SIM_KSH_SOURCES} "${SIM_KSH_ENTRY}")
target_include_directories(sim_ksh PUBLIC ${SIM_KSH_INCLUDES} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sim_ksh PUBLIC ${OpenCV_LIBS})
set_target_properties(sim_ksh PROPERTIES POSITION_INDEPENDENT_CODE ON)
sim_common(sim_ksh)

# ---- 시뮬레이터 코어 (콘솔 · MFC 공용)
file(GLOB SIM_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp")
add_library(sim_core STATIC ${SIM_CORE_SOURCES})
target_include_directories(sim_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/core" "${CMAKE_CURRENT_SOURCE_DIR}/sdk")
target_link_libraries(sim_core PUBLIC sim_ksh ${CMAKE_DL_LIBS})
sim_common(sim_core)

# ---- 콘솔 앱
add_executable(Simulator console/main.cpp)
target_link_libraries(Simulator PRIVATE sim_core)
sim_common(Simulator)
set_target_properties(Simulator PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(WIN32)
    add_custom_command(TARGET Simulator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:Simulator>")
    target_compile_definitions(sim_core PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# ---- 플러그인 SDK (팀원 추정기 DLL + 예시 ksh 플러그인)
add_subdirectory(sdk)
