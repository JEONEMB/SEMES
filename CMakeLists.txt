cmake_minimum_required(VERSION 3.15)
project(semes_alignment LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)   # OpenCV 3.4.15 에서 확인했다

# 실행 파일 하나를 만든다: add_program(이름 소스 라이브러리…)
function(add_program name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${ARGN})
    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8 /EHsc)
        target_compile_definitions(${name} PRIVATE NOMINMAX)
    endif()
endfunction()

# ---- 측정 라이브러리: include/alignment.hpp 의 함수들 (src/) ----
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

# ---- tools/try: 써 보기, 빌드 확인, 문서 그림 ----
foreach(name measure_one show_steps check_photos check_setup make_figures)
    add_program(${name} tools/try/${name}.cpp alignment)
    target_include_directories(${name} PRIVATE src)   # show_steps, make_figures 는 단계 함수를 직접 부른다
endforeach()

# ---- tools/common: 학습, 평가 도구가 같이 쓰는 코드 ----
add_library(tools_common STATIC
    tools/common/photo_files.cpp
    tools/common/tables.cpp
    tools/common/console.cpp
    tools/common/progress.cpp
    tools/common/coefficient_fit.cpp
    tools/common/dataset.cpp)
target_include_directories(tools_common PUBLIC tools src)   # 계수 자료형과 계산식은 src/steps 것을 그대로 쓴다 (실사용과 같은 계산)
target_link_libraries(tools_common PUBLIC alignment)
if(MSVC)
    target_compile_options(tools_common PRIVATE /utf-8 /EHsc)
    target_compile_definitions(tools_common PRIVATE NOMINMAX)
endif()

# ---- tools/train: 계수를 다시 만들 때 ----
foreach(name train prepare_dataset fit_coefficients)
    add_program(${name} tools/train/${name}.cpp tools_common)
endforeach()

# ---- tools/evaluate: 성능을 잴 때 ----
foreach(name evaluate_accuracy measure_speed score_photos)
    add_program(${name} tools/evaluate/${name}.cpp tools_common)
endforeach()

# ---- tools/environment: 다른 설치 환경에서 카메라가 볼 화면을 만들고 알고리즘 반응을 잰다 ----
add_library(scene_effects STATIC tools/environment/scene_effects.cpp)
target_include_directories(scene_effects PUBLIC tools/environment ${OpenCV_INCLUDE_DIRS})
target_link_libraries(scene_effects PUBLIC ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(scene_effects PRIVATE /utf-8 /EHsc)
endif()
foreach(name make_environment_images make_site_images check_new_installation)
    add_program(${name} tools/environment/${name}.cpp alignment scene_effects)
endforeach()
