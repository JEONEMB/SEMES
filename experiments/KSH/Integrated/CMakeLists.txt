# 통합 정렬 (김성현 ksh 추정 + 신윤석 판 검출·QR 검증)
#   루트 CMakeLists.txt 에서 add_subdirectory(src/Integrated) 로 함께 빌드된다
#   실행 파일은 작업 폴더 = src/Integrated 에서 실행 (config/integrated.yml 을 상대 경로로 찾음)

set(REPO_ROOT "${PROJECT_SOURCE_DIR}")
set(KSH_DIR "${REPO_ROOT}/src/KSH/Alignment")

# ---- 김성현 : ksh 추정 모듈 (ROI 정합 · 시차 특징 · 보정식)
file(GLOB KSH_SOURCES CONFIGURE_DEPENDS "${KSH_DIR}/ksh/src/*.cpp")
add_library(ksh_align STATIC ${KSH_SOURCES})
target_include_directories(ksh_align PUBLIC "${KSH_DIR}/ksh/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ksh_align PUBLIC ${OpenCV_LIBS})

# ---- 신윤석 : 판 검출 · QR 해독 · 모듈 재생성 · Fourier 코너 (src/*.cpp 중 main.cpp, alignment.cpp 제외)
#      alignment.cpp 는 검출만 하고 NaN 을 돌려주므로, 통합 쪽 team_alignment.cpp 가 같은 함수를 실제 값으로 구현한다
file(GLOB SHIN_SOURCES CONFIGURE_DEPENDS "${REPO_ROOT}/src/*.cpp")
list(FILTER SHIN_SOURCES EXCLUDE REGEX ".*/src/(main|alignment)\\.cpp$")
add_library(shin_detect STATIC ${SHIN_SOURCES})
target_include_directories(shin_detect PUBLIC "${REPO_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(shin_detect PUBLIC ${OpenCV_LIBS})

# ---- 통합 라이브러리
add_library(integrated STATIC src/integrated.cpp src/team_alignment.cpp)
target_include_directories(integrated PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(integrated PUBLIC ksh_align shin_detect)

foreach(t ksh_align shin_detect integrated)
    if(MSVC)
        target_compile_options(${t} PUBLIC /utf-8 /EHsc)
        target_compile_definitions(${t} PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
    elseif(UNIX)
        target_compile_definitions(${t} PUBLIC __error_t_defined=1)   # alignment.hpp 의 error_t 와 glibc 충돌 방지
    endif()
endforeach()

# ---- 실행 파일
add_executable(integrated_main apps/integrated_main.cpp)
target_link_libraries(integrated_main PRIVATE integrated)
add_executable(integrated_eval apps/integrated_eval.cpp)
target_link_libraries(integrated_eval PRIVATE integrated)

if(WIN32)
    foreach(app integrated_main integrated_eval)
        add_custom_command(TARGET ${app} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll"
                "$<TARGET_FILE_DIR:${app}>")
    endforeach()
endif()
