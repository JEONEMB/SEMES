# 실험/CMakeLists.txt 에서 include 해 쓴다.
#   include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
#   hye_experiment(<대상> src/main.cpp ...)
# REPO_DIR = 저장소 루트, EXPERIMENT_DIR = 이 실험의 결과/ 폴더
find_package(OpenCV 3.4 REQUIRED)
get_filename_component(HYE_REPO "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(HYE_COMMON "${CMAKE_CURRENT_LIST_DIR}")

# Python cv2(4.x 이후)와 같은 ECC. OpenCV 3.4 의 findTransformECC 는 결과가 조금 다르다.
if(NOT TARGET hye_common)
  add_library(hye_common STATIC "${HYE_COMMON}/src/ecc_compat.cpp")
  target_compile_features(hye_common PUBLIC cxx_std_17)
  target_include_directories(hye_common PUBLIC "${HYE_COMMON}/include" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(hye_common PUBLIC ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(hye_common PRIVATE /utf-8 /O2)
    target_compile_definitions(hye_common PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
  endif()
endif()

function(hye_experiment target)
  add_executable(${target} ${ARGN})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PRIVATE "${HYE_COMMON}/include" ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE hye_common ${OpenCV_LIBS})
  target_compile_definitions(${target} PRIVATE REPO_DIR="${HYE_REPO}" EXPERIMENT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../결과")
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /EHsc /O2 /openmp)
    target_compile_definitions(${target} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
  endif()
  add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_DIR}/x64/vc15/bin/opencv_world3415$<$<CONFIG:Debug>:d>.dll" "$<TARGET_FILE_DIR:${target}>")
endfunction()
