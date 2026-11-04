cmake_minimum_required(VERSION 3.15)
project(release_screw_fallback LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(release_screw_fallback src/main.cpp)
# 측정·판정 공통부는 #56 (10-21 17:53) 실험의 src 를 같이 쓴다
target_include_directories(release_screw_fallback PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../20261021_1753_후보별잡음_가우스우도/실험/src")
