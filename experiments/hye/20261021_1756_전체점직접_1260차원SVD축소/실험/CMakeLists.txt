cmake_minimum_required(VERSION 3.15)
project(fullfield_4fold LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(fullfield_4fold src/main.cpp)
# 측정·판정 공통부는 #56 (10-21 17:53) 실험의 src 를 같이 쓴다
target_include_directories(fullfield_4fold PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../20261021_1753_후보별잡음_가우스우도/실험/src")
