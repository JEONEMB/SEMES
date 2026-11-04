cmake_minimum_required(VERSION 3.15)
project(patch_windows LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
# 630점 측정·판정 공통 코드는 #50 실험 폴더(20261021_1723_센터단일기준_630패치GN추적/실험/src)의 것을 쓴다
set(S50 "${CMAKE_CURRENT_SOURCE_DIR}/../../20261021_1723_센터단일기준_630패치GN추적/실험/src")
hye_experiment(patch_windows src/main.cpp)
target_include_directories(patch_windows PRIVATE "${S50}")
