cmake_minimum_required(VERSION 3.15)
project(affine18_gls LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(affine18_gls src/main.cpp)
