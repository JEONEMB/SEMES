cmake_minimum_required(VERSION 3.15)
project(gls_wls_comparison LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(gls_wls_comparison src/main.cpp)
