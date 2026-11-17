cmake_minimum_required(VERSION 3.15)
project(center_fixed LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(center_fixed src/main.cpp)
