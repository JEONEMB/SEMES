cmake_minimum_required(VERSION 3.15)
project(corner_tracking LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(corner_tracking src/main.cpp)
