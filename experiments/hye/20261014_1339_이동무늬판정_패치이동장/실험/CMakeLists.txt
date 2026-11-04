cmake_minimum_required(VERSION 3.15)
project(patch_field LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(patch_field src/main.cpp)
