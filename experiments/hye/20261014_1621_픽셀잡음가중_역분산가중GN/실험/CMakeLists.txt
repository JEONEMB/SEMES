cmake_minimum_required(VERSION 3.15)
project(pixel_weighted_gn LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(pixel_weighted_gn src/main.cpp)
