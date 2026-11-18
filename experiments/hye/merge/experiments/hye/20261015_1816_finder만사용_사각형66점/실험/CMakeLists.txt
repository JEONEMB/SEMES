cmake_minimum_required(VERSION 3.15)
project(finder_only LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(finder_only src/main.cpp)
