cmake_minimum_required(VERSION 3.15)
project(direct_p12_alignment LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(direct_p12_alignment src/main.cpp)
