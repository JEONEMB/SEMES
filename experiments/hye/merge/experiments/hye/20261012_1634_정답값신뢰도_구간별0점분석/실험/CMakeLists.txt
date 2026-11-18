cmake_minimum_required(VERSION 3.15)
project(label_reliability LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(label_reliability src/main.cpp)
