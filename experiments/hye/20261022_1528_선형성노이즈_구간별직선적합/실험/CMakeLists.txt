cmake_minimum_required(VERSION 3.15)
project(eda_linearity LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(eda_linearity src/main.cpp)
