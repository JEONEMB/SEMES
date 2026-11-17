cmake_minimum_required(VERSION 3.15)
project(full280_frozen LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(full280_frozen src/main.cpp)
