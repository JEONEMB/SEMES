cmake_minimum_required(VERSION 3.15)
project(evaluate_archives LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(evaluate_archives src/main.cpp)
