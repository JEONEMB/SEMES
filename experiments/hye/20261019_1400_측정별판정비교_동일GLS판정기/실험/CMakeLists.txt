cmake_minimum_required(VERSION 3.15)
project(judge_compare LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(judge_compare src/main.cpp)
