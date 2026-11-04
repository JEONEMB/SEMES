cmake_minimum_required(VERSION 3.15)
project(v7b_twostage LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(v7b_twostage src/main.cpp)
