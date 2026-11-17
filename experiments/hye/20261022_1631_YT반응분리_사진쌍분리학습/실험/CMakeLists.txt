cmake_minimum_required(VERSION 3.15)
project(yt_disentangle LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(yt_disentangle src/main.cpp)
