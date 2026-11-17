cmake_minimum_required(VERSION 3.15)
project(motion_speed LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(motion_speed src/main.cpp)
