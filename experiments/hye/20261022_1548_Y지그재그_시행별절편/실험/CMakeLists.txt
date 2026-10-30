cmake_minimum_required(VERSION 3.15)
project(eda_y_trials LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(eda_y_trials src/main.cpp)
