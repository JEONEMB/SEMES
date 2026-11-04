cmake_minimum_required(VERSION 3.15)
project(fixed_id_tracking LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(fixed_id_tracking src/main.cpp)
