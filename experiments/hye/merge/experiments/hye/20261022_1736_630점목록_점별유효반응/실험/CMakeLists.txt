cmake_minimum_required(VERSION 3.15)
project(points630_inventory LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(points630_inventory src/main.cpp)
