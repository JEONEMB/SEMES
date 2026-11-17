cmake_minimum_required(VERSION 3.15)
project(qr_position_pairs LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(qr_position_pairs src/main.cpp)
