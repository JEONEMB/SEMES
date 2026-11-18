cmake_minimum_required(VERSION 3.15)
project(v8_noise_source LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(v8_noise_source src/main.cpp)
