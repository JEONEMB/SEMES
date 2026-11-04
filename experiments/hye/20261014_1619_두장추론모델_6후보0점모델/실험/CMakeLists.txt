cmake_minimum_required(VERSION 3.15)
project(two_image_six_origin LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(two_image_six_origin src/main.cpp)
