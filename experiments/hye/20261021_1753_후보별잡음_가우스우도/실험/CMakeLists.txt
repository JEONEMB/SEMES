cmake_minimum_required(VERSION 3.15)
project(likelihood_4fold LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(likelihood_4fold src/main.cpp)
