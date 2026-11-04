cmake_minimum_required(VERSION 3.15)
project(joint_yt LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(joint_yt src/main.cpp)
