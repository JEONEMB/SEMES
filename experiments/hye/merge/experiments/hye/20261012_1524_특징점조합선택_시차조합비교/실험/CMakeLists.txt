cmake_minimum_required(VERSION 3.15)
project(feature_combo_parallax LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(feature_combo_parallax src/main.cpp)
