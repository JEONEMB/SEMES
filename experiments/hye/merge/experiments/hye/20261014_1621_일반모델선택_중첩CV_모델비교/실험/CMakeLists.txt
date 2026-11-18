cmake_minimum_required(VERSION 3.15)
project(nested_cv_model_selection LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(nested_cv_model_selection src/main.cpp)
