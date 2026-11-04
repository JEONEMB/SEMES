cmake_minimum_required(VERSION 3.15)
project(qr3_ecc_regression LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../00001_common/hye_experiment.cmake")
hye_experiment(qr3_ecc_regression src/main.cpp)
